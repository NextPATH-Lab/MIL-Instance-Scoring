"""Compute UNI2-H embeddings for WSI patches saved as zarr.

Check utils-zarrify-wsi.py for details on data spec.
Written with help of Claude.

"""

import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import zarr
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(override = True)

import timm
import torch
from torch.amp import autocast
import torch.distributed as dist

from torchvision import transforms

LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)

log = logging.getLogger(__name__)
_now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")

def get_transforms():
    # UNI requires standard ImageNet preprocessing and 224x224 input
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

def extract_features(model, zarr_path, transform, device, batch_size=256):
    root = zarr.open(str(zarr_path), mode="r")

    features = []
    labels = []
    coords = []

    # Process benign patches (label 0)
    if "benign" in root:
        arr = root["benign"]
        coord_arr = root["benign_coords"]
        n_patches = arr.shape[0]

        msg = "Benign Tile Inference"
        total = n_patches // batch_size + 1
        for i in tqdm(range(0, n_patches, batch_size), msg, total = total):
            chunk_np = arr[i : i + batch_size] # (B, 256, 256, 3) uint8
            # chunk_np is (H, W, C). ToPILImage expects (H, W, C) numpy arrays correctly.
            chunk_t = [transform(chunk_np[j]) for j in range(len(chunk_np))]
            batch_t = torch.stack(chunk_t).to(device, non_blocking=True)

            with autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.float16):
                with torch.no_grad():
                    feats = model(batch_t)
            features.append(feats.cpu())
            labels.extend([0] * len(batch_t))
            coords.append(coord_arr[i : i + batch_size])

    # Process tumor patches (label 1)
    if "tumor" in root:
        arr = root["tumor"]
        coord_arr = root["tumor_coords"]
        n_patches = arr.shape[0]

        msg = "Tumor Tile Inference"
        total = n_patches // batch_size + 1
        for i in tqdm(range(0, n_patches, batch_size), msg, total = total):
            chunk_np = arr[i : i + batch_size]
            chunk_t = [transform(chunk_np[j]) for j in range(len(chunk_np))]
            batch_t = torch.stack(chunk_t).to(device, non_blocking=True)

            with autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.float16):
                with torch.no_grad():
                    feats = model(batch_t)
            features.append(feats.cpu())
            labels.extend([1] * len(batch_t))
            coords.append(coord_arr[i : i + batch_size])

    if not features:
        return None, None, None

    all_features = torch.cat(features, dim=0)
    all_labels = torch.tensor(labels, dtype=torch.long)
    all_coords = torch.tensor(np.concatenate(coords, axis=0), dtype=torch.long)
    return all_features, all_labels, all_coords

def main():
    parser = argparse.ArgumentParser(description="Extract UNI Features from Zarr slides")
    parser.add_argument("-d", "--dataset-dir", type=Path, required=True, help="Directory containing .zarr slide folders")
    parser.add_argument("-s", "--save-dir", type=Path, required=True, help="Directory to save extracted .pt files")
    parser.add_argument("-b", "--batch-size", type=int, default=225, help="Batch size for UNI extraction")
    args = parser.parse_args()
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure HuggingFace Hub does not attempt network requests on air-gapped compute nodes
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # DDP Initialization
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Each rank gets its own log file to avoid clobbering under DDP.
    logging.basicConfig(
        level = logging.INFO,
        filename = LOG_DIR / f"{_now} {Path(__file__).stem}_rank{global_rank}.log",
        format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    if global_rank == 0:
        log.info("Instantiating UNI2-h model architecture (offline mode)...")
        
    # UNI2-h requires specific architectural parameters
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    # Instantiate architecture locally without pinging HF
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)
    
    if global_rank == 0:
        log.info("Loading cached weights locally...")
        
    from huggingface_hub import hf_hub_download
    
    try:
        # First try safetensors
        weight_path = hf_hub_download(repo_id="MahmoodLab/UNI2-h", filename="model.safetensors", local_files_only=True)
        import safetensors.torch
        state_dict = safetensors.torch.load_file(weight_path)
    except Exception:
        # Fallback to pytorch_model.bin
        weight_path = hf_hub_download(repo_id="MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_files_only=True)
        state_dict = torch.load(weight_path, map_location="cpu")
        
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    model.to(device)
    
    transform = get_transforms()
    
    # Discover Zarr folders
    zarr_list = sorted([x for x in args.dataset_dir.glob("*.zarr")])
    if not zarr_list:
        raise ValueError(f"No .zarr files found in {args.dataset_dir}")
        
    # Split slides across GPUs
    my_zarrs = zarr_list[global_rank::world_size]
    
    if global_rank == 0:
        log.info("Found %d slides. Extracting features into %s", len(zarr_list), args.save_dir)
        
    start_time = time.time()
    for idx, z_path in enumerate(my_zarrs):
        out_path = args.save_dir / f"{z_path.stem}.pt"
        if out_path.exists():
            continue
            
        feats, labels, coords = extract_features(model, z_path, transform, device, args.batch_size)

        if feats is not None:
            torch.save({
                "features": feats,
                "labels": labels,
                "coords": coords
            }, out_path)
            
        log.info("[Rank %d] Processed %d/%d slides: %s", global_rank, idx + 1, len(my_zarrs), z_path.name)

    if is_ddp:
        dist.barrier()

    if global_rank == 0:
        elapsed = time.time() - start_time
        log.info("Extraction complete! Took %.1f minutes.", elapsed / 60)
        
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
