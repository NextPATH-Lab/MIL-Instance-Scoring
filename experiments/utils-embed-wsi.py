"""Merged pipeline: WSI -> UNI2-H patch embeddings, with no Zarr persisted.

Supersedes running utils-zarrify-wsi.py followed by utils-uni2-embedding.py.
Tiles are classified tumor/benign the same way as utils-zarrify-wsi.py (see
that file for the data-schema background), but each batch of patches is read
from the slide, embedded, and discarded immediately rather than written to a
Zarr store first -- avoiding the disk round-trip (and the tens of GB of raw
pixels a large slide's patches would take up) since only the embeddings are
ever needed downstream.

Output: one {features, labels, coords} .pt file per slide, saved the same
way and with the same schema as utils-uni2-embedding.py produced.
"""

import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override = True)

import numpy as np
from tqdm import tqdm

import timm
import torch
from torch.amp import autocast
import torch.distributed as dist
from torchvision import transforms

import openslide
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon, MultiPolygon, box

from src._parser import Parser, BaseReader

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

def classify_tile_rois(
        wsi_path: Path,
        tile_size: int,
        overlap: int,
        parser: Parser,
        reader: BaseReader
) -> tuple[list, list]:
    """Slice a slide's tissue ROI(s) into tiles and split them tumor/benign.

    Same classification logic as extract_patches in utils-zarrify-wsi.py,
    stopping short of reading any pixel data.
    """
    annotation_files = wsi_path.parent.rglob(f"*{wsi_path.stem}*.sec")
    tumor_annotation_file = (
        wsi_path.parent.parent / f"annotations/{wsi_path.stem}.xml")

    if (is_tumor := tumor_annotation_file.exists()):
        tumor_regions = parser.parse_asap_xml(tumor_annotation_file)
        tumor_regions = MultiPolygon(list(map(Polygon, tumor_regions)))

    tumor_rois = []
    benign_rois = []
    for f in annotation_files:
        roi_coords = parser.parse_getafics_roi(f, False)
        slices = reader._get_roi_tile_slices(roi_coords, tile_size, overlap)
        for s in slices:
            tile = box(*s)
            if is_tumor and tumor_regions.intersects(tile):
                tumor_rois.append(s)
            else:
                benign_rois.append(s)

    return tumor_rois, benign_rois

def embed_slide(
        model,
        wsi_path: Path,
        transform,
        device,
        tile_size: int,
        overlap: int,
        batch_size: int,
        parser: Parser,
        reader: BaseReader
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    tumor_rois, benign_rois = classify_tile_rois(
        wsi_path, tile_size, overlap, parser, reader)

    if len(tumor_rois) + len(benign_rois) == 0:
        log.warning("No patches found for %s. Skipping.", wsi_path.stem)
        return None, None, None

    features = []
    labels = []
    coords = []

    with openslide.OpenSlide(str(wsi_path)) as slide:
        def read_patch(s):
            # OpenSlide is thread-safe for reading from a single object in Python
            return np.asarray(
                slide.read_region(
                    (s[0], s[1]), 0, (tile_size, tile_size)).convert("RGB"))

        with ThreadPoolExecutor(max_workers=32) as executor:
            roi_groups = [
                ("Benign", benign_rois, 0),
                ("Tumor", tumor_rois, 1),
            ]
            for label_name, rois, label_val in roi_groups:
                if not rois: continue

                msg = f"[{wsi_path.stem}] {label_name} Tile Embedding"
                for i in tqdm(range(0, len(rois), batch_size), msg):
                    batch = rois[i : i + batch_size]
                    images = list(executor.map(read_patch, batch))
                    chunk_t = [transform(im) for im in images]
                    batch_t = torch.stack(chunk_t).to(device, non_blocking=True)

                    with autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.float16):
                        with torch.no_grad():
                            feats = model(batch_t)

                    features.append(feats.cpu())
                    labels.extend([label_val] * len(batch))
                    coords.append(np.array(batch, dtype=np.int64))

    all_features = torch.cat(features, dim=0)
    all_labels = torch.tensor(labels, dtype=torch.long)
    all_coords = torch.tensor(np.concatenate(coords, axis=0), dtype=torch.long)
    return all_features, all_labels, all_coords

def main():
    parser_cli = argparse.ArgumentParser(description="Zarrify + Extract UNI2 Features from WSIs in one pass")
    parser_cli.add_argument("-d", "--directory", type=Path, required=True, help="Directory of WSI (+ .sec ROI / annotations) to embed")
    parser_cli.add_argument("-s", "--save-dir", type=Path, required=True, help="Directory to save extracted .pt files")
    parser_cli.add_argument(
        "-i", "--indices",
        nargs="*", default=None, type=int,
        help="[start, end) range of indices within directory to process.")
    parser_cli.add_argument("-t", "--tile-size", type=int, default=256, help="Tile size")
    parser_cli.add_argument("-o", "--overlap-size", type=int, default=32, help="Size of tile overlaps.")
    parser_cli.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size for UNI extraction")
    args = parser_cli.parse_args()

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
    parser = Parser()
    reader = BaseReader()

    # Discover WSI files
    file_list = sorted(list(args.directory.glob("*.tif*")))
    if not file_list:
        raise ValueError(f"No WSI files found in {args.directory}")

    if args.indices is None:
        indices = list(range(len(file_list)))
    else:
        start_idx = max(0, args.indices[0])
        end_idx = min(args.indices[1], len(file_list))
        indices = list(range(start_idx, end_idx))

    # Split slides across GPUs
    my_indices = indices[global_rank::world_size]

    if global_rank == 0:
        log.info(
            "Got arguments: %d files, indices=%s, tile_size=%d, overlap=%d, batch_size=%d",
            len(file_list), indices, args.tile_size, args.overlap_size, args.batch_size
        )
        log.info("Found %d slides. Extracting features into %s", len(indices), args.save_dir)

    start_time = time.time()
    for count, idx in enumerate(my_indices):
        wsi_path = file_list[idx]
        out_path = args.save_dir / f"{wsi_path.stem}.pt"
        if out_path.exists():
            continue

        slide_start = time.time()
        feats, labels, coords = embed_slide(
            model, wsi_path, transform, device,
            args.tile_size, args.overlap_size, args.batch_size,
            parser, reader
        )

        if feats is not None:
            torch.save({
                "features": feats,
                "labels": labels,
                "coords": coords
            }, out_path)

        dur = (time.time() - slide_start) / 60
        log.info(
            "[Rank %d] Processed %d/%d slides: %s (%.3f minutes, %d tumor / %d benign)",
            global_rank, count + 1, len(my_indices), wsi_path.name, dur,
            int((labels == 1).sum()) if labels is not None else 0,
            int((labels == 0).sum()) if labels is not None else 0,
        )

    if is_ddp:
        dist.barrier()

    if global_rank == 0:
        elapsed = time.time() - start_time
        log.info("Extraction complete! Took %.1f minutes.", elapsed / 60)

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
