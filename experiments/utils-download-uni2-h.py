"""Script to download the UNI2-H weights from huggingface.
Written by Claude - I believe it requires a huggingface token in .env file
"""

import os
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override = True)

import timm
import torch
from huggingface_hub import login

LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log",
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

def main():
    log.info("Loading HuggingFace token from .env...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_TOKEN")

    if not hf_token:
        raise ValueError("Could not find HF_READ_TOKEN in .env file!")

    log.info("Logging into HuggingFace...")
    login(token=hf_token)

    log.info("Downloading MahmoodLab/UNI2-h weights to local cache (~/.cache/huggingface/hub)...")
    
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
    
    # This will trigger the download and save it to the cache directory
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    
    log.info("Download complete! The compute nodes will now use the cached weights.")

if __name__ == "__main__":
    main()
