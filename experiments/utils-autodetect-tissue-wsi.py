"""Script to autodetect tissue on WSIs and save ROI coordinates.
"""

import os, time, logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override = True)

from tqdm import tqdm

import numpy as np
from src.napari_utils import (
    auto_section_polygon,
    save_polygon_coords_as_sections
)

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

def save_auto_annotation(wsi_path: Path, level: int = 4) -> None:
    """Extract regions of the WSI with tissue, and extract tiles as file
    """

    tissue_area_polygons, _ = (
        auto_section_polygon(
            wsi_path,
            level = level,
            forgivingness = 0.5,
            polygon_buffer = 0,
            min_area_threshold = 1_000,
            merge_intersecting = True,
            with_convex_hull = False,
            as_arrays = True,
            rescale = True))

    tissue_area_polygons = list(map(np.fliplr, tissue_area_polygons))

    save_polygon_coords_as_sections(
        tissue_area_polygons, wsi_path.parent, wsi_path.stem)

    return

def annotate_wsi_dataset(
        directory: Path,
        file_ext: str = ".tif",
        level: int = 3
) -> None:
    """The main function. Run annotation/auto-detection on batch of WSI.
    """
    to_process = list(directory.glob(file_ext))
    log.info("Found %d files to process in %s", len(to_process), directory)

    for f in tqdm(to_process):
        start = time.time()
        save_auto_annotation(f, level = level)
        end = time.time()
        log.info("%s took %.3f minutes.", f.name, (end - start) / 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type = Path, help = "Folder of files to process.")
    parser.add_argument("-t", "--type", type = str, help = "File extension starting with '.' - '.tiff' for example.")
    parser.add_argument("-l", "--level", type = int, default = 3, help = "Number of levels to downsample.")

    args = parser.parse_args()

    annotate_wsi_dataset(args.directory, args.type, args.level)