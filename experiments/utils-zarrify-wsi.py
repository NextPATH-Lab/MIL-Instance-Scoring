"""Convert WSI in pyramid structure to .zarr of patches.
Data schema:

Zarr V3 store containing Whole Slide Image (WSI) patches and metadata.

The Zarr store is organized as a single root directory/group containing global
slide-level attributes and up to four arrays storing the image data and their
corresponding level-0 spatial coordinates.

Attributes:
    n_tiles (int): Total number of patches extracted (tumor + benign).
    n_tumor_tiles (int): Total number of tumor patches.
    n_benign_tiles (int): Total number of benign patches.
    tumor_fraction (float): Ratio of tumor patches to total patches 
        (n_tumor_tiles / n_tiles). Returns 0.0 if no patches exist.
    slide_label (int): 1 if a tumor annotation file exists for the slide, 0 otherwise.
    coord_format (str): Fixed string denoting the bounding box format: 
        "x0, y0, x1, y1 (level-0 pixel coordinates)".

Arrays:
    tumor:
        Description: RGB image patches intersecting with tumor annotations.
        Shape: (n_tumor_tiles, tile_size, tile_size, 3)
        Dtype: uint8
        Chunks: (1, tile_size, tile_size, 3)  # Logical access boundary
        Shards: (shard_size, tile_size, tile_size, 3)  # Physical file boundary
        Compressor: BloscCodec(cname='lz4', clevel=3, shuffle="bitshuffle")

    benign:
        Description: RGB image patches from regions outside tumor annotations.
        Shape: (n_benign_tiles, tile_size, tile_size, 3)
        Dtype: uint8
        Chunks: (1, tile_size, tile_size, 3)
        Shards: (shard_size, tile_size, tile_size, 3)
        Compressor: BloscCodec(cname='lz4', clevel=3, shuffle="bitshuffle")

    tumor_coords:
        Description: Bounding box coordinates [x0, y0, x1, y1] for each tumor patch. 
            Row indices perfectly correspond to the `tumor` array indices.
        Shape: (n_tumor_tiles, 4)
        Dtype: int64
        Chunks: (shard_size, 4)
        Shards: (shard_size, 4)  # Inherited logically from chunk behavior in Zarr V3 coords

    benign_coords:
        Description: Bounding box coordinates [x0, y0, x1, y1] for each benign patch. 
            Row indices perfectly correspond to the `benign` array indices.
        Shape: (n_benign_tiles, 4)
        Dtype: int64
        Chunks: (shard_size, 4)
        Shards: (shard_size, 4)

"""
import os, logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override = True)

import numpy as np
import zarr
from zarr.codecs import BloscCodec  # Explicit Zarr V3 Codecs
import openslide
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon, MultiPolygon, box

from src._parser import Parser, BaseReader

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

def extract_patches(
        wsi_path: Path,
        tile_size: int,
        overlap: int,
        shard_size: int = 225
) -> None:
    parser = Parser()
    reader = BaseReader()
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

    n_tumor = len(tumor_rois)
    n_benign = len(benign_rois)
    n_patches = n_tumor + n_benign
    n_max = max(n_tumor, n_benign)
    if n_patches == 0:
        log.warning("No patches found for %s. Skipping.", wsi_path.stem)
        return

    # Setup Zarr V3
    zarr_dir_path = wsi_path.parent / f"{wsi_path.stem}.zarr"
    root_grp = (
        zarr.create_group(
            store=str(zarr_dir_path), zarr_format=3, overwrite=True))
    
    image_compressor = BloscCodec(cname='lz4', clevel=3, shuffle="bitshuffle")

    # Helper to create patch arrays consistently
    def create_patch_arr(name, count):
        return root_grp.create_array(
            name=name,
            shape=(count, tile_size, tile_size, 3),
            dtype='uint8',
            shards=(shard_size, tile_size, tile_size, 3), # The physical file boundary
            chunks=(1, tile_size, tile_size, 3),          # The logical access boundary
            compressors=image_compressor
        )

    # Helper to create coordinate arrays: one row per tile -> (x0, y0, x1, y1)
    def create_coord_arr(name, count):
        return root_grp.create_array(
            name=name,
            shape=(count, 4),
            dtype='int64',
            chunks=(shard_size, 4),
        )

    tumor_arr = create_patch_arr("tumor", n_tumor)
    benign_arr = create_patch_arr("benign", n_benign)

    tumor_coord_arr = create_coord_arr("tumor_coords", n_tumor)
    benign_coord_arr = create_coord_arr("benign_coords", n_benign)

    # Coordinates are already known up front (they're just the ROI slices),
    # so write them in one shot. Order matches the image arrays because both
    # come from the same tumor_rois / benign_rois lists, in the same order.
    if n_tumor:
        tumor_coord_arr[:] = np.array(tumor_rois, dtype=np.int64)
    if n_benign:
        benign_coord_arr[:] = np.array(benign_rois, dtype=np.int64)

    cursor = 0
    with openslide.OpenSlide(str(wsi_path)) as slide:
        def read_patch(s):
            # OpenSlide is thread-safe for reading from a single object in Python
            return np.asarray(
                slide.read_region(
                    (s[0], s[1]), 0, (tile_size, tile_size)).convert("RGB"))

        with ThreadPoolExecutor(max_workers=32) as executor:
            # Combined processing to keep the console output clean
            zarr_iterable = [
                ("Tumor", tumor_rois, tumor_arr),
                ("Benign", benign_rois, benign_arr)
            ]
            for label, rois, z_arr in zarr_iterable:
                if not rois: continue
                
                for i in range(0, len(rois), shard_size):
                    batch = rois[i : i + shard_size]
                    # Map the read_patch function
                    images = list(executor.map(read_patch, batch))
                    
                    # Stack and write
                    z_arr[i : i + len(images)] = np.stack(images)
                    cursor += len(images)
                    print(f"[{wsi_path.stem}] {cursor} / {n_max}", end = "\r")
                    
    # Update attributes
    root_grp.attrs.update({
        "n_tiles" : n_patches,
        "n_tumor_tiles": n_tumor,
        "n_benign_tiles": n_benign,
        "tumor_fraction": n_tumor / n_patches if n_patches > 0 else 0,
        "slide_label": int(is_tumor),
        "coord_format": "x0, y0, x1, y1 (level-0 pixel coordinates)"
    })
    print()
    return

if __name__ == "__main__":

    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--directory",
        type=Path,
        help="Root directory of images to zarrify.")
    parser.add_argument(
        "-i", "--indices",
        nargs="*", default=None, type=int,
        help="Indices within directory to process.")
    parser.add_argument(
        "-s", "--size",
        type=int, default=256, help="Tile size")
    parser.add_argument(
        "-o", "--overlap-size",
        type=int, default=32, help="Size of tile overlaps.")

    args = parser.parse_args()
    file_list = sorted(list(args.directory.glob("*.tif*")))
    if args.indices is None:
        indices = list(range(len(file_list)))
    else:
        start_idx = max(0, args.indices[0])
        end_idx = min(args.indices[1], len(file_list))
        indices = list(range(start_idx, end_idx))
    
    log.info(
        "Got arguments: %d files, indices=%s, tile_size=%d, overlap=%d",
        len(file_list), indices, args.size, args.overlap_size
    )

    for idx in indices:
        start = time.time()
        extract_patches(file_list[idx], args.size, args.overlap_size)
        end = time.time()
        dur = (end - start) / 60
        log.info("%s took %.3f minutes.", file_list[idx], dur)

        # Delete file to save space?
        # file_list[idx].unlink()