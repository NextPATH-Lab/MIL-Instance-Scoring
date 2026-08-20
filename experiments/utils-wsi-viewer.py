"""Script to visualize CAMELYON16 (or any pyramidal .svs/.tif*).
This viewer has the added functionality of being able to read in 
annotation polygons made in `utils-autodetect-tissue-wsi.py`.
"""
import ast
import os
import json
import logging
from typing import Literal, Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override = True)

import napari
from napari.utils.notifications import show_info

import tiffslide as tsl
from magicgui import magicgui

import numpy as np
import dask.array as da

from src._parser import Parser
from src.utils import generate_discrete_colors

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

def parse_image_description(img_desc: str) -> dict:
    # 'Motic V1.0.0\r\n30347x47657 [0,0 30347x47657] [512x512] JPEG/RGB Q = 75|AppMag = 40|MPP = 0.261780'
    if img_desc.startswith("Motic"):
        components = img_desc.split("|")
        mag, res = list(map(lambda x: float(x.split("= ")[-1]), components[-2:]))
        result = {'objective' : mag, 'resolution' : res}
    else:
        result = {'objective' : 20, 'resolution' : 0.5}

    return result

def parse_color_config(json_file):
    """This JSON file MUST be configured specifically as follows.
    {
        "properties" : {
            "gleason_score" : ["section", "3", "4", "5"]
        },
        "color_cycle" : {
            "section" : "[0,0,0,1]",
            "3" : "[128, 235, 52, 1]",
            "4" : "[235, 192, 52, 1]",
            "5" : "[235, 67, 52, 1]"
        }
    }
    The 'properties' key does not matter. This is just to explain what 
    the colors represent. It is not used in the code. HOWEVER, the code
    expects this level of nesting.

    The "color_cycle" key MUST be named "color_cycle". The rest of the keys 
    should be numerical characters (str type) of numbers 0-9. The values 
    should be lists of integers in RGBA format. ONLY the alpha value (last
    number) should be 1 (for full opacity) or 0 (for transluscent). The color 
    values (RGB) MUST be in the [0,256) range.

    Args:
        json_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(json_file, "r") as f:
        config = json.load(f)
    for k, v in config['color_cycle'].items():
        log.debug("color_cycle[%s] = %s", k, v)
        config['color_cycle'][k] = ast.literal_eval(v)

    return config

def load_section_annotations(
        viewer: "napari.Viewer",
        annotation_directory: Path,
        color_rois_by: Literal["section", "custom"] = "section",
        color_cycle_json: Optional[Path] = None,
) -> None:
    """
    {
        "properties" : {
            "property_name" : ["value1", "value2"...]
        },
        "color_cycle" : {
            "value1" : "color1",
            "value2" : "color2"
        }
    }
    """
    section_paths = list(annotation_directory.rglob("*.sec"))
    asap_paths = annotation_directory.with_suffix(".xml")

    parser = Parser()

    if asap_paths.exists():
        asap_rois = parser.parse_asap_xml(asap_paths)
        viewer.add_shapes(
            asap_rois,
            shape_type = "polygon",
            name = "metastatic",
            edge_color = "black",
            edge_width = 10,
            face_color = 'cyan')

    # Hardcoding to limit at 10 colors for now.
    colors = generate_discrete_colors(10, "tab10") / 255
    colors = np.concatenate([colors, np.ones((10,1))], axis = 1)
    if color_cycle_json is None or color_rois_by == "section":
        color_schema = {
            str(i) : colors[i] for i in range(10)
        }
    else:
        color_schema = parse_color_config(color_cycle_json)["color_cycle"]
    show_info(f"Color keys: {list(color_schema.keys())}")
    for i, section_path in enumerate(section_paths):
        edge_color_array = []
        # Metadata
        # This is the naive approach of getting section names
        # TODO: Replace with proper regex
        section_name = section_path.stem.split("_")[-1]

        sec_annotations = []
        sec_polygon = (
            parser.parse_getafics_roi(section_path, apply_offset = False))

        sec_annotations.append(np.fliplr(np.array(sec_polygon)))
        section_color = (
            np.array([0,0,0,1]) if color_rois_by == "custom"
            else color_schema[str(i)])
        edge_color_array.append(section_color)

        # Getafics style annotations have ROIs nested within SECs.
        roi_paths = section_path.parent.rglob("*.roi")
        for roi_path in roi_paths:
            roi_polygon = (
                parser.parse_getafics_roi(roi_path, apply_offset = True))

            if color_rois_by == "custom":
                with open(roi_path, "r") as f:
                    header = f.readlines()[0].split(" ")
                    _color = color_schema[header[1]]
                    edge_color_array.append(_color)

            sec_annotations.append(np.fliplr(np.array(roi_polygon)))

        shapes_kwargs = {
            "shape_type" : "polygon",
            "name" : section_name,
            "opacity": 1.,
            "face_color" : [0,0,0,0],
            "edge_width" : 40,
        }
        if color_rois_by == "section":
            shapes_kwargs["edge_color"] = color_schema[str(i)]
        else:
            _edge_colors = np.stack(edge_color_array, axis = 0).astype('float32')
            _edge_colors[:, :3] /= 255
            shapes_kwargs["edge_color"] = _edge_colors

        viewer.add_shapes(sec_annotations, **shapes_kwargs)


@magicgui(
        call_button = "Open WSI",)
def load_big_wsi(
        viewer: "napari.Viewer",
        path: Path,
        with_annotations: bool = True,
        clear_workspace: bool = True,
        color_rois_by: Literal["custom (select config file)", "section"] = "section",
        custom_color_config: Optional[Path] = None,
) -> None:
    if clear_workspace:
        viewer.layers.clear()
    # Get file data
    name = Path(path).stem
    os.environ["NAPARI_ASYNC"] = "1"

    # 2. Open the slide lazily
    slide = tsl.TiffSlide(str(path))
    properties = slide.properties
    properties['width.manual'] = slide.dimensions[0]
    properties['height.manual'] = slide.dimensions[1]

    # Get metadata
    metadata = {
        'path' : path,
        'tsl.properties' : properties,
        'tiff.ImageDescription' : properties['tiff.ImageDescription']
    }

    # 3. Create the multiscale levels (the "Progressive" steps)
    # tiffslide.zarr_group makes this easy; it looks like a list of images.
    multiscale_data = [
        da.from_array(
            slide.zarr_group[str(level)],
            chunks = slide.zarr_group[str(level)].chunks)
        for level in range(len(slide.level_dimensions))
    ]

    # 5. Add the image with these specific flags:
    layer = viewer.add_image(
        multiscale_data,
        multiscale=True,          # Uses the pre-made levels for zoom
        contrast_limits=[0, 255], # CRITICAL: Skips the "scan whole file" step
        name=name,
        metadata = metadata)

    # 6. Load any Annotations made
    if with_annotations:
        anno_path = Path(path).parent / name
        anno_path2 = Path(path).parent.parent / f"annotations/{name}.xml"
        color_rois_by = "custom" if color_rois_by == "custom (select config file)" else color_rois_by
        load_section_annotations(viewer, anno_path, color_rois_by, custom_color_config)
        load_section_annotations(viewer, anno_path2)

    return

@magicgui(call_button="Open WSI")
def load_big_wsi_v2(
    viewer: "napari.Viewer",
    path: Path,
    with_annotations: bool = False
) -> None:
    path_str = str(path)
    name = Path(path).stem
    os.environ["NAPARI_ASYNC"] = "1"

    # 1. Open slide safely to grab metadata
    slide = tsl.TiffSlide(path_str)
    properties = dict(slide.properties)
    properties['width.manual'] = slide.dimensions[0]
    properties['height.manual'] = slide.dimensions[1]
    properties['tiff.ImageDescription'] = properties.get('tiff.ImageDescription', 'No description')

    metadata = {
        'path': path,
        'tsl.properties': properties,
        'tiff.ImageDescription': properties['tiff.ImageDescription']
    }

    # 2. Build a bulletproof Dask pyramid using map_blocks
    multiscale_data = []
    chunk_size = 1024

    for level_idx in range(len(slide.level_dimensions)):
        w, h = slide.level_dimensions[level_idx]
        downsample = slide.level_downsamples[level_idx]

        # Early binding of loop variables ensures correct execution inside Dask workers
        def read_chunk(block_id, path=path_str, lvl=level_idx, ds=downsample, w_lvl=w, h_lvl=h):
            y_idx, x_idx, _ = block_id
            
            y_start = y_idx * chunk_size
            x_start = x_idx * chunk_size
            
            # Handle edge chunks seamlessly
            chunk_h = min(chunk_size, h_lvl - y_start)
            chunk_w = min(chunk_size, w_lvl - x_start)
            
            if chunk_w <= 0 or chunk_h <= 0:
                return np.zeros((0, 0, 3), dtype=np.uint8)
                
            # tsl.read_region requires coordinates mapped to Level 0
            lvl0_x = int(x_start * ds)
            lvl0_y = int(y_start * ds)
            
            # Re-instantiating inside the worker makes it thread-safe
            s = tsl.TiffSlide(path)
            img = s.read_region((lvl0_x, lvl0_y), lvl, (chunk_w, chunk_h))
            return np.array(img.convert("RGB"))

        # Explicitly calculate Dask chunk sizes (required for Napari shape validation)
        y_chunks = tuple(chunk_size if i < h // chunk_size else h % chunk_size for i in range(h // chunk_size + (1 if h % chunk_size else 0)))
        x_chunks = tuple(chunk_size if i < w // chunk_size else w % chunk_size for i in range(w // chunk_size + (1 if w % chunk_size else 0)))

        if not y_chunks or not x_chunks:
            continue

        # Map the read_chunk function across the explicit grid
        level_da = da.map_blocks(
            read_chunk,
            chunks=(y_chunks, x_chunks, (3,)), # The (3,) strictly defines the RGB channels
            dtype=np.uint8
        )
        multiscale_data.append(level_da)

    # 3. Add to viewer
    viewer.add_image(
        multiscale_data,
        multiscale=True,          
        contrast_limits=[0, 255], # CRITICAL: Skips the "scan whole file" step
        name=name,
        metadata=metadata
    )

    # 4. Safely load Annotations
    if with_annotations:
        anno_path = Path(path).parent / name
        anno_path2 = Path(path).parent.parent / f"annotations/{name}.xml"
        
        # Ensures the function exists before calling
        if 'load_section_annotations' in globals():
            if anno_path.exists() or anno_path.with_suffix('.xml').exists():
                load_section_annotations(viewer, anno_path)
            if anno_path2.exists():
                load_section_annotations(viewer, anno_path2)

    return

if __name__ == "__main__":
    # for testing: open a dummy image
    viewer = napari.Viewer(title = f"Napari WSI Viewer")

    # add the widget to napari
    loader = viewer.window.add_dock_widget(load_big_wsi, name = "Big WSI Loader", area = 'right', tabify = False)

    napari.run()