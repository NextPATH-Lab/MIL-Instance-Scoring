"""Utility functions to visualize CAMELYON16 data.
"""

import os
import json
import logging

import cv2
import pyvips
import napari
import tiffslide as tsl
from pathlib import Path

import numpy as np
import dask.array as da
from sklearn.cluster import DBSCAN

from shapely.ops import unary_union
from shapely.geometry import MultiPoint, Polygon

from ._parser import Parser

logger = logging.getLogger(__name__)

type DirPath = Path
type Coordinates2D = np.ndarray[tuple[int,int], np.dtype[np.uint32]]
type Matrix3D_u8 = np.ndarray[tuple[int,int,int], np.dtype[np.uint8]]

def load_section_annotations(
        viewer: "napari.Viewer",
        annotation_directory: Path,
        root_name: str
) -> None:
    roi_paths = annotation_directory.rglob("*.roi")

    print(annotation_directory)

    parser = Parser()

    # ASPERANZA BRANCH
    with open(Path(__file__).parent / "config.json") as f:
        color_config = json.load(f)['draw-schema']

    for _layername, _enum in color_config["labels"].items():
        # Get config info
        color = color_config["colors"][str(_enum)]

        if not (_layername in viewer.layers):

            layer = viewer.add_shapes(
                name = _layername,
                face_color = [0,0,0,0],
                edge_color = color,
                edge_width = 80,
                opacity = 1.,)

    # Getafics style annotations have ROIs nested within SECs.
    for roi_path in roi_paths:
        roi_layer = roi_path.stem.split("_u")[0].split(f"{root_name}_")[-1]
        roi_polygon = (
            parser.parse_getafics_roi(roi_path, apply_offset = True))

        try:
            viewer.layers[roi_layer].add(
                np.fliplr(np.array(roi_polygon)),
                shape_type = "polygon")
        except:
            logger.warning(f"Layer: {roi_layer} not found.")

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

def get_topmost_visible_image(viewer):
    # Iterate backwards through the layers (top to bottom)
    for layer in reversed(viewer.layers):
        if isinstance(layer, napari.layers.Image) and layer.visible:
            return layer
    return None

####################
def format_coordinates_as_roi(
        polygon_coords: Coordinates2D,
        use_relative_coords: bool = False,
        return_as_strings: bool = False
) -> dict:
    """Creates an ROI format from a set of coordinates of a single polygon.
    Everything is processed in x, y space.

    Keys include 'header', a vector of numerical elements for the header
    of the ROI file and 'pts', a set of coordinate points.

    Args:
        polygon_coords (Coordinates2D): A numpy array of coordinates for a
            SINGLE polygon.

    Returns:
        dict: Data required to write an roi file
    """
    poly = Polygon(polygon_coords)
    minx, miny, maxx, maxy = poly.bounds
    header = [
        5, 0, # Hard-coded elements which I'm not sure what they do
        0, 0, # megaunit x, y - top left corner of WSI
        int(maxx), int(maxy), # megaunit dim x, y - cannot infer from polygon alone
        int(minx), int(miny),# slide unit x, y (offset from top left corner)
        int(maxx - minx), int(maxy - miny), # slide dim x, y (size of bounding box)
        0, 0, 0, 0, 0, 0, 0, # Yet again some hard coded elements
        polygon_coords.shape[0],
        0
    ]

    if use_relative_coords:
        polygon_coords[:, 0] -= int(minx)
        polygon_coords[:, 1] -= int(miny)

    if return_as_strings:
        header = " ".join(list(map(str, header)))
        polygon_coords = polygon_coords.astype(str).tolist()
        polygon_coords = list(map(lambda x: " ".join(x), polygon_coords))
        polygon_coords = "\n".join(polygon_coords).strip("\n")

    roi_data = {
        'header' : header,
        'pts' : polygon_coords
    }

    return roi_data

def save_polygon_coords_as_sections(
        polygons: list[Coordinates2D],
        output_dir: DirPath,
        name: str = "temp"
) -> None:
    """Saves provided polygon coordinates in Getafics annotation format.

    Args:
        polygons (list[Coordinates2D]): A list of NumPy arrays of coordinates
            of polygon vertices.
        output_dir (DirPath): Where the save folder should go.
        name (str): The name of the slide from which polygons are generated.
            Default is 'temp'
    """
    polygons = polygons if isinstance(polygons, list) else [polygons]
    n_pol = len(polygons)

    # Create directory to save things
    save_dir = output_dir / name
    save_dir.mkdir(parents = True, exist_ok = True)
    section_dir = name + "_s%d"

    for i, poly in enumerate(polygons):
        roi_data = format_coordinates_as_roi(poly, False, True)

        # Make the directory
        ith_name = section_dir % i
        (save_dir / ith_name).mkdir(exist_ok = True)

        with open((save_dir / f"{ith_name}/{ith_name}.sec"), "w") as f:
            f.write(roi_data['header'] + "\n")
            f.write(roi_data['pts'])

    return

def adjust_gamma(image_matrix: Matrix3D_u8, gamma: float = 1.0) -> Matrix3D_u8:
    inv_gamma = 1.0 / gamma
    gamma_corrected = (image_matrix/ 255) ** inv_gamma
    image_corrected = np.clip(255 * gamma_corrected, 0, 255).astype('uint8')
    return image_corrected

def get_downsampled_pixel_matrix(
        slide_path,
        level: int = 2,
        override: bool = False,
        gamma: float = 1.0
) -> Matrix3D_u8:
    """
    Downsamples from Level 0 to requested level using streaming.
    level 6 = 4^6 = 4096x downsample.
    """
    with tsl.TiffSlide(slide_path) as tiff:
        num_levels = tiff.level_count
        if (level < num_levels) and not override:
            _dims = tiff.level_dimensions[level]
            logging.info(f"Loading image with {_dims=}")
            im_mx = np.asarray(tiff.read_region((0,0), level, _dims))
            if gamma != 1.0:
                im_mx = adjust_gamma(im_mx, gamma)
            return im_mx
    # 1. 'sequential' access tells pyvips we will only read from start to finish once.
    # This triggers aggressive read-ahead caching and disables expensive seeking.
    image = pyvips.Image.new_from_file(slide_path, access='sequential')

    # 2. Calculate the target pixel width (e.g., Level 6 = 1/4096 scale)
    # We do not use 'resize' because 'thumbnail' is optimized for downsampling.
    scale_factor = 1 / (4**level)
    target_width = int(image.width * scale_factor)
    target_height = int(image.height * scale_factor)
    logging.info(f"{target_width=}, {target_height=}")

    # 3. Generate thumbnail
    # This runs the entire pipeline (load -> shrink -> average) in parallel threads.
    thumb = pyvips.Image.thumbnail(slide_path, target_width, height=10000000)
    # Large height to auto-scale

    # 4. Return as NumPy array
    im_mx = (
        np.frombuffer(
            thumb.write_to_memory(), dtype=np.uint8)
                .reshape(thumb.height, thumb.width, thumb.bands))

    im_mx = adjust_gamma(im_mx, gamma) if gamma != 1.0 else im_mx

    return im_mx

def merge_intersecting_polygons(polygon_list):
    """
    Merges a list of shapely Polygons, combining any that intersect.
    
    Args:
        polygon_list (list): A list of shapely.geometry.Polygon objects.
        
    Returns:
        list: A list of merged shapely.geometry.Polygon objects.
    """
    if not polygon_list:
        return []

    if not isinstance(polygon_list[0], Polygon):
        polygon_list = list(map(Polygon, polygon_list))

    # unary_union handles the complex spatial merging efficiently
    merged_geometry = unary_union(polygon_list)

    # unary_union returns a single Polygon if everything intersected,
    # or a MultiPolygon if there are disjoint groups left over.
    poly2coords = lambda x: np.array(x.exterior.coords)
    if merged_geometry.geom_type == 'Polygon':
        return list(map(poly2coords, [merged_geometry]))
        # return [merged_geometry]
    elif merged_geometry.geom_type == 'MultiPolygon':
        # Extract the individual polygons from the MultiPolygon
        
        return list(map(poly2coords, list(merged_geometry.geoms)))
    else:
        # Fallback for empty or purely invalid inputs
        return []

def auto_section_polygon(
        slide_path,
        level = 2,
        min_area_threshold = 500,
        resolution: float = 0.0005,
        forgivingness: float = 0.1,
        polygon_merge_dist: float = 1000,
        polygon_buffer: int = 0,
        gamma: float = 1.0,
        color_channel: int = -1,
        cluster_merge: bool = False,
        merge_intersecting: bool = False,
        force_downsample_compute: bool = False,
        with_convex_hull: bool = False,
        rescale: bool = True,
        with_image: bool = False,
        as_arrays: bool = False
) -> tuple[list[Coordinates2D, float]]:
    """Find the tissue contours on a downsampled image.

    Args:
        slide_path (str, Path): Path to histology slide.
        level (int, optional): Downsample level. Defaults to 2.
        min_area_threshold (int, optional): Minimum area of polygon in
            downsampled space. Smaller objects are removed. Defaults to 500.
        resolution (float, optional): Resolution of contour; smaller values
            give higher resolution. Defaults to 0.0005.
        forgivingness (float, optional): How forgiving the Otsu thresholding
            is when finding tissue regions. Defaults to 0.1.
        polygon_merge_dist (float, optional): The distance between points on
            different polygons to merge if they are closer than this value.
            This value is compared in the FINAL rescaled coordinate space
            if it is rescaled.
        polygon_buffer (float, optional): How much to dilate (or erode) the
            boundaries of the detected polygon, in rescaled space.

    Returns:
        tuple[Coordinates2D, float]: Returns the coordinates of the polygons
            in a list, and the scale factor to rescale the coordinates back
            to full resolution.
    """
    # 1. Get the image using the pyvips helper
    # We pass the path directly because pyvips handles its own file opening
    img = get_downsampled_pixel_matrix(
        slide_path, level = level,
        override = force_downsample_compute, gamma = gamma)
    logging.info("Retrieved downsampled image.")

    # 1.1 Check if specific color channels are selected
    ### TODO: IMPLMENET

    # 2. Extract Saturation channel from HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # 3. Thresholding (Standard OpenCV)
    ret, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(
        saturation, ret * (1 - forgivingness), 255, cv2.THRESH_BINARY)
    logging.info("Thresholded.")

    # 4. Cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    logging.info("Cleaned.")

    # 5. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info("Contours found.")

    # 6. Process and scale coordinates
    # CRITICAL: Open the slide briefly just to get the Level 0 dimensions
    # This ensures your scale_factor is exactly correct for the coordinate mapping
    with tsl.TiffSlide(slide_path) as slide:
        full_width = slide.dimensions[0]
        full_height = slide.dimensions[1]

    scale_factor = full_width / img.shape[1]

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_threshold:
            epsilon = resolution * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            coords = np.fliplr(approx.reshape(-1,2).astype(int))

            # Rescale to Level 0
            if rescale:
                coords = (coords * scale_factor).astype(int)

            polygon_buffer = max(0, polygon_buffer) # Ensure > 0
            # .buffer(0) fixes self intersections but can create MultiPolygon
            coords = Polygon(coords).buffer(0).buffer(polygon_buffer)

            if coords.geom_type == 'Polygon':
                geoms = [coords]
            elif coords.geom_type == "MultiPolygon":
                geoms = list(coords.geoms)
            else:
                continue

            for _geom in geoms:
                coords = np.array(_geom.exterior.coords).astype(int)
                coords[:,0] = np.clip(coords[:,0], 0, full_height)
                coords[:,1] = np.clip(coords[:,1], 0, full_width)

                if with_convex_hull:
                    hull = MultiPoint(coords).convex_hull
                    if hull.geom_type == 'Polygon':
                        coords = np.array(hull.exterior.coords).astype(int)
                    else:
                        coords = np.array(hull.coords).astype(int)
                coords = coords.tolist() if not as_arrays else coords
                polygons.append(coords)

    if merge_intersecting:
        polygons = merge_intersecting_polygons(polygons)

    if len(polygons) == 0:
        print("No polygons were found.")
    else:
        logging.info(f"{len(polygons)} polygons found.")

    if not polygons is None and len(polygons) > 1 and cluster_merge:
        poly_coords = np.concatenate(polygons, axis = 0)
        dbscan = DBSCAN(eps = polygon_merge_dist)
        labels = dbscan.fit_predict(poly_coords)

        # Organize points by their cluster ID
        polygons = [
            MultiPoint(poly_coords[labels == i]).convex_hull.exterior.coords
            for i in range(labels.max() + 1)
        ]
        polygons = list(map(np.uint32, polygons))

    result = (polygons, scale_factor)

    return result + (img,) if with_image else result
