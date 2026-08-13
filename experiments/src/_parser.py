"""Utility class for parsing region of interest data.
"""
import yaml
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

from types import NoneType
from typing import Literal, Union, Optional, Tuple

import numpy as np
from shapely import geometry as sg
from shapely.errors import GEOSException
from shapely.validation import explain_validity

from PIL import Image, ImageDraw

type FilePath = Path
type Matrix2D_u32 = np.ndarray[Tuple[int,int], np.dtype[np.uint32]]
type Matrix3D_u8 = np.ndarray[Tuple[int,int,int], np.dtype[np.uint8]]
type Coordinates2D = Matrix2D_u32


class BaseReader:
    def __init__(self):
        self.x_offset = 0
        self.y_offset = 0
        # self.imread = None # To symbol unspecialized read function

    def _repair_geometry(self, geom):
        """Best-effort repair for invalid geometries before boolean ops."""
        if geom.is_valid:
            return geom

        try:
            from shapely.validation import make_valid
            logging.warning("[Base Reader] Geometry is invalid. Attempting to repair with make_valid.")
            logging.warning(explain_validity(geom))
            repaired = make_valid(geom)
        except Exception:
            logging.warning("[Base Reader] Failed to repair geometry with make_valid. Attempting buffer(0) method.")
            repaired = geom.buffer(0)

        if repaired.is_empty:
            logging.error("[Base Reader] Geometry repair failed. Resulting geometry is empty.")
            return geom

        if repaired.geom_type == "Polygon":
            logging.info("[Base Reader] Geometry repaired successfully with type: %s", repaired.geom_type)
            return repaired

        if repaired.geom_type == "MultiPolygon":
            logging.info("[Base Reader] Geometry repaired successfully with type: %s", repaired.geom_type)
            return max(repaired.geoms, key = lambda g: g.area)

        logging.warning("[Base Reader] Geometry repair resulted in unexpected type: %s. Returning original geometry.", repaired.geom_type)
        return repaired.convex_hull if not repaired.convex_hull.is_empty else geom

    def _build_roi_polygon(self, roi_coordinates: np.ndarray):
        roi_polygon = sg.Polygon(roi_coordinates)
        if roi_polygon.is_valid:
            return roi_polygon

        repaired = self._repair_geometry(roi_polygon)
        if not repaired.is_valid:
            logging.warning("[Base Reader] ROI polygon remains invalid after repair attempt.")
        return repaired

    def _safe_intersection(self, geom_a, geom_b):
        try:
            return geom_a.intersection(geom_b)
        except GEOSException:
            repaired_a = self._repair_geometry(geom_a)
            repaired_b = self._repair_geometry(geom_b)
            return repaired_a.intersection(repaired_b)

    def _safe_intersects(self, geom_a, geom_b) -> bool:
        try:
            return geom_a.intersects(geom_b)
        except GEOSException:
            repaired_a = self._repair_geometry(geom_a)
            repaired_b = self._repair_geometry(geom_b)
            return repaired_a.intersects(repaired_b)

    def _get_roi_tile_slices(
            self,
            roi_coordinates: np.ndarray,
            tile_size: int = 5000,
            overlap: int = 256,
    ) -> list[Coordinates2D]:
        """Compute coordinates from a Polygon ROI into tile slices

        Calculate the coordinates of tile slices of size tile_size overlapped
        by overlap from coordinates of an arbitrary region of interest (ROI).

        """

        logging.info("[Base Reader] Slicing ROI...")
        # Create polygon structure to slice
        roi_polygon = self._build_roi_polygon(roi_coordinates)
        minx, miny, maxx, maxy = roi_polygon.bounds

        step_size = tile_size - overlap
        x_len = maxx - minx
        y_len = maxy - miny

        if (x_len < tile_size) and (y_len < tile_size):
            tile_size = max(x_len, y_len)

        tile_slices = []
        for y in np.arange(miny, maxy, step_size):
            for x in np.arange(minx, maxx, step_size):
                tile = sg.box(x, y, x + tile_size, y + tile_size)
                if not self._safe_intersects(roi_polygon, tile):
                    continue
                tile_slices.append(list(map(int, tile.bounds)))

        logging.info("[Base Reader] Found %d slices for ROI.", len(tile_slices))

        return tile_slices

    def planar_to_interleaved(self, matrix: Matrix3D_u8) -> Matrix3D_u8:
        """Utility function to convert planar images to interleaved.
        (C,H,W) -> (H, W, C)
        """
        if matrix.ndim == 2:
            return matrix[:,:,np.newaxis]

        if matrix.ndim != 3:
            raise ValueError(f"Expected 3D Matrix, got {matrix.ndim}.")

        if matrix.shape[0] < matrix.shape[-1]: # Check channel is the first dim
            matrix = np.moveaxis(matrix, 0, -1) # Set channel index to be last

        return matrix

    def _get_local_roi_mask(
            self,
            tile: sg.Polygon,
            roi_polygon: sg.Polygon,
            overlap_threshold: float = 0.1
    ) -> Union[Matrix3D_u8, None]:
        """Create a mask for each tile for regions inside the ROI.
        """
        minx, miny, maxx, maxy = tile.bounds
        tile_area = tile.area

        roi_polygon = self._repair_geometry(roi_polygon)
        intersection = self._safe_intersection(roi_polygon, tile)

        if intersection.area < tile_area * overlap_threshold: # If less than 10%
            return None

        def to_local_coords(coords):
            return [(x-minx, y - miny) for x, y in coords]

        mask_img = Image.new("L", (int(maxx - minx), int(maxy - miny)), 0)
        draw = ImageDraw.Draw(mask_img)

        if intersection.geom_type == "Polygon":
            geoms = [intersection]
        elif intersection.geom_type == "MultiPolygon":
            geoms = intersection.geoms
        else:
            return None

        for poly in geoms:
            exterior_coords = to_local_coords(poly.exterior.coords)
            draw.polygon(exterior_coords, fill = 1) # Boolean fill

            for interior in poly.interiors:
                interior_coords = to_local_coords(interior.coords)
                draw.polygon(interior_coords, fill = 0)

        return np.asarray(mask_img, dtype = bool)

    def mask_non_roi_regions(
            self,
            tile_matrix: list[Matrix3D_u8],
            tile_slice: list[Coordinates2D],
            roi_coords: Coordinates2D,
            background_fill: int = 250,
            intersection_threshold: float = 0.1
    ) -> Union[Matrix3D_u8, None]:
        """
        Mask only tile areas depending on the big ROI shape.
        Tiles fully inside ROI are left unchanged. Works on a SINGLE tile.
        
        Parameters
        ----------
        tiles : list[np.ndarray]
            List of tile arrays (H, W, C).
        tile_slices : list[list[int]]
            List of tile bounding boxes in format [y_min, y_max, x_min, x_max].
            Must correspond 1-to-1 with `tiles`.
        roi_polygon : shapely.geometry.Polygon
            ROI polygon in slide coordinates.
        
        Returns
        -------
        list[np.ndarray]
            List of tiles where boundary tiles are masked.
        """
        roi_polygon = self._build_roi_polygon(roi_coords)

        # for tile_mx, tile_slice in zip(tiles, tile_slices):
        tile_box = sg.box(*tile_slice)
        mask = self._get_local_roi_mask(
            tile_box,
            roi_polygon,
            intersection_threshold)

        if isinstance(mask, NoneType):
            # isinstance with NoneType is used here to avoid the warning
            # 'Truth values of arrays are ambiguous' warning/error
            return None
        tile_matrix = tile_matrix.copy()
        tile_matrix[~mask] = background_fill

        return tile_matrix

class Parser:
    def __init__(self, config_file: Optional[FilePath] = None) -> None:
        self.config_file = config_file
        # Project Metadata
        self.scalef = 99
        self.objective = 99
        self.resolution = 99
        self.image_type = 99
        self.image_preprocessing = 0
        self.image_type_enum = {
            'Absorption' : 0,
            'Concentration' : 1,
            'Fluorescence' : 2
        }

        if not config_file is None:
            self.parse_config_yaml(config_file)

        # Image Metadata
        self.ome_namespace = {
            'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
        }

    def parse_config_yaml(self, config_yaml: FilePath) -> None:
        with open(config_yaml, "r", encoding = 'utf-8') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.scalef = self.config['imaging']['scale-factor']
        self.objective = self.config['imaging']['objective']
        self.resolution = self.config['imaging']['resolution']
        self.image_type = self.config['imaging']['image-type']
        self.image_preprocessing = self.config['imaging']['image-preprocessing']
        # print("Project config: \n", self.config)

        return

    def get_ome_tag_values(
            self,
            ome_tag: str
    ) -> dict[str,Union[float,int]]:
        """Get OME Tiff tag values. Used for in-house tiff tags only.
        """
        root = ET.fromstring(ome_tag)

        def get_tag_value(find_str: str, value: str) -> str | Literal[99]:
            ome_tag = root.find(find_str, self.ome_namespace)
            ome_tag = ome_tag.get(value) if not ome_tag is None else 99
            return ome_tag

        image_type = get_tag_value(".//ome:Dataset", "ID")
        scalef = get_tag_value(".//ome:Channel", "NDFilter")
        resolution = get_tag_value(".//ome:Image/ome:Pixels", "PhysicalSizeX")
        objective = get_tag_value(
            ".//ome:Instrument/ome:Objective", "CalibratedMagnification")

        return {
            "scalef" : float(scalef),
            "objective" : int(float(objective)),
            "image_type" : self.image_type_enum[str(image_type)],
            "resolution" : float(resolution)
        }

    def parse_getafics_roi(
            self,
            annotation_roi_file: FilePath,
            apply_offset: bool = True,
    ) -> Coordinates2D:
        """Read roi file created by Getafics
        
        """
        with open(annotation_roi_file, "r", encoding = "utf-8") as f:
            file_contents = f.readlines()

        header = file_contents[0].strip().split(" ")
        x_offset = int(header[2])
        y_offset = int(header[3])
        num_vertices = int(header[-2])

        _roi_coords = file_contents[1 : num_vertices + 1]

        roi_coords = []
        for _xy in _roi_coords:
            _x, _y = _xy.strip().split(" ")
            roi_coords.append([int(float(_x)), int(float(_y))])

        roi_coords = np.array(roi_coords)

        # Correct absolute offsets
        roi_coords[:, 0] += x_offset * int(apply_offset)
        roi_coords[:, 1] += y_offset * int(apply_offset)

        return roi_coords

    def parse_aperio_roi(
            self,
            annotation_xml_file: FilePath
    ) -> Coordinates2D:
        root = ET.parse(annotation_xml_file).getroot()
        roi_coords = []
        #####
        print("Getting ROI coordinates of tissue...")
        for annotation in root.findall("Annotation"):

            for attribute in annotation.findall("Attributes/Attribute"):
                name = attribute.attrib['Value']
                if name == "Tissue":
                    break
            else:
                continue
            region = annotation.find("Regions/Region")
            for vertex in region.findall("Vertices/Vertex"):
                vertex_data = [
                    int(float(vertex.attrib['X'])),
                    int(float(vertex.attrib['Y']))
                ]

                roi_coords.append(vertex_data)

        return np.array(roi_coords)

    def parse_asap_xml(
            self,
            annotation_xml_file: FilePath
    ) -> Matrix2D_u32:
        """
        """
        tree = ET.parse(annotation_xml_file)
        root = tree.getroot()

        all_annotations = []

        for annotation in root.findall(".//Annotation"):
            coords_list = []

            for coord in annotation.findall(".//Coordinate"):
                order = int(coord.get("Order"))
                x = float(coord.get("X"))
                y = float(coord.get("Y"))
                coords_list.append((order, x, y))

            coords_list.sort(key = lambda item: item[0])

            if len(coords_list) < 4:
                # Some annotations can be dots.
                continue

            xy_array = np.array(
                [[item[2], item[1]] for item in coords_list],
                dtype = np.float32)

            all_annotations.append(xy_array)

        return all_annotations