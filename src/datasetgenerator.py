from osgeo import gdal
import shapely.geometry as geo
import os
from dataclasses import dataclass
import json
import affine
import lxml.etree as ET
from tqdm import tqdm

# No aux.xml files
gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

# No warnings
gdal.SetConfigOption('CPL_LOG', '/dev/null')


@dataclass
class PixelBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    classification: int

    def get_class_str(self):
        if self.classification == 1:
            return 'rock_underwater'
        elif self.classification == 2:
            return 'rock'
        raise KeyError("Suluzo")


@dataclass
class Bound:
    upper_left: geo.Point
    lower_right: geo.Point
    classification: int = 2

    def __post_init__(self):
        # Need to cast if someone leaves retarded marking in shapefiles again
        self.classification = int(self.classification)


class DatasetGenerator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _get_edge_coordinates(raster: gdal.Dataset,
                              point: geo.Point,
                              pixels: int):
        ulx, uly = point.x, point.y
        _, xres, _, _, _, yres = raster.GetGeoTransform()
        lrx = ulx + (pixels * xres)
        lry = uly + (pixels * yres)
        return ulx, uly, lrx, lry

    @staticmethod
    def _get_raster_polygon(raster: gdal.Dataset) -> geo.polygon.Polygon:
        ulx, xres, _, uly, _, yres = raster.GetGeoTransform()
        lrx = ulx + (raster.RasterXSize * xres)
        lry = uly + (raster.RasterYSize * yres)
        return geo.polygon.Polygon((
            geo.Point(ulx, uly),
            geo.Point(lrx, uly),
            geo.Point(lrx, lry),
            geo.Point(ulx, lry),
            geo.Point(ulx, uly)
        ))

    @staticmethod
    def _read_rock_polygon(file) -> list[Bound]:
        json_obj = None
        bounds = []
        with open(file) as f:
            json_obj = json.load(f)

        for feature in json_obj['features']:
            polygon: geo.polygon.Polygon = geo.shape(feature['geometry'])
            bounds.append(Bound(geo.Point(polygon.exterior.coords[0]),
                                geo.Point(polygon.exterior.coords[2])))
        return bounds

    @staticmethod
    def _read_rock_polygons_by_type(file) -> list[Bound]:
        json_obj = None
        bounds = []
        with open(file) as f:
            json_obj = json.load(f)

        for feature in json_obj['features']:
            try:
                classification = feature['properties']['B2'] # Hardcoded B2 value
            except KeyError:
                classification = 2
            polygon: geo.polygon.Polygon = geo.shape(feature['geometry'])
            bounds.append(Bound(geo.Point(polygon.exterior.coords[0]),
                                geo.Point(polygon.exterior.coords[2]),
                                classification))
        return bounds

    @staticmethod
    def _retrieve_pixel_value(raster: gdal.Dataset, point: geo.Point):
        """Pixel locations px and py"""
        x, y = point.x, point.y
        forward_transform =  \
            affine.Affine.from_gdal(*raster.GetGeoTransform())
        reverse_transform = ~forward_transform
        px, py = reverse_transform * (x, y)
        px, py = int(px + 0.5), int(py + 0.5)
        return px, py

    def _get_pixel_bounds(self, raster: gdal.Dataset,
                          bounds: list) -> list[PixelBounds]:
        pixel_bounds = []
        for bound in bounds:
            xmin, ymin = self._retrieve_pixel_value(raster=raster,
                                                    point=bound.upper_left)
            xmax, ymax = self._retrieve_pixel_value(raster=raster,
                                                    point=bound.lower_right)
            
            if xmax < xmin:
                xmax, xmin = xmin, xmax

            if ymax < ymin:
                ymax, ymin = ymin, ymax

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > 2000:
                xmax = 2000
            if ymax > 2000:
                ymax = 2000

            if xmax == xmin or ymax == ymin:
                continue

            pixel_bounds.append(PixelBounds(
                xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax,
                classification=bound.classification
            ))

        return pixel_bounds

    def _create_raster_element(self, output_raster: str, raster: gdal.Dataset,
                               point: geo.Point, pixels: int) -> gdal.Dataset:
        ulx, uly, lrx, lry = DatasetGenerator._get_edge_coordinates(
                            raster=raster, point=point, pixels=pixels)
        raster = gdal.Warp(
            destNameOrDestDS=output_raster,
            srcDSOrSrcDSTab=raster,
            format='GTiff',
            outputBounds=[ulx, lry, lrx, uly]
        )
        return raster

    def _create_xml_file(self, output_path: str, bounds: list[PixelBounds], pixel_size: int) -> None:
        xml_obj = ET.Element('annotation')
        image_name, _ = os.path.splitext(os.path.basename(output_path))
        image_name += '.jpg'

        ET.SubElement(xml_obj, 'filename').text = image_name

        size = ET.SubElement(xml_obj, 'size')
        ET.SubElement(size, 'width').text = str(pixel_size)
        ET.SubElement(size, 'height').text = str(pixel_size)
        ET.SubElement(size, 'depth').text = str(3)

        for bound in bounds:
            obj = ET.SubElement(xml_obj, 'object')

            ET.SubElement(obj, 'name').text = bound.get_class_str()
            ET.SubElement(obj, 'difficult').text = str(0)

            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(bound.xmin)
            ET.SubElement(bndbox, 'ymin').text = str(bound.ymin)
            ET.SubElement(bndbox, 'xmax').text = str(bound.xmax)
            ET.SubElement(bndbox, 'ymax').text = str(bound.ymax)

        tree = ET.ElementTree(xml_obj)
        tree.write(output_path, pretty_print=True)

    def warp_dataset(self, raster_path: str,
                     points_list: list[geo.Point],
                     rocks_path: str,
                     starting_counter: int,
                     dir='images'):
        # Create required dirs
        dir_annotations = f'{dir}/annotations'
        dir_images = f'{dir}/images'
        dir_rasters = f'{dir}/Rasters'

        os.makedirs(dir_annotations, exist_ok=True)
        os.makedirs(dir_images, exist_ok=True)
        os.makedirs(dir_rasters, exist_ok=True)

        prefix = f'image'
        pixels = 2000

        raster_entire = gdal.Open(raster_path)
        bounds = self._read_rock_polygon(rocks_path)
        bounds2 = self._read_rock_polygons_by_type(rocks_path)

        for i, point in tqdm(enumerate(points_list, start=starting_counter), total=len(points_list)):
            id = f'{prefix}{str(i).zfill(5)}'
            # Warp raster
            # --------------
            out_raster_name = f'{dir_rasters}/{id}.tiff'
            raster = self._create_raster_element(
                output_raster=out_raster_name,
                raster=raster_entire,
                point=point,
                pixels=pixels
            )
            raster_polygon = self._get_raster_polygon(raster=raster)
            # --------------
            # Get rocks
            filtered_bounds = [bound for bound in bounds2
                               if raster_polygon.contains(bound.upper_left) or
                               raster_polygon.contains(bound.lower_right)]
            if not filtered_bounds:
                continue
            pixel_bounds = self._get_pixel_bounds(raster=raster,
                                                  bounds=filtered_bounds)
            # Create xml file
            out_xml_name = f'{dir_annotations}/{id}.xml'
            self._create_xml_file(output_path=out_xml_name,
                                  bounds=pixel_bounds,
                                  pixel_size=pixels)
            # --------------
            # Warp jpeg image
            out_jpeg_name = f'{dir_images}/{id}.jpg'
            gdal.Warp(
                destNameOrDestDS=out_jpeg_name,
                srcDSOrSrcDSTab=raster,
                format='JPEG'
            )
            # --------------
