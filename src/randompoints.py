import shapely.geometry as geo
from osgeo import gdal
import json
import random


class RandomPointGenerator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _get_raster_edges(image_path):
        src = gdal.Open(image_path)
        ulx, xres, _, uly, _, yres = src.GetGeoTransform()
        lrx = ulx + (src.RasterXSize * xres)
        lry = uly + (src.RasterYSize * yres)
        return ulx, uly, lrx, lry

    @staticmethod
    def _read_polygon(file):
        polygons = []
        json_obj = None
        with open(file) as f:
            json_obj = json.load(f)
        for feature in json_obj['features']:
            polygons.append(geo.shape(feature['geometry']))
        return polygons

    @staticmethod
    def _read_polygons(files):
        if isinstance(files, str):
            return RandomPointGenerator._read_polygon(files)
        elif isinstance(files, list):
            polygons = []
            for file in files:
                polygons.extend(RandomPointGenerator._read_polygon(file))
            return polygons

    def generate_random_points(self, raster_path: str, amount: int,
                               polygon_restriction: str | list) -> list[geo.Point]:
        random_points = []
        ulx, uly, lrx, lry = RandomPointGenerator._get_raster_edges(
            image_path=raster_path
        )
        polygons = RandomPointGenerator._read_polygons(polygon_restriction)
        while amount != 0:
            point: geo.Point = geo.Point((random.uniform(ulx, lrx),
                                          random.uniform(uly, lry)))
            for polygon in polygons:
                if polygon.contains(point):
                    random_points.append(point)
                    amount -= 1
                    break
        return random_points
