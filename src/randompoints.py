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
        json_obj = None
        with open(file) as f:
            json_obj = json.load(f)
        polygon: geo.Polygon = geo.shape(json_obj['features'][0]['geometry'])
        return polygon

    def generate_random_points(self, raster_path: str, amount: int,
                               polygon_restriction: str) -> list[geo.Point]:
        random_points = []
        ulx, uly, lrx, lry = RandomPointGenerator._get_raster_edges(
            image_path=raster_path
        )
        polygon = RandomPointGenerator._read_polygon(polygon_restriction)
        while amount != 0:
            point: geo.Point = geo.Point((random.uniform(ulx, lrx),
                                          random.uniform(uly, lry)))
            if polygon.contains(point):
                random_points.append(point)
                amount -= 1
        return random_points
