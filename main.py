import os
from osgeo import gdal
from src.randompoints import RandomPointGenerator
from src.datasetgenerator import DatasetGenerator
import shutil


def reproject_raster(image_path, output_path, destined_srs='EPSG:4326'):
    gdal.Warp(output_path, image_path, dstSRS=destined_srs)


def main():
    dataset_dir = 'validation_dataset'
    rocks_path = 'image_shp/rocks.geojson'

    shutil.rmtree(path=dataset_dir, ignore_errors=True)

    image_path = 'image_shp/image.tiff'
    polygon_restriction = 'image_shp/outline_true.geojson'
    random_point_generator = RandomPointGenerator()
    random_points = random_point_generator.generate_random_points(
        raster_path=image_path,
        amount=20,
        polygon_restriction=polygon_restriction
    )
    dataset_generator = DatasetGenerator()
    dataset_generator.warp_dataset(raster_path=image_path,
                                   points_list=random_points,
                                   rocks_path=rocks_path,
                                   dir=dataset_dir)


if __name__ == '__main__':
    main()
