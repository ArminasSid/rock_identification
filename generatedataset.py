from src.randompoints import RandomPointGenerator
from src.datasetgenerator import DatasetGenerator


def main():
    output_dataset_folder = 'validation_dataset'
    objects_path = 'Data/Vadagiai-0.163-210803/Vadagiai-0.163-atr_b_SHP-210803/Vadagiai-0.163-atr_b-210803.geojson'
    image_path = 'Data/Vadagiai-0.163-210803/Vadagiai-0.163-orto-210803/Vadagiai-0.163-orto-210803.tif'
    area_restriction_polygon = 'outline_true.geojson'
    starting_counter = 20
    images_amount = 20 


    # shutil.rmtree(path=output_dataset_folder, ignore_errors=True)

    random_point_generator = RandomPointGenerator()
    random_points = random_point_generator.generate_random_points(
        raster_path=image_path,
        amount=images_amount,
        polygon_restriction=area_restriction_polygon
    )
    dataset_generator = DatasetGenerator()
    dataset_generator.warp_dataset(raster_path=image_path,
                                   points_list=random_points,
                                   rocks_path=objects_path,
                                   starting_counter=starting_counter,
                                   dir=output_dataset_folder)


if __name__ == '__main__':
    main()
