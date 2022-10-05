from src.randompoints import RandomPointGenerator
from src.datasetgenerator import DatasetGenerator
from glob import glob
import os


def generate_dataset(output_dataset_folder: str, input_objects_path: str, 
                     input_raster_path: str, input_outline_path: str,
                     starting_counter: int, amount_to_generate: int):
    
    # Generate random points
    random_point_generator = RandomPointGenerator()
    random_points = random_point_generator.generate_random_points(raster_path=input_raster_path,
                                                                  amount=amount_to_generate,
                                                                  polygon_restriction=input_outline_path)

    # Generate dataset for points
    dataset_generator = DatasetGenerator()
    dataset_generator.warp_dataset(raster_path=input_raster_path,
                                   points_list=random_points,
                                   rocks_path=input_objects_path,
                                   starting_counter=starting_counter,
                                   dir=output_dataset_folder)


def create_dataset(main_folder: str, output_folder: str, dataset_name: str, iteration: int):
    starting_counter = 0
    all_subdirectories = glob(f'{main_folder}/*/')

    for subdirectory in all_subdirectories:
        objects_path = glob(f'{subdirectory}/**/*atr_b2*.geojson', recursive=True)[0]
        image_path = glob(f'{subdirectory}/**/*orto*.tif', recursive=True)[0]
        outline_path = glob(f'{subdirectory}/*{dataset_name}*.geojson')[0]
        generate_dataset(output_dataset_folder=f'{output_folder}/{dataset_name}_dataset',
                         input_objects_path=objects_path,
                         input_raster_path=image_path,
                         input_outline_path=outline_path,
                         starting_counter=starting_counter, 
                         amount_to_generate=iteration)
        starting_counter += iteration


def main():
    main_folder = 'Data'
    output_dataset_folder = 'dataset_b2_pascal'

    # Training dataset
    print('Generating training dataset...')
    create_dataset(main_folder=main_folder,
                   output_folder=output_dataset_folder,
                   dataset_name='training',
                   iteration=100)

    # Validation dataset
    print('Generating validation dataset...')
    create_dataset(main_folder=main_folder,
                   output_folder=output_dataset_folder,
                   dataset_name='validation',
                   iteration=5)


if __name__ == '__main__':
    main()
