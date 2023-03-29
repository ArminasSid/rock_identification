from src.randompoints import RandomPointGenerator
from src.datasetgenerator import DatasetGenerator
from osgeo import gdal
import geopandas as gpd
from glob import glob
from tqdm import tqdm
import os
import tempfile


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
                                   

def get_outlines(subdirectory: str, outlines: str | list):
    # Deprecated for now
    if isinstance(outlines, str):
        return glob(f'{subdirectory}/*{outlines}*.geojson')[0]
    elif isinstance(outlines, list):
        paths = []
        for single_outline in outlines:
            paths.append(glob(f'{subdirectory}/**/*{single_outline}*.geojson', recursive=True)[0])
        return paths


def filter_objects(bounds_path: str, objects_path: str, output_path: str):
    bounds_polygons: gpd.GeoDataFrame = gpd.read_file(filename=bounds_path)
    object_polygons: gpd.GeoDataFrame = gpd.read_file(filename=objects_path)

    intersecting: gpd.GeoDataFrame = object_polygons.loc[object_polygons.intersects(bounds_polygons.unary_union)].reset_index(drop=True)

    res_intersection: gpd.GeoDataFrame = object_polygons.overlay(bounds_polygons, how='intersection')
    res_intersection.to_file(filename=output_path)


def create_dataset(main_folder: str, output_folder: str, dataset_name: str, dataset_type: str, input_prefix: str, iteration: int):
    starting_counter = 0
    all_subdirectories = glob(f'{main_folder}/*/')

    all_subdirectories = [subdirectory for subdirectory in all_subdirectories if 'Musa' not in subdirectory]

    for subdirectory in all_subdirectories:
        shapes_path = glob(f'{subdirectory}/{input_prefix}/{dataset_name}/{dataset_type}/*.geojson')[0]
        image_path = glob(f'{subdirectory}/{input_prefix}/{dataset_name}/{dataset_type}/*.tif')[0]

        objects_path = glob(f'{subdirectory}/**/*atr_b3*.geojson', recursive=True)[0]

        with tempfile.NamedTemporaryFile(suffix='.geojson') as fp:
            # Filter object rectangles that are within HMU-clip
            filter_objects(
                bounds_path=shapes_path,
                objects_path=objects_path,
                output_path=fp.name
            )

            generate_dataset(
                output_dataset_folder=f'{output_folder}/{dataset_type}_dataset',
                input_objects_path=fp.name,
                input_raster_path=image_path,
                input_outline_path=shapes_path,
                starting_counter=starting_counter,
                amount_to_generate=iteration
            )

            starting_counter += iteration


def create_single_dataset(main_folder: str, dataset: dict, output_prefix: str, input_prefix: str):
    dataset_name = dataset.get('name')

    # Training dataset
    print('Generating training dataset...')
    create_dataset(
        main_folder=main_folder,
        output_folder=f'{output_prefix}-{dataset_name}',
        dataset_name=dataset_name,
        dataset_type='train',
        input_prefix=input_prefix,
        iteration=100
    )

    # Validation dataset
    print('Generating validation dataset...')
    create_dataset(
        main_folder=main_folder,
        output_folder=f'{output_prefix}-{dataset_name}',
        dataset_name=dataset_name,
        dataset_type='valid',
        input_prefix=input_prefix,
        iteration=25
    )


def warp_image_to_cutline(input_file: str, output_file: str, outline: str):
    gdal.Warp(
        destNameOrDestDS=output_file,
        srcDSOrSrcDSTab=input_file,
        options=gdal.WarpOptions(
            cutlineDSName=outline,
            cropToCutline=True,
            dstNodata=0,
            dstAlpha=False
        )
    )


def merge_polygons(input_files: list, output_file: str):
    shapes = []
    for input_file in input_files:
        shapes.append(gpd.read_file(input_file))

    merged_shape: gpd.GeoDataFrame = gpd.pd.concat(shapes)

    # Buffer so that shapes would properly align
    merged_shape = merged_shape.buffer(0.1)

    merged_shape.to_file(filename=output_file)


def create_dataset_preparation_data(dataset: dict, main_folder: str, working_dir: str, main_orto: str, dataset_type: str):
    shapes = []
    for shape_name in dataset[dataset_type]:
        shapes.extend(glob(f'{main_folder}/**/*{shape_name}*.geojson'))

    train_folder = f'{working_dir}/{dataset_type}'
    os.makedirs(train_folder, exist_ok=True)
    
    output_shape = f'{train_folder}/{dataset_type}.geojson'
    merge_polygons(
        input_files=shapes,
        output_file=output_shape
    )

    output_orto = f'{train_folder}/{dataset_type}.tif'
    warp_image_to_cutline(
        input_file=main_orto,
        output_file=output_orto,
        outline=output_shape
    )


def warp_for_datasets(main_folder: str, prefix: str, dataset: dict, main_orto: str):
    dataset_name = dataset.get('name')

    # Make dir for dataset
    working_dir = f'{main_folder}/{prefix}/{dataset_name}'
    os.makedirs(f'{main_folder}/{prefix}/{dataset_name}', exist_ok=True)

    # Make training
    create_dataset_preparation_data(
        dataset=dataset,
        main_folder=main_folder,
        working_dir=working_dir,
        main_orto=main_orto,
        dataset_type='train'
    )

    # Make validation
    create_dataset_preparation_data(
        dataset=dataset,
        main_folder=main_folder,
        working_dir=working_dir,
        main_orto=main_orto,
        dataset_type='valid'
    )


def prepare_folder_for_dataset_generation(datasets: list, orto: str, folder: str, prefix: str):
    # Prepare single folder train and valid rasters and geojsons for dataset generation
    for dataset in tqdm(datasets):
        warp_for_datasets(
            main_folder=folder,
            prefix=prefix,
            dataset=dataset,
            main_orto=orto
        )


def invalidate_shapefile(shape: str, output_file: str):
    polygons: gpd.GeoDataFrame = gpd.read_file(shape)

    # Buffer so that shapes would properly align
    polygons = polygons.buffer(0.1)

    polygons.to_file(filename=output_file)


def create_parts_outline(folder: str, output_file: str):
    parts = [
        glob(f'{folder}/**/*part_1*.geojson')[0],
        glob(f'{folder}/**/*part_2*.geojson')[0],
        glob(f'{folder}/**/*part_3*.geojson')[0],
        glob(f'{folder}/**/*part_4*.geojson')[0],
        glob(f'{folder}/**/*part_5*.geojson')[0]
    ]

    merge_polygons(input_files=parts, output_file=output_file)


def prepare_main_folder_image(folder: str, prefix: str):
    # Prepare main orto image for folder
    main_orto_image = glob(f'{folder}/**/*orto*.tif')[0]
    # main_outline = glob(f'{folder}/**/*HMU-clip*.geojson')[0]
    output_orto = None

    with tempfile.NamedTemporaryFile(suffix='.geojson') as fp:
        # invalidate_shapefile(shape=main_outline, output_file=fp.name)
        create_parts_outline(folder=folder, output_file=fp.name)

        output_directory = f'{folder}/{prefix}'

        os.makedirs(output_directory, exist_ok=True)
        output_orto = f'{output_directory}/orto_clipped.tif'


        warp_image_to_cutline(
            input_file=main_orto_image,
            output_file=output_orto,
            outline=fp.name
        )

    return output_orto


def prepare_data_for_dataset_creation(datasets: list, main_folders: list, prefix: str):
    # Iterate over main folders and create all dataset rasters and geojsons
    for folder in main_folders:
        print(f'Preparing folder: {folder} for dataset generation.')

        # Prepare main orto image for folder
        orto = prepare_main_folder_image(folder=folder, prefix=prefix)

        prepare_folder_for_dataset_generation(
            datasets=datasets,
            orto=orto,
            folder=folder,
            prefix=prefix
        )


def main():
    gdal.UseExceptions()

    datasets = [{
            'name': 'train1234-valid5',
            'train': ['part_1', 'part_2', 'part_3', 'part_4'],
            'valid': ['part_5']
        }, {
            'name': 'train1235-valid4',
            'train': ['part_1', 'part_2', 'part_3', 'part_5'],
            'valid': ['part_4']
        }, {
            'name': 'train1245-valid3',
            'train': ['part_1', 'part_2', 'part_4', 'part_5'],
            'valid': ['part_3']
        }, {
            'name': 'train1345-valid2',
            'train': ['part_1', 'part_3', 'part_4', 'part_5'],
            'valid': ['part_2']
        }, {
            'name': 'train2345-valid1',
            'train': ['part_2', 'part_3', 'part_4', 'part_5'],
            'valid': ['part_1']
        },
    ]

    main_folder = 'Data'
    prefix = 'dataset_generation'
    output_prefix = 'dataset/dataset'

    # Get a list of working main folders (Vadagiai-0.967, Vadagiai-0.163, etc.)
    main_folders = glob(f'{main_folder}/*')

    main_folders = [subdirectory for subdirectory in main_folders if 'Musa' not in subdirectory]

    # Prepare dataset generation info (rasters, polygons)
    prepare_data_for_dataset_creation(
        datasets=datasets,
        main_folders=main_folders,
        prefix=prefix
    )

    # Create individual datasets
    for dataset in datasets:
        create_single_dataset(
            main_folder=main_folder,
            dataset=dataset,
            output_prefix=output_prefix,
            input_prefix=prefix
        )


if __name__ == '__main__':
    main()
