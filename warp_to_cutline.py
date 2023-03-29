from osgeo import gdal
import geopandas as gpd
from glob import glob
from tqdm import tqdm
import os


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


def warp_for_datasets(main_folder: str, prefix: str, dataset: dict):
    main_orto = glob(f'{main_folder}/{prefix}/*orto*.tif')[0]
    dataset_name = dataset['name']

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


def create_datasets(main_folder: str):
    print(f'Warping out {main_folder}')
    prefix = 'dataset_generation'

    main_orto_image = glob(f'{main_folder}/**/*orto*.tif')[0]
    main_outline = glob(f'{main_folder}/**/*HMU-clip*.geojson')[0]
    output_orto = f'{main_folder}/{prefix}/orto_clipped.tif'

    os.makedirs(f'{main_folder}/{prefix}', exist_ok=True)
    warp_image_to_cutline(
        input_file=main_orto_image,
        output_file=output_orto,
        outline=main_outline
    )

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

    for dataset in datasets:
        warp_for_datasets(
            main_folder=main_folder,
            prefix=prefix,
            dataset=dataset
        )


def main():
    # Do not glob Musa-22
    main_folders = glob(f'Data/*21*')

    for folder in tqdm(main_folders):
        create_datasets(main_folder=folder)
    

def main1():
    # Quick random clip
    input_raster = 'Data/Musa-x.xx-220830/Musa-x.xx-orto-220830/Musa-x.xx-orto-220830.tif'
    output_raster = 'Data/Musa-x.xx-220830/Musa-x.xx-orto-220830/orto_clipped.tif'
    outline = 'Data/Musa-x.xx-220830/Musa-x.xx-HMU_SHP-clip-220830/Musa-x.xx-HMU-clip-220830.geojson'

    warp_image_to_cutline(
        input_file=input_raster,
        output_file=output_raster,
        outline=outline
    )


if __name__ == '__main__':
    main()
