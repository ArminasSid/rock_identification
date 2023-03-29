import geopandas as gpd
from osgeo import gdal
import os
from glob import glob
from tqdm import tqdm


def change_vector_format(infile: str, outfile: str) -> None:
    """Output format is decided by outfile extension eg. .geojson or .shp"""
    gdal.VectorTranslate(
        destNameOrDestDS=outfile,
        srcDS=infile
    )

def add_offset(input_file: str, output_file: str, offset: float = 0.1):
    data: gpd.GeoDataFrame = gpd.read_file(filename=input_file)

    data = data.buffer(distance=offset)

    data.to_file(filename=output_file)

def main():
    # Gather all input files
    input_files = glob('Data/**/*.shp', recursive=True)

    # Gather all output file paths
    output_files = []
    for input_file in input_files:
        output_files.append(os.path.splitext(input_file)[0] + '.geojson')

    # Convert all .shp files to .geojson format
    iterable = zip(input_files, output_files)
    for input_file, output_file in tqdm(iterable=iterable, total=len(input_files)):
        change_vector_format(infile=input_file, outfile=output_file)
        add_offset(input_file=output_file, output_file=output_file, offset=0.01)





if __name__ == '__main__':
    main()
