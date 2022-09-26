from ctypes import sizeof
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

def main():
    # Gather all input files
    input_files = glob('**/*.shp', recursive=True)

    # Gather all output file paths
    output_files = []
    for input_file in input_files:
        output_files.append(os.path.splitext(input_file)[0] + '.geojson')

    # Convert all .shp files to .geojson format
    iterable = zip(input_files, output_files)
    for input_file, output_file in tqdm(iterable=iterable, total=len(input_files)):
        change_vector_format(infile=input_file, outfile=output_file)



if __name__ == '__main__':
    main()
