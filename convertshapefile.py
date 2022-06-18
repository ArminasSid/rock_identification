from osgeo import gdal


def change_vector_format(infile: str, outfile: str) -> None:
    """Output format is decided by outfile extension eg. .geojson or .shp"""
    gdal.VectorTranslate(
        destNameOrDestDS=outfile,
        srcDS=infile
    )


if __name__ == '__main__':
    input_file = 'Data/Jura-1.98-210916/Jura-1.98-atr_b-210916-0.9-thresh-pred/Jura-1.98-atr_b-210916-0.9-thresh-pred.geojson'
    output_file = 'Data/Jura-1.98-210916/Jura-1.98-atr_b-210916-0.9-thresh-pred/Jura-1.98-atr_b-210916-0.9-thresh-pred.shp'

    change_vector_format(
        infile=input_file, 
        outfile=output_file
    )
