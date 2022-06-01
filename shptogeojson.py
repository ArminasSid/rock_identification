from osgeo import gdal


def shapefile2geojson(infile, outfile):

    options = gdal.VectorTranslateOptions(format="GeoJSON", dstSRS="EPSG:4326")
    gdal.VectorTranslate(outfile, infile, options=options)


if __name__ == '__main__':
    input_file = 'image_shp/Vadagiai-0.163-atr_b-210803.shp'
    output_file = 'image_shp/rocks.geojson'

    shapefile2geojson(infile=input_file, outfile=output_file)
