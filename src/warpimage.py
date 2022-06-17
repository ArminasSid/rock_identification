from osgeo import gdal
import os


class Warper:
    def __init__(self) -> None:
        pass

    def _create_folder(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)

    def warp_to_pieces(self, raster: str, output_folder: str, size: int = 2000, step_size: int = 1600):
        self._create_folder(output_folder=output_folder)
        raster: gdal.Dataset = gdal.Open(raster)
        counter = 0
        for i in range(0, raster.RasterXSize, step_size):
            for j in range(0, raster.RasterYSize, step_size):
                output_name = f'{output_folder}/image{str(counter).zfill(5)}.tiff'
                counter += 1
                options = gdal.TranslateOptions(
                    srcWin=[i, j, size, size]
                )
                gdal.Translate(destName=output_name, srcDS=raster, options=options)
