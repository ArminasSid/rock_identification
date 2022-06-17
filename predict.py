import shutil
from detecto import core, utils
import numpy as np
import torch
import shapely.geometry as geo
from osgeo import gdal
from geojsonformer import geojsonformer
from src.warpimage import Warper
import os
from tqdm import tqdm


print(f'Cuda is available - {torch.cuda.is_available()}')


def load_model(model_file: str, model_name: core.Model = core.Model.DEFAULT):
    model = core.Model(classes=['rock'], model_name=model_name)
    model.get_internal_model().load_state_dict(torch.load(f=model_file, map_location=model._device))
    return model


def predict(model: core.Model, image: str, threshold: float = 0.7):
    image = utils.read_image(image)
    predictions = model.predict(image)

    # predictions format: (labels, boxes, scores)
    labels, boxes, scores = predictions

    # Filter based on threshold
    filtered_indices=np.where(scores>threshold)
    filtered_scores=scores[filtered_indices]
    filtered_boxes=boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]

    return filtered_labels, filtered_boxes, filtered_scores


def pixel(image: gdal.Dataset, dx, dy):
    xoffset, px_w, rot1, yoffset, rot2, px_h = image.GetGeoTransform()

    # supposing x and y are your pixel coordinate this 
    # is how to get the coordinate in space.
    posX = px_w * dx + rot1 * dy + xoffset
    posY = rot2 * dx + px_h * dy + yoffset

    # shift to the center of the pixel
    posX += px_w / 2.0
    posY += px_h / 2.0
    return posX, posY


def get_raster_polygon(raster: gdal.Dataset) -> geo.polygon.Polygon:
    ulx, xres, _, uly, _, yres = raster.GetGeoTransform()
    lrx = ulx + (raster.RasterXSize * xres)
    lry = uly + (raster.RasterYSize * yres)
    return geo.polygon.Polygon((
        geo.Point(ulx, uly),
        geo.Point(lrx, uly),
        geo.Point(lrx, lry),
        geo.Point(ulx, lry),
        geo.Point(ulx, uly)
    ))


def get_raster_subarea_polygon(raster: gdal.Dataset, offset: int = 200) -> geo.polygon.Polygon:
    '''offset default adjusted for 2000x2000 raster'''
    ulx, xres, _, uly, _, yres = raster.GetGeoTransform()
    new_ulx = ulx + (offset * xres)
    new_uly = uly + (offset * yres)
    new_lrx, new_lry = None, None
    new_lrx = ulx + ((raster.RasterXSize - offset) * xres)
    new_lry = uly + ((raster.RasterYSize - offset) * yres)
    return geo.polygon.Polygon((
        geo.Point(new_ulx, new_uly),
        geo.Point(new_lrx, new_uly),
        geo.Point(new_lrx, new_lry),
        geo.Point(new_ulx, new_lry),
        geo.Point(new_ulx, new_uly)
    ))


def get_polygon_from_pixels(box, image: gdal.Dataset):
    # Get pixel coordinates
    xmin, ymin, xmax, ymax = box[0].item(), box[1].item(), box[2].item(), box[3].item()

    # Get geocoordinates
    ul = geo.Point(pixel(image=image, dx=xmin, dy=ymin))
    ur = geo.Point(pixel(image=image, dx=xmax, dy=ymin))
    lr = geo.Point(pixel(image=image, dx=xmax, dy=ymax))
    ll = geo.Point(pixel(image=image, dx=xmin, dy=ymax))

    return geo.polygon.Polygon((ul, ur, lr, ll, ul))


def form_geojson(image: str, boxes: list, output_name):
    img = gdal.Open(image)
    geojson = geojsonformer.GeoJSON()

    for box in boxes:
        geojson.add_polygon(get_polygon_from_pixels(box=box, image=img))
    
    geojson.write_to_file(file_path=output_name)


def create_polygons(images: list[str], model: core.Model, threshold: float, coord_sys: str):
    geojson = geojsonformer.GeoJSON(epsg=coord_sys)
    print(f'Predicting images.')
    for image in tqdm(images):
        raster: gdal.Dataset = gdal.Open(image)
        
        raster_subpolygon = get_raster_subarea_polygon(raster=raster)

        _, boxes, _ = predict(model=model, image=image, threshold=threshold)

        for box in boxes:
            box_polygon = get_polygon_from_pixels(box=box, image=raster)
            intersection_polygon = raster_subpolygon.intersection(box_polygon)
            if intersection_polygon.area/box_polygon.area >= 0.5:
                geojson.add_polygon(polygon=box_polygon)

    return geojson



def main():
    pieces_folder = 'image_pieces'
    main_raster = 'Data/Verkne-3.31-210827/Verkne-3.31-orto-210827/Verkne-3.31-orto-210827.tif'
    model = 'model.pth'
    prediction_threshold = 0.9
    epsg = '32634'

    # End output of polygons
    output_dir = f'Data/Verkne-3.31-210827'
    output_name = f'Verkne-3.31-atr_b_SHP-210827-{prediction_threshold}-thresh-pred'
    output_path = f'{output_dir}/{output_name}'
    os.makedirs(output_path, exist_ok=True)
    output_path = f'{output_path}/{output_name}.geojson'

    # Warp main image into pieces
    warper = Warper()
    warper.warp_to_pieces(raster=main_raster, output_folder=pieces_folder)

    # Object detection model
    model = load_model(model_file=model, model_name=core.Model.DEFAULT)

    images_to_predict = [f'{pieces_folder}/{file}' for file in os.listdir(pieces_folder)]

    geojson = create_polygons(images=images_to_predict, model=model, threshold=prediction_threshold, coord_sys=epsg)
    geojson.write_to_file(output_path)

    # Cleanup
    shutil.rmtree(pieces_folder, ignore_errors=True)


if __name__=='__main__':
    main()
