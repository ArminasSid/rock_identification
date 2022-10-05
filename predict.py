from concurrent.futures import thread
import shutil
from xmlrpc.client import Boolean
from detecto import core, utils
import numpy as np
import torch
import shapely.geometry as geo
from osgeo import gdal
from geojsonformer import geojsonformer
from src.warpimage import Warper
import os
from tqdm import tqdm
from glob import glob
import tempfile
import json

from src.yolopredictor import load_model as load_yolo_model
from src.yolopredictor import predict as predict_yolo


print(f'Cuda is available - {torch.cuda.is_available()}')

def read_polygons(file):
    polygons = []
    json_obj = None
    with open(file) as f:
        json_obj = json.load(f)
    for feature in json_obj['features']:
        polygons.append(geo.shape(feature['geometry']))
    return polygons


def load_model(model_file: str, classes: list, model_name: core.Model = core.Model.DEFAULT):
    model = core.Model(classes=classes, model_name=model_name)
    model.get_internal_model().load_state_dict(torch.load(f=model_file, map_location=model._device))
    return model


def predict_image(model: core.Model, image: str, threshold: float = 0.7):
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

def get_polygon_from_pixels_yolo(box, image: gdal.Dataset):
    # Get pixel coordinates
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

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


def get_feature_from_string(name: str) -> geojsonformer.Feature:
    feature = geojsonformer.Feature(0)
    if name == 'rock_underwater':
        feature.add_feature(key='B2', value=1)
    elif name == 'rock':
        feature.add_feature(key='B2', value=2)
    return feature


def intersects_with_bounds(polygon: geo.Polygon, bound_polygons: list[geo.Polygon]) -> Boolean:
    for bound in bound_polygons:
        if polygon.intersection(bound).area > 0:
            return True
    return False


def create_polygons(images: list[str], model: core.Model, threshold: float, coord_sys: str, bound_polygons: list[geo.Polygon]):
    geojson = geojsonformer.GeoJSON(epsg=coord_sys)
    print(f'Predicting images.')
    for image in tqdm(images):
        raster: gdal.Dataset = gdal.Open(image)
        
        raster_subpolygon = get_raster_subarea_polygon(raster=raster)

        # labels, boxes, _ = predict_image(model=model, image=image, threshold=threshold)
        labels, boxes, _ = predict_yolo(model=model, image_to_predict=image, conf_thresh=threshold)

        for label, box in zip(labels, boxes):
            feature = get_feature_from_string(label)
            box_polygon = get_polygon_from_pixels_yolo(box=box, image=raster)
            intersection_polygon = raster_subpolygon.intersection(box_polygon)
            if intersection_polygon.area/box_polygon.area >= 0.5:
                if intersects_with_bounds(polygon=box_polygon, bound_polygons=bound_polygons):
                    geojson.add_polygon(polygon=box_polygon, feature=feature)

    return geojson


def create_output_path(subdirectory: str, threshold: float, prefix: str) -> str:
    name = os.path.basename(os.path.normpath(subdirectory))
    output_name = f'{name}-pred-{prefix}-{threshold}-thresh-pred'
    output_path = f'{subdirectory}/{output_name}'
    os.makedirs(output_path, exist_ok=True)
    output_path = f'{output_path}/{output_name}.geojson'
    return output_path

def classify_directory(subdirectory: str, img_size: int, model: core.Model, epsg: str, thresholds_to_predict: list, prefix: str):
    # Get outline polygons of prediction area
    outline_polygons = read_polygons(file=glob(f'{subdirectory}/**/*HMU-clip*.geojson')[0])

    # Create image warper instance
    warper = Warper()

    step_size = int(img_size * 0.8)

    orto_img = glob(f'{subdirectory}/**/*orto*.tif', recursive=True)[0]
    print(f'Predicting image: {orto_img}')

    # Initialize temporary directory to warp images into, cleans up afterwards
    with tempfile.TemporaryDirectory() as dir:
        warper.warp_to_pieces(raster=orto_img, output_folder=dir,
                                size=img_size, step_size=step_size)
        images_to_predict = [f'{dir}/{file}' for file in os.listdir(dir)]
        for threshold in thresholds_to_predict:
            print(f'Predicting with threshold: {threshold}')
            output_path = create_output_path(subdirectory=subdirectory, threshold=threshold, prefix=prefix)
            geojson = create_polygons(images=images_to_predict, model=model, 
                                      threshold=threshold, coord_sys=epsg,
                                      bound_polygons=outline_polygons)
            geojson.write_to_file(output_path)


def predict(folder: str, model: core.Model, thresholds: float, epsg: str, img_size: int, prefix: str):
    # Get all subdirectories in main data folder
    all_subdirectories = glob(f'{folder}/*/')

    for subdirectory in all_subdirectories:
        classify_directory(subdirectory=subdirectory,
                           img_size=img_size,
                           model=model,
                           epsg=epsg,
                           thresholds_to_predict=thresholds,
                           prefix=prefix)


def main():
    main_folder = 'Data'
    prefix = 'b'
    path_to_model = 'model_b.pth'
    prediction_thresholds = [0.5, 0.7, 0.9]
    piece_size = 2000
    epsg = '32634'

    # fasterrcnn_resnet50_fpn
    model_name = core.Model.DEFAULT
    classes = ['rock']

    # Object detection model
    model = load_model(model_file=path_to_model, 
                       classes=classes, 
                       model_name=model_name)

    # Predict folder of orto images
    predict(folder=main_folder,
            model=model,
            thresholds=prediction_thresholds,
            epsg=epsg,
            img_size=piece_size,
            prefix=prefix)

def main_yolo():
    main_folder = 'Data'
    prefix = 'b'
    path_to_model = 'yolo_b/best.pt'
    path_to_data = 'yolo_b/data.yaml'
    prediction_thresholds = [0.5, 0.7, 0.9]
    piece_size = 2000
    epsg = '32634'

    # Object detection model
    model = load_yolo_model(weights=path_to_model, data=path_to_data)

    # Predict folder of orto images
    predict(folder=main_folder,
            model=model,
            thresholds=prediction_thresholds,
            epsg=epsg,
            img_size=piece_size,
            prefix=prefix)


if __name__=='__main__':
    main_yolo()
