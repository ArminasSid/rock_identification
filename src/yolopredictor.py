from pathlib import Path

import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import Profile, check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, smart_inference_mode

def load_model(weights: str):
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    dnn = False  # use OpenCV DNN for ONNX inference
    half = False  # use FP16 half-precision inference

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)

    return model

@smart_inference_mode()
def predict(model, image_to_predict: str, conf_thresh: float = 0.5):
    imgsz = (2000, 2000)
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    augment = True
    agnostic_nms = False

    labels = []
    boxes = []
    confidence = []

    # Load model
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(image_to_predict, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thresh, iou_thres, None, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    vals =  (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    xmin, ymin, xmax, ymax = ([int(x) for x in vals])
                    c = int(cls)  # integer class
                    label = names[c]
                    line = (label, xmin, ymin, xmax, ymax, float(conf))
                    labels.append(label)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confidence.append(float(conf))

    return labels, boxes, confidence


def main():
    weights = 'yolo_b2/best.pt'
    data = 'yolo_b2/data.yaml'
    model = load_model(weights=weights, data=data)

    image_to_predict = '/home/arminius/repos/rock_identification/dataset_b2/training_dataset/images/image00166.jpg'
    predict(model=model, image_to_predict=image_to_predict, conf_thresh=0.5)



if __name__ == "__main__":
    main()
