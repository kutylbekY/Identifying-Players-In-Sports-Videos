# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import easyocr
import os
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pytesseract
from PIL import Image, ImageDraw
import re

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Modify these colors based on your preferences
COLOR_TEAM_1 = (0, 0, 255, 255)  # Red
COLOR_TEAM_2 = (255, 0, 0, 255)  # Yellow
COLOR_REF = (0, 0, 0, 0)  # Blue
COLOR_UN = (0, 0, 0, 0) 

# Define a mapping of class combinations to colors
class_color_mapping = {
    ('team_blue', 'team_red'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_yellow', 'team_red'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_yellow', 'team_black'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_yellow', 'team_white'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_black', 'team_red'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_black', 'team_white'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_red', 'team_white'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_blue', 'team_white'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_blue', 'team_yellow'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_blue', 'team_black'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_red', 'team_yellow'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_red', 'team_black'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_white', 'team_yellow'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('team_white', 'team_black'): (COLOR_TEAM_1, COLOR_TEAM_2),
    ('_ref',): (COLOR_REF,),
    # Add more combinations as needed
}

def colors_s(class_idx, use_rgb=False):
    # Define your color mapping based on class indices
    color_mapping = {
        'team_1': (0, 0, 255),  # Red for "team 1"
        'team_2': (0, 255, 0),  # Blue for "team 2"
        'ref': (0, 255, 255),  # Yellow for "ref"
        # Add more mappings for other classes as needed
    }

    if use_rgb:
        return color_mapping.get(class_idx, (255, 255, 255))  # Default to white if class not found
    else:
        return class_idx

cnt = 0
team_1 = -1
team_2 = -1
team1_color = ""
team2_color = ""

def calculate_weighted_average(hist_combined): 
    return np.sum(np.arange(len(hist_combined)) * hist_combined) / np.sum(hist_combined)

def calculate_histogram_and_mask(roi, lower_color, upper_color):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # region of interest
    
    mask = cv2.inRange(hsv_roi, lower_color, upper_color)
    roi_color = cv2.bitwise_and(roi, roi, mask=mask)
    
    hist_hue = cv2.calcHist([roi_color], [0], None, [180], [0, 180])
    hist_saturation = cv2.calcHist([roi_color], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([roi_color], [2], None, [256], [0, 256])
    
    hist_hue /= hist_hue.sum() + 1e-10
    hist_saturation /= hist_saturation.sum() + 1e-10
    hist_value /= hist_value.sum() + 1e-10
    
    hist_combined = np.concatenate((hist_hue, hist_saturation, hist_value), axis=None)
    
    return hist_combined

def assign_class_from_color_histogram(img, xyxy, save_plot=True):
    global cnt
    global team_1
    global team_2
    global team1_color
    global team2_color
    x1, y1, x2, y2 = map(int, xyxy)
    roi = img[y1:y2, x1:x2]  # Extract region of interest (ROI)

    colors = {
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'black': ([0, 0, 0], [180, 255, 75]),
        'purple': ([110, 50, 30], [160, 255, 255]),
        'blue': ([100, 50, 50], [120, 255, 255]),
        'red': ([0, 50, 50], [10, 255, 255])
        # 'white': ([0, 0, 200], [180, 50, 255])
    }

    hist_combined = []
    
    for color_name, (lower_color, upper_color) in colors.items():
        hist_combined.append(calculate_histogram_and_mask(roi, np.array(lower_color), np.array(upper_color)))

    # Calculate the weighted averages of x-axis values based on histogram values
    weighted_avgs = [round(calculate_weighted_average(hist)) for hist in hist_combined]

    # Classify teams and refs
    if team_1 == -1 and weighted_avgs[0] > max(weighted_avgs[1], weighted_avgs[2], weighted_avgs[3], weighted_avgs[4]): team_1, team1_color = 0, "yellow" # yellow
    elif team_1 == -1 and weighted_avgs[1] > max(weighted_avgs[0], weighted_avgs[2], weighted_avgs[3], weighted_avgs[4]): team_1, team1_color = 1, "black" # black
    elif team_1 == -1 and weighted_avgs[3] > max(weighted_avgs[0], weighted_avgs[2], weighted_avgs[1], weighted_avgs[4]): team_1, team1_color = 3, "blue" # blue
    elif team_1 == -1 and weighted_avgs[4] > max(weighted_avgs[0], weighted_avgs[2], weighted_avgs[1], weighted_avgs[3]): team_1, team1_color = 4, "red" # red
    # elif team_1 == -1 and weighted_avgs[5] > max(weighted_avgs[0], weighted_avgs[1], weighted_avgs[2], weighted_avgs[3], weighted_avgs[4]): team_1 = 5 # white

    if team_2 == -1 and (team_1 != -1 and team_1 != 0) and weighted_avgs[0] > max(weighted_avgs[1], weighted_avgs[2], weighted_avgs[3], weighted_avgs[4]): team_2, team2_color = 0, "yellow" # yellow
    elif team_2 == -1 and (team_1 != -1 and team_1 != 1) and weighted_avgs[1] > max(weighted_avgs[0], weighted_avgs[2], weighted_avgs[3], weighted_avgs[4]): team_2, team2_color = 1, "black" # black
    elif team_2 == -1 and (team_1 != -1 and team_1 != 3) and weighted_avgs[3] > max(weighted_avgs[0], weighted_avgs[2], weighted_avgs[1], weighted_avgs[4]): team_2, team2_color = 3, "blue" # blue
    elif team_2 == -1 and (team_1 != -1 and team_1 != 4) and weighted_avgs[4] > max(weighted_avgs[0], weighted_avgs[2], weighted_avgs[1], weighted_avgs[3]): team_2, team2_color = 4, "red" # red
    # elif team_2 == -1 and (team_1 != -1 and team_1 != 5) and weighted_avgs[5] > max(weighted_avgs[0], weighted_avgs[1], weighted_avgs[2], weighted_avgs[3], weighted_avgs[4]): team_2 = 5 # white

    # print("team_1: ", team_1)
    # print("team_2: ", team_2)
    # print(str(cnt) + "_team_1: ", weighted_avgs[team_1])
    # print(str(cnt) + "_team_2: ", weighted_avgs[team_2])
    # print(str(cnt) + "_ref: ", weighted_avgs[2])

    if (team_1 != -1 and team_2 != -1):
        if weighted_avgs[team_1] > weighted_avgs[team_2]: return 'team_1' # return str(cnt) + '_team_1'
        elif weighted_avgs[team_2] > weighted_avgs[team_1]: return 'team_2'
        # elif weighted_avgs[2] > weighted_avgs[team_1] and weighted_avgs[2] > weighted_avgs[team_2]: return 'ref'
        else: return 'unknown'
    elif (team_1 != -1 and team_2 == -1):
        return 'team_1'
        # if weighted_avgs[team_1] > weighted_avgs[2]: return 'team_1'
        # elif weighted_avgs[2] > weighted_avgs[team_1]: return 'ref'
        # else: return 'unknown'

def change_white(image):
    if isinstance(image, str):
        # Read the input image if a file path is provided
        input_image = cv2.imread(image)
    else:
        # Use the provided NumPy array directly
        input_image = image.copy()

    # Convert BGR to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment players (adjust the threshold value accordingly)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with black pixels
    mask = np.zeros_like(input_image)

    # Draw contours on the mask with white color
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the mask to the appropriate data type
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Bitwise AND operation with the input image and the inverse mask
    result = cv2.bitwise_and(input_image, input_image, mask=mask)

    # Make the ice field completely green (adjust the color if needed)
    result[mask == 0] = [0, 255, 0]

    # Return the result
    return result

def change_gray(image):
    if isinstance(image, str):
        # Read the input image if a file path is provided
        input_image = cv2.imread(image)
    else:
        # Use the provided NumPy array directly
        input_image = image.copy()

    # Define the lower and upper bounds for dark gray to white in RGB
    lower_bound = np.array([100, 100, 100], dtype=np.uint8)
    upper_bound = np.array([130, 130, 130], dtype=np.uint8)

    # Create a mask for colors between dark gray and white
    mask = cv2.inRange(input_image, lower_bound, upper_bound)

    # Convert the mask to the appropriate data type
    mask = mask.astype(np.uint8)

    # Bitwise AND operation with the input image and the inverse mask
    result = cv2.bitwise_and(input_image, input_image, mask=~mask)

    # Make the areas between dark gray and white purple
    result[mask > 0] = [128, 0, 128]  # Purple color in BGR format

    # Return the result
    return result

def ocr_on_jersey(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    roi = img[y1:y2, x1:x2]  # Extract region of interest (ROI)

    # Apply OCR on the extracted region
    jersey_number = pytesseract.image_to_string(roi, config='--psm 8')
    # reader = easyocr.Reader(['en'], gpu = False)
    # jersey_number = reader.readtext(roi)
    
    # Extract only numeric characters using regular expression
    # numeric_jersey_number = re.sub(r'\D', '', jersey_number)
    save_path = f"hist/bw_bb/{str(cnt)}.jpg"
    cv2.imwrite(save_path, roi)
    print(f"Bounding box images saved at: {save_path}")
    
    return jersey_number

# Add this function to convert bounding box to black and white image
# def save_bounding_box_image(img, bbox, team_color):
#     global cnt
#     x1, y1, x2, y2 = map(int, bbox)
#     roi = img[y1:y2, x1:x2]

#     colors = {
#         'yellow': ([20, 100, 100], [30, 255, 255]), 
#         'black': ([0, 0, 0], [180, 255, 75]),
#         'blue': ([100, 50, 50], [120, 255, 255]),
#         'red': ([0, 50, 50], [10, 255, 255])
#     }

#     # Convert image to HSV for color filtering
#     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#     # Filter pixels based on team_color
#     lower_color, upper_color = colors.get(team_color, ([0, 0, 0], [0, 0, 0]))
#     mask = cv2.inRange(hsv_roi, np.array(lower_color), np.array(upper_color))

#     # Make pixels white if in the color range, otherwise make them black
#     roi[mask > 0] = [0, 0, 0]
#     # roi[mask == 0] = [0, 0, 0]
    
#     # Apply OCR on the extracted region
#     # jersey_number = pytesseract.image_to_string(roi, config='--psm 8')
    
#     # # Extract only numeric characters using regular expression
#     # numeric_jersey_number = re.sub(r'\D', '', jersey_number)
    
#     # return numeric_jersey_number

#     # Save the black and white image
#     save_path = f"hist/bw_bb/{str(cnt)}.jpg"
#     cv2.imwrite(save_path, roi)
#     print(f"Bounding box images saved at: {save_path}")

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global cnt
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
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
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # create im_changed to deleted_white_image and then deleted_gray_image
            im_changed = change_white(im0)
            # im_changed = change_gray(im_no_white)
            # cv2.imwrite("hist/hist_data_unseen/result_" + str(cnt) + ".jpg", im_changed)
            # print("Result saved to 'hist/hist_data_unseen' folder.")

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Process results
                stored_bounding_boxes = []  # List to store bounding boxes with colors and assigned classes

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        # Assign classes "team_1", "team_2", or "ref" based on color histogram
                        assigned_class = assign_class_from_color_histogram(im_changed, xyxy) # change im0 to im_changed

                        ocr_on_jersey(im0, xyxy)
                        # jersey_number = ""
                        # if assigned_class.endswith('team_1'): save_bounding_box_image(im_changed, xyxy, team1_color)
                        # elif assigned_class.endswith('team_2'): save_bounding_box_image(im_changed, xyxy, team2_color)
                        cnt += 1
                        # OCR on jersey number
                        # jersey_number = ocr_on_jersey(im0, xyxy)
                        # assigned_class = str(jersey_number) + assigned_class
                        # print("assigned_class", assigned_class)
                        # print("jersey_number", jersey_number)

                        # Assign colour based on class
                        if assigned_class.endswith('team_1'): color = COLOR_TEAM_1
                        elif assigned_class.endswith('team_2'): color = COLOR_TEAM_2
                        elif assigned_class.endswith('ref'): color = COLOR_REF
                        else: color = COLOR_UN

                        stored_bounding_boxes.append({'xyxy': xyxy, 'assigned_class': assigned_class, 'color': color})
                        # annotator.box_label(xyxy, assigned_class, color=color)
                        # annotator.box_label(xyxy, assigned_class, color=colors_s(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / assigned_class / f'{p.stem}.jpg', BGR=True)
                
                # Apply stored bounding boxes onto the image
                for stored_bbox in stored_bounding_boxes:
                    annotator.box_label(stored_bbox['xyxy'], stored_bbox['assigned_class'], color=stored_bbox['color'])
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)