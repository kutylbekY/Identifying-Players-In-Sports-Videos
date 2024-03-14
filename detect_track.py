import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import pickle
from typing import Any
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from scipy.ndimage import gaussian_filter
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

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

# Colours
COLOR_TEAM_1 = (0, 0, 255, 255)  # Red
COLOR_TEAM_2 = (255, 0, 0, 255)  # Yellow
COLOR_REF = (0, 255, 0, 255)  # Green
COLOR_POINTS = (255, 165, 0, 255)  # Orange
COLOR_UN = (0, 0, 0, 0) 
COLOR_WHITE = (255, 255, 255, 255)  # White
COLOR_BLACK = (0, 0, 0, 255)    # Black

feature_vectors = []  # To accumulate feature vectors from initial detections
kmeans_model = None  # To store the K-means model once clustering is done
team_centers = None  # To store team centers for classification
homography_matrices = {} # To store homography matrices

left_side_labels = {"TLP", "TLI", "TLG", "BLG", "BLI"}
right_side_labels = {"TRP", "TRI", "TRG", "BRG", "BRI"}
back_labels = {"TLB", "BLB", "TRB", "BRB"}

# Corresponding coordinates on the 2D field
field_points = [
    (53, 60), (53, 169), (53, 187), (53, 224), (53, 242),
    (587, 60), (587, 169), (587, 187), (587, 224), (587, 242),
    (216, 52), (424, 52), (320, 52), (231, 129), (231, 283),
    (409, 129), (409, 283), (124, 129), (124, 283), (516, 129), (516, 283),
    (320, 161), (320, 251), (11, 176), (11, 235), (629, 176), (629, 235)
]

point_names = {
    "TLP": 0, "TLI": 0, "TLG": 0, "BLG": 0, "BLI": 0,
    "TRP": 0, "TRI": 0, "TRG": 0, "BRG": 0, "BRI": 0,
    "TLMP": 0, "TRMP": 0, "MP": 0, "TLMC": 0, "BLMC": 0, "TRMC": 0, "BRMC": 0,
    "TLC": 0, "BLC": 0, "TRC": 0, "BRC": 0,
    "TMP": 0, "BMP": 0, "TLB": 0, "BLB": 0, "TRB": 0, "BRB": 0
}

# Bundary coordinates
top_boundary = (320, 60)
bottom_boundary = (320, 355)
right_boundary = (620, 202)
left_boundary = (20, 202)

# Define a dictionary to map labels to 2D field coordinates
label_to_field = {
    "TLP": (53, 60), "TLI": (53, 169), "TLG": (53, 187), "BLG": (53, 224), "BLI": (53, 242),
    "TRP": (587, 60), "TRI": (587, 169), "TRG": (587, 187), "BRG": (587, 224), "BRI": (587, 242),
    "TLMP": (216, 52), "TRMP": (424, 52), "MP": (320, 52), "TLMC": (231, 129), "BLMC": (231, 283), "TRMC": (409, 129), "BRMC": (409, 283),
    "TLC": (124, 129), "BLC": (124, 283), "TRC": (516, 129), "BRC": (516, 283),
    "TMP": (320, 161), "BMP": (320, 251), "TLB": (11, 176), "BLB": (11, 235), "TRB": (629, 176), "BRB": (629, 235)
}

# heatmap initialisation
################################################################

 # Load the 2D field image
field_image_path = 'inputs/2d_field.png'
heatmap_image = cv2.imread(field_image_path)
if heatmap_image is None:
    raise ValueError(f"Image not found at the path: {field_image_path}")

height, width, _ = heatmap_image.shape

# Initialize empty heatmaps for each team
heatmap_team_1 = np.zeros((height, width), dtype=np.float32)
heatmap_team_2 = np.zeros((height, width), dtype=np.float32)

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
    # result[mask == 0] = [255, 255, 255]

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

# Check corner boundries
########################################################################################################################################
def is_left_of_diagonal(p, tl, br):
    return (p[0] - tl[0]) * (br[1] - tl[1]) - (p[1] - tl[1]) * (br[0] - tl[0]) < 0

def is_right_of_diagonal(p, tl, br):
    return (p[0] - tl[0]) * (br[1] - tl[1]) - (p[1] - tl[1]) * (br[0] - tl[0]) > 0

def is_point_inside_square_2(p, t, b):
    return t[0] <= p[0] <= b[0] and t[1] <= p[1] <= b[1]

def is_point_inside_square(p, t, b):
    return b[0] <= p[0] <= t[0] and t[1] <= p[1] <= b[1]

def mirror_across_diagonal(p, tl, br):
    # Vector from tl to br
    diagonal_vec = np.array([br[0] - tl[0], br[1] - tl[1]])

    # Vector from tl to p
    tl_to_p_vec = np.array([p[0] - tl[0], p[1] - tl[1]])
    
    # Project tl_to_p_vec onto diagonal_vec
    proj_length = np.dot(tl_to_p_vec, diagonal_vec) / np.linalg.norm(diagonal_vec)
    proj_vec = proj_length * diagonal_vec / np.linalg.norm(diagonal_vec)
    
    # Calculate the perpendicular vector from p to the diagonal
    perp_vec = tl_to_p_vec - proj_vec
    
    # Mirrored point is p minus 2 times the perpendicular vector
    mirrored_p = p - 2 * perp_vec
    
    return mirrored_p.astype(int)

def adjust_point_based_on_corner(field_point):
    # Top Left Corner
    tr_top_left = np.array([73, 51])
    bl_top_left = np.array([11, 114])

    # Bottom Left
    tl_point = np.array([11, 302])
    br_point = np.array([73, 360])

    # Top Right Corner
    tl_top_right = np.array([565, 51])
    br_top_right = np.array([629, 114])

    # Bottom Right Corner
    tr_bottom_right = np.array([629, 302])
    bl_bottom_right = np.array([565, 360])

    if is_point_inside_square(field_point, tr_top_left, bl_top_left) and is_left_of_diagonal(field_point, tr_top_left, bl_top_left):
        field_point = mirror_across_diagonal(field_point, tr_top_left, bl_top_left)
    elif is_point_inside_square_2(field_point, tl_point, br_point) and is_left_of_diagonal(field_point, tl_point, br_point):
        field_point = mirror_across_diagonal(field_point, tl_point, br_point)
    elif is_point_inside_square_2(field_point, tl_top_right, br_top_right) and is_right_of_diagonal(field_point, tl_top_right, br_top_right):
        field_point = mirror_across_diagonal(field_point, tl_top_right, br_top_right)
    elif is_point_inside_square(field_point, tr_bottom_right, bl_bottom_right) and is_right_of_diagonal(field_point, tr_bottom_right, bl_bottom_right):
        field_point = mirror_across_diagonal(field_point, tr_bottom_right, bl_bottom_right)

    return field_point
########################################################################################################################################

# Team classification
########################################################################################################################################
def calculate_rgb_histogram(roi):
    # Calculate the histograms for each channel
    hist_r = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([roi], [2], None, [256], [0, 256])
    
    # Normalize histograms
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    
    return np.concatenate((hist_r, hist_g, hist_b))

def classify_player(feature_vector, team_centers):
    # Find the closest cluster center and classify the player
    distances = np.linalg.norm(team_centers - feature_vector, axis=1)
    return np.argmin(distances)

def get_feature_vector(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = img[y1:y2, x1:x2]  # Extract the region of interest (ROI)
    jersey_roi = jersey_cutout(roi)  # Focus on the jersey area
    blurred_jersey_roi = apply_gaussian_blur(jersey_roi)  # Apply Gaussian blur to the jersey ROI
    # feature_vector = calculate_rgb_histogram(blurred_jersey_roi)  # Calculate the RGB histogram and flatten it
    feature_vector = calculate_hsv_histogram(blurred_jersey_roi) 

    return feature_vector

def calculate_hsv_histogram(roi, bins=32):
    # Convert ROI to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate the histograms for the H, S, and V channels with fewer bins
    hist_h = cv2.calcHist([hsv_roi], [0], None, [bins], [0, 180])  # Hue values range from 0 to 180
    hist_s = cv2.calcHist([hsv_roi], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_roi], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    return np.concatenate((hist_h, hist_s, hist_v))

def jersey_cutout(roi, reduce_percentage=0.5):
    h, w, _ = roi.shape
    center_x, center_y = w // 2, h // 2

    new_w = int(w * reduce_percentage)
    new_h = int(h * reduce_percentage)

    start_x = max(center_x - new_w // 2, 0)
    start_y = max(center_y - new_h // 2, 0)
    end_x = start_x + new_w
    end_y = start_y + new_h

    return roi[start_y:end_y, start_x:end_x]

def apply_gaussian_blur(roi, kernel_size=(5, 5)):
    blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
    return blurred_roi

########################################################################################################################################

# Heatmap generation
########################################################################################################################################

def generate_heatmap(heatmap):
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)  # Apply Gaussian Blur
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to fit [0, 255] scale
    heatmap_normalized = np.uint8(heatmap_normalized)  # Convert to uint8
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)  # Apply color map
    return heatmap_color

# Overlay heatmaps on the original field image
def overlay_heatmap(field_image, heatmap, alpha=0.6):
    return cv2.addWeighted(heatmap, alpha, field_image, 1 - alpha, 0)

########################################################################################################################################

# Side points filtering
########################################################################################################################################

def filter_top_conf_points(side_labels, points):
    # Extract points and their confidences for the specified side
    side_points = {k: v for k, v in points.items() if k in side_labels}
    # Sort by confidence, high to low, and get the top 2
    sorted_by_conf = sorted(side_points.items(), key=lambda x: float(x[1]['label'].split()[-1]), reverse=True)[:2]
    return {k: v for k, v in sorted_by_conf}

########################################################################################################################################

# Function to transform points using the homography matrix
def apply_homography(H, points):
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the homography matrix
    points_transformed = np.dot(H, points_homogeneous.T).T

    # Convert back from homogeneous to 2D coordinates
    points_transformed = points_transformed[:, :2] / points_transformed[:, 2:]

    return points_transformed

# Update DeepSORT tracker with detections
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt', source=ROOT / 'data/images', data=ROOT / 'data/coco128.yaml', imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45,
        max_det=1000, device='', view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, 
        visualize=False, update=False, project=ROOT / 'runs/detect', name='exp', exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, 
        half=False, dnn=False, vid_stride=1, 
):
    global kmeans_model
    source = str(source)
    save_img = not nosave and not source.endswith('.txt') 
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Extract the filename from the path
    filename = os.path.basename(weights[0])

    # Now, assuming the version part is always formatted as "_v<number>.pt" and you want "v<number>.pt":
    version_part = filename.split('_')[-1]

    if (version_part == "v4.pt"):
        # Load the homography_matrices dictionary from the file
        with open('hom_matrix/homography_matrices_short_45_new.pkl', 'rb') as f:
            loaded_homography_matrices = pickle.load(f)

    # Dataloader
    bs = 1
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
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz)) 
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

        # Process predictions and DeepSORT tracking
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            im_changed = change_white(im0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            annotator_track = Annotator(im0, line_width=line_thickness, example=str(names))

            # Load the 2D field image
            field_image_path = 'inputs/2d_field.png' 
            field_image = cv2.imread(field_image_path)
            if field_image is None:
                raise ValueError(f"Image not found at the path: {field_image_path}")

            # Process results
            ref_bbox = []  # List to store bounding boxes with colors and assigned classes
            detections_to_track = []  # List to store bounding boxes with colors and assigned classes
            bbox_conf = []

            # Define the coordinates in the video frame (to be filled with actual detection data)
            frame_points = []  # List of tuples (x, y) of detected points in the video frame
            field_points = [] 

            points_dict  = {}
            points_back = {}
            count_R = 0
            count_L = 0

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        x1, y1, x2, y2 = map(int, xyxy)
                        w, h = x2 - x1, y2 - y1
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        if label[0] == 'R':
                            assigned_class = 'referee'
                            ref_bbox.append({'xyxy': xyxy})
                        elif label[0] == 'p':
                            detections_to_track.append(np.array([x1, y1, w, h]))
                            bbox_conf.append(conf)
                        else:
                            point = label.split()[0]

                            if point in back_labels:
                                if point in points_back:
                                    # If the label already exists, compare confidence values
                                    curr_conf = float(points_back[point]['label'].split()[-1])
                                    new_conf = float(label.split()[-1])
                                    if new_conf > curr_conf:
                                        # Update the dictionary with the detection of higher confidence
                                        points_back[point] = {'xyxy': xyxy, 'label': label}
                                else:
                                    # If the label does not exist, add it to the dictionary
                                    if label[1] == 'R':
                                        count_R += 1
                                    elif label[1] == 'L':
                                        count_L += 1
                                    points_back[point] = {'xyxy': xyxy, 'label': label}
                            elif point in points_dict:
                                # If the label already exists, compare confidence values
                                curr_conf = float(points_dict[point]['label'].split()[-1])
                                new_conf = float(label.split()[-1])
                                if new_conf > curr_conf:
                                    # Update the dictionary with the detection of higher confidence
                                    points_dict[point] = {'xyxy': xyxy, 'label': label}
                            else:
                                # If the label does not exist, add it to the dictionary
                                if len(point) == 3:  # Check if the label has 3 letters
                                    if label[1] == 'R':
                                        count_R += 1
                                    elif label[1] == 'L':
                                        count_L += 1
                                points_dict[point] = {'xyxy': xyxy, 'label': label}
                    if save_crop:
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / assigned_class / f'{p.stem}.jpg', BGR=True)
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
                dominant_letter = 'R' if count_R > count_L else 'L'
                updated_points = {}

                while len(points_dict) < 4 and points_back:
                    # Sort points_back items by their confidence values, highest first
                    sorted_points_back = sorted(points_back.items(), key=lambda x: float(x[1]['label'].split()[-1]), reverse=True)

                    # Take the highest confidence item from points_back
                    highest_conf_point, highest_conf_data = sorted_points_back[0]

                    # Add this item to points_dict
                    points_dict[highest_conf_point] = highest_conf_data

                    # Remove the added item from points_back to avoid adding it again
                    del points_back[highest_conf_point]

                # Adjust labels in points_dict
                for key, value in points_dict.items():
                    label, conf_str = value['label'].rsplit(' ', 1)
                    conf = float(conf_str)
                    if (len(label) == 3 and count_R != 0 and count_L != 0):  # Check if the label has 3 letters
                        # Replace the second letter with the dominant letter
                        new_label = label[0] + dominant_letter + label[2]
                        new_label_conf = new_label + ' ' + conf_str

                        if (new_label in points_dict) and label[1] != dominant_letter:
                            # If the label already exists, compare confidence values
                            curr_conf = float(points_dict[new_label]['label'].split()[-1])
                            if conf > curr_conf:
                                # Update the dictionary with the detection of higher confidence
                                updated_points[new_label] = {'xyxy': value['xyxy'], 'label': new_label_conf}
                        else:
                            updated_points[new_label] = {'xyxy': value['xyxy'], 'label': new_label_conf}
                    else:
                        updated_points[label] = {'xyxy': value['xyxy'], 'label': value['label']}

                # Filter left and right side points
                left_points_filtered = filter_top_conf_points(left_side_labels, updated_points)
                right_points_filtered = filter_top_conf_points(right_side_labels, updated_points)

                # Update original points dictionary to include only the filtered points plus any points not in left/right side labels
                filtered_points = {**{k: v for k, v in updated_points.items() if k not in left_side_labels and k not in right_side_labels},
                                **left_points_filtered,
                                **right_points_filtered}

                points = list(filtered_points.values())

                # Apply stored bounding boxes onto the image
                if (len(points) >= 4):
                    for stored_bbox in points:
                        bbox = stored_bbox['xyxy']
                        label = stored_bbox['label']
                        label_point = label.split()[0]
                        # Get the center of the bbox 

                        if label_point in label_to_field:
                            frame_points.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))    
                            field_point = label_to_field[label_point]
                            field_points.append(field_point)
                            annotator.box_label(bbox, label, COLOR_POINTS) # Delete after finished implementing 
                        else:
                            print(f"Label {label_point} not found in label_to_field mapping.")

                    
                    # Convert lists to numpy arrays for OpenCV functions
                    frame_points = np.array(frame_points, dtype='float32')
                    field_points= np.array(field_points, dtype='float32')

                    # Calculate the Homography matrix
                    H, status = cv2.findHomography(frame_points, field_points)
                    if H is not None:
                        homography_matrices[seen] = H

                if (len(detections_to_track) != 0):
                    detections_to_track = np.array(detections_to_track, dtype=np.float32)
                    bbox_conf = np.array(bbox_conf, dtype=np.float32) 
                    tracks = tracker.update(detections_to_track, bbox_conf, im0)
                    update_interval = 20  # Interval to retrain KMeans model
                    new_data_counter = 0 

                    for track in tracks:
                        track_id = track[4]
                        x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
                        bbox = x1, y1, x2, y2

                        feature_vector = get_feature_vector(im_changed, bbox)  # Extract feature vector
                        feature_vectors.append(feature_vector)

                        if len(feature_vectors) >= 20:  # Condition to start or update clustering
                            # Convert list to array for K-means
                            feature_vectors_array = np.array(feature_vectors)

                            if kmeans_model is None or new_data_counter >= update_interval:
                                kmeans_model = KMeans(n_clusters=2, random_state=42).fit(feature_vectors_array)
                                new_data_counter = 0  # Reset counter after update
                            team_centers = kmeans_model.cluster_centers_

                        assigned_class = None
                        if kmeans_model is not None:
                            assigned_class = classify_player(feature_vector, team_centers)  # Classify based on closest center

                        # Assign colour based on class
                        if assigned_class == 0: 
                            color = COLOR_TEAM_1
                            label = "team_1: " + str(track_id)
                        elif assigned_class == 1: 
                            color = COLOR_TEAM_2
                            label = "team_2: " + str(track_id)
                        else: 
                            color = COLOR_UN
                            label = "unknown: " + str(track_id)

                        annotator_track.box_label([x1, y1, x2, y2], label, color)

                        temp = seen
                        while (temp != 1 and loaded_homography_matrices.get(temp) is None):
                            temp -= 1
                        
                        H = loaded_homography_matrices.get(temp)
                        
                        if (H is not None):
                            # Get the center of the bounding box
                            bbox_center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                            # Map the center point onto the 2D field using the homography matrix
                            field_point = apply_homography(H, bbox_center)[0]
                            x, y = field_point  # Mapped point

                            # Adjust for boundaries
                            x = max(left_boundary[0], min(x, right_boundary[0]))  # Between Left and Right
                            y = max(top_boundary[1], min(y, bottom_boundary[1]))  # Between Top and Bottom
                            field_point = adjust_point_based_on_corner(np.array([x, y]))

                            x, y = field_point  # Mapped point
                            field_point = (int(x), int(y))

                            if (assigned_class == 0):
                                heatmap_team_1[field_point[1], field_point[0]] += 1
                            elif (assigned_class == 1):
                                heatmap_team_2[field_point[1], field_point[0]] += 1

                            # Draw the point on the field image
                            field_point = tuple(np.round(field_point).astype(int))  # Convert to integer tuple
                            cv2.circle(field_image, field_point, radius=5, color=color, thickness=-1)  # -1 fills the circle

                            # Add track_id at the center of the circle
                            # Define font scale and thickness
                            font_scale = 0.3
                            thickness = 1
                            text = str(track_id)  # Convert track_id to string
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                            text_x = field_point[0] - text_size[0] // 2
                            text_y = field_point[1] + text_size[1] // 2
                            cv2.putText(field_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                        font_scale, COLOR_WHITE, thickness)

                
                if (len(ref_bbox) != 0):
                    for stored_bbox in ref_bbox:
                        bbox = stored_bbox['xyxy']
                        annotator.box_label(bbox, "referee", COLOR_REF)

                        x1, y1, x2, y2 = stored_bbox['xyxy']

                        # Map the center point onto the 2D field using the homography matrix
                        temp = seen
                        while (temp != 1 and loaded_homography_matrices.get(temp) is None):
                            temp -= 1
                        
                        H = loaded_homography_matrices.get(temp)
                        if (H is not None):
                            # Get the center of the bounding box
                            bbox_center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                            field_point = apply_homography(H, bbox_center)[0]
                            x, y = field_point  # Mapped point

                            # Adjust for boundaries
                            x = max(left_boundary[0], min(x, right_boundary[0]))  # Between Left and Right
                            y = max(top_boundary[1], min(y, bottom_boundary[1]))  # Between Top and Bottom
                            field_point = adjust_point_based_on_corner(np.array([x, y]))

                            x, y = field_point  # Mapped point
                            field_point = (int(x), int(y))
                            
                            # Draw the point on the field image
                            field_point = tuple(np.round(field_point).astype(int))  # Convert to integer tuple
                            cv2.circle(field_image, field_point, radius=5, color=COLOR_REF, thickness=-1)  # -1 fills the circle

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
                # Calculate padding to add to the shorter image (empty space)
                field_image = cv2.resize(field_image, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_LINEAR)

                h1, w1 = im0.shape[:2]
                h2, w2 = field_image.shape[:2]
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + w2  # Adjust width to include field_image
                            h = max(h1, h2)  # Use the taller height
                        else:  # stream
                            fps, w, h = 30, w1 + w2, max(h1, h2)
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    # Prepare combined image for each frame before writing to the video
                    if h1 > h2:  # If im0 is taller
                        padding = np.zeros((h1 - h2, w2, 3), dtype=np.uint8)
                        field_image_padded = np.vstack((field_image, padding))
                        combined_image = np.hstack((im0, field_image_padded))
                    else:  # If field_image is taller or they are the same height
                        padding = np.zeros((h2 - h1, w1, 3), dtype=np.uint8)
                        im0_padded = np.vstack((im0, padding))
                        combined_image = np.hstack((im0_padded, field_image))
                    vid_writer[i].write(combined_image)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Save the homography_matrices dictionary to a file
    if (version_part == "v7.pt"):
        with open('hom_matrix/homography_matrices_short_45_new.pkl', 'wb') as f:
            pickle.dump(homography_matrices, f)

    if (version_part == "v3.pt"):
        heatmap_color_team_1 = generate_heatmap(heatmap_team_1)
        heatmap_color_team_2 = generate_heatmap(heatmap_team_2)
        
        field_with_heatmap_team_1 = overlay_heatmap(heatmap_image.copy(), heatmap_color_team_1)
        field_with_heatmap_team_2 = overlay_heatmap(heatmap_image.copy(), heatmap_color_team_2)

        # Save the heatmap images
        cv2.imwrite('heatmaps/heatmap_team_1.png', field_with_heatmap_team_1)
        cv2.imwrite('heatmaps/heatmap_team_2.png', field_with_heatmap_team_2)

        print("Heatmap for team_1 save at: heatmaps/heatmap_team_1.png")
        print("Heatmap for team_2 save at: heatmaps/heatmap_team_2.png")

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
