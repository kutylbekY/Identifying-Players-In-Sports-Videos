import cv2
import easyocr
import pytesseract
import numpy as np
import re

# Path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'<path_to_tesseract_executable>'

def extract_jersey_numbers(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Extract the region of interest (ROI)
    # x, y, w, h = roi
    # roi_image = image[y:y+h, x:x+w]

    # Preprocess the ROI (if necessary)
    # You can apply techniques like thresholding, denoising, or morphological operations here

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment dark gray areas
    _, thresh_roi = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)

    # Invert the thresholded image to get dark gray areas as white
    thresh_roi = cv2.bitwise_not(thresh_roi)

    # Create a green color mask for the dark gray areas
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = 255  # Set green channel to 255 (full intensity)

    # Apply the green mask to the original image
    green_regions = cv2.bitwise_and(image, green_mask)

    # Replace dark gray areas in the original image with green regions
    result_image = cv2.add(green_regions, cv2.cvtColor(thresh_roi, cv2.COLOR_GRAY2BGR))

    # Apply thresholding only to the grayscale ROI
    # _, thresh_roi = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    output_path = r'C:\Users\User\Desktop\Project\Code\yolov5\inputs\test_ocr\result_image.jpg'
    cv2.imwrite(output_path, result_image)
    output_path = r'C:\Users\User\Desktop\Project\Code\yolov5\inputs\test_ocr\gray_image.jpg'
    cv2.imwrite(output_path, gray_image)
    output_path = r'C:\Users\User\Desktop\Project\Code\yolov5\inputs\test_ocr\thresh_roi.jpg'
    cv2.imwrite(output_path, thresh_roi)

    # Perform OCR using pytesseract
    jersey_numbers = pytesseract.image_to_string(result_image, config='--psm 6')  # PSM 6 assumes a single uniform block of text

    # Extract only numeric characters
    jersey_numbers = re.sub(r'\D', '', jersey_numbers)

    # Create EasyOCR reader
    # reader = easyocr.Reader(['en'])

    # # Perform OCR using EasyOCR
    # result = reader.readtext(result_image)

    # # Extract only numeric characters
    # jersey_numbers = ""
    # for detection in result:
    #     text = detection[1]
    #     if text.isdigit():
    #         jersey_numbers += text

    return jersey_numbers

# Example usage
image_path = r'C:\Users\User\Desktop\Project\Code\yolov5\inputs\test_ocr\148.jpg'
# roi = (100, 100, 200, 50)  # Example ROI (x, y, width, height)
jersey_numbers = extract_jersey_numbers(image_path)
print("Jersey Numbers:", jersey_numbers)
