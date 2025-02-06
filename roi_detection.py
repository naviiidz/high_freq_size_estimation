'''
name: Navid Zarrabi
date: Aug 28, 2024
email: navid.zarrabii@gmail.com

Bounding box using fastSAM and simple thresholding
'''


import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import stats
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from ultralytics.models.fastsam import FastSAMPredictor



def fastSam(image_path):
    # Create FastSAMPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-x.pt", save=True, imgsz=1024)
    predictor = FastSAMPredictor(overrides=overrides)

    # Segment everything
    everything_results = predictor(image_path)

    # Extract first result and relevant information
    first_result = everything_results[0].boxes
    bboxes = first_result.xyxy.cpu()  # Move to CPU

    # Load the original image to get dimensions
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
    img_height, img_width = image.shape[:2]

    # Define a minimum size threshold (e.g., 20x20 pixels)
    min_width = 200
    min_height = 200

    # Filter out small boxes
    filtered_indices = []
    for i, box in enumerate(bboxes):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        if width >= min_width and height >= min_height:
            filtered_indices.append(i)

    # Keep only the boxes that meet the size criteria
    filtered_boxes = bboxes[filtered_indices]

    # Apply Non-Maximum Suppression (NMS)
    nms_threshold = 0.2
    keep_indices = torch.ops.torchvision.nms(filtered_boxes, torch.ones_like(filtered_boxes[:, 0]), nms_threshold)

    # Final filtered boxes after NMS
    final_boxes = filtered_boxes[keep_indices]

    # Plot the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Loop through final boxes and draw them on the image
    for i, box in enumerate(final_boxes):
        x_min, y_min, x_max, y_max = map(int, box.numpy())  # Convert to integer coordinates

        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)


        
        # Annotate the box with the class label
        plt.text(x_min, y_min, 'x', fontsize=12, color='white', verticalalignment='top')

    plt.axis('off')  # Hide axes
    plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def thresholding_box_fitting(image_path, original_image_path, classification_image, gt_dict):

    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
    img_height, img_width = image.shape[:2]

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding (you can adjust the threshold value)
    equalized = cv2.equalizeHist(gray_image)

    # Apply a threshold to isolate the bright regions
    _, binary_image = cv2.threshold(equalized, 220, 255, cv2.THRESH_BINARY)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Resize classification_image to match the image dimensions
    classification_resized = cv2.resize(classification_image, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    # Define a minimum size threshold (e.g., 20x20 pixels)
    min_width = 10
    min_height = 10

    # Filter contours based on size and create bounding boxes
    filtered_boxes = []
    for contour in contours:
        x_min, y_min, width, height = cv2.boundingRect(contour)
        if width >= min_width and height >= min_height:
            x_max = x_min + width
            y_max = y_min + height
            filtered_boxes.append([x_min, y_min, x_max, y_max])

    # Convert filtered_boxes to a NumPy array for consistency with previous code
    filtered_boxes = np.array(filtered_boxes)

    # Load the original image
    image = cv2.imread(original_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Plot the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Loop through final boxes and draw them on the image
    for i, box in enumerate(filtered_boxes):
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Determine the class label from resized classification_image using the mode
        sub_region = classification_resized[y_min:y_max, x_min:x_max]
        sub_region.flatten()
        mode_result = stats.mode(sub_region, axis=None)

        # Handle cases where the mode result is a scalar or array
        if mode_result.count.size > 0:
            class_label_value = int(mode_result.mode)
        else:
            class_label_value = int(mode_result)  # Fallback to scalar mode result if necessary

        class_label = gt_dict.get(class_label_value, 'Unknown')  # Map to actual class name using gt_dict
        
        # Annotate the box with the class label
        plt.text(x_min, y_min, class_label, fontsize=12, color='white', verticalalignment='top')

    plt.axis('off')  # Hide axes
    plt.show()

    # Example usage
    # thresholding_box_fitting(image_path, original_image_path, classification_image, gt_dict)
    return filtered_boxes
