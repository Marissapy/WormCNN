import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Author: Yan Pan
# Affiliation: School of Medicine, University of Electronic Science and Technology of China (UESTC)
# GitHub: https://github.com/Marissapy
# Contact: yanpan@zohomail.com

# Purpose: This script processes segmented worm images to prepare them for input into WormCNN.
# Usage: It resamples skeleton coordinates, generates dense skeleton points, computes angles for cutting,
# and straightens the worm images. This preprocessing is essential for preparing data for WormCNN analysis.

# Resample skeleton coordinates to a fixed number of points
def resample_skeleton_coords(coords, num_points):
    distances = np.sqrt(np.diff(coords[:, 0])**2 + np.diff(coords[:, 1])**2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    f_x = interp1d(cumulative_distances, coords[:, 0], kind='linear')
    f_y = interp1d(cumulative_distances, coords[:, 1], kind='linear')
    new_distances = np.linspace(0, cumulative_distances[-1], num_points)
    resampled_coords = np.column_stack((f_x(new_distances), f_y(new_distances)))
    return resampled_coords

# Generate a dense set of skeleton points for detailed analysis
def generate_dense_skeleton(resampled_coords, total_points):
    dense_skeleton = []
    num_segments = len(resampled_coords) - 1
    points_per_segment = total_points // num_segments

    for i in range(num_segments):
        start_point = resampled_coords[i]
        end_point = resampled_coords[i + 1]
        line_points = np.linspace(start_point, end_point, points_per_segment, endpoint=False)
        dense_skeleton.extend(line_points)

    dense_skeleton.append(resampled_coords[-1])  # Add the last point
    return np.array(dense_skeleton)

# Compute the angle for vertical cuts based on skeleton points
def compute_angle(prev, curr):
    dy = curr[0] - prev[0]
    dx = curr[1] - prev[1]
    angle = np.arctan2(dy, dx)
    return angle

# Straighten the worm image based on skeleton coordinates
def straighten_worm(image, skeleton_coords, segment_half_width=15, segment_height=1):
    sorted_coords = skeleton_coords[skeleton_coords[:, 1].argsort()]  # Sort by y-coordinate
    worm_segments = []

    for i in range(1, len(sorted_coords) - 1):
        prev = sorted_coords[i - 1]
        curr = sorted_coords[i]
        next = sorted_coords[i + 1]

        angle = compute_angle(prev, next)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        for offset in range(-segment_half_width, segment_half_width):
            x_offset = int(curr[0] + offset * cos_angle)
            y_offset = int(curr[1] - offset * sin_angle)
            if 0 <= x_offset < image.shape[0] and 0 <= y_offset < image.shape[1]:
                worm_segments.append(image[x_offset, y_offset])

    straight_worm = np.array(worm_segments).reshape(-1, 2 * segment_half_width)
    straight_worm = cv2.GaussianBlur(straight_worm, (1, 1), 0)  # Apply Gaussian blur to the straightened image
    
    return straight_worm

# Process a single image
def process_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        skeleton = skeletonize(opening // 255)
        skeleton_coords = np.column_stack(np.where(skeleton > 0))

        total_skeleton_points = 400
        resampled_skeleton_coords = resample_skeleton_coords(skeleton_coords, num_points=20)
        dense_skeleton_coords = generate_dense_skeleton(resampled_skeleton_coords, total_points=total_skeleton_points)

        straight_worm_image = straighten_worm(image, dense_skeleton_coords, segment_half_width=15, segment_height=1)
        return straight_worm_image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Save image to specified path
def save_image(image, path):
    try:
        if image is not None:
            cv2.imwrite(path, image)
    except Exception as e:
        print(f"Error saving {path}: {e}")

# Process all images in a directory
def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                straight_worm_image = process_image(input_path)
                save_image(straight_worm_image, output_path)

input_directory = 'dataset'
output_directory = 'dataset_straighted'
process_directory(input_directory, output_directory)
