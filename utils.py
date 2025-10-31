import os
import typing
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def get_image_paths(directory: str | os.PathLike) -> typing.List[str]:
    '''Gets all image file paths from specified directory. Non-recursive.
    Args:
        directory (str | os.PathLike): Path to the directory containing images.
        Returns:
            List[str]: List of image file paths.
    '''
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def plot_ellipse_overlay(img: np.ndarray, data: np.ndarray) -> None:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(data), 6):
        centroid = (np.round(data[i][1]).astype(int), np.round(data[i][0]).astype(int))
        cv2.circle(img, centroid, 5, (0,0, 255), -1)
        end = (np.round(data[i][1] + data[i+1][1]/2 * math.cos(data[i+2])).astype(int),
                np.round(data[i][0] - data[i+1][1]/2 * math.sin(data[i+2])).astype(int))
        cv2.line(img, centroid, end, (255,0,0),5)
        cv2.ellipse(img, centroid, (np.round(data[i+1][1]/2).astype(int), np.round(data[i+1][0]/2).astype(int)),
                    -math.degrees(data[i+2]), 0, 360, (0,0,255), 2)