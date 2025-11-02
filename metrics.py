import numpy as np
import cv2
from skimage.measure import label, regionprops, find_contours
from skimage import filters, morphology
from processor import calculate_sharp_edges

import time
import pandas as pd
import math
from utils.plotutils import plot_ellipse_overlay
from utils.utils import info, warn, err, header

def analyse_image(image: np.ndarray, thresh: int = 100, visual=False, save=False) -> tuple[list, float]:
    """Continuously processes images from the queue until the process is stopped."""

    # Initialization of image counter and data container

    # Define the size of a pixel
    pixel_size = 5.86 # in [um]

    # Get image from queue and flip it 180 degrees

    # Remove the high frequency noise with the gaussian blur filter
    smoothed_image = cv2.GaussianBlur(image, (25, 25), sigmaX=2, sigmaY=2)
    data = []

    # Save image if the amount of sharp edges in it are above a defined threshold
    start = time.time_ns()
    if calculate_sharp_edges(smoothed_image) > thresh: 
        # Create file in previously generated folder

        # Create binary image with defined threshold
        
        
        #thresh = 30
        #binary_image = ((smoothed_image > thresh) * 255)
        
        threshold = filters.threshold_otsu(smoothed_image)
        binary_image = smoothed_image > threshold
        binary_image = morphology.remove_small_objects(binary_image, 50)
        binary_image = morphology.remove_small_holes(binary_image, 50)
        #cv2.imshow("Binary Image", binary_image.astype(np.uint8)*255)
        # Morphological closing to fill small holes inside snowlakes
        kernel = np.ones((20, 20), np.uint8)
        closed_binary_image = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
        # Calculate regions of snowflakes in image
        label_img = label(closed_binary_image)
        snowflakes = regionprops(label_img)
        # Initialize a list to store characteristic values of snowflakes
        snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)

        for snowflake in snowflakes:
            # Only save the snowflakes that are bigger than 50 pixel in diameter
            
            if snowflake.equivalent_diameter_area >= 100:
                # Append center and axes of snowflake
                data.append(snowflake.centroid)
                data.append((snowflake.axis_major_length, snowflake.axis_minor_length))
                # Append orientation of snowflake in grad
                data.append(snowflake.orientation)
                # Append aspect ratio of snowflake
                data.append(snowflake.axis_minor_length/snowflake.axis_major_length)
                # Append diameter in micrometers
                data.append(snowflake.equivalent_diameter_area*pixel_size)
                # Append complexity parameter of snowflake
                data.append(snowflake.perimeter/(math.pi*snowflake.equivalent_diameter_area))                 
    else:
        err("Image discarded due to insufficient sharp edges.")
    
    if visual:
        plot_ellipse_overlay(image, data, 1000)
        
    if save:
        pass # TODO: implement saving functionality  
    
    end = time.time_ns()
    elapsed = end - start
        

    return (data, elapsed)

