import os
import numpy as np
import cv2
from skimage.measure import label, regionprops, find_contours
from skimage import filters, morphology
from processor import calculate_sharp_edges, gamma

import time
import pandas as pd
import math
from utils.plotutils import plot_ellipse_overlay, skimage_show_plot
from utils.utils import info, warn, err, header

def analyse_image(image: np.ndarray, thresh: int = 300, visual=False, save=False, save_path: str = "") -> tuple[list, float]:
    """Continuously processes images from the queue until the process is stopped."""

    # Initialization of image counter and data container

    # Define the size of a pixel
    pixel_size = 5.86 # in [um]

    # Get image from queue and flip it 180 degrees

    # Remove the high frequency noise with the gaussian blur filter
    # smoothed_image = cv2.GaussianBlur(image, (25, 25), sigmaX=2, sigmaY=2)
    res = cv2.GaussianBlur(image, (7, 7), sigmaX=2, sigmaY=2) # 11,5
    data = []

    # Save image if the amount of sharp edges in it are above a defined threshold
    start = time.time_ns()
    if calculate_sharp_edges(res) > thresh: 
        # Create file in previously generated folder

        # Create binary image with defined threshold
        
        
        # thresh = 10
        # binary_image = ((res > thresh) * 255)
        
        # threshold = filters.threshold_otsu(res)
        # res = (res > threshold).astype(np.uint8)*255
        
        cv2.normalize(src=res, dst=res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, res = cv2.threshold(res, 10, 255, cv2.THRESH_OTSU)
        # kernel = np.ones((3,3),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        scale = 70
        res = morphology.remove_small_objects(res, scale)
        res = morphology.remove_small_holes(res, scale)
        #cv2.imshow("Binary Image", binary_image.astype(np.uint8)*255)
        # Morphological closing to fill small holes inside snowlakes
        kernel = np.ones((20, 20), np.uint8)
        closed_binary_image = cv2.morphologyEx(res.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
        # Calculate regions of snowflakes in image
        
        label_img = label(closed_binary_image)
        snowflakes = regionprops(label_img)
        # Initialize a list to store characteristic values of snowflakes
        snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)
        flake = 0

        for snowflake in snowflakes:
            # Only save the snowflakes that are bigger than 50 pixel in diameter
            
            if snowflake.equivalent_diameter_area >= scale: # 100
                label_i = snowflake.label
                contour = find_contours(label_img == label_i, 0.5)
                if visual:
                    skimage_show_plot(snowflake, gamma(image, gamma=0.4), contour)
                
                flake += 1
                snowflake_img = snowflake.image_filled
                bbox = snowflake.bbox
                sliced_img = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # cv2.imshow("Detected Snowflake", snowflake_img.astype(np.uint8)*255)
                # print(f"Intensity average: {np.mean(sliced_img)}, std: {np.std(sliced_img)}")
                if save:
                    filename = f"snowflake_{flake}_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png"
                    filename2 = f"snowflake_{flake}_binary_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png"
                    filename2 = os.path.join(save_path, filename2)
                    filename = os.path.join(save_path, filename)
                    cv2.imwrite(filename, sliced_img)
                    cv2.imwrite(filename2, snowflake_img.astype(np.uint8)*255)
                
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
                # Append area in square micrometers
                data.append(snowflake.area*pixel_size*pixel_size)
                # Append perimeter in micrometers
                data.append(snowflake.perimeter*pixel_size) # perimeter_crofton
                # Append solidity of snowflake
                data.append(snowflake.solidity) # ratio of pixels in the convex hull to those in the region             
    else:
        err("Image discarded due to insufficient sharp edges.")
    
    if visual:
        # plot_ellipse_overlay(image, data, 10000)
        pass
    
    end = time.time_ns()
    elapsed = end - start
        

    return (data, elapsed)

