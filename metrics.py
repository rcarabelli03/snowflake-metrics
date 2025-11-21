import os
import numpy as np
import cv2
from skimage.measure import label, regionprops, find_contours
from skimage import filters, morphology
from processor import calculate_sharp_edges, gamma, gradient_angle

import time
import pandas as pd
import math
from utils.plotutils import plot_ellipse_overlay, skimage_show_plot
from utils.utils import info, warn, err, header


snowflake_nr = 0


def analyse_image(image: np.ndarray, thresh: int = 300, plot=False, display_plot=True, save=False, save_path: str = "", folder_desc: str = "") -> tuple[list, float]:
    """Continuously processes images from the queue until the process is stopped."""

    # Initialization of image counter and data container
    global snowflake_nr
    # Define the size of a pixel
    pixel_size = 5.86 # in [um]
    
    original_img = image.copy()
    # Get image from queue and flip it 180 degrees

    # Remove the high frequency noise with the gaussian blur filter
    # smoothed_image = cv2.GaussianBlur(image, (25, 25), sigmaX=2, sigmaY=2)
    res = cv2.GaussianBlur(image, (7, 7), sigmaX=2, sigmaY=2) # 11,5
    data = []

    # Save image if the amount of sharp edges in it are above a defined threshold
    start = time.time_ns()
    number_of_sharp_edges = calculate_sharp_edges(res)
    if number_of_sharp_edges > thresh:
        
        # thresh = 10
        # binary_image = ((res > thresh) * 255)
        
        # threshold = filters.threshold_otsu(res)
        # res = (res > threshold).astype(np.uint8)*255
        
        cv2.normalize(src=res, dst=res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        normalised_image = res.copy()
        
        cv2.imshow("Normalisation", res.astype(np.uint8)*255)
        cv2.waitKey(1)
        
        _, res = cv2.threshold(res, 20, 255, cv2.THRESH_OTSU)
        # res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
        # kernel = np.ones((3,3),np.uint8)
        # res = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel, iterations = 2)
        scale = 70
        res = morphology.remove_small_objects(res, scale)
        res = morphology.remove_small_holes(res, scale)
        #cv2.imshow("Binary Image", binary_image.astype(np.uint8)*255)
        # Morphological closing to fill small holes inside snowlakes
        kernel = np.ones((20, 20), np.uint8)
        closed_binary_image = cv2.morphologyEx(res.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
        
        cv2.imshow("Closed Binary Image", closed_binary_image.astype(np.uint8)*255)
        cv2.waitKey(1)
        
        label_img = label(closed_binary_image)
        snowflakes = regionprops(label_img)
        # Initialize a list to store characteristic values of snowflakes
        snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)
        flake = 0

        for snowflake in snowflakes:
            # Only save the snowflakes that are bigger than 50 pixel in diameter
            snowflake_nr += 1
            if snowflake.equivalent_diameter_area >= scale: # 100
                display_plot = False if snowflake.equivalent_diameter_area > 1000 else display_plot # filter tooo big flakes for visualizations
                
                label_i = snowflake.label
                contour = find_contours(label_img == label_i, 0.5)
                descriptor = f"area-{int(snowflake.equivalent_diameter_area)}um"
                flake += 1
                
                snowflake_img = snowflake.image_filled
                bbox = snowflake.bbox
                sliced_img = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # cv2.imshow("Detected Snowflake", snowflake_img.astype(np.uint8)*255)
                # print(f"Intensity average: {np.mean(sliced_img)}, std: {np.std(sliced_img)}")
                subfolder = f"snowflake_{snowflake_nr}_" + folder_desc # prevents overlap
                snowflake_path = os.path.join(save_path, subfolder)
                
                if save:
                    os.makedirs(snowflake_path, exist_ok=True)
                    original = f"snowflake_{flake}_original_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png"
                    normalised_imv = f"snowflake_{flake}_normalised_inv_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png"
                    isolated_flake = f"snowflake_{flake}_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png"
                    binarised_isolated_flake = f"snowflake_{flake}_binary_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png"
                    original = os.path.join(snowflake_path, original)
                    normalised_imv = os.path.join(snowflake_path, normalised_imv)
                    isolated_flake = os.path.join(snowflake_path, isolated_flake)
                    binarised_isolated_flake = os.path.join(snowflake_path, binarised_isolated_flake)
                    cv2.imwrite(original, original_img)
                    cv2.imwrite(normalised_imv, normalised_image)
                    cv2.imwrite(isolated_flake, sliced_img)
                    cv2.imwrite(binarised_isolated_flake, snowflake_img.astype(np.uint8)*255)
                
                if plot:
                    skimage_show_plot(snowflake, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image), contour, display=display_plot, save=save, save_path=snowflake_path, descriptor=descriptor, flake_id=flake)
                    
                # Append center and axes of snowflake
                tmp = []
                tmp.append(snowflake.centroid[1]) # centroid is (row, col) -> (y,x)
                tmp.append(snowflake.centroid[0]) # centroid is (row, col) -> (y,x)
                tmp.append(snowflake.axis_major_length)
                tmp.append(snowflake.axis_minor_length)
                # Append orientation of snowflake in grad
                tmp.append(snowflake.orientation)
                # Append aspect ratio of snowflake
                tmp.append(snowflake.axis_minor_length/snowflake.axis_major_length)
                # Append diameter in micrometers
                tmp.append(snowflake.equivalent_diameter_area*pixel_size)
                # Append complexity parameter of snowflake
                tmp.append(snowflake.perimeter/(math.pi*snowflake.equivalent_diameter_area)) 
                # Append area in square micrometers
                tmp.append(snowflake.area*pixel_size*pixel_size)
                # Append perimeter in micrometers
                tmp.append(snowflake.perimeter*pixel_size) # perimeter_crofton
                # Append solidity of snowflake
                tmp.append(snowflake.solidity) # ratio of pixels in the convex hull to those in the region 
                
                data.extend(tmp)
                
                if save:
                    metrics_array = np.array(tmp).squeeze()
                    print(f"Metrics array: {metrics_array}")        
                    df = pd.DataFrame(columns=['centroid_x', 'centroid_y', 'axis_major_length', 'axis_minor_length', 'orientation_rad', 'aspect_ratio', 'diameter_um', 'complexity', 'area_um2', 'perimeter_um', 'solidity'])
                    df.loc[0] = metrics_array
                    csv_filename = f"snowflake_{flake}_{int(snowflake.equivalent_diameter_area*pixel_size)}um_metrics.csv"
                    csv_filepath = os.path.join(snowflake_path, csv_filename)
                    df.to_csv(csv_filepath)
                
                if plot:
                    plot_ellipse_overlay(gamma(image,0.4), tmp, 1, save=save, save_path=snowflake_path, descriptor=descriptor, flake_id=flake)
    else:
        err("Image discarded due to insufficient sharp edges.")
        
        
    end = time.time_ns()
    elapsed = end - start
        

    return (data, elapsed)

