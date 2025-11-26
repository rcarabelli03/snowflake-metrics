import os
import numpy as np
import cv2
from skimage.measure import label, regionprops, find_contours
from skimage import filters, morphology
from processor import calculate_sharp_edges, gamma, gradient_angle

import time
import typing
import pandas as pd
import math
from src.utils.plotutils import plot_ellipse_overlay, skimage_show_plot, plot_histogram, write_image
from src.utils.utils import info, warn, err, header

snowflake_nr = 0

class Analyser:
    def __init__(self, config: dict, save_path: str = ""):
        assert len(config) > 0, "Configuration dictionary is empty."
        self.save_path = save_path
        self.config = config
        
        self.ksize = config["analysis"]["gaussian_blur"]["kernel_size"]
        self.sigma = config["analysis"]["gaussian_blur"]["sigma"]
        self.thresh = config["analysis"]["thresholding"]["value"]
        
        self.sharp_angle_thresh = config["analysis"]["flake_sharp_angle_thresh"]
        self.scale = config["analysis"]["remove_small_objects"]["small_object_size"]
        
        self.closing_ksize = config["analysis"]["morphology"]["closing"]["kernel_size"]
        self.iterations = config["analysis"]["morphology"]["closing"]["iterations"]
        
        self.area_thresh = config["plot"]["max_display_area"]
        self.contour_level = config["analysis"]["find_contour_level"]
        
        self.enable_remove_small = config["analysis"]["remove_small_objects"]["enabled"]
        
        self.plot = config["analysis"]["plot"]["enable"]
        self.display_plot = config["analysis"]["plot"]["display"]
        self.save = config["analysis"]["plot"]["save"]
        self.show_intermediate = config["debug"]["show_intermediate_images"]
        
    def analyse_image(self, image: np.ndarray, folder_desc: str = "") -> tuple[list, float, typing.Optional[str]]:
                
        # Initialization of image counter and data container
        global snowflake_nr
        subfolder = None
        # Define the size of a pixel
        pixel_size = 5.86 # in [um]
        
        original_img = image.copy()
        # Get image from queue and flip it 180 degrees

        # Remove the high frequency noise with the gaussian blur filter
        # smoothed_image = cv2.GaussianBlur(image, (25, 25), sigmaX=2, sigmaY=2)
        res = cv2.GaussianBlur(image, (self.ksize, self.ksize), sigmaX=self.sigma, sigmaY=self.sigma) # 11,5
        data = []

        # Save image if the amount of sharp edges in it are above a defined threshold
        start = time.time_ns()
        number_of_sharp_edges = calculate_sharp_edges(res)
            
            # thresh = 10
            # binary_image = ((res > thresh) * 255)
            
            # threshold = filters.threshold_otsu(res)
            # res = (res > threshold).astype(np.uint8)*255
            
        cv2.normalize(src=res, dst=res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        normalised_image = res.copy()
        inversion = cv2.bitwise_not(res)
        
        if self.show_intermediate:
            cv2.imshow("Normalisation", res.astype(np.uint8))
            cv2.imshow("Inversion", inversion.astype(np.uint8))
            cv2.waitKey(1)
            
        if number_of_sharp_edges > self.sharp_angle_thresh:
            _, res = cv2.threshold(res, self.thresh, 255, cv2.THRESH_OTSU)
            # res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
            # kernel = np.ones((3,3),np.uint8)
            # res = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel, iterations = 2)
            if self.enable_remove_small:
                res = morphology.remove_small_objects(res, self.scale)
                res = morphology.remove_small_holes(res, self.scale)
                
            # Morphological closing to fill small holes inside snowflakes
            
            kernel = np.ones((self.closing_ksize, self.closing_ksize), np.uint8)
            closed_binary_image = cv2.morphologyEx(res.astype(np.uint8),
                                                cv2.MORPH_CLOSE,
                                                kernel,
                                                iterations=self.iterations)
            
            cv2.imshow("Closed Binary Image", closed_binary_image.astype(np.uint8)*255)
            cv2.waitKey(1)
            
            label_img = label(closed_binary_image)
            snowflakes = regionprops(label_img)
            # Initialize a list to store characteristic values of snowflakes
            snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)
            flake = 0 # local snowflake counter, resets for each image

            for snowflake in snowflakes:
                # Only save the snowflakes that are bigger than 50 pixel in diameter
                if snowflake.equivalent_diameter_area >= self.scale: # 100
                    snowflake_nr += 1 # global snowflake counter, unique across images
                    
                    # filter tooo big flakes for visualizations
                    display_plot = False if snowflake.equivalent_diameter_area > self.area_thresh else self.display_plot
                    
                    label_i = snowflake.label
                    contour = find_contours(label_img == label_i, self.contour_level)
                    flake += 1
                    
                    snowflake_img = snowflake.image_filled
                    bbox = snowflake.bbox
                    sliced_img = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                    normalised_slice = normalised_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                    # cv2.imshow("Detected Snowflake", snowflake_img.astype(np.uint8)*255)
                    # print(f"Intensity average: {np.mean(sliced_img)}, std: {np.std(sliced_img)}")
                    subfolder = f"{snowflake_nr}_{flake}_" + folder_desc # prevents overlap
                    snowflake_path = os.path.join(self.save_path, subfolder)
                    
                    if self.save:
                        os.makedirs(snowflake_path, exist_ok=True)
                        write_image(original_img, save_path=snowflake_path, filename=f"snowflake_{snowflake_nr}_{flake}_original_image_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png")
                        write_image(normalised_image, save_path=snowflake_path, filename=f"snowflake_{snowflake_nr}_{flake}_normalised_image_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png")
                        write_image(inversion, save_path=snowflake_path, filename=f"snowflake_{snowflake_nr}_{flake}_inverted_image_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png")
                        write_image(sliced_img, save_path=snowflake_path, filename=f"snowflake_{snowflake_nr}_{flake}_cropped_flake_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png")
                        write_image(normalised_slice, save_path=snowflake_path, filename=f"snowflake_{snowflake_nr}_{flake}_normalised_cropped_flake_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png")
                        write_image(snowflake_img.astype(np.uint8)*255, save_path=snowflake_path, filename=f"snowflake_{snowflake_nr}_{flake}_binary_image_{int(snowflake.equivalent_diameter_area*pixel_size)}um.png")
                    
                    if self.plot:
                        skimage_show_plot(snowflake, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image), contour, display=display_plot, save=self.save, save_path=snowflake_path, flake_id=flake)
                        plot_histogram(original_img, title=f"intensity_histogram_{snowflake_nr}_{flake}_original", xlabel="Intensity", ylabel="Frequency", bins=256, visual=display_plot, save=self.save, save_path=snowflake_path)
                        plot_histogram(normalised_image, title=f"intensity_histogram_{snowflake_nr}_{flake}_normalised", xlabel="Intensity", ylabel="Frequency", bins=256, visual=display_plot, save=self.save, save_path=snowflake_path)
                        plot_histogram(sliced_img, title=f"intensity_histogram_{snowflake_nr}_{flake}_cropped_flake", xlabel="Intensity", ylabel="Frequency", bins=256, visual=display_plot, save=self.save, save_path=snowflake_path)
                        
                    # Append center and axes of snowflake
                    tmp = []
                    tmp.append(snowflake.centroid[1]) # centroid is (row, col) -> (y,x)
                    tmp.append(snowflake.centroid[0]) # centroid is (row, col) -> (y,x)
                    tmp.append(snowflake.axis_major_length)
                    tmp.append(snowflake.axis_minor_length)
                    # Append orientation of snowflake in grad
                    tmp.append(snowflake.orientation)
                    # # bbox is (min_row, min_col, max_row, max_col)
                    # tmp.append(snowflake.bbox[1]) # min_col (x)
                    # tmp.append(snowflake.bbox[0]) # min_row (y)
                    # tmp.append(snowflake.bbox[3]) # max_col (x)
                    # tmp.append(snowflake.bbox[2]) # max_row (y)                
                    # Append aspect ratio of snowflake
                    tmp.append(snowflake.axis_minor_length/snowflake.axis_major_length)
                    # Append diameter in micrometers
                    tmp.append(snowflake.equivalent_diameter_area) # in [pixels] 
                    # Append complexity parameter of snowflake
                    tmp.append(snowflake.perimeter/(math.pi*snowflake.equivalent_diameter_area)) 
                    # Append area in square micrometers
                    tmp.append(snowflake.area) # number of pixels in region
                    # Append perimeter in micrometers
                    tmp.append(snowflake.perimeter) # perimeter_crofton
                    # Append solidity of snowflake
                    tmp.append(snowflake.solidity) # ratio of pixels in the convex hull to those in the region 
                    
                    data.extend(tmp)
                    
                    if self.save:
                        metrics_array = np.array(tmp).squeeze()
                        print(f"Metrics array: {metrics_array}")        
                        df = pd.DataFrame(columns=['centroid_x', 'centroid_y', 'axis_major_length', 'axis_minor_length', 'orientation_rad', 'aspect_ratio', 'diameter_um', 'complexity', 'area_um2', 'perimeter_um', 'solidity'])
                        df.loc[0] = metrics_array
                        csv_filename = f"snowflake_{flake}_{int(snowflake.equivalent_diameter_area*pixel_size)}um_metrics.csv"
                        csv_filepath = os.path.join(snowflake_path, csv_filename)
                        df.to_csv(csv_filepath)
                    
                    if self.plot:
                        plot_ellipse_overlay(gamma(image,0.4), tmp, 1, save=self.save, save_path=snowflake_path, flake_id=flake)
        else:
            err("Image discarded due to insufficient sharp edges.")
            
            
        end = time.time_ns()
        elapsed = end - start
            

        return (data, elapsed, subfolder)


overlap_prevention_counter = 0

def alternative_analyse_image(image: np.ndarray, config: dict, save_path: str = "", folder_desc: str = "") -> list:
    
    ## stuff
    ksize = 11
    sigma = 5
    s_ksize = 3
    threshold = 1000
    threshold_angle = 50
    global overlap_prevention_counter
    overlap_prevention_counter += 1
    
    res = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) # 11,5
    res = gradient_angle(res, kernel_size=s_ksize).astype(np.float32)
    assert res is not None, "Gradient angle computation failed."
    print(f"Gradient angle stats: min={np.min(res)}, max={np.max(res)}, mean={np.mean(res)}, std={np.std(res)}")
    
    path = os.path.join(save_path, folder_desc)
    if folder_desc == "default": # FIXME: remove in future
        path = os.path.join(path, f"flake_alternative_{overlap_prevention_counter}")
    os.makedirs(path, exist_ok=True)
    # calculate histogram
    plot_histogram(res, title="gradient_angle_histogram", xlabel="Angle (degrees)", ylabel="Frequency", bins=256, visual=False, save=True, save_path=path)
    res = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    sharp_edges = int(np.sum(res > threshold_angle))
    print(f"Number of sharp edges (>{threshold_angle}): {sharp_edges}")
    
    
    _, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    if sharp_edges > threshold:
        cv2.imwrite(os.path.join(path, "reference_selected_by_grad.png"), image)
        cv2.imwrite(os.path.join(path, "selected_by_grad_angle.png"), res)
        cv2.imwrite(os.path.join(path, "selected_by_grad_angle_binary.png"), opening)
        
        scale = config["analysis"]["remove_small_objects"]["small_object_size"]
        if config["analysis"]["remove_small_objects"]["enabled"]:
            res = morphology.remove_small_objects(res, scale)
            res = morphology.remove_small_holes(res, scale)
            
        # Morphological closing to fill small holes inside snowflakes
        n = config["analysis"]["morphology"]["closing"]["kernel_size"]
        iterations = config["analysis"]["morphology"]["closing"]["iterations"]
        
        kernel = np.ones((n, n), np.uint8)
        closed_binary_image = cv2.morphologyEx(res.astype(np.uint8), # opening
                                               cv2.MORPH_CLOSE,
                                               kernel,
                                               iterations=iterations)
        
        cv2.imshow("Closed Binary Image", closed_binary_image.astype(np.uint8)*255)
        cv2.waitKey(1)
        
        label_img = label(closed_binary_image)
        snowflakes = regionprops(label_img)
        # Initialize a list to store characteristic values of snowflakes
        snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)
        flake = 0 # local snowflake counter, resets for each image

        for snowflake in snowflakes:
            # Only save the snowflakes that are bigger than 50 pixel in diameter
            if snowflake.equivalent_diameter_area >= scale: # 100
                flake += 1
                label_i = snowflake.label
                contour = find_contours(label_img == label_i, config["analysis"]["find_contour_level"])
                
                snowflake_img = snowflake.image_filled
                bbox = snowflake.bbox
                sliced_img = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                normalised_slice = res[bbox[0]:bbox[2], bbox[1]:bbox[3]].astype(np.uint8)
                # cv2.imshow("Detected Snowflake", snowflake_img.astype(np.uint8)*255)
                # print(f"Intensity average: {np.mean(sliced_img)}, std: {np.std(sliced_img)}")
                snowflake_path = path
                isolated_flake = f"alt_snowflake_isolated_{snowflake_nr}_{flake}_{int(snowflake.equivalent_diameter_area)}um.png"
                normalised_flake = f"alt_snowflake_{snowflake_nr}_{flake}_normalised_slice_image_{int(snowflake.equivalent_diameter_area)}um.png"
                binarised_isolated_flake = f"alt_snowflake_{snowflake_nr}_{flake}_binary_image_{int(snowflake.equivalent_diameter_area)}um.png"
                isolated_flake = os.path.join(snowflake_path, isolated_flake)
                normalised_flake = os.path.join(snowflake_path, normalised_flake)
                binarised_isolated_flake = os.path.join(snowflake_path, binarised_isolated_flake)
                cv2.imwrite(isolated_flake, sliced_img)
                cv2.imwrite(normalised_flake, normalised_slice)
                cv2.imwrite(binarised_isolated_flake, snowflake_img.astype(np.uint8)*255)
        
    
    return []

