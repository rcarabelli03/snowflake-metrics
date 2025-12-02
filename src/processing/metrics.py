import os
import numpy as np
import cv2
import skimage as ski
from skimage.measure import label, regionprops, regionprops_table, find_contours
from skimage import filters, morphology

import time
import typing
import pandas as pd
import math
from processing.processor import calculate_sharp_edges, gamma, gradient_angle
from utils.io.plotutils import plot_ellipse_overlay, skimage_show_plot, plot_histogram, write_image
from utils.logger import info, warn, err, header
from utils.results_wrapper import IntermediateImage, AnalysisResult

snowflake_nr = 0
snowflake_2_nr = 0
H, W = (1200, 1920) 

class Analyser:
    def __init__(self, config: dict, save_path: str = ""):
        assert len(config) > 0, "Configuration dictionary is empty."
        self.save_path = save_path
        self.config = config
        
        self.props = config["metrics"]["metrics"]
        
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
        
        self.plot = config["plot"]["enable"]
        self.display_plot = config["plot"]["display"]["matplotlib"]
        self.cv2_display = config["plot"]["display"]["cv2_ellipses"]
        self.save = config["plot"]["save"]
        self.show_intermediate = config["debug"]["show_intermediate_images"]
        self.print = config["verbose"]["enabled"] or config["debug"]["enabled"]
        
        self.previous_image: np.ndarray = np.ones((H, W), dtype=np.uint8) * 255  # initialize with a white image
        
        info(f"""Analyser initialized with config:
            Gaussian Blur: kernel_size={self.ksize}, sigma={self.sigma}
            Thresholding value: {self.thresh}
            Sharp angle threshold: {self.sharp_angle_thresh}
            Remove small objects enabled: {self.enable_remove_small}, size: {self.scale}
            Morphological Closing: kernel_size={self.closing_ksize}, iterations={self.iterations}
            Plotting enabled: {self.plot}, display: {self.display_plot}, save: {self.save}
            Debug show intermediate images: {self.show_intermediate}""")
            
        self.duplicate_count = 0
        
    def __del__(self):
        info(f"Detected {self.duplicate_count} duplicates (or even multiply duplicate images) in original dataset.")
        
    def analyse_image(self, image: np.ndarray, folder_desc: str = "") -> tuple[typing.Optional[pd.DataFrame], float]:
                
        # Initialization of image counter and data container
        global snowflake_nr
        global snowflake_2_nr
        # Define the size of a pixel
        pixel_size = 5.86 # in [um]
        
        # So, for reasons of lazyness, and unwillingness to refactor too much code, I will only return analysis results from the first algorithm
        # this is mostly bc i dont wanna handle receiveing arbitrary number of dataframes from multiple algorithms
        # in main
        # also it's just for verbose output anyway, so who cares
        df_sel = pd.DataFrame()
        
        # Check for identical image to previous (skip)
        original_img = image.copy()
        if np.array_equal(image, self.previous_image):
            warn("Identical image detected as previous one; skipping analysis.") # causes skew in resulting data
            self.duplicate_count += 1
            return (None, 0.0)
            
        self.previous_image = image.copy()

        # Run and time analysis algorithms
        start = time.time_ns()
        result_1 = self._analysis_algorithm_1(image)
        result_2 = self._analysis_algorithm_2(image)
        end = time.time_ns()
        elapsed = end - start
        
        if (result_1 is not None) and result_1.has_detections: # i mean, it should have detections if we got here
            info(f"Analysis algorithm detected {len(result_1.detections)} potential snowflakes.")          
            # extract intermediate images
            assert result_1.intermediates is not None, "Intermediates should not be None when detections are present."
            contour = find_contours(result_1.labelled_image, level=self.contour_level)
            normalised_image = result_1.intermediates[1].image
            
            save_path = self.save_path + "/algorithm_1/"
            
            # extract data and sort by size
            snowflakes = result_1.detections
            snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)
            snowflakes = [(s, potential_flake) for potential_flake, s in enumerate(snowflakes) if s.equivalent_diameter_area > self.scale]
            
            props = regionprops_table(                  # yes I know, this runs regionprops again, but i get a nice df, so whatever
                label_image=result_1.labelled_image,
                properties=self.props,
            )
            df_props = pd.DataFrame(props)
            df_sel = df_props.sort_values(by="equivalent_diameter_area", ascending=False).reset_index(drop=True)
            
            # rename columns to have consistent naming (centroid and centroid_local)
            df_sel = df_sel.rename(columns={
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
                "centroid_local-0": "centroid_y_local",
                "centroid_local-1": "centroid_x_local",
            })
            # add other conventional metrics (complexity, aspect ratio, etc.)
            df_sel["complexity"] = df_sel["perimeter"]/(df_sel["equivalent_diameter_area"]*math.pi)
            df_sel["aspect_ratio"] = df_sel["axis_minor_length"]/df_sel["axis_major_length"]
            
            flake_id = 0
            for snowflake, potential_flake in snowflakes:
                snowflake_nr += 1
                
                path = os.path.join(save_path, f"{snowflake_nr}_{flake_id}_" + folder_desc)
                tmp = df_sel.iloc[potential_flake].to_dict()
                flake_metrics = df_sel[df_sel["equivalent_diameter_area"] == snowflake.equivalent_diameter_area]
                info(str(flake_metrics))
                
                display_plot = self.display_plot if snowflake.equivalent_diameter_area*pixel_size < self.area_thresh else False
                
                snowflake_img = snowflake.image_filled
                bbox = snowflake.bbox
                sliced_img = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                normalised_slice = normalised_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                
                avg_intensity = np.mean(sliced_img)
                std_intensity = np.std(sliced_img)
                df_sel[f"avg_intensity"] = avg_intensity
                df_sel[f"std_intensity"] = std_intensity
                
                avg_gradient_angle = np.mean(gradient_angle(
                    sliced_img,
                    3
                ).astype(np.float32))
                
                if avg_gradient_angle > 30.:
                    header(f"High average gradient angle detected: {avg_gradient_angle:.2f} degrees for flake {snowflake_nr}_{flake_id}")
                
                df_sel[f"gradient_angle"] = avg_gradient_angle
        
                # print(f"Intensity average: {np.mean(sliced_img)}, std: {np.std(sliced_img)}")
                if self.save:
                    os.makedirs(path, exist_ok=True)
                    write_image(original_img, save_path=path, filename="original_image.png")
                    result_1.save(save_path=path, folder_desc=folder_desc)
                    
                    flake_metrics.to_csv(os.path.join(path, "metrics.csv"), index=False)
                    write_image(sliced_img, save_path=path, filename=f"{folder_desc}_cropped_flake_image.png")
                    write_image(normalised_slice, save_path=path, filename=f"{folder_desc}_normalised_cropped_flake_image.png")
                    write_image(snowflake_img.astype(np.uint8)*255, save_path=path, filename=f"{folder_desc}_cropped_binarised_flake_image.png")
                
                if self.plot:
                    skimage_show_plot(snowflake, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image), contour, display=display_plot, save=self.save, save_path=path, flake_id=flake_id)
                    plot_histogram(sliced_img, title=f"intensity_histogram_cropped_flake", xlabel="Intensity", ylabel="Frequency", bins=256, visual=display_plot, save=self.save, save_path=path)
                    plot_histogram(original_img, title=f"intensity_histogram_{snowflake_nr}_{flake_id}_original", xlabel="Intensity", ylabel="Frequency", bins=256, visual=display_plot, save=self.save, save_path=path)
                    plot_histogram(normalised_image, title=f"intensity_histogram_{snowflake_nr}_{flake_id}_normalised", xlabel="Intensity", ylabel="Frequency", bins=256, visual=display_plot, save=self.save, save_path=path)
                    
                    plot_ellipse_overlay(gamma(image,0.4), tmp, 1, visual=self.cv2_display, save=self.save, save_path=path, flake_id=flake_id)
                
                flake_id += 1
                
        if result_2 is not None and result_2.has_detections:
            info(f"Second analysis algorithm detected {len(result_2.detections)} potential snowflakes.")
            contour = find_contours(result_2.labelled_image, level=self.contour_level)
            
            save_path = self.save_path + "/algorithm_2/"

            # extract data and sort by size
            snowflakes = result_2.detections
            snowflakes.sort(key=lambda x: x.equivalent_diameter_area, reverse=True)
            snowflakes = [(s, potential_flake) for potential_flake, s in enumerate(snowflakes) if s.equivalent_diameter_area > self.scale]
            
            flake_id = 0
            for snowflake, potential_flake in snowflakes:
                snowflake_2_nr += 1
                
                data = self.snowflake_data(snowflake, image) # dict of metrics
                df_tmp = pd.DataFrame.from_dict(data, orient='index').T
                
                normalised = image.copy()
                cv2.normalize(image, normalised, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                inversion = cv2.bitwise_not(normalised)
                _, sharp_edges_image = calculate_sharp_edges(image=normalised)
                
                sliced_img = image[snowflake.bbox[0]:snowflake.bbox[2], snowflake.bbox[1]:snowflake.bbox[3]]
                sliced_normalised = normalised[snowflake.bbox[0]:snowflake.bbox[2], snowflake.bbox[1]:snowflake.bbox[3]]
                
                               
                path = os.path.join(save_path, f"{snowflake_2_nr}_{flake_id}_" + folder_desc)
                if self.save:
                    os.makedirs(path, exist_ok=True)
                    write_image(original_img, save_path=path, filename=f"{folder_desc}_original_image.png")
                    write_image(normalised, save_path=path, filename=f"{folder_desc}_normalised_image.png")
                    write_image(inversion, save_path=path, filename=f"{folder_desc}_inversion_image.png")
                    write_image(sharp_edges_image, save_path=path, filename=f"{folder_desc}_sharp_edges_image.png")
                    write_image(sliced_img, save_path=path, filename=f"{folder_desc}_cropped_flake_image.png")
                    write_image(sliced_normalised, save_path=path, filename=f"{folder_desc}_normalised_cropped_flake_image.png")
                    result_2.save(save_path=path, folder_desc=folder_desc)
                    
                    df_tmp.to_csv(os.path.join(path, "metrics.csv"), index=False)
                
                if self.plot:
                    skimage_show_plot(snowflake, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image), contour, display=self.display_plot, save=self.save, save_path=path, flake_id=flake_id)
                
                flake_id += 1


        return (df_sel, elapsed)

    
    
    def snowflake_data(self, snowflake: ski.measure._regionprops.RegionProperties, image: np.ndarray) -> dict:
        # Extract bounding box
        bbox = snowflake.bbox
        sliced_img = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        centroid = snowflake.centroid
        centroid_local = snowflake.centroid_local
        
        # Calculate average intensity and standard deviation
        avg_intensity = np.mean(sliced_img)
        std_intensity = np.std(sliced_img)
        
        # Calculate average gradient angle
        avg_gradient_angle = np.mean(gradient_angle(
            image=sliced_img,
            kernel_size=3
        ).astype(np.float32))
        
        # compile data dictionary
        data = {
            "centroid_y": centroid[0],
            "centroid_x": centroid[1],
            "centroid_y_local": centroid_local[0],
            "centroid_x_local": centroid_local[1],
            "axis_major_length": snowflake.axis_major_length,
            "axis_minor_length": snowflake.axis_minor_length,
            "orientation": snowflake.orientation,
            "equivalent_diameter_area": snowflake.equivalent_diameter_area,
            "area": snowflake.area,
            "area_convex": snowflake.area_convex,
            "perimeter": snowflake.perimeter,
            "feret_diameter_max": snowflake.feret_diameter_max,
            "solidity": snowflake.solidity,
            "avg_intensity": avg_intensity,
            "std_intensity": std_intensity,
            "avg_gradient_angle": avg_gradient_angle
            }
        
        return data
    
    
    
    #############################################################################################
    # Analysis algorithms
    #############################################################################################
    
    def _analysis_algorithm_1(self, image: np.ndarray) -> typing.Optional[AnalysisResult]:        
        res = np.clip(image, 6, 255)
        # Remove the high frequency noise with the gaussian blur filter
        res = cv2.GaussianBlur(res, (self.ksize, self.ksize), sigmaX=self.sigma, sigmaY=self.sigma) # 11,5

        # Save image if the amount of sharp edges in it are above a defined threshold
        number_of_sharp_edges, _ = calculate_sharp_edges(image=res)
        # NOTE: normalise before sharp edge calculation to increase acceptance rate
        cv2.normalize(src=res, dst=res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Gives a nicer visual representation of the edges
        _, sharp_edges_image = calculate_sharp_edges(image=res)
        
        normalised_image = res.copy()
        image_for_overlay = res.copy()
        inversion = cv2.bitwise_not(res)
            
        if number_of_sharp_edges > self.sharp_angle_thresh:
            info(f"Image accepted for analysis: {number_of_sharp_edges} sharp edges detected.")
            _, res = cv2.threshold(res, self.thresh, 255, cv2.THRESH_OTSU)
            
            # overlay thresh and gradient on original for debugging
            alpha = 0.6
            overlay_thresh = cv2.addWeighted(res.astype(np.uint8), alpha, image.astype(np.uint8), 1-alpha, 0)
            overlay_thresh_normalised = cv2.addWeighted(res.astype(np.uint8), alpha, image_for_overlay.astype(np.uint8), 1-alpha, 0)
            overlay_gradient = cv2.addWeighted(sharp_edges_image.astype(np.uint8), alpha, image_for_overlay.astype(np.uint8), 1-alpha, 0)
            # the same but grad is in different color channel to original
            overlay_gradient_color = cv2.cvtColor(normalised_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlay_gradient_color[:, :, 1] = cv2.addWeighted(sharp_edges_image.astype(np.uint8), alpha, image_for_overlay.astype(np.uint8), 1-alpha, 0)
            overlay_gradient_color[:, :, 0] = normalised_image.astype(np.uint8)
            overlay_gradient_color[:, :, 2] = normalised_image.astype(np.uint8)
            
            if self.enable_remove_small:
                res = morphology.remove_small_objects(res, self.scale)
                res = morphology.remove_small_holes(res.astype(bool), self.scale).astype(np.uint8)
                
            # Morphological closing to fill small holes inside snowflakes
            kernel = np.ones((self.closing_ksize, self.closing_ksize), np.uint8)
            closed_binary_image = cv2.morphologyEx(res.astype(np.uint8),
                                                cv2.MORPH_CLOSE,
                                                kernel,
                                                iterations=self.iterations).astype(np.uint8)*255
            
            
            label_img = label(closed_binary_image)
            data = regionprops(label_img)
            assert isinstance(label_img, np.ndarray), "Apparently I guessed wrong about skimage.measure.label output type."
            
            intermediates = [
                IntermediateImage("closed_binary_image", closed_binary_image),
                IntermediateImage("normalised_image", normalised_image),
                IntermediateImage("inversion_image", inversion),
                IntermediateImage("sharp_edges_image", sharp_edges_image),
                IntermediateImage("overlay_threshold", overlay_thresh),
                IntermediateImage("overlay_threshold_normalised", overlay_thresh_normalised),
                IntermediateImage("overlay_gradient", overlay_gradient),
                IntermediateImage("overlay_gradient_color", overlay_gradient_color),
            ]
            
            result = AnalysisResult(
                pipeline_name="analysis_algorithm_1",
                detections=data,
                labelled_image=label_img,
                intermediates=intermediates
            )
            if self.show_intermediate:
                result.show_intermediates()
            return result
        
        # default
        err("Image discarded due to insufficient sharp edges.")
        return None
    
    def _analysis_algorithm_2(self, image: np.ndarray) -> typing.Optional[AnalysisResult]:
        # Placeholder for a second analysis algorithm
        image = np.clip(image, 5, 255)
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  
            
        thresholds = ski.filters.threshold_multiotsu(image, classes=3)
        cells = image > thresholds[0]

        label_img = label(label_image=cells)
        snowflakes = regionprops(label_img)
        
        full_grad = gradient_angle(image).astype(np.float32) #image)
        
        intermediates = [
            IntermediateImage("binarized_image", cells.astype(np.uint8)*255),
            IntermediateImage("gradient_image", full_grad),
            # IntermediateImage("unsharp_mask", sharp)
        ]
        
        flake_nr = 0
        accepted_snowflakes = []
        snowflakes = sorted(snowflakes, key=lambda x: x.equivalent_diameter_area, reverse=True)
        for snowflake in snowflakes:
            if snowflake.area < 600:
                continue
            flake_nr += 1
            crop = snowflake.bbox
            flake_image = image[crop[0]:crop[2], crop[1]:crop[3]]
            grad_angle = gradient_angle(flake_image, 3).astype(np.float32)
            avg_gradient_angle = np.mean(grad_angle)
            sharp_edges, sharp_edges_image = calculate_sharp_edges(flake_image)
            # print(f"average grad angle {avg_gradient_angle}, sharp edges {sharp_edges}")
            
            if avg_gradient_angle > 15. and sharp_edges > 500:
                header(f"High average gradient angle detected: {avg_gradient_angle:.2f} degrees for flake {flake_nr}, accepted for analysis.")
                intermediates.append(IntermediateImage(f"gradient_angle_{flake_nr}", grad_angle))
                intermediates.append(IntermediateImage(f"sharp_edges_{flake_nr}", sharp_edges_image))
                accepted_snowflakes.append(snowflake)
                
        return AnalysisResult(
            pipeline_name="analysis_algorithm_2",
            detections=accepted_snowflakes,
            labelled_image=label_img,
            intermediates=intermediates
        )





