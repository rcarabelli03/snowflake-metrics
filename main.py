import os
import cv2
from cv2.typing import MatLike
from typing import Optional
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import tqdm

from utils.pathutils import get_snowflake_id_from_path, get_image_paths, get_image_filename, get_filtered_image_paths, __deprecated___get_image_paths_filtered, get_image_folder, get_str_image_id_from_path
from utils.plotutils import plot_ellipse_overlay, plot_histogram
from utils.consolecolors import bcolors
from utils.utils import initial_setup, info, warn, err, header
from processor import preprocess_image, gamma, image_stats
from metrics import analyse_image

# base_path = "../images/pictures_Test"
# dir_list = ["10-7_15-42-5/", "10-7_15-45-41/", "10-7_15-51-6/", "10-28_15-27-6"]
# path = base_path + "/" + dir_list[-1]

base_path = "/mnt/f/snowflake_analysis_plots"


# __import__('pdb').set_trace()
if __name__=="__main__":
    
    save_path, csv_filepath = initial_setup(base_path=base_path)
            
    df = pd.DataFrame(columns=['min_intensity', 'max_intensity', 'mean_intensity', 'std_intensity', 'variance_intensity', 'entropy', 'snr', 'mean_gradient_magnitude', 'median_gradient_magnitude', 'std_gradient_magnitude', 'mean_laplace', 'median_laplace', 'std_laplace', 'mean_angle', 'median_angle', 'std_angle', 'sharp_edge_count'] + ['image_name'])
    
    # images = get_image_paths("/mnt/e/pictures_Vikram/10-29_11-52-33/")
    # images = get_filtered_image_paths()
    # images = __deprecated___get_image_paths_filtered(image_dir="/mnt/e/pictures_Test/", start_from="11-12_15-50-21") # test_old, 10-22_12-27-20
    # images = get_image_paths("../images/nice_flakes")
    # images = ["../../images/nice_flakes/Snowflake_20.bmp"] #
    images = get_image_paths("/mnt/f/davos_1/11-17_1-57-46/") # 11-17_2-29-24
    # images = ["/mnt/f/davos_1/11-17_2-29-24/Snowflake_284.bmp"]
    
    '''
    NOTE: When imaging with a bright background, ensure to invert the image colors before processing. The snowflake should appear brighter than the background for accurate analysis.
    
    # Example of inverting an image using OpenCV
    image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)   
    inverted_image = cv2.bitwise_not(image)
    '''
    
    # sort images by snowflake id extracted from filename
    images = [img for _, img in sorted((get_snowflake_id_from_path(img), img) for img in images)]
    
    
    try:
        from tqdm import tqdm
        images_iterable = tqdm(images)
    except ImportError:
        images_iterable = images
        
    try:
        for img_path in images_iterable:
            
            name = get_image_filename(img_path)
            if name.startswith("processed_"):
                continue
            header(f"Processing image: {img_path}, id: {get_snowflake_id_from_path(img_path)} from folder: {get_image_folder(img_path)} with str id: {get_str_image_id_from_path(img_path)}")
            
            res: Optional[MatLike] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if res is None:
                err(f"Failed to read image: {img_path}")
                continue
            
            frame: MatLike = res.copy()
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(src=frame)
            frame = gamma(frame, gamma=0.8)
            cv2.imshow("CLAHE Result", frame)
            cv2.waitKey(1)
            
            processed_image, elapsed_processor  = preprocess_image(res)
            stats                               = image_stats(res) # returns an np array
            metrics, elapsed_analyser           = analyse_image(image=res, thresh=300, plot=True, display_plot=False, save=True, save_path=save_path, folder_desc=f"{get_str_image_id_from_path(img_path)}")
            
            print(f"Metrics for {os.path.basename(img_path)}: {metrics}")
            
            df.loc[len(df)] = np.concatenate((stats, [name]))
            
            
            info(f"Processing time: {elapsed_processor/1e6:.4f} ms, Analysis time: {elapsed_analyser/1e6:.4f} ms")
            
    except KeyboardInterrupt:
        warn("Processing interrupted by user.")
        
    df.to_csv(csv_filepath, index=False)
    
    cv2.destroyAllWindows()