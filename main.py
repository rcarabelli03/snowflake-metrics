import os
import cv2
from cv2.typing import MatLike
from typing import Optional
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import tqdm

from utils.pathutils import get_snowflake_id_from_path, get_image_paths, get_image_filename, get_filtered_image_paths, __deprecated___get_image_paths_filtered
from utils.plotutils import plot_ellipse_overlay
from utils.consolecolors import bcolors
from utils.utils import info, warn, err, header
from processor import preprocess_image, gamma, image_stats
from metrics import analyse_image

# base_path = "../images/pictures_Test"
# dir_list = ["10-7_15-42-5/", "10-7_15-45-41/", "10-7_15-51-6/", "10-28_15-27-6"]
# path = base_path + "/" + dir_list[-1]

save_path = "/mnt/e/snowflake_analysis_plots/run-2"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# __import__('pdb').set_trace()
if __name__=="__main__":
    
    csv_filename = "snowflake_image_metrics.csv"
    csv_filepath = os.path.join(save_path, csv_filename)
    if os.path.exists(csv_filepath):
        warn(f"CSV file {csv_filepath} already exists and will be overwritten.")
        os.remove(csv_filepath)
        
    df = pd.DataFrame(columns=['min_intensity', 'max_intensity', 'mean_intensity', 'std_intensity', 'variance_intensity', 'entropy', 'snr', 'mean_gradient_magnitude', 'median_gradient_magnitude', 'std_gradient_magnitude', 'mean_laplace', 'median_laplace', 'std_laplace', 'mean_angle', 'median_angle', 'std_angle', 'sharp_edge_count'] + ['image_name'])
    
    # images = get_image_paths("/mnt/e/pictures_Vikram/10-29_11-52-33/")
    # images = get_filtered_image_paths()
    # images = __deprecated___get_image_paths_filtered(image_dir="/mnt/e/pictures_Test_old/", start_from="10-22_12-27-20")
    images = get_image_paths("../images/nice_flakes")
    # images = ["../../images/nice_flakes/Snowflake_20.bmp"] #
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
            header(f"Processing image: {img_path}, id: {get_snowflake_id_from_path(img_path)}")
            
            res: Optional[MatLike] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if res is None:
                err(f"Failed to read image: {img_path}")
                continue
            
            # res = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(res)
            # res = gamma(res, gamma=0.8)
            # cv2.imshow("CLAHE Result", res)
            # cv2.waitKey(10000)
            processed_image, elapsed_processor  = preprocess_image(res)
            stats                               = image_stats(res) # returns an np array
            metrics, elapsed_analyser           = analyse_image(res, visual=True, save=True, save_path=save_path)
            print(f"Metrics for {os.path.basename(img_path)}: {metrics}")
            
            df.loc[len(df)] = np.concatenate((stats, [name]))
            
            
            info(f"Processing time: {elapsed_processor/1e6:.4f} ms, Analysis time: {elapsed_analyser/1e6:.4f} ms")
            
    except KeyboardInterrupt:
        warn("Processing interrupted by user.")
        
    df.to_csv(csv_filepath, index=False)
    
    cv2.destroyAllWindows()