import os
from pathlib import Path
import cv2
import yaml
import argparse
from cv2.typing import MatLike
from typing import Optional
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils.pathutils import get_snowflake_id_from_path, get_image_paths, get_image_filename, get_filtered_image_paths, __deprecated___get_image_paths_filtered, get_image_folder, get_str_image_id_from_path
from utils.io.plotutils import plot_ellipse_overlay, plot_histogram
from utils.consolecolors import bcolors
from utils.print_wrapper import initial_setup, info, warn, err, header
from utils.configurator.config import load_config
from processing.processor import preprocess_image, gamma, image_stats
from processing.metrics import Analyser
from processing.experimental import multi_otsu_thresholding

# base_path = "../images/pictures_Test"
# dir_list = ["10-7_15-42-5/", "10-7_15-45-41/", "10-7_15-51-6/", "10-28_15-27-6"]
# path = base_path + "/" + dir_list[-1]

    
# __import__('pdb').set_trace()
if __name__=="__main__":
    
    config = load_config()
    image_directory = config["paths"]["image_directory"]
    base_path = config["paths"]["output_directory"]
    
    save_path, csv_filepath = initial_setup(image_path=image_directory, out_path=base_path)
    
    analyser = Analyser(config=config, save_path=save_path)
    
    stats = config["metrics"]["stats"]
            
    df = pd.DataFrame(columns=stats + ["image_name"])
    
    # images = get_filtered_image_paths()
    # images = __deprecated___get_image_paths_filtered(image_dir="/mnt/e/pictures_Test/", start_from="11-12_15-50-21") # test_old, 10-22_12-27-20
    # images = get_image_paths("../images/nice_flakes")
    # images = ["../../images/nice_flakes/Snowflake_20.bmp"] #
    images = get_image_paths(image_directory) #
    # images = ["/mnt/f/davos_1/11-17_2-29-24/Snowflake_284.bmp"]
    
    # sort images by snowflake id extracted from filename
    images = [img for _, img in sorted((get_snowflake_id_from_path(img), img) for img in images)]
    
    
    try:
        for img_path in tqdm(images, desc="Processing images", unit="image"):
            
            name = get_image_filename(img_path)
            if name.startswith("processed_"): # legacy check
                continue
            header(f"Processing image: {img_path}, id: {get_snowflake_id_from_path(img_path)} from folder: {get_image_folder(img_path)} with str id: {get_str_image_id_from_path(img_path)}")
            
            # Read image in grayscale
            res: Optional[MatLike] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if res is None:
                err(f"Failed to read image: {img_path}")
                continue
            
            # Show slightly improved contrast using CLAHE and gamma correction for visual inspection
            # frame: MatLike = res.copy()
            # cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(src=frame)
            # frame = gamma(frame, gamma=0.8)
            # cv2.imshow("CLAHE Result", frame)
            # cv2.waitKey(1)
            
            # Preprocess image and analyse
            processed_image, elapsed_processor      = preprocess_image(image=res, config=config)
            stats                                   = image_stats(image=res, config=config) # returns an np array
            metrics, elapsed_analyser    = analyser.analyse_image(image=res, folder_desc=f"{get_str_image_id_from_path(img_path)}")
            
            # multi_otsu_thresholding(image=res)
            
            # print(f"Subfolder used: {subfolder}")
            # if subfolder is None:
            #     subfolder = "default"
                                
            # alternative_metrics = alternative_analyse_image(image=res, config=config, save_path=save_path, folder_desc=subfolder)
            
            # debug: print metrics
            if metrics is None:
                warn(f"No metrics extracted for image: {img_path}")
                continue
            print(f"Metrics for {os.path.basename(img_path)}:\n{metrics}")
            
            df.loc[len(df)] = np.concatenate((stats, [name]))
            
            
            info(f"Processing time: {elapsed_processor/1e6:.4f} ms, Analysis time: {elapsed_analyser/1e6:.4f} ms")
            
    except KeyboardInterrupt:
        warn("Processing interrupted by user.")
        
    df.to_csv(csv_filepath, index=False)
    config_filepath = os.path.join(save_path, "config_used.yaml")
    with open(config_filepath, "w") as f:
        yaml.dump(config, f)
    info(f"Saved results to {csv_filepath} and config to {config_filepath}")
    cv2.destroyAllWindows()