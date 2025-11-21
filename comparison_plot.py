import os
import cv2
from cv2.typing import MatLike
from typing import Optional
import time
# import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# format table output to TeX with booktabs (df.to_latex requires Jinja2 --- install it via pip if not already installed)
def latex_table(df: pd.DataFrame) -> str:
    return df.to_latex(index=False)

from utils.pathutils import get_snowflake_id_from_path, get_image_paths, get_image_filename, get_filtered_image_paths, __deprecated___get_image_paths_filtered, get_image_folder, get_str_image_id_from_path
from utils.plotutils import plot_ellipse_overlay
from utils.csv_utils import read_csv_metrics, read_single_img_metrics, get_csv_filename_from_path
from utils.consolecolors import bcolors
from utils.utils import info, warn, err, header
from processor import preprocess_image, gamma, image_stats
from metrics import analyse_image

data_path = "/mnt/e/snowflake_analysis_plots/sorted-run-3-test"
output_path = "/mnt/e/plots_comparison"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
    
if __name__=="__main__":

    csv_filepath = os.path.join(data_path, "snowflake_image_metrics.csv")
    if not os.path.exists(csv_filepath):
        err(f"Could not find CSV file with image metrics at {csv_filepath}")
        exit(1)
    image_metrics_df = read_csv_metrics(csv_filepath)
    print(f"Read image metrics DataFrame: {image_metrics_df.describe()}")
    
    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    # images
    for folder in subfolders:
        snowflake_metric_csv_path = get_snowflake_id_from_path(folder)
        if snowflake_metric_csv_path is None:
            err(f"Could not find CSV file in folder: {folder}")
            continue
        csv_file = get_csv_filename_from_path(folder)
        if csv_file is None: