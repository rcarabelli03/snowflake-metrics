import os
import cv2
from cv2.typing import MatLike
from typing import Optional
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import tqdm

from utils.pathutils import get_image_paths, get_snowflake_id_from_path, get_image_filename, filter_paths_by_selection
from utils.plotutils import plot_ellipse_overlay
from utils.consolecolors import bcolors
from utils.utils import info, warn, err, header
from processor import preprocess_image
from metrics import analyse_image

# __import__('pdb').set_trace()
if __name__=="__main__":
    IMAGE_PATH: str = "/mnt/e/pictures_Vikram/10-29_11-52-33/"
    FLAKE_SELECTION_PATH: str = "/mnt/c/Users/vikra/Desktop/ETHZ/semester_project/code_cv/output_img/2025-11-01_Sat_07-27-32_14/processed/"
    images: list[str] = get_image_paths(IMAGE_PATH)
    info(f"Found {len(images)} images in {IMAGE_PATH}")
    selected_snowflakes: list[str] = [get_image_filename(p) for p in get_image_paths(FLAKE_SELECTION_PATH)]
    snowflake_ids: list[int] = [get_snowflake_id_from_path(p) for p in selected_snowflakes if not p.startswith("processed_")]
    
    print(f"Filtering images based on selection of {len(snowflake_ids)} snowflakes...")
    images = filter_paths_by_selection(images, snowflake_ids)
    info(f"{len(images)} images remaining after filtering. Corresponding ids: {snowflake_ids}")
    
    for img_path in images:
        name = get_image_filename(img_path)
        if name.startswith("processed_"):
            continue
        header(f"Processing image: {img_path}, id: {get_snowflake_id_from_path(img_path)}")
        image: Optional[MatLike] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            err(f"Failed to read image: {img_path}")
            continue
        res = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
        processed_image, elapsed_processor  = preprocess_image(res)
        metrics, elapsed_analyser = analyse_image(res)
        print(f"Metrics for {os.path.basename(img_path)}: {metrics}")
        if len(metrics) == 0:
            continue
        plot_ellipse_overlay(image, metrics, 3000)
        info(f"Processing time: {elapsed_processor/1e6:.4f} ms, Analysis time: {elapsed_analyser/1e6:.4f} ms")
    cv2.destroyAllWindows()