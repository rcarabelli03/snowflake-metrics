import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import get_image_paths
from processor import preprocess_image
from metrics import analyse_image

__import__('pdb').set_trace()

if __name__=="__main__":
    IMAGE_PATH: str = ""
    images: list = get_image_paths(IMAGE_PATH)
    for img_path in images:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        res = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
        processed_image = preprocess_image(res)
        metrics = analyse_image(processed_image)
        print(f"Metrics for {os.path.basename(img_path)}: {metrics}")