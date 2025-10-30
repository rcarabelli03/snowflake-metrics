import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

__import__('pdb').set_trace()

if __name__=="__main__":
    IMAGE_PATH: str = ""
    images: list = get_image_paths(IMAGE_PATH)
    processed_images = process_images(images)

