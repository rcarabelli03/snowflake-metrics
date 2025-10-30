import numpy as np
import cv2
import pandas as pd
import time

def gradient_angle(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    x_grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    y_grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    angle = cv2.phase(x_grad, y_grad, angleInDegrees=True)
    return angle

def process_image(image: np.ndarray, idx: int, dir: str | os.PathLike) -> (np.array, float):    
    start_time = time.time()

    ## stuff
    g_ksize = 11
    g_sigma = 5
    s_ksize = 3
    
    
    res = cv2.GaussianBlur(out, (g_ksize, g_ksize), g_sigma)
    res = gradient_angle(res, kernel_size=s_ksize)
    
    # (_, _, res) = processor.sobel_detector(res, kernel_size=s_ksize)
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    end_time = time.time()
    elapsed_time = end_time - start_time

    #comparison = cv2.hconcat([gray, res])
    #cv2.imshow("Original --- Processed Image", comparison)
    cv2.imshow("Processed Image", opening)
    cv2.waitKey(1)
    
    return (opening, elapsed_time)