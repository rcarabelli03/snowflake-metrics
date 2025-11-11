import numpy as np
import cv2
from cv2.typing import MatLike
import time
from skimage.measure import shannon_entropy


def gamma(img_original: MatLike, gamma: float = 0.5) -> MatLike:
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img_original, lookUpTable)

def laplace_detector(image: MatLike, kernel_size: int = 3) -> MatLike:
    laplace = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    absolute = cv2.convertScaleAbs(laplace) ## alpha=255/laplace.max()
    return absolute

def sobel_detector(image: MatLike, kernel_size: int = 3):
    x_grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    y_grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    mag = cv2.magnitude(x_grad, y_grad)
    return (x_grad, y_grad, mag)

def gradient_angle(image: MatLike, kernel_size: int = 3) -> MatLike:
    x_grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    y_grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    angle = cv2.phase(x_grad, y_grad, angleInDegrees=True)
    return angle

def calculate_sharp_edges(image: np.ndarray, threshold: float = 10.0) -> int:
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    sharp_edges = int(np.sum(grad_magnitude > threshold))
    return sharp_edges

    
def SNR(a: MatLike, axis: int = None, ddof: int = 0) -> MatLike:
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, float]:    
    start_time = time.time_ns()

    ## stuff
    g_ksize = 11
    g_sigma = 5
    s_ksize = 3
    
    # smoothed_image = cv2.GaussianBlur(image, (25, 25), sigmaX=2, sigmaY=2)
    res = cv2.GaussianBlur(image, (g_ksize, g_ksize), g_sigma)
    res = gradient_angle(res, kernel_size=s_ksize)
    
    # (_, _, res) = processor.sobel_detector(res, kernel_size=s_ksize)
    # normalize in-place (dst must be a MatLike according to type hints)
    cv2.normalize(src=res, dst=res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    end_time = time.time_ns()
    elapsed_time = end_time - start_time

    #comparison = cv2.hconcat([gray, res])
    #cv2.imshow("Original --- Processed Image", comparison)
    # cv2.imshow("Processed Image", opening)
    # cv2.waitKey(1)
    
    return (opening, elapsed_time)

def image_stats(image: np.ndarray):
    """Compute basic statistics (min, max, mean, std, entropy, ...) for the image."""
    return [np.min(image),
            np.max(image),
            np.mean(image),
            np.std(image),
            np.var(image),
            shannon_entropy(image),
            SNR(image),
            np.mean(sobel_detector(image)[2]),
            np.median(sobel_detector(image)[2]),
            np.std(sobel_detector(image)[2]),
            np.mean(laplace_detector(image)),
            np.median(laplace_detector(image)),
            np.std(laplace_detector(image)),
            np.mean(gradient_angle(image)),
            np.median(gradient_angle(image)),
            np.std(gradient_angle(image)),
            np.sum(gradient_angle(image) >= 70)]