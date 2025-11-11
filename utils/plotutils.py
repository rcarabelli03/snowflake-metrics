import os
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
import cv2
from cv2.typing import MatLike
import math


def plot_ellipse_overlay(img: MatLike, data: np.ndarray, display_time: int) -> None:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(data), 9):
        centroid = (np.round(data[i][1]).astype(int), np.round(data[i][0]).astype(int))
        cv2.circle(img, centroid, 5, (0,0, 255), -1)        
        end_short = (np.round(data[i][1] + data[i+1][1]/2 * math.cos(data[i+2])).astype(int),
                np.round(data[i][0] - data[i+1][1]/2 * math.sin(data[i+2])).astype(int))
        cv2.line(img, centroid, end_short, (255,0,0),5)
        cv2.ellipse(img, centroid, (np.round(data[i+1][1]/2).astype(int), np.round(data[i+1][0]/2).astype(int)),
                    -math.degrees(data[i+2]), 0, 360, (0,0,255), 2)
    cv2.imshow("Ellipses Overlay", img)
    cv2.waitKey(display_time)
            
def skimage_show_plot(snowflake, binary_image):
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray') ##     ax.imshow(binary_image, cmap=plt.cm.gray)

    y0, x0 = snowflake.centroid
    orientation = snowflake.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * snowflake.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * snowflake.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * snowflake.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * snowflake.axis_major_length
    
    ellipse = Ellipse((x0, y0), snowflake.axis_minor_length, snowflake.axis_major_length,
                      angle=-math.degrees(orientation), edgecolor='red', facecolor='none', linewidth=2.5)
    ax.add_patch(ellipse)

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = snowflake.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
    
    fig.set_size_inches(20, 13)
    plt.tight_layout()
    plt.show()