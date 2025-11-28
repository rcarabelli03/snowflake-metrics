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

def write_image(img: MatLike, save_path: str, filename: str) -> None:
    filepath = os.path.join(save_path, filename)
    cv2.imwrite(filepath, img)
    

def plot_ellipse_overlay(img: MatLike, data: dict, display_time: int, visual=False, save=None, save_path=None, flake_id=None) -> None:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    centroid = (np.round(data["centroid_x"]).astype(int), np.round(data["centroid_y"]).astype(int))
    end_short = (np.round(centroid[0] + data["axis_minor_length"]/2 * math.cos(data["orientation"])).astype(int),
            np.round(centroid[1] - data["axis_minor_length"]/2 * math.sin(data["orientation"])).astype(int))
    end_long = (np.round(centroid[0] - data["axis_major_length"]/2 * math.sin(data["orientation"])).astype(int),
            np.round(centroid[1] - data["axis_major_length"]/2 * math.cos(data["orientation"])).astype(int))
    cv2.line(img, centroid, end_short, (255,0,0),5)
    cv2.line(img, centroid, end_long, (255,0,0),5)
    cv2.circle(img, centroid, 5, (0,0, 255), -1)        
    cv2.ellipse(img, centroid, (np.round(data["axis_minor_length"]/2).astype(int), np.round(data["axis_major_length"]/2).astype(int)),
                -math.degrees(data["orientation"]), 0, 360, (0,0,255), 2)
    
    if visual:
        cv2.imshow("Ellipses Overlay", img)
        cv2.waitKey(display_time)
    if save and save_path is not None and flake_id is not None:
        filename = f"snowflake_{flake_id}_ellipse_overlay.png"
        filename = os.path.join(save_path, filename)
        cv2.imwrite(filename, img)
            
def skimage_show_plot(snowflake, binary_image, contour=None, display=True, save=None, save_path=None, flake_id=None) -> None:
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
    
    if contour is not None:
        for contour_part in contour:
            ax.plot(contour_part[:, 1], contour_part[:, 0], '-g', linewidth=2.0)
            
    
    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = snowflake.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
    
    fig.set_size_inches(10, 13/2)
    plt.tight_layout()
    if display:
        plt.show()
    
    if save and save_path is not None and flake_id is not None:
        filename = f"snowflake_{flake_id}_analysis.png"
        filename = os.path.join(save_path, filename)
        fig.set_dpi(1000)
        fig.savefig(filename)
    plt.close()
        
def plot_histogram(image: np.ndarray, title: str, xlabel: str, ylabel: str, bins: int = 256, log: bool = True, visual=False, save: bool = False, save_path: str = "") -> None:
    fig = plt.figure()
    # pdf
    hist_vals, bin_edges = np.histogram(image.flatten(), bins=bins, range=(0,256))
    # cdf
    cdf = hist_vals.cumsum()
    cdf_normalized = cdf * float(hist_vals.max()) / cdf.max()
    # plot
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(image.flatten(), bins=bins, range=(0,256), color = 'r', log=log)
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set ticks every 20 units and rotate x ticks by 45 degrees
    plt.xticks(np.arange(0, 257, 20), rotation=45)
    fig.set_size_inches(10, 13/2)
    plt.tight_layout()
    
    
    if save and save_path:
        filename = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
        fig.set_dpi(1000)
        plt.savefig(filename)
    
    if visual:
        plt.show()
    
    plt.close()  
    
def plot_line_graph(x: list, y: list, title: str, xlabel: str, ylabel: str, visual=False, save: bool = False, save_path: str = "") -> None:
    fig = plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    fig.set_size_inches(10, 13/2)
    plt.tight_layout()
    
    if save and save_path:
        filename = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
        fig.set_dpi(1000)
        plt.savefig(filename)
    
    if visual:
        plt.show()
    
    plt.close()
    