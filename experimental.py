import skimage as ski
from skimage import filters, io, color
from skimage.measure import label, regionprops
import numpy as np
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def multi_otsu_thresholding(image: np.ndarray):
    # flat lower threshold at 6, remove noise below that
    image = np.clip(image, 5, 255)
        
    thresholds = ski.filters.threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)
    
    cells = image > thresholds[0]
    dividing = image > thresholds[1]
    labeled_cells = ski.measure.label(cells)
    labeled_dividing = ski.measure.label(dividing)
    naive_mi = labeled_dividing.max() / labeled_cells.max()
    print(naive_mi)
    
    div = cells

    smoother_dividing = ski.filters.rank.mean(
        ski.util.img_as_ubyte(div), ski.morphology.disk(3)
    )

    binary_smoother_dividing = smoother_dividing > 20
    
    label_img = label(div)
    snowflakes = regionprops(label_img)


    _, ax = plt.subplots(ncols=4, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[0].set_axis_off()
    
    for snowflake in snowflakes:
        if snowflake.area < 600:
            continue
        
        y0, x0 = snowflake.centroid
        orientation = snowflake.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * snowflake.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * snowflake.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * snowflake.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * snowflake.axis_major_length
        
        ellipse = Ellipse((x0, y0), snowflake.axis_minor_length, snowflake.axis_major_length,
                        angle=-math.degrees(orientation), edgecolor='red', facecolor='none', linewidth=2.5)
        ax[0].add_patch(ellipse)
        
        ax[0].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax[0].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax[0].plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = snowflake.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax[0].plot(bx, by, '-b', linewidth=2.5)
    
    ax[1].imshow(dividing)
    ax[1].set_title('Dividing nuclei?')
    ax[1].set_axis_off()
    ax[2].imshow(cells)
    ax[2].set_title('All nuclei?')
    ax[2].set_axis_off()
    ax[3].imshow(binary_smoother_dividing)
    ax[3].set_title('Dividing nuclei')
    ax[3].set_axis_off()
    plt.tight_layout()
    plt.show()
    
    # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # ax[0].imshow(image)
    # ax[0].set_title('Original')
    # ax[0].set_axis_off()
    # ax[1].imshow(regions)
    # ax[1].set_title('Multi-Otsu thresholding')
    # ax[1].set_axis_off()
    # plt.show()