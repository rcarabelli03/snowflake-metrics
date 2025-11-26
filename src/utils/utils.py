import os
import time
import typing
from utils.consolecolors import bcolors

def initial_setup(image_path: str, out_path: str) -> typing.Tuple[str, str]:
    run_number = time.strftime("%Y-%m-%d_%a_%H-%M-%S", time.localtime())
    save_path = out_path + "/" + run_number
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    readme_path = os.path.join(save_path, "README.md")    
    README = f"""# Snowflake Metrics Analysis - {run_number}

Generated on {run_number} by Snowflake Metrics Analysis Tool.

Based on the images located in: `{out_path}`.

## Description

This directory contains the results of snowflake image analysis, including computed metrics and processed images.
Each image has been analyzed to extract various metrics related to snowflake morphology and characteristics.

## Naming Conventions

- Processed images and metrics files are named using the format:
    `snowflake_<global_id>_<local_id>_<size_um>um.<extension>`
    where `<global_id>` is a unique identifier for the snowflake across all images, `<local_id>` is the identifier within the current image, and `<size_um>` is the equivalent diameter in micrometers.

## Contents

- `snowflake_image_metrics.csv`: A CSV file containing computed metrics for all processed images
- `<global_id>_<local_id>_snowflake_<original_image_id>_<run_date>/`: Subdirectories for each image containing:
  - `snowflake_image_metrics.csv`: A CSV file with metrics for individual snowflakes in that image.
  - Processed images: Individual images of isolated snowflakes, binarized versions, and overlays, etc.
- info.txt: A text file containing metadata about the analysis run.

## Usage

The metrics can be analyzed using standard data analysis tools that support CSV format, such as Python (pandas), R, or spreadsheet software.

## Notes

- Ensure that the images in the base path are in a supported format (e.g., PNG, JPG).
- The analysis assumes that snowflakes are brighter than the background in the images. If this is not the case, please preprocess the images accordingly.
"""

    with open(readme_path, 'w') as f:
        f.write(README)

    csv_filename = "snowflake_image_metrics.csv"
    csv_filepath = os.path.join(save_path, csv_filename)
    if os.path.exists(csv_filepath):
        warn(f"CSV file {csv_filepath} already exists and will be overwritten. Are you sure? [y/N]")
        user_input = input().strip().lower()
        if user_input != 'y':
            info("Exiting program.")
            exit(0)        
        os.remove(csv_filepath)
        
    info_txt = f"metadata.txt"
    info_filepath = os.path.join(save_path, info_txt)
    with open(info_filepath, 'w') as f:
        f.write(f"run_date:{run_number}\n")
        f.write(f"image_path:{image_path}\n")
        f.write(f"csv_filepath:{csv_filepath}\n")
        
    return (save_path, csv_filepath)

def info(str: str) -> None:
    print(f"{bcolors.OKCYAN}[INFO] {str}{bcolors.ENDC}")
    
def warn(str: str) -> None:
    print(f"{bcolors.WARNING}[WARNING] {str}{bcolors.ENDC}")
    

def header(str: str) -> None:
    print(f"{bcolors.HEADER}[INFO] {str}{bcolors.ENDC}")
   
def err(str: str) -> None:
    print(f"{bcolors.FAIL}[ERROR] {str}{bcolors.ENDC}")