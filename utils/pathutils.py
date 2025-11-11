import os
from os import walk
import typing
import numpy as np
import cv2
from utils.utils import info, warn, err, header

def get_image_paths(directory: str) -> typing.List[str]:
    if not os.path.exists(directory):
        err(f"Directory does not exist: {directory}")
        return []
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    info(f"Found {len(image_paths)} image files in directory: {directory}")
    return image_paths

def get_image_filename(image_path: str) -> str:
    base_name = os.path.basename(image_path)
    name = os.path.splitext(base_name)[0]
    return name

def get_snowflake_id_from_path(image_path: str) -> int:
    name = get_image_filename(image_path)
    try:
        snowflake_id = int(name.split('_')[1])
    except (IndexError, ValueError):
        err(f"Could not extract snowflake ID from path: {image_path}")
        snowflake_id = -1
    return snowflake_id

def filter_paths_by_selection(image_paths: typing.List[str], selection: typing.List[int]) -> typing.List[str]:
    filtered_paths: list[str] = [path for path in image_paths if get_snowflake_id_from_path(path) in selection]
    return filtered_paths

def get_filtered_image_paths() -> typing.List[str]:
    IMAGE_PATH: str = "/mnt/e/pictures_Vikram/10-29_11-52-33/"
    FLAKE_SELECTION_PATH: str = "/mnt/c/Users/vikra/Desktop/ETHZ/semester_project/code_cv/output_img/2025-11-01_Sat_07-27-32_14/processed/"
    images: list[str] = get_image_paths(IMAGE_PATH)
    info(f"Found {len(images)} images in {IMAGE_PATH}")
    selected_snowflakes: list[str] = [get_image_filename(p) for p in get_image_paths(FLAKE_SELECTION_PATH)]
    snowflake_ids: list[int] = [get_snowflake_id_from_path(p) for p in selected_snowflakes if not p.startswith("processed_")]
    
    print(f"Filtering images based on selection of {len(snowflake_ids)} snowflakes...")
    images = filter_paths_by_selection(images, snowflake_ids)
    info(f"{len(images)} images remaining after filtering. Corresponding ids: {snowflake_ids}")
    return images


def __deprecated___get_image_paths_filtered(image_dir: str = "/mnt/e/pictures_Test",
                                            key: typing.Callable[[str], tuple[int, int, int]] = lambda x: (int(x.split('_')[0].split('-')[1]), int(x.split('_')[1].split('-')[0]), int(x.split('_')[1].split('-')[1])),
                                            start_from: str = "10-28_15-27-6") -> typing.List[str]:
    ''' **DEPRECATED**\\
    Get all image file paths from all subdirectories of image_dir, sorted by subdirectory name and with optionally a subdirectory to start from. If start_from is given, only subdirectories from that one (inclusive) are considered.
    It's been assumed that subdirectory names are in the format "MM-DD_HH-MM-SS".
    
    This function has been replaced by get_filtered_image_paths().
    
    Tbh, this function is kinda messy but it works for now. It's a quick-and-dirty solution from the old codebase and idk it's just easier to leave it like this for now.
    
    Args:
        image_dir (str, optional): Directory containing subdirectories with images. Defaults to "../images/pictures_Test".
        key (typing.Callable[[str], tuple[int, int, int]], optional): Sorting key for subdirectory names. Defaults to lambda x: (int(x.split('_')[0].split('-')[1]), int(x.split('_')[1].split('-')[0]), int(x.split('_')[1].split('-')[1])).
        start_from (str, optional): Subdirectory name to start from (inclusive). Defaults to "10-28_15-27-6".
    Returns:
        typing.List[str]: List of image file paths.
    '''
    info(f"test sorting key for '10-7_15-4-42': {key('10-7_15-4-42')}")
    
    subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))] # no, i wont use os.path.walk bc i need the subdirs sorted first
    subdirs.sort(key=key)
    print(f"Sorted subdirectories: {subdirs}")
    
    NOTES = "10-7_15-4-42"
    paths: typing.List[str] = []
    # start_from = "10-22_12-27-20"
    ok = False
    for subdir in subdirs:
        if not subdir.startswith(start_from) and not ok: # franky this is ugly but idc
            continue
        ok = True
        subdir = os.path.join(image_dir, subdir)
        for (dirpath, dirnames, filenames) in walk(subdir):
            info(f"Found {len(filenames)} files in {dirpath}")
            tmp = lambda x: int(x.split('_')[1].strip(".png"))
            filenames.sort( key=tmp )   
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    file_path = os.path.join(dirpath, filename)
                    print(f"Found image file: {file_path}")
                    paths.append(file_path)
    info(f"Total images found: {len(paths)}")
    return paths