import os
import typing
import numpy as np
import cv2

def get_image_paths(directory: str) -> typing.List[str]:
    '''Gets all image file paths from specified directory. Non-recursive.
    Args:
        directory (str | os.PathLike): Path to the directory containing images.
        Returns:
            List[str]: List of image file paths.
    '''
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_image_filename(image_path: str) -> str:
    base_name = os.path.basename(image_path)
    name = os.path.splitext(base_name)[0]
    return name


def get_snowflake_id_from_path(image_path: str) -> int:
    name = get_image_filename(image_path)
    snowflake_id = int(name.split('_')[1])
    return snowflake_id

def filter_paths_by_selection(image_paths: typing.List[str], selection: typing.List[int]) -> typing.List[str]:
    filtered_paths: list[str] = [path for path in image_paths if get_snowflake_id_from_path(path) in selection]
    return filtered_paths