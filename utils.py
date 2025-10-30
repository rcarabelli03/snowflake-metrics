import os
import typing

def get_image_paths(directory: str | os.PathLike) -> typing.List[str]:
    '''Gets all image file paths from specified directory. Non-recursive.
    Args:
        directory (str | os.PathLike): Path to the directory containing images.
        Returns:
            List[str]: List of image file paths.
    '''
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths