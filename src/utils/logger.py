import os
import time
import typing
from utils.consolecolors import bcolors
import logging
import logging.handlers
import subprocess
import sys
import shutil
from pathlib import Path

LOGGER: typing.Optional[logging.Logger] = None
PRINT_TO_CONSOLE: bool = True

def initial_setup(image_path: str, out_path: str, out: bool) -> typing.Tuple[str, str]:
    run_number = time.strftime("%Y-%m-%d_%a_%H-%M-%S", time.localtime())
    save_path = out_path + "/" + run_number
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # setup logging to file and console
    setup_logging(save_path, console=out)
    # save code state and environment for reproducibility
    try:
        save_code_state(save_path, image_path, run_number)
    except Exception as e:
        # don't crash on snapshot errors; log them
        if LOGGER:
            LOGGER.exception("Failed to save code state: %s", e)
        else:
            print(f"{bcolors.WARNING}[WARNING] Failed to save code state: {e}{bcolors.ENDC}")
         
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
        if out:
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
        if LOGGER:
            f.write(f"logfile:{os.path.join(save_path, 'run.log')}\n")
         
    return (save_path, csv_filepath)
 
def info(str: str) -> None:
    msg = f"{bcolors.OKCYAN}[INFO] {str}{bcolors.ENDC}"
    if PRINT_TO_CONSOLE:
        print(msg)
    if LOGGER:
        LOGGER.info(str)
      
def warn(str: str) -> None:
    msg = f"{bcolors.WARNING}[WARNING] {str}{bcolors.ENDC}"
    if PRINT_TO_CONSOLE:
        print(msg)
    if LOGGER:
        LOGGER.warning(str)
      

def header(str: str) -> None:
    msg = f"{bcolors.HEADER}[INFO] {str}{bcolors.ENDC}"
    if PRINT_TO_CONSOLE:
        print(msg)
    if LOGGER:
        LOGGER.info(str)
     
def err(str: str, exc: typing.Optional[BaseException] = None) -> None:
    msg = f"{bcolors.FAIL}[ERROR] {str}{bcolors.ENDC}"
    if PRINT_TO_CONSOLE:
        print(msg)
    if LOGGER:
        # if an exception object is supplied, include the traceback in the log
        if exc is not None:
            LOGGER.exception(str)
        else:
            LOGGER.error(str)
 
 
def setup_logging(save_path: str, level: int = logging.INFO, console: bool = True) -> logging.Logger:
    """
    Create a module-level logger that writes to save_path/run.log and (optionally) to console.
    """
    global LOGGER, PRINT_TO_CONSOLE
    # control whether wrappers print to stdout
    PRINT_TO_CONSOLE = bool(console)
    log_file = os.path.join(save_path, "run.log")
    logger = logging.getLogger("snowflake_metrics")
    logger.setLevel(level)
    # Avoid duplicate handlers when re-initializing
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    LOGGER = logger
    logger.info("Logging initialized. Logfile: %s", log_file)
    return logger


def save_code_state(save_path: str, image_path: str, run_number: str) -> None:
    """
    Save a small code + environment snapshot:
    - git commit, branch, status
    - pip freeze -> requirements_{run}.txt
    - zipped repo or src/ into code_snapshot.zip
    - environment metadata file
    """
    meta_path = Path(save_path) / "metadata.txt"
    with open(meta_path, "a", encoding="utf-8") as m:
        m.write(f"run_date:{run_number}\n")
        m.write(f"image_path:{image_path}\n")
        m.write(f"python:{sys.version.replace(os.linesep, ' ')}\n")
        m.write(f"platform:{sys.platform}\n")

    # try git info
    try:
        repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode().strip()
        with open(os.path.join(save_path, "git_info.txt"), "w", encoding="utf-8") as g:
            g.write(f"repo_root:{repo_root}\ncommit:{commit}\nbranch:{branch}\nstatus:\n{status}\n")
        # make a zip snapshot of the repo (may be large); user can opt to remove large files later
        archive_path = os.path.join(save_path, "code_snapshot")
        shutil.make_archive(archive_path, 'zip', repo_root)
        if LOGGER:
            LOGGER.info("Saved git info and zipped repo to %s.zip", archive_path)
    except Exception:
        # fallback: copy src directory if present
        try:
            src_dir = Path(__file__).resolve().parents[1]  # points to src/
            dest = Path(save_path) / "src_snapshot"
            if src_dir.exists():
                shutil.copytree(src_dir, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
                if LOGGER:
                    LOGGER.info("Copied src/ to %s", dest)
        except Exception as e:
            if LOGGER:
                LOGGER.warning("Failed to snapshot code: %s", e)

    # save pip freeze
    try:
        req = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
        req_file = os.path.join(save_path, f"requirements_{run_number}.txt")
        with open(req_file, "w", encoding="utf-8") as r:
            r.write(req)
        if LOGGER:
            LOGGER.info("Saved pip freeze to %s", req_file)
    except Exception as e:
        if LOGGER:
            LOGGER.warning("Failed to save pip freeze: %s", e)
    return
