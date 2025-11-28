import argparse
from pathlib import Path
import yaml

def get_args():
    parser = argparse.ArgumentParser(description="Snowflake Image Processor and Analyzer")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/conf.yaml",
        help="Path to the configuration YAML file.",
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing snowflake images. Overrides config file if provided.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output results. Overrides config file if provided.",
    )
    
    # parser.add_argument(
    #     "--debug",
    #     action="store_true",
    #     default=False,
    #     help="Enable debug mode with additional outputs.",
    # )
    
    # parser.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     help="Enable verbose logging.",
    # )
    
    parser.add_argument(
        "--save_disabled",
        action="store_true",
        help="Run the program without saving any outputs.",
    )
    
    parser.add_argument(
        "--display_disabled",
        "-d",
        action="store_true",
        help="Run the program without displaying any outputs.",
    )
    
    return parser.parse_args()

def load_config():
    args = get_args()
    with open(Path(args.config), "r") as f:
        config =  yaml.safe_load(f)
    # Override config paths if command-line arguments are provided
    if args.image_dir is not None:
        config["paths"]["image_directory"] = args.image_dir
    if args.output_dir is not None:
        config["paths"]["output_directory"] = args.output_dir
    
    # Set debug and verbose flags
    # config["debug"]["enabled"] = args.debug
    # config["debug"]["show_intermediate_images"] = args.debug
    
    if args.save_disabled:
        config["plot"]["save"] = False
    if args.display_disabled:
        config["plot"]["display"] = False
    return config