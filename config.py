import json
from pathlib import Path

DEFAULT_CONFIG = {
    "pixels_per_um": 18.2044,
    "nuclear_channel": 2,
    "signal_channel": 0,
    "sigma": 2,
    "thresh_factor_low": 0.7,
    "thresh_factor_high": 1.3,
    "min_distance": 10,
    "min_nucleus_area": 2000,
    "max_nucleus_area": 10000,
    "image_extensions": [".lsm", ".tif", ".tiff", ".png"],
}

def load_config(config_path="config.json"):
    config = DEFAULT_CONFIG.copy()
    try:
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            config.update(user_config)
            print("Loaded configuration from", config_path)
        else:
            print("No config file found; using default settings.")
    except Exception as e:
        print(f"Error loading config file: {e}; using defaults.")
    return config

def compute_scale(config):
    pixels_per_um = config["pixels_per_um"]
    config["um_per_pixel"] = 1 / pixels_per_um
    config["area_um2_per_pixel2"] = config["um_per_pixel"] ** 2
    return config