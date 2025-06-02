import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread as tiff_imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_dilation, binary_opening, disk
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, binary_fill_holes
import pandas as pd
from skimage.feature import peak_local_max
from pathlib import Path
import json
import sys

# Check for GUI support to toggle interactivity
try:
    matplotlib.use('TkAgg')  # Default to TkAgg for interactivity
    INTERACTIVE = True
except:
    matplotlib.use('Agg')  # Fall back to non-interactive if no GUI
    INTERACTIVE = False
    print("No GUI support detected; plots will be saved instead of displayed.")

# Configuration defaults (can be overridden by a config file or user input)
DEFAULT_CONFIG = {
    "pixels_per_um": 18.2044,  # 18.2044 pixels = 1 μm
    "nuclear_channel": 2,      # Channel for nuclear staining (e.g., purple, lamin)
    "signal_channel": 0,       # Channel for signal to quantify (e.g., green)
    "sigma": 2,                # Gaussian smoothing parameter
    "thresh_factor_low": 0.7,  # Low threshold factor for lamin rings
    "thresh_factor_high": 1.3, # High threshold factor (unused but kept for flexibility)
    "min_distance": 10,        # Minimum distance between nuclei seeds
    "min_nucleus_area": 2000,  # Minimum nucleus area in pixels
    "max_nucleus_area": 10000, # Maximum nucleus area in pixels
    "image_extensions": [".lsm", ".tif", ".tiff", ".png"],  # Supported image formats
}

# Load configuration from file if it exists
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

# Get directory from user input or command-line argument
def get_directory():
    if len(sys.argv) > 1:
        dir_path = sys.argv[1]
    else:
        dir_path = input("Enter the directory path containing image files: ").strip()
    return Path(dir_path)

# Compute scale-dependent values
def compute_scale(config):
    pixels_per_um = config["pixels_per_um"]
    config["um_per_pixel"] = 1 / pixels_per_um
    config["area_um2_per_pixel2"] = config["um_per_pixel"] ** 2
    return config

# Segment 1: Load and Validate Image
def load_image(file_path):
    print(f"Loading file: {file_path}")
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist.")
        return None
    try:
        image = tiff_imread(file_path)
        print("Image shape:", image.shape)
        print("Image min/max:", image.min(), image.max())
        # Check if image is in (height, width, channels) format
        if len(image.shape) == 3 and image.shape[-1] in [3, 4]:
            image = np.transpose(image, (2, 0, 1))  # Convert to (channels, height, width)
            print("Transposed image shape:", image.shape)
        elif len(image.shape) != 3:
            print("Error: Image must be 3D (channels, height, width) or (height, width, channels).")
            return None
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Segment 2: Separate Channels
def separate_channels(image, config):
    if image is None or image.ndim != 3:
        print("Error: Invalid image data for channel separation.")
        return None, None
    max_channel = image.shape[0] - 1
    nuclear_channel = min(config["nuclear_channel"], max_channel)
    signal_channel = min(config["signal_channel"], max_channel)
    if nuclear_channel == signal_channel:
        print(f"Warning: Nuclear and signal channels are the same ({nuclear_channel}); check config.")
    nuclear = image[nuclear_channel]
    signal = image[signal_channel]
    print(f"Nuclear channel ({nuclear_channel}) shape:", nuclear.shape)
    print(f"Nuclear channel min/max:", nuclear.min(), nuclear.max())
    print(f"Signal channel ({signal_channel}) shape:", signal.shape)
    print(f"Signal channel min/max:", signal.min(), signal.max())
    return nuclear, signal

# Segment 3: Segment Nuclei
def segment_nuclei(nuclear, config):
    if nuclear is None:
        print("Error: No nuclear channel data for segmentation.")
        return None, None, None
    smoothed = gaussian(nuclear, sigma=config["sigma"])
    base_thresh = threshold_otsu(smoothed)
    lamin_rings = smoothed > (base_thresh * config["thresh_factor_low"])
    lamin_rings = binary_opening(lamin_rings, disk(3))
    filled_nuclei = binary_fill_holes(lamin_rings)
    distance = distance_transform_edt(filled_nuclei)
    if distance.max() == 0:
        print("Error: Distance transform is empty.")
        return None, None, None
    seeds = peak_local_max(distance, min_distance=config["min_distance"], labels=filled_nuclei)
    if len(seeds) == 0:
        print("Error: No seeds detected.")
        return None, None, None
    lamin_seeds = np.zeros_like(filled_nuclei, dtype=np.int32)
    for idx, (y, x) in enumerate(seeds):
        lamin_seeds[y, x] = idx + 1
    labeled_nuclei = watershed(-distance, lamin_seeds, mask=filled_nuclei)
    for region in regionprops(labeled_nuclei):
        if region.area < config["min_nucleus_area"]:
            labeled_nuclei[labeled_nuclei == region.label] = 0
    labeled_nuclei = label(labeled_nuclei)
    print(f"Final number of labeled nuclei: {len(np.unique(labeled_nuclei)) - 1}")
    return lamin_rings, filled_nuclei, labeled_nuclei

# Segment 4: Quantify Nuclei
def quantify_nuclei(labeled_nuclei, signal_channel, config):
    if labeled_nuclei is None or signal_channel is None:
        print("Error: Invalid data for quantification.")
        return [], []
    initial_results = []
    valid_nuclei_labels = []
    for region in regionprops(labeled_nuclei, intensity_image=signal_channel):
        if config["min_nucleus_area"] <= region.area <= config["max_nucleus_area"]:
            nucleus_id = region.label
            area_pixels = region.area
            area_um2 = area_pixels * config["area_um2_per_pixel2"]
            mean_intensity = region.mean_intensity
            raw_intden = np.sum(region.intensity_image[region.image])
            initial_results.append({
                'Nucleus_ID': nucleus_id,
                'Area_pixels': area_pixels,
                'Area_um2': area_um2,
                'Mean_Intensity': mean_intensity,
                'RawIntDen': raw_intden
            })
            valid_nuclei_labels.append(nucleus_id)
    return initial_results, valid_nuclei_labels

# Segment 5: Visualize Segmentation
def visualize_segmentation(nuclear, lamin_rings, overlay, labeled_nuclei, valid_nuclei_labels, file_name, title, save_path=None):
    if any(x is None for x in [nuclear, lamin_rings, overlay, labeled_nuclei]):
        return
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(nuclear, cmap='gray')
    plt.title(f'Nuclear Channel - {file_name}')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(lamin_rings, cmap='gray')
    plt.title(f'Nuclear Lamin Rings - {file_name}')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    for region in regionprops(labeled_nuclei):
        if region.label in valid_nuclei_labels:
            y, x = region.centroid
            plt.text(x, y, str(region.label), color='white', fontsize=8, ha='center', va='center')
    plt.title(title)
    plt.axis('off')
    if save_path and not INTERACTIVE:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# Segment 6: Interactive Nucleus Selection
def interactive_selection(nuclear, labeled_nuclei, valid_nuclei_labels, file_name, output_dir):
    if any(x is None for x in [nuclear, labeled_nuclei, valid_nuclei_labels]):
        print("Error: Invalid data for interactive selection.")
        return []
    print(f"\nValid nucleus IDs for {file_name}: {valid_nuclei_labels}")
    print("Left-click to exclude nuclei, right-click to include back. Close window to finish.")

    def create_overlay(nuclear, labeled_nuclei, valid_nuclei_labels, exclude_ids):
        overlay = np.stack([nuclear, nuclear, nuclear], axis=-1)
        overlay_max = overlay.max()
        if overlay_max == 0:
            overlay = np.zeros_like(overlay, dtype=np.uint8) + 128
        else:
            overlay = (overlay / overlay_max * 255).astype(np.uint8)
        valid_mask = np.isin(labeled_nuclei, valid_nuclei_labels)
        exclude_mask = np.isin(labeled_nuclei, exclude_ids)
        overlay[valid_mask & ~exclude_mask, 0] = 255  # Yellow for included
        overlay[valid_mask & ~exclude_mask, 1] = 255
        overlay[valid_mask & ~exclude_mask, 2] = 0
        overlay[exclude_mask, 0] = 255  # Red for excluded
        overlay[exclude_mask, 1] = 0
        overlay[exclude_mask, 2] = 0
        alpha = 0.3
        overlay = (alpha * overlay + (1 - alpha) * np.stack([nuclear, nuclear, nuclear], axis=-1)).astype(np.uint8)
        return overlay

    exclude_ids = []
    if INTERACTIVE:
        fig, ax = plt.subplots(figsize=(12, 12))
        overlay = create_overlay(nuclear, labeled_nuclei, valid_nuclei_labels, [])
        ax.imshow(overlay)
        for region in regionprops(labeled_nuclei):
            if region.label in valid_nuclei_labels:
                y, x = region.centroid
                ax.text(x, y, str(region.label), color='white', fontsize=8, ha='center', va='center')
        ax.set_title(f'Left-click to exclude, right-click to include - {file_name}')
        ax.axis('off')

        def on_click(event):
            if event.inaxes != ax:
                return
            x, y = int(event.xdata), int(event.ydata)
            tolerance = 10
            label_at_click = 0
            min_dist = tolerance
            for region in regionprops(labeled_nuclei):
                if region.label not in valid_nuclei_labels:
                    continue
                ry, rx = region.centroid
                dist = np.sqrt((y - ry)**2 + (x - rx)**2)
                if dist < min_dist:
                    min_dist = dist
                    label_at_click = region.label
            if label_at_click in valid_nuclei_labels:
                if event.button == 1:  # Left-click to exclude
                    if label_at_click not in exclude_ids:
                        exclude_ids.append(label_at_click)
                        print(f"Excluding nucleus ID: {label_at_click}")
                elif event.button == 3:  # Right-click to include
                    if label_at_click in exclude_ids:
                        exclude_ids.remove(label_at_click)
                        print(f"Including nucleus ID: {label_at_click}")
                ax.clear()
                overlay = create_overlay(nuclear, labeled_nuclei, valid_nuclei_labels, exclude_ids)
                ax.imshow(overlay)
                for region in regionprops(labeled_nuclei):
                    if region.label in valid_nuclei_labels:
                        y, x = region.centroid
                        ax.text(x, y, str(region.label), color='white', fontsize=8, ha='center', va='center')
                ax.set_title(f'Left-click to exclude, right-click to include - {file_name}')
                ax.axis('off')
                plt.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)
    else:
        print("Interactive mode unavailable; skipping nucleus selection.")

    invalid_ids = [id for id in exclude_ids if id not in valid_nuclei_labels]
    if invalid_ids:
        print(f"Warning: Invalid nucleus IDs {invalid_ids} ignored.")
        exclude_ids = [id for id in exclude_ids if id in valid_nuclei_labels]
    print(f"Excluded nuclei: {exclude_ids}")
    selection_file = Path(output_dir) / f"{file_name}_excluded_ids.txt"
    with open(selection_file, 'w') as f:
        f.write(','.join(map(str, exclude_ids)))
    print(f"Excluded IDs saved to {selection_file}")
    return exclude_ids

# Segment 7: Final Quantification
def final_quantification(initial_results, exclude_ids):
    results = [r for r in initial_results if r['Nucleus_ID'] not in exclude_ids]
    print(f"\nQuantified results after excluding nuclei {exclude_ids}:")
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        return results
    else:
        print("No nuclei remain after exclusion.")
        return []

# Segment 8: Update Summary
def update_summary(file_name, results, config):
    if results:
        avg_area_pixels = np.mean([r['Area_pixels'] for r in results])
        avg_area_um2 = np.mean([r['Area_um2'] for r in results])
        avg_mean_intensity = np.mean([r['Mean_Intensity'] for r in results])
        avg_raw_intden = np.mean([r['RawIntDen'] for r in results])
        total_nuclei = len(results)
    else:
        avg_area_pixels = 0
        avg_area_um2 = 0
        avg_mean_intensity = 0
        avg_raw_intden = 0
        total_nuclei = 0
    summary = {
        'File_Name': file_name,
        'Total_Nuclei': total_nuclei,
        'Average_Area_pixels': avg_area_pixels,
        'Average_Area_um2': avg_area_um2,
        'Average_Mean_Intensity': avg_mean_intensity,
        'Average_RawIntDen': avg_raw_intden
    }
    return summary

# Segment 9: Visualize Updated Segmentation
def visualize_updated(nuclear, lamin_rings, overlay, labeled_nuclei, valid_nuclei_labels, file_name, output_dir):
    if any(x is None for x in [nuclear, lamin_rings, overlay, labeled_nuclei]):
        print("Error: Invalid data for updated visualization.")
        return
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(nuclear, cmap='gray')
    plt.title(f'Nuclear Channel - {file_name}')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(lamin_rings, cmap='gray')
    plt.title(f'Nuclear Lamin Rings - {file_name}')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    for region in regionprops(labeled_nuclei):
        if region.label in valid_nuclei_labels:
            y, x = region.centroid
            plt.text(x, y, str(region.label), color='white', fontsize=8, ha='center', va='center')
    plt.title(f'Retained Nuclei (Yellow, Labeled) - {file_name}')
    plt.axis('off')
    save_path = Path(output_dir) / f"{file_name}_updated_segmentation.png"
    if not INTERACTIVE:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# Helper: Create Overlay
def create_overlay(nuclear, labeled_nuclei, valid_nuclei_labels, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = []
    overlay = np.stack([nuclear, nuclear, nuclear], axis=-1)
    overlay_max = overlay.max()
    if overlay_max == 0:
        overlay = np.zeros_like(overlay, dtype=np.uint8) + 128
    else:
        overlay = (overlay / overlay_max * 255).astype(np.uint8)
    valid_mask = np.isin(labeled_nuclei, valid_nuclei_labels)
    exclude_mask = np.isin(labeled_nuclei, exclude_ids)
    overlay[valid_mask & ~exclude_mask, 0] = 255  # Yellow for included
    overlay[valid_mask & ~exclude_mask, 1] = 255
    overlay[valid_mask & ~exclude_mask, 2] = 0
    overlay[exclude_mask, 0] = 255  # Red for excluded
    overlay[exclude_mask, 1] = 0
    overlay[exclude_mask, 2] = 0
    alpha = 0.3
    overlay = (alpha * overlay + (1 - alpha) * np.stack([nuclear, nuclear, nuclear], axis=-1)).astype(np.uint8)
    return overlay

# Process a Single File
def process_file(file_path, file_name, config, output_dir):
    image = load_image(file_path)
    if image is None:
        return [], None, None, None, None, None
    nuclear, signal = separate_channels(image, config)
    if nuclear is None or signal is None:
        return [], None, None, None, None, None
    lamin_rings, filled_nuclei, labeled_nuclei = segment_nuclei(nuclear, config)
    if labeled_nuclei is None:
        return [], None, None, None, None, None
    initial_results, valid_nuclei_labels = quantify_nuclei(labeled_nuclei, signal, config)
    overlay = create_overlay(nuclear, labeled_nuclei, valid_nuclei_labels)
    return initial_results, nuclear, lamin_rings, labeled_nuclei, valid_nuclei_labels, overlay

# Main Program
def main():
    # Load configuration
    config = load_config("config.json")
    config = compute_scale(config)

    # Get directories
    input_dir = get_directory()
    output_dir = input("Enter the output directory path (press Enter to use input directory): ").strip()
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # List image files
    image_files = []
    for ext in config["image_extensions"]:
        image_files.extend(input_dir.glob(f"*{ext}"))
    if not image_files:
        print(f"No image files found in {input_dir} with extensions {config['image_extensions']}.")
        return

    all_results = []
    all_files_summary = []

    # Process each file
    for file_path in image_files:
        file_name = file_path.stem
        print(f"\nProcessing file: {file_name}")
        initial_results, nuclear, lamin_rings, labeled_nuclei, valid_nuclei_labels, overlay = process_file(file_path, file_name, config, output_dir)
        if not initial_results:
            continue
        save_path = output_dir / f"{file_name}_initial_segmentation.png"
        visualize_segmentation(nuclear, lamin_rings, overlay, labeled_nuclei, valid_nuclei_labels, file_name, title=f"Initial Segmentation - {file_name}", save_path=save_path)
        exclude_ids = interactive_selection(nuclear, labeled_nuclei, valid_nuclei_labels, file_name, output_dir)
        results = final_quantification(initial_results, exclude_ids)
        all_results.extend(results)
        summary = update_summary(file_name, results, config)
        all_files_summary.append(summary)
        updated_overlay = create_overlay(nuclear, labeled_nuclei, [id for id in valid_nuclei_labels if id not in exclude_ids])
        visualize_updated(nuclear, lamin_rings, updated_overlay, labeled_nuclei, valid_nuclei_labels, file_name, output_dir)

    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = output_dir / "all_quantification.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nAll quantification results saved to {output_file}")
    else:
        print("\nNo quantification results to save.")

    # Display and save summary
    print("\nSummary across all files:")
    summary_df = pd.DataFrame(all_files_summary)
    print(summary_df.to_string(index=False))
    summary_file = output_dir / "summary_quantification.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")
    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()