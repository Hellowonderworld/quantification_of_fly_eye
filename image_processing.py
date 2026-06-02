import numpy as np
from pathlib import Path
from skimage.io import imread as tiff_imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_opening, disk
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.feature import peak_local_max

def load_image(file_path):
    print(f"Loading file: {file_path}")
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist.")
        return None
    try:
        image = tiff_imread(file_path)
        if len(image.shape) == 3 and image.shape[-1] in [3, 4]:
            image = np.transpose(image, (2, 0, 1))
        elif len(image.shape) != 3:
            print("Error: Image must be 3D.")
            return None
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def separate_channels(image, config):
    if image is None or image.ndim != 3:
        return None, None
    max_channel = image.shape[0] - 1
    nuclear_channel = min(config["nuclear_channel"], max_channel)
    signal_channel = min(config["signal_channel"], max_channel)
    nuclear = image[nuclear_channel]
    signal = image[signal_channel]
    return nuclear, signal

def segment_nuclei(nuclear, config):
    if nuclear is None:
        return None, None, None
    smoothed = gaussian(nuclear, sigma=config["sigma"])
    base_thresh = threshold_otsu(smoothed)
    lamin_rings = smoothed > (base_thresh * config["thresh_factor_low"])
    lamin_rings = binary_opening(lamin_rings, disk(3))
    filled_nuclei = binary_fill_holes(lamin_rings)
    distance = distance_transform_edt(filled_nuclei)
    
    if distance.max() == 0:
        return None, None, None
    seeds = peak_local_max(distance, min_distance=config["min_distance"], labels=filled_nuclei)
    if len(seeds) == 0:
        return None, None, None
        
    lamin_seeds = np.zeros_like(filled_nuclei, dtype=np.int32)
    for idx, (y, x) in enumerate(seeds):
        lamin_seeds[y, x] = idx + 1
        
    labeled_nuclei = watershed(-distance, lamin_seeds, mask=filled_nuclei)
    for region in regionprops(labeled_nuclei):
        if region.area < config["min_nucleus_area"]:
            labeled_nuclei[labeled_nuclei == region.label] = 0
            
    labeled_nuclei = label(labeled_nuclei)
    return lamin_rings, filled_nuclei, labeled_nuclei

def quantify_nuclei(labeled_nuclei, signal_channel, config):
    if labeled_nuclei is None or signal_channel is None:
        return [], []
    initial_results = []
    valid_nuclei_labels = []
    
    for region in regionprops(labeled_nuclei, intensity_image=signal_channel):
        if config["min_nucleus_area"] <= region.area <= config["max_nucleus_area"]:
            nucleus_id = region.label
            area_pixels = region.area
            area_um2 = area_pixels * config["area_um2_per_pixel2"]
            mean_intensity = region.mean_intensity
            raw_int_den = np.sum(region.intensity_image[region.image])
            
            initial_results.append({
                'Nucleus_ID': nucleus_id,
                'Area_pixels': area_pixels,
                'Area_um2': area_um2,
                'Mean_Intensity': mean_intensity,
                'RawIntDen': raw_int_den
            })
            valid_nuclei_labels.append(nucleus_id)
            
    initial_results = calculate_intden(labeled_nuclei, signal_channel, initial_results)
    return initial_results, valid_nuclei_labels

def calculate_intden(labeled_nuclei, signal_channel, initial_results):
    if labeled_nuclei is None or signal_channel is None or not initial_results:
        return initial_results
    for result in initial_results:
        result['IntDen'] = result['Area_um2'] * result['Mean_Intensity']
    return initial_results

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