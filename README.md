Nucleus Quantification App
Overview
This Python application processes multi-channel microscopy images to segment and quantify nuclei, providing measurements such as area (in pixels and μm²), mean intensity, and integrated density (RawIntDen). It features a Tkinter-based GUI for interactive nucleus selection, allowing users to exclude or include nuclei via mouse clicks. Results are saved as CSV files, and excluded nucleus IDs are logged for traceability.
Features

Image Loading: Supports .lsm
Channel Separation: Extracts nuclear and signal channels based on configuration.
Nucleus Segmentation: Uses Gaussian smoothing, Otsu thresholding, and watershed segmentation.
Quantification: Computes area, mean intensity, and RawIntDen for each nucleus.
Interactive GUI: Left-click to exclude, right-click to include nuclei in the visualization.
Output: Saves per-image results, excluded IDs, and a summary across all files as CSV.

Requirements

Python 3.x
Libraries: numpy, pandas, scikit-image, scipy, Pillow, tkinter
Install via pip:pip install numpy pandas scikit-image scipy Pillow


Note: Tkinter is typically included with Python; ensure it's available.

Usage

Run the App:
Execute the script: python nucleus_quant_app.py


Select Directories:
Click "Select Input Directory" to choose a folder with microscopy images.
Click "Select Output Directory" to specify where results and excluded IDs are saved.


Load Images:
Click "Load Images" to process all supported files in the input directory.


Interact with Nuclei:
View the current image with labeled nuclei.
Left-click a nucleus to exclude it (turns red).
Right-click an excluded nucleus to include it (turns yellow).


Navigate:
Use "Previous Image" and "Next Image" to process multiple files.


Results:
Per-image quantification appears in the "Quantification Results" table.
After processing all images, a summary table is populated.
Results are saved to all_quantification.csv and summary_quantification.csv in the output directory.
Excluded nucleus IDs are saved as <filename>_excluded_ids.txt.



Configuration

A config.json file (optional) customizes settings. If absent, defaults are used:
pixels_per_um: 18.2044 (for area conversion to μm²)
nuclear_channel: 2 (channel index for nucleus detection)
signal_channel: 0 (channel for intensity measurement)
sigma: 2 (Gaussian smoothing parameter)
thresh_factor_low: 0.7 (low threshold for lamin rings)
thresh_factor_high: 1.3 (unused in current version)
min_distance: 10 (minimum distance for peak detection in watershed)
min_nucleus_area: 2000 (minimum area in pixels)
max_nucleus_area: 10000 (maximum area in pixels)
image_extensions: [".lsm", ".tif", ".tiff", ".png"]


Edit config.json to adjust these values, e.g.:{
  "pixels_per_um": 20.0,
  "sigma": 1.5
}



Quantification Metrics

Nucleus_ID: Unique identifier for each detected nucleus.
Area_pixels: Area of the nucleus in pixels.
Area_um2: Area in square micrometers, calculated as Area_pixels * (1 / pixels_per_um)².
Mean_Intensity: Average pixel intensity within the nucleus in the signal channel.
RawIntDen: Raw Integrated Density, the sum of all pixel intensities within the nucleus in the signal channel.

Understanding IntDen and RawIntDen

Integrated Density (IntDen): A common term in image analysis, it represents the sum of pixel intensities within a region (e.g., a nucleus). It reflects the total "signal" present, combining area and intensity.
RawIntDen: In this app, RawIntDen is the raw integrated density, calculated as the sum of all pixel values within the nucleus region in the signal channel. It is computed using scikit-image's regionprops by summing the intensities in the intensity_image masked by the nucleus region. Formula:
RawIntDen = Σ(pixel intensities within nucleus)


Note: "IntDen" is often used interchangeably with "RawIntDen" in literature. Here, we report RawIntDen explicitly as the unnormalized sum, without adjustments for background or scaling.

Output Files

all_quantification.csv: Contains per-nucleus results: Nucleus_ID, Area_pixels, Area_um2, Mean_Intensity, RawIntDen.
summary_quantification.csv: Summarizes per file: File_Name, Total_Nuclei, Average_Area_pixels, Average_Area_um2, Average_Mean_Intensity, Average_RawIntDen.
_excluded_ids.txt: Lists IDs of excluded nuclei for each image.

Notes

Ensure input images are 3D (multi-channel) microscopy files.
The app uses watershed segmentation, which may require tuning via config.json for optimal results.
Excluded nuclei are removed from quantification but marked red in the visualization for reference.


