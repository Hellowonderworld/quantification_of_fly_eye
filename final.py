import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use PyQt5 backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
                             QLabel, QMessageBox)
from PyQt5.QtCore import Qt
from skimage.io import imread as tiff_imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_dilation, binary_opening, disk
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.feature import peak_local_max
import pandas as pd
from pathlib import Path
import json
import os
import sys
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage.segmentation import find_boundaries


# Configuration defaults
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

# Load configuration
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

# Compute scale-dependent values
def compute_scale(config):
    pixels_per_um = config["pixels_per_um"]
    config["um_per_pixel"] = 1 / pixels_per_um
    config["area_um2_per_pixel2"] = config["um_per_pixel"] ** 2
    return config

# Load and validate image
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

# Separate channels
def separate_channels(image, config):
    if image is None or image.ndim != 3:
        return None, None
    max_channel = image.shape[0] - 1
    nuclear_channel = min(config["nuclear_channel"], max_channel)
    signal_channel = min(config["signal_channel"], max_channel)
    nuclear = image[nuclear_channel]
    signal = image[signal_channel]
    return nuclear, signal

# Segment nuclei
# 請確保在檔案最上方加入這個 import
from skimage.morphology import remove_small_holes

def segment_nuclei(nuclear, config):
    if nuclear is None:
        return None, None, None
        
    smoothed = gaussian(nuclear, sigma=config["sigma"])
    base_thresh = threshold_otsu(smoothed)
    
    # 產生初始光環遮罩
    lamin_rings = smoothed > (base_thresh * config["thresh_factor_low"])
    lamin_rings = binary_opening(lamin_rings, disk(3))
    
    # 🚨 關鍵修正區域：取代原本的 binary_fill_holes 🚨
    # 使用 remove_small_holes，並給定一個細胞內部空洞的極限大小。
    # 這裡預設為 5000 像素，代表大於 5000 像素的背景黑洞不會被錯誤填滿。
    # (您可以將 "max_hole_area" 加入您的 config 檔中，以便未來隨時微調)
    max_hole = config.get("max_hole_area", 5000)
    filled_nuclei = remove_small_holes(lamin_rings, area_threshold=max_hole)
    
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

# Quantify nuclei
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
    # Integrate IntDen and CorrectedIntDen calculation
    initial_results = calculate_intden(labeled_nuclei, signal_channel, initial_results)
    return initial_results, valid_nuclei_labels

# Calculate IntDen 
def calculate_intden(labeled_nuclei, signal_channel, initial_results):
    if labeled_nuclei is None or signal_channel is None or not initial_results:
        return initial_results
    # Estimate background intensity for CorrectedIntDen
    non_nucleus_mask = labeled_nuclei == 0
    background_intensity = np.mean(signal_channel[non_nucleus_mask]) if non_nucleus_mask.sum() > 0 else 0
    for result in initial_results:
        # IntDen = Area (pixels) × Mean Gray Value (per user definition)
        result['IntDen'] = result['Area_um2'] * result['Mean_Intensity']
        nucleus_id = result['Nucleus_ID']
        nucleus_mask = labeled_nuclei == nucleus_id
        total_intensity = np.sum(signal_channel[nucleus_mask])
    return initial_results

# Create overlay for visualization
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

# Main application window
class NucleusQuantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nucleus Quantification App")
        self.state('zoomed')
        self.config = load_config()
        self.config = compute_scale(self.config)
        self.input_dir = None
        self.output_dir = None
        self.image_files = []
        self.current_file_idx = 0
        self.initial_results = []
        self.nuclear = None
        self.lamin_rings = None
        self.labeled_nuclei = None
        self.valid_nuclei_labels = []
        self.overlay = None
        self.exclude_ids = []
        self.all_results = []
        self.all_files_summary = []
        self.photo = None
        self.init_ui()

    def init_ui(self):
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Button(left_panel, text="Select Input Directory", command=self.select_input_dir).pack(fill=tk.X)
        self.input_label = tk.Label(left_panel, text="No input directory selected")
        self.input_label.pack(fill=tk.X)

        tk.Button(left_panel, text="Select Output Directory", command=self.select_output_dir).pack(fill=tk.X)
        self.output_label = tk.Label(left_panel, text="No output directory selected")
        self.output_label.pack(fill=tk.X)

        tk.Button(left_panel, text="Load Images", command=self.load_images).pack(fill=tk.X)

        nav_frame = tk.Frame(left_panel)
        nav_frame.pack(fill=tk.X, pady=5)
        tk.Button(nav_frame, text="Previous Image", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next Image", command=self.next_image).pack(side=tk.RIGHT)

        tk.Label(left_panel, text="Quantification Results").pack()
        self.results_table = ttk.Treeview(left_panel, columns=('ID', 'Area_pixels', 'Area_um2', 'Mean_Intensity', 'RawIntDen', 'IntDen'), show='headings')
        self.results_table.heading('ID', text='Nucleus ID')
        self.results_table.heading('Area_pixels', text='Area (pixels)')
        self.results_table.heading('Area_um2', text='Area (μm²)')
        self.results_table.heading('Mean_Intensity', text='Mean Intensity')
        self.results_table.heading('RawIntDen', text='RawIntDen')
        self.results_table.heading('IntDen', text='IntDen')
        self.results_table.column('ID', width=80)
        self.results_table.column('Area_pixels', width=100)
        self.results_table.column('Area_um2', width=100)
        self.results_table.column('Mean_Intensity', width=100)
        self.results_table.column('RawIntDen', width=100)
        self.results_table.column('IntDen', width=100)
        self.results_table.pack(fill=tk.BOTH, expand=True)

        tk.Label(left_panel, text="Summary Across Files").pack()
        self.summary_table = ttk.Treeview(left_panel, columns=('File', 'Nuclei', 'Avg_Pixels', 'Avg_um2', 'Avg_Intensity', 'Avg_RawIntDen', 'Avg_IntDen'), show='headings')
        self.summary_table.heading('File', text='File Name')
        self.summary_table.heading('Nuclei', text='Total Nuclei')
        self.summary_table.heading('Avg_Pixels', text='Avg Area (pixels)')
        self.summary_table.heading('Avg_um2', text='Avg Area (μm²)')
        self.summary_table.heading('Avg_Intensity', text='Avg Intensity')
        self.summary_table.heading('Avg_RawIntDen', text='Avg RawIntDen')
        self.summary_table.heading('Avg_IntDen', text='Avg IntDen')
        self.summary_table.column('File', width=150)
        self.summary_table.column('Nuclei', width=80)
        self.summary_table.column('Avg_Pixels', width=100)
        self.summary_table.column('Avg_um2', width=100)
        self.summary_table.column('Avg_Intensity', width=100)
        self.summary_table.column('Avg_RawIntDen', width=100)
        self.summary_table.column('Avg_IntDen', width=100)
        self.summary_table.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, bg='white', width=1000, height=600)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Button-3>', self.on_canvas_click)
        # 原本的 Load Images 按鈕
        tk.Button(left_panel, text="Load Images", command=self.load_images).pack(fill=tk.X)
        
        # --- 新增這行：Clear All 按鈕 ---
        tk.Button(left_panel, text="Clear All Data", command=self.clear_all, bg="#ff4c4c", fg="white", font=('Arial', 10, 'bold')).pack(fill=tk.X, pady=5)
        # ------------------------------

    def select_input_dir(self):
        self.input_dir = filedialog.askdirectory(title="Select Input Directory")
        if self.input_dir:
            self.input_label.config(text=f"Input: {self.input_dir}")

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")
        if self.output_dir:
            self.output_label.config(text=f"Output: {self.output_dir}")
        else:
            self.output_dir = self.input_dir

    def load_images(self):
        if not self.input_dir:
            messagebox.showwarning("Error", "Please select an input directory.")
            return
        self.image_files = []
        for ext in self.config["image_extensions"]:
            self.image_files.extend(Path(self.input_dir).glob(f"*{ext}"))
        if not self.image_files:
            messagebox.showwarning("Error", f"No image files found in {self.input_dir}.")
            return
        self.current_file_idx = 0
        self.process_current_file()

    def process_current_file(self):
        if not self.image_files or self.current_file_idx >= len(self.image_files):
            return
        file_path = self.image_files[self.current_file_idx]
        file_name = file_path.stem
        image = load_image(file_path)
        if image is None:
            return
        self.nuclear, signal = separate_channels(image, self.config)
        if self.nuclear is None or signal is None:
            return
        self.lamin_rings, _, self.labeled_nuclei = segment_nuclei(self.nuclear, self.config)
        if self.labeled_nuclei is None:
            return
        self.initial_results, self.valid_nuclei_labels = quantify_nuclei(self.labeled_nuclei, signal, self.config)
        self.exclude_ids = []
        self.update_visualization(file_name)
        self.update_results_table()

    def update_visualization(self, file_name):
        self.canvas.delete("all")
        
        import numpy as np
        from skimage.segmentation import find_boundaries
        from skimage.morphology import dilation, square
        
        bright_nuclear = np.clip(self.nuclear.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
        self.overlay = create_overlay(bright_nuclear, self.labeled_nuclei, self.valid_nuclei_labels, self.exclude_ids)
        
        # --- 修正核心：只針對有效的細胞核 (valid_nuclei_labels) 畫邊界 ---
        filtered_labels = np.where(np.isin(self.labeled_nuclei, self.valid_nuclei_labels), self.labeled_nuclei, 0)
        boundaries = find_boundaries(filtered_labels, mode='outer')
        thick_boundaries = dilation(boundaries, square(3))
        # -------------------------------------------------------------
        
        if len(self.overlay.shape) == 3: 
            self.overlay[thick_boundaries] = [255, 0, 0] 
        else:
            self.overlay[thick_boundaries] = 255

        max_size = (1000, 600)
        img = Image.fromarray(self.overlay)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        scale_x = img.width / self.overlay.shape[1]
        scale_y = img.height / self.overlay.shape[0]
        
        for region in regionprops(self.labeled_nuclei):
            if region.label in self.valid_nuclei_labels:
                y, x = region.centroid
                text_color = 'red' if region.label in self.exclude_ids else 'green'
                self.canvas.create_text(x * scale_x, y * scale_y, text=str(region.label), fill=text_color, font=('Arial', 12, 'bold'))
                
        self.canvas.create_text(10, 10, anchor=tk.NW, text=f'Select Nuclei - {file_name} (Left-click: exclude, Right-click: include)', fill='yellow', font=('Arial', 14, 'bold'))
    def on_canvas_click(self, event):
        if not self.valid_nuclei_labels:
            return
        img = Image.fromarray(self.overlay)
        img.thumbnail((1000, 600), Image.Resampling.LANCZOS)
        scale_x = self.overlay.shape[1] / img.width
        scale_y = self.overlay.shape[0] / img.height
        x, y = int(event.x * scale_x), int(event.y * scale_y)
        tolerance = 15
        label_at_click = 0
        min_dist = tolerance
        for region in regionprops(self.labeled_nuclei):
            if region.label not in self.valid_nuclei_labels:
                continue
            ry, rx = region.centroid
            dist = np.sqrt((y - ry)**2 + (x - rx)**2)
            if dist < min_dist:
                min_dist = dist
                label_at_click = region.label
        if label_at_click in self.valid_nuclei_labels:
            if event.num == 1:  # Left-click to exclude
                if label_at_click not in self.exclude_ids:
                    self.exclude_ids.append(label_at_click)
                    print(f"Excluding nucleus ID: {label_at_click}")
            elif event.num == 3:  # Right-click to include
                if label_at_click in self.exclude_ids:
                    self.exclude_ids.remove(label_at_click)
                    print(f"Including nucleus ID: {label_at_click}")
            self.update_visualization(self.image_files[self.current_file_idx].stem)
            self.update_results_table()

    def update_results_table(self):
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        results = [r for r in self.initial_results if r['Nucleus_ID'] not in self.exclude_ids]
        for result in results:
            self.results_table.insert('', 'end', values=(
                result['Nucleus_ID'],
                f"{result['Area_pixels']:.2f}",
                f"{result['Area_um2']:.2f}",
                f"{result['Mean_Intensity']:.2f}",
                f"{result['RawIntDen']:.2f}",
                f"{result['IntDen']:.2f}",

            ))

    def save_current_results(self):
        if not self.initial_results or not self.output_dir:
            return
        file_name = self.image_files[self.current_file_idx].stem
        
        # 1. Filter out excluded nuclei AND add 'File_Name' to each individual row
        filtered_results = []
        for r in self.initial_results:
            if r['Nucleus_ID'] not in self.exclude_ids:
                # This prepends 'File_Name' as the first column for the detailed CSV
                nucleus_record = {'File_Name': file_name, **r}
                filtered_results.append(nucleus_record)
                
        # Append the detailed rows to the global list (for all_quantification.csv)
        self.all_results.extend(filtered_results)
        
        # 2. Build the summary dictionary (for summary_quantification.csv)
        # It already includes 'File_Name' as its first column key!
        summary = {
            'File_Name': file_name,
            'Total_Nuclei': len(filtered_results),
            'Average_Area_pixels': np.mean([r['Area_pixels'] for r in filtered_results]) if filtered_results else 0,
            'Average_Area_um2': np.mean([r['Area_um2'] for r in filtered_results]) if filtered_results else 0,
            'Average_Mean_Intensity': np.mean([r['Mean_Intensity'] for r in filtered_results]) if filtered_results else 0,
            'Average_RawIntDen': np.mean([r['RawIntDen'] for r in filtered_results]) if filtered_results else 0,
            'Average_IntDen': np.mean([r['IntDen'] for r in filtered_results]) if filtered_results else 0,
        }
        
        # Append the summary row to the global summary list
        self.all_files_summary.append(summary)
        
        # 3. Save the txt log of excluded IDs if there are any
        if self.exclude_ids:
            try:
                with open(Path(self.output_dir) / f"{file_name}_excluded_ids.txt", 'w') as f:
                    f.write(','.join(map(str, self.exclude_ids)))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save excluded IDs for {file_name}: {e}")

    def finalize_processing(self):
        if not self.all_results or not self.output_dir:
            messagebox.showwarning("Warning", "No results to save or output directory not set.")
            return
        try:
            results_df = pd.DataFrame(self.all_results)
            results_df.to_csv(Path(self.output_dir) / "all_quantification.csv", index=False)
            summary_df = pd.DataFrame(self.all_files_summary)
            summary_df.to_csv(Path(self.output_dir) / "summary_quantification.csv", index=False)
            for item in self.summary_table.get_children():
                self.summary_table.delete(item)
            for summary in self.all_files_summary:
                self.summary_table.insert('', 'end', values=(
                    summary['File_Name'],
                    summary['Total_Nuclei'],
                    f"{summary['Average_Area_pixels']:.2f}",
                    f"{summary['Average_Area_um2']:.2f}",
                    f"{summary['Average_Mean_Intensity']:.2f}",
                    f"{summary['Average_RawIntDen']:.2f}",
                    f"{summary['Average_IntDen']:.2f}",
                ))
            messagebox.showinfo("Done", "All files processed and results saved to output directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def prev_image(self):
        if self.current_file_idx > 0:
            self.save_current_results()
            self.current_file_idx -= 1
            self.process_current_file()

    def next_image(self):
        if self.current_file_idx < len(self.image_files) - 1:
            self.save_current_results()
            self.current_file_idx += 1
            self.process_current_file()
        else:
            self.save_current_results()
            self.finalize_processing()
    def clear_all(self):
        """Clears all accumulated data and resets the tables."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all accumulated results and summaries?"):
            # Reset data structures
            self.all_results = []
            self.all_files_summary = []
            self.initial_results = []
            self.exclude_ids = []
            
            # Clear Results Table
            for item in self.results_table.get_children():
                self.results_table.delete(item)
                
            # Clear Summary Table
            for item in self.summary_table.get_children():
                self.summary_table.delete(item)
            
            # Clear Image Canvas
            self.canvas.delete("all")
            self.photo = None
            
            messagebox.showinfo("Cleared", "All data has been cleared.")

if __name__ == "__main__":
    app = NucleusQuantApp()
    app.mainloop()  