import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from pathlib import Path
from skimage.measure import regionprops

# Import our modularized backend logic
from config import load_config, compute_scale
import image_processing as ip

class NucleusQuantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nucleus Quantification App")
        self.geometry("1400x900")
        
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
        for col in self.results_table['columns']:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=90)
        self.results_table.pack(fill=tk.BOTH, expand=True)

        tk.Label(left_panel, text="Summary Across Files").pack()
        self.summary_table = ttk.Treeview(left_panel, columns=('File', 'Nuclei', 'Avg_Pixels', 'Avg_um2', 'Avg_Intensity', 'Avg_RawIntDen', 'Avg_IntDen'), show='headings')
        for col in self.summary_table['columns']:
            self.summary_table.heading(col, text=col)
            self.summary_table.column(col, width=90)
        self.summary_table.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, bg='white', width=1000, height=600)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Button-3>', self.on_canvas_click)

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
        
        image = ip.load_image(file_path)
        if image is None: return
        
        self.nuclear, signal = ip.separate_channels(image, self.config)
        if self.nuclear is None or signal is None: return
        
        self.lamin_rings, _, self.labeled_nuclei = ip.segment_nuclei(self.nuclear, self.config)
        if self.labeled_nuclei is None: return
        
        self.initial_results, self.valid_nuclei_labels = ip.quantify_nuclei(self.labeled_nuclei, signal, self.config)
        self.exclude_ids = []
        
        self.update_visualization(file_name)
        self.update_results_table()

    def update_visualization(self, file_name):
        self.canvas.delete("all")
        self.overlay = ip.create_overlay(self.nuclear, self.labeled_nuclei, self.valid_nuclei_labels, self.exclude_ids)
        
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
                self.canvas.create_text(x * scale_x, y * scale_y, text=str(region.label), fill='white', font=('Arial', 12))
        
        self.canvas.create_text(10, 10, anchor=tk.NW, text=f'Select Nuclei - {file_name} (Left: exclude, Right: include)', fill='black', font=('Arial', 14, 'bold'))

    def on_canvas_click(self, event):
        if not self.valid_nuclei_labels: return
        
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
            elif event.num == 3:  # Right-click to include
                if label_at_click in self.exclude_ids:
                    self.exclude_ids.remove(label_at_click)
            
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
        results = [r for r in self.initial_results if r['Nucleus_ID'] not in self.exclude_ids]
        self.all_results.extend(results)
        
        summary = {
            'File_Name': file_name,
            'Total_Nuclei': len(results),
            'Average_Area_pixels': np.mean([r['Area_pixels'] for r in results]) if results else 0,
            'Average_Area_um2': np.mean([r['Area_um2'] for r in results]) if results else 0,
            'Average_Mean_Intensity': np.mean([r['Mean_Intensity'] for r in results]) if results else 0,
            'Average_RawIntDen': np.mean([r['RawIntDen'] for r in results]) if results else 0,
            'Average_IntDen': np.mean([r['IntDen'] for r in results]) if results else 0,
        }
        self.all_files_summary.append(summary)
        
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
            pd.DataFrame(self.all_results).to_csv(Path(self.output_dir) / "all_quantification.csv", index=False)
            pd.DataFrame(self.all_files_summary).to_csv(Path(self.output_dir) / "summary_quantification.csv", index=False)
            
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