import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from PIL import Image, ImageTk
import tkinter as tk

#########################################
# Utility Function: Point-Line Distance
#########################################
def point_line_distance(point, line_start, line_end):
    """
    Computes the perpendicular distance from a point to a line segment.
    
    Args:
        point (tuple): (x, y) coordinates of the point.
        line_start (tuple): (x, y) coordinates of the start of the line segment.
        line_end (tuple): (x, y) coordinates of the end of the line segment.
        
    Returns:
        float: The perpendicular distance from the point to the line segment.
    """
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.dot(line_vec, line_vec)
    t = np.clip(np.dot(point_vec, line_vec) / line_len, 0, 1)
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

#########################################
# GroundTruthGenerator Class
#########################################
class GroundTruthGenerator:
    """
    Orchestrates the workflow for processing RGB and Thermal image pairs.
    
    Workflow:
      1. Loads an RGB and Thermal image pair from the data directory.
      2. Extracts keypoints and descriptors from each image using SuperPoint.
      3. Matches keypoints between images using SuperGlue.
      4. Visualizes the matches and provides an interactive UI for user quality control.
         - The user can select/deselect matches (selected matches highlighted in green).
         - Zoom and pan functionality are added.
         - The match currently hovered over is highlighted in dark blue.
      5. Upon user confirmation, writes the approved matches to a ground truth file in the SuperGlue format.
      6. Waits for a 'Next' command from the user to process the next image pair.
    """
    def __init__(self, data_dir: str, backend_dir: str, pair_ids: list,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 debug: bool = False):
        self.device = device
        self.debug = debug

        # SuperPoint Model Configuration
        self.sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }
        self.superpoint = SuperPoint(self.sp_config).eval().to(self.device)

        # SuperGlue Model Configuration
        self.sg_config = {'weights': 'indoor'}
        self.superglue = SuperGlue(self.sg_config).eval().to(self.device)

        if self.debug:
            print(f"[GroundTruthGenerator] Models loaded on {self.device}")
            
        self.data_dir = data_dir
        self.backend_dir = backend_dir
        self.pair_ids = pair_ids

        if self.debug:
            print(f"[DEBUG] Initializing GroundTruthGenerator with data_dir: {data_dir} and backend_dir: {backend_dir}")

        # Initialize helper components.
        self.image_loader = ImageLoader(data_dir)
        self.sp_extractor = SuperPointExtractor()
        self.sg_matcher = SuperGlueMatcher()
        self.file_writer = GroundTruthFileWriter(backend_dir)

        self.current_pair_id = None
    
    def on_matches_selected(self, selected_matches):
        if self.debug:
            print(f"[DEBUG] User approved {len(selected_matches)} matches.")
        if not hasattr(self, "current_pair_id"):
            print("[ERROR] current_pair_id is not set before saving matches.")
            return
        self.file_writer.write_ground_truth_file(self.current_pair_id, selected_matches)
        if self.debug:
            print(f"[DEBUG] Ground truth file written for pair: {self.current_pair_id}")

    def process_image_pair(self, pair_id: str, debug: bool = False):
        if debug:
            print(f"[DEBUG] Starting processing for image pair: {pair_id}")
        
        self.current_pair_id = pair_id
        # Step 1: Load the image pair.
        rgb_image, thermal_image = self.image_loader.load_image_pair(pair_id)

        # Convert images to RGB for display.
        org_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        org_thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)

        # Create grayscale versions for feature extraction.
        grayscale_rgb_image = cv2.cvtColor(org_rgb_image, cv2.COLOR_BGR2GRAY)
        grayscale_thermal_image = cv2.cvtColor(org_thermal_image, cv2.COLOR_BGR2GRAY)

        grayscale_rgb_image = grayscale_rgb_image.astype(np.float32) / 255.0
        grayscale_thermal_image = grayscale_thermal_image.astype(np.float32) / 255.0

        grayscale_rgb_image = torch.from_numpy(grayscale_rgb_image).unsqueeze(0).unsqueeze(0).to(self.device)
        grayscale_thermal_image = torch.from_numpy(grayscale_thermal_image).unsqueeze(0).unsqueeze(0).to(self.device)

        if debug:
            print(f"[DEBUG] Loaded images and grayscale images for pair {pair_id}.")

        # Step 2: Extract keypoints and descriptors.
        kpts1, desc1, scores1 = self.sp_extractor.extract_keypoints(grayscale_rgb_image)
        kpts2, desc2, scores2 = self.sp_extractor.extract_keypoints(grayscale_thermal_image)
        if debug:
            print("[DEBUG] Extracted keypoints for RGB and Thermal.")

        kpts1 = kpts1.unsqueeze(0)
        kpts2 = kpts2.unsqueeze(0)
        scores1 = scores1.unsqueeze(0)
        scores2 = scores2.unsqueeze(0)

        # Step 3: Prepare data dictionary for SuperGlue.
        data = {
            'image0': grayscale_rgb_image, 
            'image1': grayscale_thermal_image,
            'keypoints0': kpts1, 
            'keypoints1': kpts2,
            'descriptors0': desc1, 
            'descriptors1': desc2,
            'scores0': scores1, 
            'scores1': scores2
        }
        if data['scores0'].dim() == 3:
            data['scores0'] = data['scores0'].squeeze(-1)
        if data['scores1'].dim() == 3:
            data['scores1'] = data['scores1'].squeeze(-1)

        # Step 4: Compute matches using SuperGlue.
        matches = self.sg_matcher.match_keypoints(data)
        if debug:
            print(f"[DEBUG] Computed {len(matches)} matches.")

        # Step 5: Visualize matches with the interactive UI (with zoom/pan and dark-blue hover).
        visualizer_ui = MatchVisualizerUI(org_rgb_image, org_thermal_image, matches, self.on_matches_selected)
        visualizer_ui.show()

    def run(self, debug: bool = False):
        if debug:
            print("[DEBUG] Starting main processing loop...")
        pair_ids = self.pair_ids
        for pair_id in pair_ids:
            if debug:
                print(f"[DEBUG] Processing next image pair: {pair_id}")
            self.process_image_pair(pair_id, debug=debug)
        if debug:
            print("[DEBUG] Completed processing all image pairs.")

    def get_all_pair_ids(self, debug: bool = False):
        # Placeholder implementation.
        pair_ids = ["pair_1", "pair_2", "pair_3"]
        if debug:
            print(f"[DEBUG] Retrieved pair identifiers: {pair_ids}")
        return pair_ids

#########################################
# Dependency 1: ImageLoader
#########################################
class ImageLoader:
    def __init__(self, data_dir: str, debug: bool = False):
        self.data_dir = data_dir
        self.debug = debug
        if self.debug:
            print(f"[DEBUG] ImageLoader initialized with directory: {data_dir}")
    
    def load_image_pair(self, pair_id: str, debug: bool = False):
        rgb_path = os.path.join(self.data_dir, "rgb", f"{pair_id}.jpg")
        thermal_path = os.path.join(self.data_dir, "thermal", f"{pair_id}.jpg")
        if debug or self.debug:
            print(f"[DEBUG] Loading RGB image from: {rgb_path}")
            print(f"[DEBUG] Loading Thermal image from: {thermal_path}")
        rgb_image = cv2.imread(rgb_path)
        thermal_image = cv2.imread(thermal_path)
        rgb_image = cv2.resize(rgb_image, (640, 480))
        thermal_image = cv2.resize(thermal_image, (640, 480))
        if rgb_image is None or thermal_image is None:
            raise FileNotFoundError(f"Could not load images for pair {pair_id}")
        if debug or self.debug:
            print(f"[DEBUG] Loaded images: RGB {rgb_image.shape}, Thermal {thermal_image.shape}")
        return rgb_image, thermal_image

#########################################
# Dependency 2: SuperPointExtractor
#########################################
class SuperPointExtractor:
    def __init__(self, debug: bool = False):
        self.orb = cv2.ORB_create()
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }
        self.superpoint = SuperPoint(self.sp_config).eval().to(self.device)
        if self.debug:
            print("[DEBUG] SuperPointExtractor initialized.")
    
    def extract_keypoints(self, image, debug: bool = False):
        if image is None:
            raise ValueError("Failed to load image.")
        with torch.no_grad():
            output = self.superpoint({'image': image})
        keypoints = output['keypoints'][0].cpu()
        descriptors = output['descriptors'][0].cpu()
        scores = output['scores'][0].cpu()
        if self.debug:
            print(f"[DEBUG] Extracted {len(keypoints)} keypoints.")
        return keypoints, descriptors, scores

#########################################
# Dependency 3: SuperGlueMatcher
#########################################
class SuperGlueMatcher:
    def __init__(self, debug: bool = False):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sg_config = {'weights': 'indoor'}
        self.superglue = SuperGlue(self.sg_config).eval().to(self.device)
        if self.debug:
            print("[DEBUG] SuperGlueMatcher initialized.")
    
    def match_keypoints(self, data, debug: bool = False):
        with torch.no_grad():
            raw_matches = self.superglue(data)['matches0'][0].cpu().numpy()
        if self.debug or debug:
            num_valid = np.sum(raw_matches > -1)
            print(f"[DEBUG] Found {num_valid} valid matches out of {raw_matches.shape[0]} keypoints.")
        keypoints0 = data['keypoints0'][0].cpu().numpy()  
        keypoints1 = data['keypoints1'][0].cpu().numpy()  
        match_pairs = []
        for i, match_idx in enumerate(raw_matches):
            if match_idx > -1:
                pt0 = keypoints0[i]
                pt1 = keypoints1[int(match_idx)]
                match_pairs.append((pt0, pt1))
        if self.debug or debug:
            print(f"[DEBUG] Converted matches to {len(match_pairs)} coordinate pairs.")
        return match_pairs

#########################################
# Dependency 4: Updated MatchVisualizerUI
# (with Zoom/Pan, dark-blue hover, and debounce for mouse wheel)
#########################################
class MatchVisualizerUI:
    def __init__(self, rgb_image, thermal_image, matches, save_callback):
        """
        Args:
            rgb_image (np.array): OpenCV BGR image for RGB.
            thermal_image (np.array): OpenCV BGR image for Thermal.
            matches (list): List of tuples: ((x_rgb, y_rgb), (x_thermal, y_thermal)).
            save_callback (function): Callback when saving approved matches.
        """
        self.org_rgb_image = rgb_image
        self.org_thermal_image = thermal_image
        self.matches = matches  # In base coordinates.
        self.save_callback = save_callback
        self.selected_matches = set()
        self.hover_threshold = 5
        self.hovered_match = None

        # Zoom & pan state.
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start = None

        # For debouncing zoom (mouse wheel) events.
        self.zoom_debounce_id = None

        self.rgb_width = self.org_rgb_image.shape[1]
        self.rgb_height = self.org_rgb_image.shape[0]
        self.thermal_width = self.org_thermal_image.shape[1]
        self.thermal_height = self.org_thermal_image.shape[0]
        self.canvas_width = self.rgb_width + self.thermal_width
        self.canvas_height = max(self.rgb_height, self.thermal_height)

        self.root = tk.Tk()
        self.root.title("Match Visualizer (Interactive with Zoom/Pan)")
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(self.root, text="Hover over a match to see its index.")
        self.status_label.pack()
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)
        self.end_button = tk.Button(self.button_frame, text="End Pair", command=self.end_pair)
        self.end_button.pack(side=tk.LEFT, padx=5)
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next_pair)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.next_button.config(state=tk.DISABLED)

        # Bind events.
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux
        self.root.bind("<KeyPress-plus>", self.on_key_zoom_in)
        self.root.bind("<KeyPress-equal>", self.on_key_zoom_in)
        self.root.bind("<KeyPress-minus>", self.on_key_zoom_out)
        self.root.bind("<Left>", self.on_arrow_left)
        self.root.bind("<Right>", self.on_arrow_right)
        self.root.bind("<Up>", self.on_arrow_up)
        self.root.bind("<Down>", self.on_arrow_down)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)

        self.draw_scene()

    def make_composite_image(self, rgb_bgr, thermal_bgr):
        h = max(rgb_bgr.shape[0], thermal_bgr.shape[0])
        w_rgb = rgb_bgr.shape[1]
        w_thermal = thermal_bgr.shape[1]
        composite = np.zeros((h, w_rgb + w_thermal, 3), dtype=np.uint8)
        composite[:rgb_bgr.shape[0], :w_rgb] = rgb_bgr
        composite[:thermal_bgr.shape[0], w_rgb:] = thermal_bgr
        return composite

    def draw_scene(self):
        self.canvas.delete("all")
        composite = self.make_composite_image(self.org_rgb_image, self.org_thermal_image)
        scaled_w = int(composite.shape[1] * self.scale)
        scaled_h = int(composite.shape[0] * self.scale)
        scaled_w = max(1, scaled_w)
        scaled_h = max(1, scaled_h)
        composite_scaled = cv2.resize(composite, (scaled_w, scaled_h))
        pil_img = Image.fromarray(cv2.cvtColor(composite_scaled, cv2.COLOR_BGR2RGB))
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        base_thermal_offset = self.rgb_width
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
            x2_disp = x2 + base_thermal_offset
            sx1 = x1 * self.scale + self.offset_x
            sy1 = y1 * self.scale + self.offset_y
            sx2 = x2_disp * self.scale + self.offset_x
            sy2 = y2 * self.scale + self.offset_y
            if idx in self.selected_matches:
                color = "green"
            elif self.hovered_match == idx:
                color = "#00008B"  # Dark blue
            else:
                color = "gray"
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=2)

    def canvas_to_base_coords(self, cx, cy):
        x_no_offset = cx - self.offset_x
        y_no_offset = cy - self.offset_y
        bx = x_no_offset / self.scale
        by = y_no_offset / self.scale
        return bx, by

    def on_mouse_move(self, event):
        bx, by = self.canvas_to_base_coords(event.x, event.y)
        found_hover = False
        base_thermal_offset = self.rgb_width
        hover_idx = None
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
            x2_disp = x2 + base_thermal_offset
            dist = point_line_distance((bx, by), (x1, y1), (x2_disp, y2))
            if dist < self.hover_threshold:
                hover_idx = idx
                found_hover = True
                break
        self.hovered_match = hover_idx
        if found_hover:
            self.status_label.config(text=f"Hovering over match {hover_idx}")
        else:
            self.status_label.config(text="Hover over a match to see its index.")
        self.draw_scene()

    def on_mouse_click(self, event):
        bx, by = self.canvas_to_base_coords(event.x, event.y)
        base_thermal_offset = self.rgb_width
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
            x2_disp = x2 + base_thermal_offset
            dist = point_line_distance((bx, by), (x1, y1), (x2_disp, y2))
            if dist < self.hover_threshold:
                if idx in self.selected_matches:
                    self.selected_matches.remove(idx)
                else:
                    self.selected_matches.add(idx)
                self.draw_scene()
                break

    def on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            factor = 0.9
        elif event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 1.0
        bx, by = self.canvas_to_base_coords(event.x, event.y)
        self.scale *= factor
        self.scale = max(0.1, min(10.0, self.scale))
        self.offset_x = event.x - bx * self.scale
        self.offset_y = event.y - by * self.scale
        if self.zoom_debounce_id is not None:
            self.root.after_cancel(self.zoom_debounce_id)
        self.zoom_debounce_id = self.root.after(50, self.draw_scene)

    def on_key_zoom_in(self, event):
        factor = 1.1
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        bx, by = self.canvas_to_base_coords(center_x, center_y)
        self.scale *= factor
        self.scale = min(10.0, self.scale)
        self.offset_x = center_x - bx * self.scale
        self.offset_y = center_y - by * self.scale
        self.draw_scene()

    def on_key_zoom_out(self, event):
        factor = 0.9
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        bx, by = self.canvas_to_base_coords(center_x, center_y)
        self.scale *= factor
        self.scale = max(0.1, self.scale)
        self.offset_x = center_x - bx * self.scale
        self.offset_y = center_y - by * self.scale
        self.draw_scene()

    def on_arrow_left(self, event):
        self.offset_x += 20
        self.draw_scene()

    def on_arrow_right(self, event):
        self.offset_x -= 20
        self.draw_scene()

    def on_arrow_up(self, event):
        self.offset_y += 20
        self.draw_scene()

    def on_arrow_down(self, event):
        self.offset_y -= 20
        self.draw_scene()

    def on_pan_start(self, event):
        self.pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.draw_scene()

    def end_pair(self):
        selected = [self.matches[i] for i in sorted(self.selected_matches)]
        self.save_callback(selected)
        self.end_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)

    def next_pair(self):
        self.root.destroy()

    def show(self):
        self.root.mainloop()

#########################################
# Dependency 5: GroundTruthFileWriter
#########################################
class GroundTruthFileWriter:
    def __init__(self, backend_dir: str, debug: bool = False):
        self.backend_dir = backend_dir
        self.debug = debug
        if self.debug:
            print(f"[DEBUG] GroundTruthFileWriter initialized with backend directory: {backend_dir}")
    
    def write_ground_truth_file(self, pair_id: str, selected_matches, debug: bool = False):
        filename = os.path.join(self.backend_dir, f"{pair_id}_ground_truth.txt")
        if debug or self.debug:
            print(f"[DEBUG] Writing ground truth file to: {filename}")
        with open(filename, 'w') as f:
            for (pt_rgb, pt_thermal) in selected_matches:
                line = f"{pt_rgb[0]:.2f} {pt_rgb[1]:.2f} {pt_thermal[0]:.2f} {pt_thermal[1]:.2f}\n"
                f.write(line)
                if debug or self.debug:
                    print(f"[DEBUG] Wrote match: {line.strip()}")
        if debug or self.debug:
            print("[DEBUG] Ground truth file writing complete.")

#########################################
# End of File
#########################################