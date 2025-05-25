import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from PIL import Image, ImageTk
import tkinter as tk

import torch
print(torch.version.cuda)        # e.g. "11.8" or None
print(torch.backends.cudnn.enabled)  # should be True if CUDA build

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: " + device)

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
    point = np.array([float(point[0]), float(point[1])])
    line_start = np.array([float(line_start[0]), float(line_start[1])])
    line_end = np.array([float(line_end[0]), float(line_end[1])])
    
    if np.allclose(line_start, line_end):
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
    def __init__(self, data_dir: str, backend_dir: str, pair_ids: list, debug: bool = False):
        self.debug = debug

        # SuperPoint Model Configuration
        self.sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }
        self.superpoint = SuperPoint(self.sp_config).eval().to(device)

        # SuperGlue Model Configuration
        self.sg_config = {'weights': 'indoor'}
        self.superglue = SuperGlue(self.sg_config).eval().to(device)

        if self.debug:
            print(f"[GroundTruthGenerator] Models loaded on {device}")
            
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

        grayscale_rgb_image = torch.from_numpy(grayscale_rgb_image).unsqueeze(0).unsqueeze(0).to(device)
        grayscale_thermal_image = torch.from_numpy(grayscale_thermal_image).unsqueeze(0).unsqueeze(0).to(device)

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
        # Resize images to have the same height (e.g., 480px), preserving aspect ratio
        target_height = 480
        # Compute new width for RGB
        rgb_h, rgb_w = rgb_image.shape[:2]
        rgb_aspect = rgb_w / rgb_h
        new_rgb_w = int(target_height * rgb_aspect)
        rgb_image = cv2.resize(rgb_image, (new_rgb_w, target_height))
        # Compute new width for Thermal
        thermal_h, thermal_w = thermal_image.shape[:2]
        thermal_aspect = thermal_w / thermal_h
        new_thermal_w = int(target_height * thermal_aspect)
        thermal_image = cv2.resize(thermal_image, (new_thermal_w, target_height))
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
        self.sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }
        self.superpoint = SuperPoint(self.sp_config).eval().to(device)
        if self.debug:
            print("[DEBUG] SuperPointExtractor initialized.")
    
    def extract_keypoints(self, image, debug: bool = False):
        if image is None:
            raise ValueError("Failed to load image.")
        with torch.no_grad():
            output = self.superpoint({'image': image})
        keypoints = output['keypoints'][0].to(device)
        descriptors = output['descriptors'][0].to(device)
        scores = output['scores'][0].to(device)
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
        self.sg_config = {'weights': 'indoor'}
        self.superglue = SuperGlue(self.sg_config).eval().to(device)
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

        # Region zoom state
        self.cutout_size = 300  # Size of the cutout frame
        self.cutout_extract_factor = 0.7  # Extract a region smaller than the frame for more zoom
        self.active_region = None  # Currently selected region (x, y, image_type)
        self.region_matches = []  # Matches that fall into selected region
        
        # Cutout region variables
        self.rgb_cutout_region = None
        self.thermal_cutout_region = None
        self.rgb_cutout_display = None
        self.thermal_cutout_display = None
        
        # For debouncing zoom (mouse wheel) events.
        self.zoom_debounce_id = None

        # Set image dimensions based on actual input images
        self.rgb_width = self.org_rgb_image.shape[1]
        self.rgb_height = self.org_rgb_image.shape[0]
        self.thermal_width = self.org_thermal_image.shape[1]
        self.thermal_height = self.org_thermal_image.shape[0]
        self.canvas_width = self.rgb_width + self.thermal_width
        self.canvas_height = max(self.rgb_height, self.thermal_height) + self.cutout_size  # Only enough for images and cutouts

        self.root = tk.Tk()
        self.root.title("Match Visualizer (Interactive with Zoom/Pan)")
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(self.root, text="Click on a region to see zoom details. Hover over a match to see its index.")
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
        
        # Main composite image (RGB + Thermal side by side)
        composite = self.make_composite_image(self.org_rgb_image, self.org_thermal_image)
        scaled_w = int(composite.shape[1] * self.scale)
        scaled_h = int(composite.shape[0] * self.scale)
        scaled_w = max(1, scaled_w)
        scaled_h = max(1, scaled_h)
        composite_scaled = cv2.resize(composite, (scaled_w, scaled_h))
        pil_img = Image.fromarray(cv2.cvtColor(composite_scaled, cv2.COLOR_BGR2RGB))
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        
        # Draw selection boxes
        self.draw_selection_boxes()
        
        # Draw all matches on the main image
        self.draw_matches(self.matches)
        
        # Draw cutout area and region matches
        if self.active_region:
            self.draw_cutout_regions()
            
    def draw_selection_boxes(self):
        """Draw boxes to indicate where users can click on the images"""
        # RGB image region indicator
        rgb_box_color = "#00FF00"  # Green frame
        self.canvas.create_rectangle(
            self.offset_x, 
            self.offset_y, 
            self.offset_x + self.rgb_width * self.scale, 
            self.offset_y + self.rgb_height * self.scale, 
            outline=rgb_box_color, 
            width=2
        )
        
        # Thermal image region indicator
        thermal_box_color = "#FF0000"  # Red frame
        self.canvas.create_rectangle(
            self.offset_x + self.rgb_width * self.scale, 
            self.offset_y, 
            self.offset_x + (self.rgb_width + self.thermal_width) * self.scale, 
            self.offset_y + self.thermal_height * self.scale, 
            outline=thermal_box_color, 
            width=2
        )
            
    def draw_matches(self, matches_to_draw, is_region_match=False):
        """Draw the given matches on the canvas
        
        Args:
            matches_to_draw: List of matches to draw
            is_region_match: Whether the matches are from the selected region
        """
        base_thermal_offset = self.rgb_width
        
        line_width = 3 if is_region_match else 2
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(matches_to_draw):
            # Find match index in self.matches using array-safe comparison
            match_idx = -1
            for i, ((mx1, my1), (mx2, my2)) in enumerate(self.matches):
                if (np.isclose(x1, mx1) and np.isclose(y1, my1) and 
                    np.isclose(x2, mx2) and np.isclose(y2, my2)):
                    match_idx = i
                    break
            
            x2_disp = x2 + base_thermal_offset
            sx1 = x1 * self.scale + self.offset_x
            sy1 = y1 * self.scale + self.offset_y
            sx2 = x2_disp * self.scale + self.offset_x
            sy2 = y2 * self.scale + self.offset_y
            
            if match_idx in self.selected_matches:
                color = "green"
            elif self.hovered_match == match_idx:
                color = "#FF00FF"  # Bright magenta
            else:
                color = "yellow" if is_region_match else "gray"
                
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=line_width)
    
    def draw_cutout_regions(self):
        """Draw the cutout regions from both RGB and thermal images"""
        if not self.active_region:
            return
            
        click_x, click_y, img_type = self.active_region
        
        # Calculate cutout positions (below the main images)
        cutout_y_pos = self.offset_y + self.rgb_height * self.scale + 20
        
        # Get cutout regions from both images
        half_size = int(self.cutout_size * self.cutout_extract_factor // 2)
        
        # RGB cutout
        rgb_center_x = click_x if img_type == "rgb" else self.get_corresponding_x(click_x, "thermal", "rgb")
        rgb_center_y = click_y if img_type == "rgb" else self.get_corresponding_y(click_y)
        
        rgb_start_x = max(0, int(rgb_center_x - half_size))
        rgb_start_y = max(0, int(rgb_center_y - half_size))
        rgb_end_x = min(self.rgb_width, int(rgb_center_x + half_size))
        rgb_end_y = min(self.rgb_height, int(rgb_center_y + half_size))
        
        # Store cutout region bounds for drawing points later
        self.rgb_cutout_region = {
            'start_x': rgb_start_x,
            'start_y': rgb_start_y,
            'end_x': rgb_end_x,
            'end_y': rgb_end_y,
            'center_x': rgb_center_x,
            'center_y': rgb_center_y
        }
        
        # Thermal cutout
        thermal_center_x = click_x if img_type == "thermal" else self.get_corresponding_x(click_x, "rgb", "thermal")
        thermal_center_y = click_y if img_type == "thermal" else self.get_corresponding_y(click_y)
        
        thermal_start_x = max(0, int(thermal_center_x - half_size))
        thermal_start_y = max(0, int(thermal_center_y - half_size))
        thermal_end_x = min(self.thermal_width, int(thermal_center_x + half_size))
        thermal_end_y = min(self.thermal_height, int(thermal_center_y + half_size))
        
        # Store thermal cutout region bounds for drawing points later
        self.thermal_cutout_region = {
            'start_x': thermal_start_x,
            'start_y': thermal_start_y,
            'end_x': thermal_end_x,
            'end_y': thermal_end_y,
            'center_x': thermal_center_x,
            'center_y': thermal_center_y
        }
        
        # Draw RGB cutout if the region is valid
        if rgb_end_x - rgb_start_x > 0 and rgb_end_y - rgb_start_y > 0:
            rgb_cutout = self.org_rgb_image[rgb_start_y:rgb_end_y, rgb_start_x:rgb_end_x]
            # Do not scale the cutout, just center it in the cutout frame
            rgb_h, rgb_w = rgb_cutout.shape[:2]
            rgb_cutout_x = self.offset_x + self.rgb_width * self.scale // 4 - self.cutout_size // 2
            rgb_cutout_y = cutout_y_pos
            rgb_pil = Image.fromarray(cv2.cvtColor(rgb_cutout, cv2.COLOR_BGR2RGB))
            self.rgb_cutout_img = ImageTk.PhotoImage(rgb_pil)
            # Center the cutout in the frame
            center_x = rgb_cutout_x + (self.cutout_size - rgb_w) // 2
            center_y = rgb_cutout_y + (self.cutout_size - rgb_h) // 2
            self.canvas.create_rectangle(rgb_cutout_x, rgb_cutout_y, rgb_cutout_x + self.cutout_size, rgb_cutout_y + self.cutout_size, outline="#00FF00", width=2)
            self.canvas.create_image(center_x, center_y, anchor=tk.NW, image=self.rgb_cutout_img)
            self.canvas.create_text(rgb_cutout_x + self.cutout_size // 2, rgb_cutout_y - 10, text="RGB Cutout", fill="black")
            self.rgb_cutout_display = {
                'x': center_x,
                'y': center_y,
                'size': self.cutout_size,
                'w': rgb_w,
                'h': rgb_h
            }
        # Draw Thermal cutout if the region is valid
        if thermal_end_x - thermal_start_x > 0 and thermal_end_y - thermal_start_y > 0:
            thermal_cutout = self.org_thermal_image[thermal_start_y:thermal_end_y, thermal_start_x:thermal_end_x]
            # Do not scale the cutout, just center it in the cutout frame
            thermal_h, thermal_w = thermal_cutout.shape[:2]
            thermal_cutout_x = self.offset_x + self.rgb_width * self.scale + self.thermal_width * self.scale // 4 - self.cutout_size // 2
            thermal_cutout_y = cutout_y_pos
            thermal_pil = Image.fromarray(cv2.cvtColor(thermal_cutout, cv2.COLOR_BGR2RGB))
            self.thermal_cutout_img = ImageTk.PhotoImage(thermal_pil)
            # Center the cutout in the frame
            center_x = thermal_cutout_x + (self.cutout_size - thermal_w) // 2
            center_y = thermal_cutout_y + (self.cutout_size - thermal_h) // 2
            self.canvas.create_rectangle(thermal_cutout_x, thermal_cutout_y, thermal_cutout_x + self.cutout_size, thermal_cutout_y + self.cutout_size, outline="#FF0000", width=2)
            self.canvas.create_image(center_x, center_y, anchor=tk.NW, image=self.thermal_cutout_img)
            self.canvas.create_text(thermal_cutout_x + self.cutout_size // 2, thermal_cutout_y - 10, text="Thermal Cutout", fill="black")
            self.thermal_cutout_display = {
                'x': center_x,
                'y': center_y,
                'size': self.cutout_size,
                'w': thermal_w,
                'h': thermal_h
            }
        
        # Draw matches in the main image
        self.draw_matches(self.region_matches, is_region_match=True)
        
        # Draw points in cutout areas
        self.draw_cutout_match_points()
        # Draw lines between matches in the cutout frames
        self.draw_cutout_match_lines()

    def draw_cutout_match_points(self):
        """Draw match points as dots in the cutout regions"""
        if not hasattr(self, 'rgb_cutout_display') or not hasattr(self, 'thermal_cutout_display'):
            return
            
        # Dot size and color setup
        dot_radius = 4
        selected_color = "#00FF00"  # Green
        hovered_color = "#FF00FF"   # Bright magenta
        regular_color = "#FFFF00"   # Yellow
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.region_matches):
            # Find match index in the full match list
            match_idx = -1
            for i, ((mx1, my1), (mx2, my2)) in enumerate(self.matches):
                if (np.isclose(x1, mx1) and np.isclose(y1, my1) and 
                    np.isclose(x2, mx2) and np.isclose(y2, my2)):
                    match_idx = i
                    break
                    
            # Determine color based on match status
            if match_idx in self.selected_matches:
                color = selected_color
            elif self.hovered_match == match_idx:
                color = hovered_color
            else:
                color = regular_color
                
            # Calculate position in RGB cutout
            if hasattr(self, 'rgb_cutout_region') and hasattr(self, 'rgb_cutout_display'):
                # Map the point from original image coordinates to cutout display coordinates
                rgb_region = self.rgb_cutout_region
                rgb_display = self.rgb_cutout_display
                
                # Calculate relative position within cutout region (0.0 to 1.0)
                rel_x = (float(x1) - rgb_region['start_x']) / (rgb_region['end_x'] - rgb_region['start_x'])
                rel_y = (float(y1) - rgb_region['start_y']) / (rgb_region['end_y'] - rgb_region['start_y'])
                
                # Map to display position
                disp_x = rgb_display['x'] + int(rel_x * rgb_display['size'])
                disp_y = rgb_display['y'] + int(rel_y * rgb_display['size'])
                
                # Draw point if it's within bounds
                if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
                    self.canvas.create_oval(
                        disp_x - dot_radius, disp_y - dot_radius,
                        disp_x + dot_radius, disp_y + dot_radius,
                        fill=color, outline="black"
                    )
                    
            # Calculate position in Thermal cutout
            if hasattr(self, 'thermal_cutout_region') and hasattr(self, 'thermal_cutout_display'):
                # Map the point from original image coordinates to cutout display coordinates
                thermal_region = self.thermal_cutout_region
                thermal_display = self.thermal_cutout_display
                
                # Calculate relative position within cutout region (0.0 to 1.0)
                rel_x = (float(x2) - thermal_region['start_x']) / (thermal_region['end_x'] - thermal_region['start_x'])
                rel_y = (float(y2) - thermal_region['start_y']) / (thermal_region['end_y'] - thermal_region['start_y'])
                
                # Map to display position
                disp_x = thermal_display['x'] + int(rel_x * thermal_display['size'])
                disp_y = thermal_display['y'] + int(rel_y * thermal_display['size'])
                
                # Draw point if it's within bounds
                if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
                    self.canvas.create_oval(
                        disp_x - dot_radius, disp_y - dot_radius,
                        disp_x + dot_radius, disp_y + dot_radius,
                        fill=color, outline="black"
                    )

    def draw_cutout_match_lines(self):
        """Draw lines and dots between the matches in the cutout frames below, only for matches inside the cutout."""
        if not (hasattr(self, 'rgb_cutout_display') and hasattr(self, 'thermal_cutout_display')):
            return
        rgb_disp = self.rgb_cutout_display
        thermal_disp = self.thermal_cutout_display
        rgb_region = self.rgb_cutout_region
        thermal_region = self.thermal_cutout_region
        dot_radius = 6
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.region_matches):
            # Find match index in the full match list
            match_idx = -1
            for i, ((mx1, my1), (mx2, my2)) in enumerate(self.matches):
                if (np.isclose(x1, mx1) and np.isclose(y1, my1) and np.isclose(x2, mx2) and np.isclose(y2, my2)):
                    match_idx = i
                    break
            # Determine color based on match status
            if match_idx in self.selected_matches:
                color = "#00FF00"  # Green
            elif self.hovered_match == match_idx:
                color = "#FF00FF"  # Magenta
            else:
                color = "#FFFF00"  # Yellow
            # Calculate position in RGB cutout
            rel_x1 = (float(x1) - rgb_region['start_x']) / (rgb_region['end_x'] - rgb_region['start_x'])
            rel_y1 = (float(y1) - rgb_region['start_y']) / (rgb_region['end_y'] - rgb_region['start_y'])
            disp_x1 = rgb_disp['x'] + int(rel_x1 * rgb_disp['w'])
            disp_y1 = rgb_disp['y'] + int(rel_y1 * rgb_disp['h'])
            # Calculate position in Thermal cutout
            rel_x2 = (float(x2) - thermal_region['start_x']) / (thermal_region['end_x'] - thermal_region['start_x'])
            rel_y2 = (float(y2) - thermal_region['start_y']) / (thermal_region['end_y'] - thermal_region['start_y'])
            disp_x2 = thermal_disp['x'] + int(rel_x2 * thermal_disp['w'])
            disp_y2 = thermal_disp['y'] + int(rel_y2 * thermal_disp['h'])
            # Only draw if both points are within their respective cutout frames
            if (0 <= rel_x1 <= 1 and 0 <= rel_y1 <= 1 and 0 <= rel_x2 <= 1 and 0 <= rel_y2 <= 1):
                self.canvas.create_line(disp_x1, disp_y1, disp_x2, disp_y2, fill=color, width=2)
                # Draw dots at the exact match points
                self.canvas.create_oval(
                    disp_x1 - dot_radius, disp_y1 - dot_radius,
                    disp_x1 + dot_radius, disp_y1 + dot_radius,
                    fill=color, outline="black"
                )
                self.canvas.create_oval(
                    disp_x2 - dot_radius, disp_y2 - dot_radius,
                    disp_x2 + dot_radius, disp_y2 + dot_radius,
                    fill=color, outline="black"
                )

    def get_corresponding_x(self, x, from_img="rgb", to_img="thermal"):
        """Get corresponding x position in the other image based on matches"""
        if not self.matches:
            return x  # If no matches, assume same position
            
        # Find the closest match point in the source image
        min_dist = float('inf')
        closest_match = None
        
        for ((x1, y1), (x2, y2)) in self.matches:
            x1_f, x2_f = float(x1), float(x2)
            if from_img == "rgb":
                dist = abs(x - x1_f)
                if dist < min_dist:
                    min_dist = dist
                    closest_match = ((x1_f, float(y1)), (x2_f, float(y2)))
            else:  # from_img == "thermal"
                dist = abs(x - x2_f)
                if dist < min_dist:
                    min_dist = dist
                    closest_match = ((x1_f, float(y1)), (x2_f, float(y2)))
        
        if not closest_match:
            return x
            
        ((x1, y1), (x2, y2)) = closest_match
        
        # Calculate offset difference between the two images
        if from_img == "rgb" and to_img == "thermal":
            return x2 + (x - x1)
        else:  # from thermal to rgb
            return x1 + (x - x2)
            
    def get_corresponding_y(self, y):
        """Get corresponding y position in the other image based on matches"""
        # For simplicity, assuming y coordinates are roughly aligned
        return y

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
            dist = point_line_distance((float(bx), float(by)), (float(x1), float(y1)), (float(x2_disp), float(y2)))
            if dist < self.hover_threshold:
                hover_idx = idx
                found_hover = True
                break
        self.hovered_match = hover_idx
        if found_hover:
            self.status_label.config(text=f"Hovering over match {hover_idx}")
        else:
            self.status_label.config(text="Click on a region to see zoom details. Hover over a match to see its index.")
        self.draw_scene()

    def on_mouse_click(self, event):
        bx, by = self.canvas_to_base_coords(event.x, event.y)
        
        # Check if click is in main image area
        if by < self.rgb_height:
            # Check if we clicked on a match to select/deselect it
            base_thermal_offset = self.rgb_width
            match_clicked = False
            
            for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
                x2_disp = x2 + base_thermal_offset
                dist = point_line_distance((float(bx), float(by)), (float(x1), float(y1)), (float(x2_disp), float(y2)))
                if dist < self.hover_threshold:
                    if idx in self.selected_matches:
                        self.selected_matches.remove(idx)
                    else:
                        self.selected_matches.add(idx)
                    match_clicked = True
                    break
            
            # If we didn't click on a match, select a region
            if not match_clicked:
                # Determine if click is in RGB or thermal image
                if bx < self.rgb_width:
                    img_type = "rgb"
                    click_x = bx
                else:
                    img_type = "thermal"
                    click_x = bx - self.rgb_width
                
                self.active_region = (click_x, by, img_type)
                self.find_region_matches()
                
            self.draw_scene()
                
    def find_region_matches(self):
        """Find matches that fall within the selected region"""
        if not self.active_region:
            return
            
        click_x, click_y, img_type = self.active_region
        half_size = int(self.cutout_size * self.cutout_extract_factor // 2)
        
        self.region_matches = []
        
        for match in self.matches:
            ((x1, y1), (x2, y2)) = match
            
            if img_type == "rgb":
                # Check if RGB point is in region
                if (abs(float(x1) - click_x) <= half_size and 
                    abs(float(y1) - click_y) <= half_size):
                    self.region_matches.append(match)
            else:  # img_type == "thermal"
                # Check if thermal point is in region
                if (abs(float(x2) - click_x) <= half_size and 
                    abs(float(y2) - click_y) <= half_size):
                    self.region_matches.append(match)
                    
        self.status_label.config(text=f"Found {len(self.region_matches)} matches in selected region")

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