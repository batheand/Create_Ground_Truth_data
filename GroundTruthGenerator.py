import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

class GroundTruthGenerator:
    """
    Orchestrates the workflow for processing RGB and Thermal image pairs.

    Workflow:
    1. Loads an RGB and Thermal image pair from the data directory.
    2. Extracts keypoints and descriptors from each image using SuperPoint.
    3. Matches keypoints between images using SuperGlue.
    4. Visualizes the matches and provides an interactive UI for user quality control.
       - The user can select/deselect matches (selected matches highlighted in green).
    5. Upon user confirmation, writes the approved matches to a ground truth file in the SuperGlue format.
    6. Waits for a 'Next' command from the user to process the next image pair.

    Note:
    - This class assumes that helper classes/methods for image loading, feature extraction,
      matching, visualization, and file writing exist.
    - For interactive user interface (UI), tools like Google Colab's ipywidgets can be used.
    """

    def __init__(self, data_dir: str, backend_dir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", debug: bool = False):
        """
        Initializes the GroundTruthGenerator with specified directories and instantiates
        helper components.

        Args:
            data_dir (str): Directory containing the image pairs.
            backend_dir (str): Directory where the ground truth files will be saved.
            debug (bool): If True, prints debug information.
        """

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
        self.debug = debug

        if debug:
            print(f"[DEBUG] Initializing GroundTruthGenerator with data_dir: {data_dir} and backend_dir: {backend_dir}")

        # Initialize helper components (assumed to be defined elsewhere)
        self.image_loader = ImageLoader(data_dir)
        self.sp_extractor = SuperPointExtractor()
        self.sg_matcher = SuperGlueMatcher()
        self.file_writer = GroundTruthFileWriter(backend_dir)

        self.current_pair_id = None
    
    def on_matches_selected(self, selected_matches):
        """
        Callback function to handle user-selected matches.

        Args:
            selected_matches (list): List of approved matches after UI selection.
        """
        if self.debug:
            print(f"[DEBUG] User approved {len(selected_matches)} matches.")


        if not hasattr(self, "current_pair_id"):
            print("[ERROR] current_pair_id is not set before saving matches.")
            return
        
        self.file_writer.write_ground_truth_file(self.current_pair_id, selected_matches)


        if self.debug:
            print(f"[DEBUG] Ground truth file written for pair: {self.current_pair_id}")


    def process_image_pair(self, pair_id: str, debug: bool = False):
        """
        Processes a single image pair using the following steps:

        1. Loads the RGB and Thermal images corresponding to the given pair_id.
        2. Extracts keypoints and descriptors for both images using the SuperPoint model.
        3. Computes the matches between the images using the SuperGlue matcher.
        4. Visualizes the matches for user quality control:
           - The UI allows toggling selection of matches (selected matches highlighted in green).
        5. After the user finalizes the selection, writes the approved matches to a ground truth file.
        6. Displays a confirmation message upon successful file creation.

        Args:
            pair_id (str): Unique identifier for the image pair.
            debug (bool): If True, prints debug information during processing.
        """
        if debug:
            print(f"[DEBUG] Starting processing for image pair: {pair_id}")
        
        self.current_pair_id = pair_id  # Ensure this is set before UI interaction
        # Step 1: Load the image pair
        rgb_image, thermal_image = self.image_loader.load_image_pair(pair_id)

        org_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        org_thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)

        grayscale_rgb_image = cv2.cvtColor(org_rgb_image, cv2.COLOR_BGR2GRAY)
        grayscale_thermal_image = cv2.cvtColor(org_thermal_image, cv2.COLOR_BGR2GRAY)

        grayscale_rgb_image = grayscale_rgb_image.astype(np.float32) / 255.0
        grayscale_thermal_image = grayscale_thermal_image.astype(np.float32) / 255.0

        grayscale_rgb_image = torch.from_numpy(grayscale_rgb_image).unsqueeze(0).unsqueeze(0).to(self.device)
        grayscale_thermal_image = torch.from_numpy(grayscale_thermal_image).unsqueeze(0).unsqueeze(0).to(self.device)

        if debug:
            print(f"[DEBUG] Loaded images for pair {pair_id}: RGB image: {org_rgb_image}, Thermal image: {org_thermal_image}")
            print(f"[DEBUG] Loaded grayscale images for pair {pair_id}: RGB image: {grayscale_rgb_image}, Thermal image: {grayscale_thermal_image}")

        # Step 2: Extract keypoints and descriptors from each image
        kpts1, desc1, scores1 = self.sp_extractor.extract_keypoints(grayscale_rgb_image)
        kpts2, desc2, scores2 = self.sp_extractor.extract_keypoints(grayscale_thermal_image)
        if debug:
            print(f"[DEBUG] Extracted keypoints for RGB: {kpts1} and Thermal: {kpts2}")
            print(f"[DEBUG] Extracted descriptors for RGB: {desc1} and Thermal: {desc2}")

        # Ensure keypoints and scores have a batch dimension:
        kpts1 = kpts1.unsqueeze(0)  # from [N,2] to [1,N,2]
        kpts2 = kpts2.unsqueeze(0)
        scores1 = scores1.unsqueeze(0)  # from [N] to [1,N]
        scores2 = scores2.unsqueeze(0)

        # Prepare data dictionary according to the SuperGlue protocol.
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

        # IMPORTANT: Make sure that scores are 2D (shape [B, N]) and NOT already unsqueezed to [B, N, 1].
        # The official SuperGlue protocol expects scores of shape [B, N] so that inside the model they get unsqueezed.
        if data['scores0'].dim() == 3:
            data['scores0'] = data['scores0'].squeeze(-1)
        if data['scores1'].dim() == 3:
            data['scores1'] = data['scores1'].squeeze(-1)

        # Now pass the data dictionary to the SuperGlue matcher.
        matches = self.sg_matcher.match_keypoints(data)

        if debug:
            print(f"[DEBUG] Computed matches: {matches}")

        # Use the new UI:
        # modifications start from here
        # Step 4: Visualize matches and perform UI-based quality control
        visualizer_ui = MatchVisualizerUI(org_rgb_image, org_thermal_image, matches, self.on_matches_selected)
        visualizer_ui.show()

        """
        # Step 4: Visualize matches and perform UI-based quality control
        visualizer = MatchVisualizer(org_rgb_image, org_thermal_image, matches)
        visualizer.display_matches()
        if debug:
            print(f"[DEBUG] Displayed matches. Awaiting user confirmation for match selection...")
        # Placeholder for user interaction: wait until the user finalizes selection.
        input("Press Enter once you have finalized the match selection...")
        selected_matches = visualizer.get_selected_matches()
        if debug:
            print(f"[DEBUG] User selected matches: {selected_matches}")

        # Step 5: Write the ground truth file using the selected matches.
        self.file_writer.write_ground_truth_file(pair_id, selected_matches)
        if debug:
            print(f"[DEBUG] Ground truth file written for pair: {pair_id}")

        # Step 6: Provide UI confirmation.
        visualizer.show_confirmation()
        if debug:
            print(f"[DEBUG] Displayed confirmation to user for pair: {pair_id}")
        """

    def run(self, debug: bool = False):
        """
        Main loop to process all available image pairs.

        This method performs the following:
        - Retrieves a list of image pair identifiers.
        - Iterates over each pair, calling process_image_pair.
        - Waits for the user to signal "Next" (simulated via input) before processing the next pair.

        Args:
            debug (bool): If True, prints debug information during the run.
        """
        if debug:
            print("[DEBUG] Starting main processing loop...")
        pair_ids = self.get_all_pair_ids(debug=debug)
        for pair_id in pair_ids:
            if debug:
                print(f"[DEBUG] Processing next image pair: {pair_id}")
            self.process_image_pair(pair_id, debug=debug)
            #input("Press Enter to load the next image pair...")
        if debug:
            print("[DEBUG] Completed processing all image pairs.")

    def get_all_pair_ids(self, debug: bool = False):
        """
        Retrieves a list of image pair identifiers from the data directory.

        Returns:
            list: A list of pair identifiers (e.g., filenames or unique IDs).

        Note:
            This is a placeholder implementation. In a real-world scenario, this method would
            scan the data directory to find and return valid image pair identifiers.

        Args:
            debug (bool): If True, prints debug information during retrieval.
        """
        # Placeholder: simulate three image pairs.
        pair_ids = ["pair_1", "pair_2", "pair_3"]
        if debug:
            print(f"[DEBUG] Retrieved pair identifiers: {pair_ids}")
        return pair_ids

def point_line_distance(point, line_start, line_end):
    """
    Computes the distance from a point to a line segment defined by line_start and line_end.
    
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

# ---------------------------------
# Dependency 1: ImageLoader
# ---------------------------------
class ImageLoader:
    """
    Loads RGB and Thermal image pairs from a dataset folder with separate subfolders.
    
    Expected file naming convention:
        - RGB images: dataset/rgb/{pair_id}.jpg
        - Thermal images: dataset/thermal/{pair_id}.jpg
    """
    def __init__(self, data_dir: str, debug: bool = False):
        self.data_dir = data_dir  # e.g., "dataset"
        self.debug = debug
        if self.debug:
            print(f"[DEBUG] ImageLoader initialized with directory: {data_dir}")
    
    def load_image_pair(self, pair_id: str, debug: bool = False):
        """
        Loads the RGB and Thermal images based on the pair_id.
        
        Args:
            pair_id (str): Unique identifier for the image pair.
            debug (bool): If True, prints debug information.
            
        Returns:
            tuple: (rgb_image, thermal_image) as numpy arrays.
        """
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
            print(f"[DEBUG] Loaded images shapes: RGB {rgb_image.shape}, Thermal {thermal_image.shape}")
        
        return rgb_image, thermal_image


# ---------------------------------
# Dependency 2: SuperPointExtractor
# ---------------------------------
class SuperPointExtractor:
    """
    Extracts keypoints and descriptors from an image.
    
    Here we use ORB as a placeholder for the SuperPoint model.
    """
    def __init__(self, debug: bool = False):
        self.orb = cv2.ORB_create()
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # SuperPoint Model Configuration
        self.sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }
        self.superpoint = SuperPoint(self.sp_config).eval().to(self.device)
        if self.debug:
            print("[DEBUG] SuperPointExtractor (using ORB) initialized.")
    
    def extract_keypoints(self, image, debug: bool = False):
        """
        Detects keypoints and computes descriptors using ORB.
        
        Args:
            image (numpy.array): Input image.
            debug (bool): If True, prints debug information.
            
        Returns:
            tuple: (keypoints, descriptors)
                   keypoints: list of cv2.KeyPoint objects.
                   descriptors: numpy array of descriptors.
        """
        if image is None:
            raise ValueError(f"Failed to load image.")
        
        # Now, pass it to the SuperPoint model
        with torch.no_grad():
            output = self.superpoint({'image': image})

        keypoints = output['keypoints'][0].cpu()
        descriptors = output['descriptors'][0].cpu()
        scores = output['scores'][0].cpu()
        
        if self.debug:
            print(f"[GroundTruthGenerator] Extracted {len(keypoints)} keypoints.")
            print(f"[GroundTruthGenerator] Computed {descriptors.shape[0]} descriptors.")

        return keypoints, descriptors, scores

# ---------------------------------
# Dependency 3: SuperGlueMatcher
# ---------------------------------
class SuperGlueMatcher:
    """
    Matches keypoints between two images.
    
    Here we use BFMatcher with Hamming norm (suitable for ORB) as a placeholder for SuperGlue.
    """
    def __init__(self, debug: bool = False):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # SuperGlue Model Configuration
        self.sg_config = {'weights': 'indoor'}
        self.superglue = SuperGlue(self.sg_config).eval().to(self.device)
        if self.debug:
            print("[DEBUG] SuperGlueMatcher (using BFMatcher) initialized.")
    
    def match_keypoints(self, data, debug: bool = False):
        """
        Matches keypoints between RGB and Thermal images.
        
        Args:
            rgb_data (tuple): (keypoints_rgb, descriptors_rgb)
            thermal_data (tuple): (keypoints_thermal, descriptors_thermal)
            debug (bool): If True, prints debug information.
            
        Returns:
            list: A list of match tuples: [((x_rgb, y_rgb), (x_thermal, y_thermal)), ...]
        """
        with torch.no_grad():
            # Get raw match indices from SuperGlue; shape [N]
            raw_matches = self.superglue(data)['matches0'][0].cpu().numpy()

        if self.debug or debug:
            num_valid = np.sum(raw_matches > -1)
            print(f"[DEBUG] Found {num_valid} valid matches out of {raw_matches.shape[0]} keypoints.")

        # Retrieve keypoints from the data dictionary.
        keypoints0 = data['keypoints0'][0].cpu().numpy()  # shape [N, 2]
        keypoints1 = data['keypoints1'][0].cpu().numpy()  # shape [M, 2]
        
        # Convert match indices to coordinate pairs.
        match_pairs = []
        for i, match_idx in enumerate(raw_matches):
            if match_idx > -1:
                pt0 = keypoints0[i]
                pt1 = keypoints1[int(match_idx)]
                match_pairs.append((pt0, pt1))
            else:
                # Optionally skip or log unmatched keypoints.
                continue

        if self.debug or debug:
            print(f"[DEBUG] Converted matches to {len(match_pairs)} coordinate pairs.")

        return match_pairs

# ---------------------------------
# Dependency 4: MatchVisualizer
# ---------------------------------
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import tkinter as tk
from tkinter import Canvas, Button, Label
import cv2
from PIL import Image, ImageTk
import numpy as np

def point_line_distance(point, line_start, line_end):
    """
    Computes the perpendicular distance from a point to a line segment.
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

class MatchVisualizerUI:
    def __init__(self, rgb_image, thermal_image, matches, save_callback):
        """
        Args:
            rgb_image (np.array): OpenCV BGR image for RGB.
            thermal_image (np.array): OpenCV BGR image for thermal.
            matches (list): List of tuples: ((x_rgb, y_rgb), (x_thermal, y_thermal)).
            save_callback (function): Function to call when saving approved matches.
        """
        self.rgb_image = rgb_image
        self.thermal_image = thermal_image
        self.matches = matches  # Each match: ((x, y), (x, y))
        self.save_callback = save_callback
        self.selected_matches = set()  # Indices of approved matches
        self.hover_threshold = 5  # pixels
        
        # Create the main window.
        self.root = tk.Tk()
        self.root.title("Match Visualizer")
        
        # Convert images to PhotoImage (keep reference to avoid garbage collection)
        self.rgb_tk = self.convert_to_tk_image(self.rgb_image)
        self.thermal_tk = self.convert_to_tk_image(self.thermal_image)
        
        # Get dimensions and create one canvas for both images side-by-side.
        self.rgb_width = self.rgb_image.shape[1]
        self.rgb_height = self.rgb_image.shape[0]
        self.thermal_width = self.thermal_image.shape[1]
        self.thermal_height = self.thermal_image.shape[0]
        self.canvas_width = self.rgb_width + self.thermal_width
        self.canvas_height = max(self.rgb_height, self.thermal_height)
        
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        
        # Draw the images.
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.rgb_tk)
        self.canvas.create_image(self.rgb_width, 0, anchor=tk.NW, image=self.thermal_tk)
        
        # Draw match lines and store their info.
        self.match_lines = []  # Each entry is a dict: {'id', 'idx', 'pt1', 'pt2'}
        for idx, match in enumerate(self.matches):
            pt_rgb, pt_thermal = match
            x1, y1 = int(pt_rgb[0]), int(pt_rgb[1])
            # Adjust thermal x-coordinate by the width of the RGB image.
            x2, y2 = int(pt_thermal[0]) + self.rgb_width, int(pt_thermal[1])
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="gray", width=2)
            self.match_lines.append({'id': line_id, 'idx': idx, 'pt1': (x1, y1), 'pt2': (x2, y2)})
        
        # Status label for hover information.
        self.status_label = Label(self.root, text="Hover over a match to see its index.")
        self.status_label.pack()
        
        # Buttons: "End Pair" to save and show the "Next" button.
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)
        self.end_button = Button(self.button_frame, text="End Pair", command=self.end_pair)
        self.end_button.pack(side=tk.LEFT, padx=5)
        self.next_button = Button(self.button_frame, text="Next", command=self.next_pair)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.next_button.config(state=tk.DISABLED)  # Initially disabled.
        
        # Bind mouse events.
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
    
    def convert_to_tk_image(self, cv_image):
        """Convert an OpenCV image (BGR) to a Tkinter PhotoImage."""
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_image)
        return ImageTk.PhotoImage(img)
    
    def on_mouse_move(self, event):
        found_hover = False
        # Check all match lines.
        for line in self.match_lines:
            dist = point_line_distance((event.x, event.y), line['pt1'], line['pt2'])
            if dist < self.hover_threshold:
                # If not already approved, change to blue.
                if line['idx'] not in self.selected_matches:
                    self.canvas.itemconfig(line['id'], fill="blue")
                self.status_label.config(text=f"Hovering over match {line['idx']}")
                found_hover = True
            else:
                # If not selected, revert to gray.
                if line['idx'] not in self.selected_matches:
                    self.canvas.itemconfig(line['id'], fill="gray")
        if not found_hover:
            self.status_label.config(text="Hover over a match to see its index.")
    
    def on_mouse_click(self, event):
        # Toggle match selection based on click position.
        for line in self.match_lines:
            dist = point_line_distance((event.x, event.y), line['pt1'], line['pt2'])
            if dist < self.hover_threshold:
                if line['idx'] in self.selected_matches:
                    self.selected_matches.remove(line['idx'])
                    self.canvas.itemconfig(line['id'], fill="gray")
                else:
                    self.selected_matches.add(line['idx'])
                    self.canvas.itemconfig(line['id'], fill="green")
                break
    
    def end_pair(self):
        """Finalize current selection, call the save callback, and enable Next button."""
        # Prepare selected matches.
        selected = [self.matches[i] for i in sorted(self.selected_matches)]
        # Call the callback provided by GroundTruthGenerator.
        self.save_callback(selected)
        # Disable end button and enable next button.
        self.end_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)
    
    def next_pair(self):
        """Close this UI so that the next image pair can be processed."""
        self.root.destroy()
    
    def show(self):
        self.root.mainloop()


class MatchVisualizer:
    """
    Visualizes matches between RGB and Thermal images and allows interactive selection.
    
    Displays the images side by side with match lines. Clicking near a match toggles its selection.
    Selected matches are drawn in green; unselected matches in red.
    """
    def __init__(self, rgb_image, thermal_image, matches, debug: bool = False):
        self.rgb_image = rgb_image
        self.thermal_image = thermal_image
        self.matches = matches  # List of tuples: ((x1, y1), (x2, y2))
        self.selected_indices = set()
        self.fig = None
        self.ax = None
        self.debug = debug
        if self.debug:
            print("[DEBUG] MatchVisualizer initialized with", len(matches), "matches.")
    
    def display_matches(self, debug: bool = False):
        """
        Displays the concatenated RGB and Thermal images with match lines.
        Registers a mouse click event to toggle selection.
        
        Args:
            debug (bool): If True, prints debug information.
        """
        if debug or self.debug:
            print("[DEBUG] Preparing to display matches interactively.")
        # Concatenate images side by side.
        self.height = max(self.rgb_image.shape[0], self.thermal_image.shape[0])
        self.width_rgb = self.rgb_image.shape[1]
        self.width_thermal = self.thermal_image.shape[1]
        
        concatenated = np.zeros((self.height, self.width_rgb + self.width_thermal, 3), dtype=np.uint8)
        concatenated[:self.rgb_image.shape[0], :self.width_rgb] = self.rgb_image
        concatenated[:self.thermal_image.shape[0], self.width_rgb:] = self.thermal_image
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(concatenated)
        self.lines = []  # To store line objects along with their match index.
        # Draw each match line.
        for idx, (pt_rgb, pt_thermal) in enumerate(self.matches):
            pt1 = pt_rgb
            pt2 = (pt_thermal[0] + self.width_rgb, pt_thermal[1])
            line, = self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', picker=5)
            self.lines.append((line, idx))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        if debug or self.debug:
            print("[DEBUG] Displaying figure with match lines. Click near a line to toggle selection.")
        plt.show(block=False)
    
    def _on_click(self, event):
        """
        Internal method to handle click events on the figure.
        If the click is close to any match line, toggle its selection.
        """
        if self.debug:
            print("[DEBUG] Mouse click event at:", (event.xdata, event.ydata))
        if event.xdata is None or event.ydata is None:
            return  # Clicked outside the axes
        threshold = 5.0  # pixels
        for line, idx in self.lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            pt1 = (xdata[0], ydata[0])
            pt2 = (xdata[1], ydata[1])
            dist = point_line_distance((event.xdata, event.ydata), pt1, pt2)
            if self.debug:
                print(f"[DEBUG] Distance to match {idx}: {dist}")
            if dist < threshold:
                self.toggle_match_selection(idx, debug=self.debug)
                self.fig.canvas.draw_idle()
                break

    def toggle_match_selection(self, match_index, debug: bool = False):
        """
        Toggles the selection state of a match.
        
        Args:
            match_index (int): The index of the match to toggle.
            debug (bool): If True, prints debug information.
        """
        if match_index in self.selected_indices:
            self.selected_indices.remove(match_index)
            if debug or self.debug:
                print(f"[DEBUG] Deselected match index: {match_index}")
            color = 'r'
        else:
            self.selected_indices.add(match_index)
            if debug or self.debug:
                print(f"[DEBUG] Selected match index: {match_index}")
            color = 'g'
        
        # Update line color.
        for line, idx in self.lines:
            if idx == match_index:
                line.set_color(color)
                break

    def get_selected_matches(self, debug: bool = False):
        """
        Returns the list of matches that have been selected by the user.
        
        Args:
            debug (bool): If True, prints debug information.
            
        Returns:
            list: List of match tuples that are selected.
        """
        if debug or self.debug:
            print("[DEBUG] Returning selected matches:", self.selected_indices)
        selected = [self.matches[idx] for idx in sorted(self.selected_indices)]
        return selected

    def show_confirmation(self, debug: bool = False):
        """
        Displays a confirmation message on the visualization.
        
        Args:
            debug (bool): If True, prints debug information.
        """
        if debug or self.debug:
            print("[DEBUG] Showing confirmation message on the figure.")
        self.ax.text(0.5, 0.95, 'Ground Truth File Created',
                     transform=self.ax.transAxes, fontsize=16,
                     color='blue', ha='center')
        self.fig.canvas.draw_idle()
        # Keep the figure open briefly.
        plt.pause(2)
        plt.close(self.fig)

# ---------------------------------
# Dependency 5: GroundTruthFileWriter
# ---------------------------------
class GroundTruthFileWriter:
    """
    Writes the approved (selected) matches to a ground truth file.
    
    The file is written in a simple text format where each line corresponds to a match:
        x_rgb y_rgb x_thermal y_thermal
    """
    def __init__(self, backend_dir: str, debug: bool = False):
        self.backend_dir = backend_dir
        self.debug = debug
        if self.debug:
            print(f"[DEBUG] GroundTruthFileWriter initialized with backend directory: {backend_dir}")
    
    def write_ground_truth_file(self, pair_id: str, selected_matches, debug: bool = False):
        """
        Writes the selected matches to a file named {pair_id}_ground_truth.txt in the backend directory.
        
        Args:
            pair_id (str): Unique identifier for the image pair.
            selected_matches (list): List of match tuples: [((x_rgb, y_rgb), (x_thermal, y_thermal)), ...]
            debug (bool): If True, prints debug information.
        """
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

