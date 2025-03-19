import tkinter as tk
from tkinter import Canvas, Button, Label
import cv2
from PIL import Image, ImageTk
import numpy as np

##############################
# Utility: Point-Line Distance
##############################
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

########################################
# The 1:1 MatchVisualizerUI from your code
########################################
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
        self.hover_threshold = 5  # Distance threshold in pixels for hover

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Match Visualizer")

        # Convert images to PhotoImage (and keep references to avoid GC)
        self.rgb_tk = self.convert_to_tk_image(self.rgb_image)
        self.thermal_tk = self.convert_to_tk_image(self.thermal_image)

        # Dimensions for side-by-side
        self.rgb_width = self.rgb_image.shape[1]
        self.rgb_height = self.rgb_image.shape[0]
        self.thermal_width = self.thermal_image.shape[1]
        self.thermal_height = self.thermal_image.shape[0]
        self.canvas_width = self.rgb_width + self.thermal_width
        self.canvas_height = max(self.rgb_height, self.thermal_height)

        # Create a Tkinter Canvas for drawing
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Draw the two images side by side
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.rgb_tk)
        self.canvas.create_image(self.rgb_width, 0, anchor=tk.NW, image=self.thermal_tk)

        # Draw match lines in gray by default
        self.match_lines = []  # Will store dicts with {'id', 'idx', 'pt1', 'pt2'}
        for idx, match in enumerate(self.matches):
            pt_rgb, pt_thermal = match
            x1, y1 = int(pt_rgb[0]), int(pt_rgb[1])
            # Offset the thermal x-coordinate by the width of the RGB image
            x2, y2 = int(pt_thermal[0]) + self.rgb_width, int(pt_thermal[1])
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="gray", width=2)
            self.match_lines.append({
                'id': line_id,
                'idx': idx,
                'pt1': (x1, y1),
                'pt2': (x2, y2)
            })

        # Status label
        self.status_label = Label(self.root, text="Hover over a match to see its index.")
        self.status_label.pack()

        # Buttons: End Pair & Next
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.end_button = Button(self.button_frame, text="End Pair", command=self.end_pair)
        self.end_button.pack(side=tk.LEFT, padx=5)

        self.next_button = Button(self.button_frame, text="Next", command=self.next_pair)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.next_button.config(state=tk.DISABLED)  # Initially disabled

        # Bind mouse events
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)

    def convert_to_tk_image(self, cv_image):
        """Convert an OpenCV BGR image to a Tkinter PhotoImage."""
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_image)
        return ImageTk.PhotoImage(pil_img)

    def on_mouse_move(self, event):
        """Highlight a line in blue when hovering, if not already selected."""
        found_hover = False
        for line in self.match_lines:
            dist = point_line_distance((event.x, event.y), line['pt1'], line['pt2'])
            if dist < self.hover_threshold:
                if line['idx'] not in self.selected_matches:
                    self.canvas.itemconfig(line['id'], fill="blue")
                self.status_label.config(text=f"Hovering over match {line['idx']}")
                found_hover = True
            else:
                # If not selected, revert to gray
                if line['idx'] not in self.selected_matches:
                    self.canvas.itemconfig(line['id'], fill="gray")
        if not found_hover:
            self.status_label.config(text="Hover over a match to see its index.")

    def on_mouse_click(self, event):
        """Toggle match selection (green vs. gray) on click."""
        for line in self.match_lines:
            dist = point_line_distance((event.x, event.y), line['pt1'], line['pt2'])
            if dist < self.hover_threshold:
                idx = line['idx']
                if idx in self.selected_matches:
                    self.selected_matches.remove(idx)
                    self.canvas.itemconfig(line['id'], fill="gray")
                else:
                    self.selected_matches.add(idx)
                    self.canvas.itemconfig(line['id'], fill="green")
                break

    def end_pair(self):
        """Finalize current selection and call the save callback."""
        selected = [self.matches[i] for i in sorted(self.selected_matches)]
        self.save_callback(selected)
        # Disable End button, enable Next
        self.end_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)

    def next_pair(self):
        """Close the UI window, simulating the move to the next pair."""
        self.root.destroy()

    def show(self):
        """Run the Tkinter main loop."""
        self.root.mainloop()

############################################
# Example usage with dummy images & matches
############################################
def dummy_save_callback(selected_matches):
    print("=== User clicked End Pair! ===")
    print("User approved these matches:")
    for i, m in enumerate(selected_matches):
        print(f"{i}: {m}")

def main():
    # Dummy image paths (change these to real paths if desired)
    rgb_path = "test_data/rgb/IM_00006.jpg"
    thermal_path = "test_data/thermal/IM_00006.jpg"

    # Load images as BGR
    img_rgb = cv2.imread(rgb_path)
    img_thermal = cv2.imread(thermal_path)

    if img_rgb is None or img_thermal is None:
        raise FileNotFoundError("Could not load dummy images. Check your file paths.")

    # Resize to 640x480 to match your pipeline
    img_rgb = cv2.resize(img_rgb, (640, 480))
    img_thermal = cv2.resize(img_thermal, (640, 480))

    # Define a few dummy matches: ((x_rgb, y_rgb), (x_thermal, y_thermal))
    matches = [
        ((100, 150), (110, 160)),
        ((200, 100), (210, 110)),
        ((300, 250), (310, 260)),
        ((400, 300), (410, 320)),
    ]

    # Create the UI with the same look & feel
    ui = MatchVisualizerUI(img_rgb, img_thermal, matches, dummy_save_callback)
    ui.show()

if __name__ == "__main__":
    main()