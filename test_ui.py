import tkinter as tk
from tkinter import Canvas, Button, Label
import cv2
from PIL import Image, ImageTk
import numpy as np
import random

def point_line_distance(point, line_start, line_end):
    """
    Computes the perpendicular distance from a point to a line segment.
    """
    px, py = point
    ax, ay = line_start
    bx, by = line_end
    if (ax == bx) and (ay == by):
        return np.hypot(px - ax, py - ay)
    line_len_sq = (bx - ax)**2 + (by - ay)**2
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / line_len_sq
    t = max(0, min(1, t))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    return np.hypot(px - proj_x, py - proj_y)

def generate_less_spaghetti_matches():
    """
    Generates a 10x10 grid of match points in the 640x480 space.
    For each (x, y) on the RGB side, we add a small random offset
    to get (x2, y2) for the thermal side.
    Produces exactly 100 matches.
    """
    matches = []
    rows = 10
    cols = 10

    # Define a region for the grid, e.g., from (50,50) to (600,400)
    x_start, x_end = 50, 600
    y_start, y_end = 50, 400
    x_step = (x_end - x_start) / (cols - 1)
    y_step = (y_end - y_start) / (rows - 1)

    for i in range(rows):
        for j in range(cols):
            x = int(x_start + j * x_step)
            y = int(y_start + i * y_step)
            # Thermal side has a small random offset of ±10 pixels
            x2 = x + random.randint(-10, 10)
            y2 = y + random.randint(-10, 10)
            matches.append(((x, y), (x2, y2)))
    return matches

class MatchVisualizerUI:
    """
    A re-rendering approach for zoom/pan. We keep the original images
    and match coordinates, and re-draw everything whenever we zoom or pan.
    """

    def __init__(self, rgb_path, thermal_path, matches, save_callback):
        """
        Args:
            rgb_path (str): Path to the RGB image.
            thermal_path (str): Path to the Thermal image.
            matches (list): List of tuples: [((x_rgb, y_rgb), (x_thermal, y_thermal)), ...]
            save_callback (function): Callback when the user ends the pair.
        """
        self.rgb_path = rgb_path
        self.thermal_path = thermal_path
        self.matches = matches
        self.save_callback = save_callback

        # Load images (BGR) and resize to base size (640x480).
        self.rgb_image = cv2.imread(self.rgb_path)
        self.thermal_image = cv2.imread(self.thermal_path)
        if self.rgb_image is None or self.thermal_image is None:
            raise FileNotFoundError("Could not load images. Check your paths.")
        self.rgb_image = cv2.resize(self.rgb_image, (640, 480))
        self.thermal_image = cv2.resize(self.thermal_image, (640, 480))

        # Selected match indices.
        self.selected_matches = set()

        # Zoom and pan state.
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start = None
        self.hover_threshold = 5

        # Initialize Tkinter UI.
        self.root = tk.Tk()
        self.root.title("Match Visualizer (100 Less-Spaghetti Matches)")

        # Base dimensions for side-by-side images.
        self.base_width = self.rgb_image.shape[1] + self.thermal_image.shape[1]
        self.base_height = max(self.rgb_image.shape[0], self.thermal_image.shape[0])
        self.canvas = Canvas(self.root, width=self.base_width, height=self.base_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status label.
        self.status_label = Label(self.root, text="Hover over a match to see its index.")
        self.status_label.pack()

        # Buttons.
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)
        self.end_button = Button(self.button_frame, text="End Pair", command=self.end_pair)
        self.end_button.pack(side=tk.LEFT, padx=5)
        self.next_button = Button(self.button_frame, text="Next", command=self.next_pair)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.next_button.config(state=tk.DISABLED)

        # Bind events.
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        self.root.bind("<KeyPress-plus>", self.on_key_zoom_in)
        self.root.bind("<KeyPress-equal>", self.on_key_zoom_in)
        self.root.bind("<KeyPress-minus>", self.on_key_zoom_out)
        # Arrow keys for panning.
        self.root.bind("<Left>", self.on_arrow_left)
        self.root.bind("<Right>", self.on_arrow_right)
        self.root.bind("<Up>", self.on_arrow_up)
        self.root.bind("<Down>", self.on_arrow_down)
        # Middle mouse for panning.
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)

        # Draw the initial scene.
        self.draw_scene()

    def make_composite_image(self, rgb_bgr, thermal_bgr):
        """
        Create a side-by-side composite image (BGR) from the two images.
        """
        h = max(rgb_bgr.shape[0], thermal_bgr.shape[0])
        w_rgb = rgb_bgr.shape[1]
        w_thermal = thermal_bgr.shape[1]
        composite = np.zeros((h, w_rgb + w_thermal, 3), dtype=np.uint8)
        composite[:rgb_bgr.shape[0], :w_rgb] = rgb_bgr
        composite[:thermal_bgr.shape[0], w_rgb:] = thermal_bgr
        return composite

    def draw_scene(self):
        """
        Clear the canvas and re-draw the composite image and match lines
        using the current scale and offset.
        """
        self.canvas.delete("all")
        composite = self.make_composite_image(self.rgb_image, self.thermal_image)
        scaled_w = int(composite.shape[1] * self.scale)
        scaled_h = int(composite.shape[0] * self.scale)
        if scaled_w < 1: scaled_w = 1
        if scaled_h < 1: scaled_h = 1
        composite_scaled = cv2.resize(composite, (scaled_w, scaled_h))
        pil_img = Image.fromarray(cv2.cvtColor(composite_scaled, cv2.COLOR_BGR2RGB))
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        # Draw match lines.
        base_thermal_offset = 640  # offset for second image
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
            x2_display = x2 + base_thermal_offset
            sx1 = x1 * self.scale + self.offset_x
            sy1 = y1 * self.scale + self.offset_y
            sx2 = x2_display * self.scale + self.offset_x
            sy2 = y2 * self.scale + self.offset_y
            color = "green" if idx in self.selected_matches else "gray"
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=2, tags="match_line")

    def canvas_to_base_coords(self, cx, cy):
        """
        Convert canvas coordinates back to base coordinates.
        """
        x_no_offset = cx - self.offset_x
        y_no_offset = cy - self.offset_y
        if self.scale != 0:
            bx = x_no_offset / self.scale
            by = y_no_offset / self.scale
        else:
            bx, by = x_no_offset, y_no_offset
        return bx, by

    def on_mouse_move(self, event):
        """
        Update the status label if hovering near a match line.
        """
        base_x, base_y = self.canvas_to_base_coords(event.x, event.y)
        found_hover = False
        base_thermal_offset = 640
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
            x2_display = x2 + base_thermal_offset
            dist = point_line_distance((base_x, base_y), (x1, y1), (x2_display, y2))
            if dist < self.hover_threshold:
                self.status_label.config(text=f"Hovering over match {idx}")
                found_hover = True
                break
        if not found_hover:
            self.status_label.config(text="Hover over a match to see its index.")

    def on_mouse_click(self, event):
        """
        Toggle match selection if clicking near a match line.
        """
        base_x, base_y = self.canvas_to_base_coords(event.x, event.y)
        base_thermal_offset = 640
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.matches):
            x2_display = x2 + base_thermal_offset
            dist = point_line_distance((base_x, base_y), (x1, y1), (x2_display, y2))
            if dist < self.hover_threshold:
                if idx in self.selected_matches:
                    self.selected_matches.remove(idx)
                else:
                    self.selected_matches.add(idx)
                self.draw_scene()
                break

    def on_mousewheel(self, event):
        """
        Zoom in/out around the mouse cursor.
        """
        if event.num == 5 or event.delta < 0:
            factor = 0.9
        elif event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 1.0

        base_x, base_y = self.canvas_to_base_coords(event.x, event.y)
        self.scale *= factor
        self.scale = max(0.1, min(10.0, self.scale))
        self.offset_x = event.x - base_x * self.scale
        self.offset_y = event.y - base_y * self.scale
        self.draw_scene()

    def on_key_zoom_in(self, event):
        """
        Zoom in around the canvas center using the '+' key.
        """
        factor = 1.1
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        base_cx, base_cy = self.canvas_to_base_coords(center_x, center_y)
        self.scale *= factor
        self.scale = min(10.0, self.scale)
        self.offset_x = center_x - base_cx * self.scale
        self.offset_y = center_y - base_cy * self.scale
        self.draw_scene()

    def on_key_zoom_out(self, event):
        """
        Zoom out around the canvas center using the '-' key.
        """
        factor = 0.9
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        base_cx, base_cy = self.canvas_to_base_coords(center_x, center_y)
        self.scale *= factor
        self.scale = max(0.1, self.scale)
        self.offset_x = center_x - base_cx * self.scale
        self.offset_y = center_y - base_cy * self.scale
        self.draw_scene()

    def on_arrow_left(self, event):
        """Pan view to the left."""
        self.offset_x += 20
        self.draw_scene()

    def on_arrow_right(self, event):
        """Pan view to the right."""
        self.offset_x -= 20
        self.draw_scene()

    def on_arrow_up(self, event):
        """Pan view upward."""
        self.offset_y += 20
        self.draw_scene()

    def on_arrow_down(self, event):
        """Pan view downward."""
        self.offset_y -= 20
        self.draw_scene()

    def on_pan_start(self, event):
        """Record starting position for panning."""
        self.pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        """Update offset based on middle mouse drag."""
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.draw_scene()

    def end_pair(self):
        """
        Finalize selection and call the save callback.
        """
        selected = [self.matches[i] for i in sorted(self.selected_matches)]
        self.save_callback(selected)
        self.end_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)

    def next_pair(self):
        """Close the UI window to proceed to the next pair."""
        self.root.destroy()

    def show(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

def dummy_save_callback(selected_matches):
    print("=== End Pair Clicked ===")
    print("User approved these matches:")
    for i, m in enumerate(selected_matches):
        print(f"{i}: {m}")

def main():
    # Paths to your images
    rgb_path = "test_data/rgb/IM_00006.jpg"
    thermal_path = "test_data/thermal/IM_00006.jpg"

    # Generate a 100-match grid with small offsets
    matches_100 = generate_less_spaghetti_matches()

    ui = MatchVisualizerUI(rgb_path, thermal_path, matches_100, dummy_save_callback)
    ui.show()

if __name__ == "__main__":
    main()
