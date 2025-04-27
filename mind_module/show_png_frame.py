import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import platform

class ZoomPanImageViewer(tk.Frame):
    """
    A Tkinter Frame component that supports zooming and panning of images.
    """
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # --- Widget Setup ---
        # Changed bg="gray" to bg="white"
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0) # <--- Modified here
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Image Properties ---
        self.original_image = None
        self.tk_image = None
        self.image_id = None
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # --- Drag Properties ---
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._is_dragging = False

        # --- Event Bindings ---
        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_button_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)

        if platform.system() == "Windows" or platform.system() == "Darwin":
             self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        else:
             self.canvas.bind("<Button-4>", self._on_mouse_wheel) # Linux scroll up
             self.canvas.bind("<Button-5>", self._on_mouse_wheel) # Linux scroll down

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.focus_set()

    # ... (Other methods like load_image, _clear_image, _display_image, etc. remain unchanged) ...
    def load_image(self, filepath):
        """Load the image from the specified path"""
        try:
            self.original_image = Image.open(filepath)
            print(f"Image loaded successfully: {filepath}, Size: {self.original_image.size}") # Image loaded successfully, Size
            self.zoom_level = 1.0 # Reset zoom level
            self._display_image() # Display the image
        except FileNotFoundError:
            print(f"Error: File not found {filepath}") # Error: File not found
            self._clear_image()
        except Exception as e:
            print(f"Error loading image: {e}") # Error loading image
            self._clear_image()

    def _clear_image(self):
        """Clear the currently displayed image"""
        if self.image_id:
            self.canvas.delete(self.image_id)
        self.original_image = None
        self.tk_image = None
        self.image_id = None
        self.zoom_level = 1.0

    def _display_image(self):
        """Display the image based on the current zoom level"""
        if not self.original_image:
            return

        if self.image_id:
            self.canvas.delete(self.image_id)

        width = max(1, int(self.original_image.width * self.zoom_level))
        height = max(1, int(self.original_image.height * self.zoom_level))

        try:
            # Use LANCZOS for resizing, good quality
            resized_img = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_img)
            self.image_id = self.canvas.create_image(0, 0, anchor=tk.CENTER, image=self.tk_image)
            self.canvas.image = self.tk_image # Keep reference
            self._center_image()
        except Exception as e:
            print(f"Error resizing or displaying image: {e}") # Error resizing or displaying image
            self.image_id = None
            self.tk_image = None

    def _center_image(self):
        """Center the image on the Canvas"""
        if not self.image_id:
            return
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            self.canvas.coords(self.image_id, canvas_width / 2, canvas_height / 2)

    def _on_configure(self, event):
        """Recenter the image when the Canvas is resized"""
        self._center_image()

    def _on_button_press(self, event):
        """Handle mouse left button press event (start dragging)"""
        if not self.image_id: return
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._is_dragging = True
        self.canvas.config(cursor="fleur") # Change cursor to indicate dragging

    def _on_button_motion(self, event):
        """Handle mouse motion event (dragging)"""
        if not self.image_id or not self._is_dragging: return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.canvas.move(self.image_id, dx, dy)
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_button_release(self, event):
        """Handle mouse left button release event (end dragging)"""
        if not self.image_id: return
        self._is_dragging = False
        self.canvas.config(cursor="") # Restore default cursor

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel event (zooming)"""
        if not self.image_id: return

        factor = 0.0
        if platform.system() == "Windows" or platform.system() == "Darwin":
            # event.delta is typically 120 or -120 on Windows/macOS
            factor = 1.1 if event.delta > 0 else 0.9
        else: # Linux (Button-4 is scroll up, Button-5 is scroll down)
            if event.num == 4: factor = 1.1
            elif event.num == 5: factor = 0.9

        if factor == 0.0: return # No zoom action detected

        new_zoom_level = self.zoom_level * factor
        # Clamp zoom level within min/max bounds
        new_zoom_level = max(self.min_zoom, min(self.max_zoom, new_zoom_level))

        if new_zoom_level == self.zoom_level: return # No effective change

        actual_factor = new_zoom_level / self.zoom_level
        self.zoom_level = new_zoom_level

        # Calculate new image position based on mouse position
        mouse_x, mouse_y = event.x, event.y
        img_x, img_y = self.canvas.coords(self.image_id)

        # Calculate the offset of the mouse from the image center
        offset_x = mouse_x - img_x
        offset_y = mouse_y - img_y

        # Scale the offset
        new_offset_x = offset_x * actual_factor
        new_offset_y = offset_y * actual_factor

        # Calculate the new image center position
        new_img_x = mouse_x - new_offset_x
        new_img_y = mouse_y - new_offset_y

        # Calculate new image dimensions
        new_width = max(1, int(self.original_image.width * self.zoom_level))
        new_height = max(1, int(self.original_image.height * self.zoom_level))

        try:
            # Resize the original image
            resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Create a new PhotoImage
            self.tk_image = ImageTk.PhotoImage(resized_img)
        except Exception as e:
            print(f"Error resizing image during zoom: {e}") # Error resizing image during zoom
            self.zoom_level /= actual_factor # Revert zoom level if resize fails
            return

        # Update the image on the canvas and set its new position
        self.canvas.itemconfig(self.image_id, image=self.tk_image)
        self.canvas.coords(self.image_id, new_img_x, new_img_y)
        self.canvas.image = self.tk_image # Keep reference


# --- Example Usage (Frame background color also removed for clarity) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Viewer (Zoom/Pan - White Background)") # Image Viewer (Zoom/Pan - White Background)
    root.geometry("800x600")

    # Create image viewer instance (removed Frame bg color)
    image_viewer = ZoomPanImageViewer(root) # <--- Removed bg="dark slate gray"
    image_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def open_image_file():
        filepath = filedialog.askopenfilename(
            title="Select PNG Image", # Select PNG Image
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg;*.jpeg"), ("All Files", "*.*")] # PNG Files, JPEG Files, All Files
        )
        if filepath:
            image_viewer.load_image(filepath)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)
    open_button = ttk.Button(button_frame, text="Open Image", command=open_image_file)
    open_button.pack()

    # <--- Change this to the path of the image you want to test
    default_image_path = "test_image.png"
    try:
        image_viewer.load_image(default_image_path)
    except Exception as e:
         print(f"Could not load default image '{default_image_path}': {e}") # Could not load default image
         print("Please click the 'Open Image' button to select a file.") #
