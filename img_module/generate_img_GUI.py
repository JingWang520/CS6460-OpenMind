# frontend.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import base64
from PIL import Image, ImageTk
from io import BytesIO
import threading


class JanusImageGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Janus Image Generator")
        self.root.geometry("600x400")
        self.root.minsize(600, 650)

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Janus Pro 7B Image Generator", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Prompt input
        prompt_frame = ttk.Frame(main_frame)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        prompt_label = ttk.Label(prompt_frame, text="Enter your prompt:", font=("Arial", 12))
        prompt_label.pack(anchor="w", pady=(0, 5))

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=8, wrap=tk.WORD, font=("Arial", 11))
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # Example prompts
        examples_label = ttk.Label(main_frame, text="Example prompts:", font=("Arial", 10, "italic"))
        examples_label.pack(anchor="w", pady=(10, 5))

        examples = [
            "A serene lake at sunset, mountains in background, birds flying, orange sky",
            "Futuristic cityscape, flying cars, neon lights, tall skyscrapers, cloudy weather",
            "Enchanted forest, glowing mushrooms, fairy lights, misty atmosphere, ancient trees"
        ]

        for example in examples:
            example_btn = ttk.Button(main_frame, text=example[:40] + "...",
                                     command=lambda e=example: self.set_example(e))
            example_btn.pack(anchor="w", pady=2, fill=tk.X)

        # Generate button with progress indicator
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)

        self.generate_btn = ttk.Button(button_frame, text="Generate Image", command=self.generate_image)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.progress = ttk.Progressbar(button_frame, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(anchor="w", pady=(5, 0))

    def set_example(self, example):
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(tk.END, example)

    def generate_image(self):
        prompt = self.prompt_text.get(1.0, tk.END).strip()

        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a prompt for image generation.")
            return

        # Disable button and show progress
        self.generate_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Generating image... This may take a minute or two.")

        # Use threading to prevent UI freeze
        threading.Thread(target=self.process_generation, args=(prompt,), daemon=True).start()

    def process_generation(self, prompt):
        try:
            # Send request to the backend
            response = requests.post(
                "http://10.25.10.144:8008/generate-image/",
                json={"prompt": prompt}
            )

            if response.status_code == 200:
                data = response.json()
                image_data = base64.b64decode(data["image"])

                # Display the image in a new window
                self.root.after(0, lambda: self.show_image(image_data, prompt))
                self.root.after(0, lambda: self.status_label.config(text=f"Image generated successfully!"))
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                self.root.after(0, lambda: messagebox.showerror("Generation Failed", error_msg))
                self.root.after(0, lambda: self.status_label.config(text="Generation failed."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status_label.config(text="Error occurred."))

        finally:
            # Re-enable the button and stop progress
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress.stop())

    def show_image(self, image_data, prompt):
        # Create a new window for the image
        img_window = tk.Toplevel(self.root)
        img_window.title("Generated Image")

        # Load the image
        img = Image.open(BytesIO(image_data))

        # Create a frame for the image and info
        frame = ttk.Frame(img_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Show prompt
        prompt_label = ttk.Label(frame, text=f"Prompt: {prompt}", wraplength=500, justify=tk.LEFT)
        prompt_label.pack(pady=(0, 10), anchor="w")

        # Create a PhotoImage object from the PIL Image
        photo = ImageTk.PhotoImage(img)

        # Keep a reference to prevent garbage collection
        img_window.photo = photo

        # Display the image
        img_label = ttk.Label(frame, image=photo)
        img_label.pack(padx=10, pady=10)

        # Add save button
        save_btn = ttk.Button(frame, text="Save Image",
                              command=lambda: self.save_image(img))
        save_btn.pack(pady=10)

    def save_image(self, img):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            img.save(filename)
            messagebox.showinfo("Success", f"Image saved to {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = JanusImageGeneratorGUI(root)
    root.mainloop()
