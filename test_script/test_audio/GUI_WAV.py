import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import numpy as np
from transformers import pipeline
import threading
import queue
import soundfile as sf  # Install soundfile library: pip install soundfile


class VoiceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Input System")

        # Initialize parameters
        self.sample_rate = 16000  # Target sample rate
        self.audio_data = []

        # Create UI components
        self.create_widgets()

        # Initialize speech recognition model
        self.model_queue = queue.Queue()
        self.init_model()

        # Initialize result queue
        self.result_queue = queue.Queue()
        self.root.after(100, self.check_queue)

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Text input box
        self.text_input = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=15)
        self.text_input.pack(pady=5, fill=tk.BOTH, expand=True)

        # Button container
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # File selection button
        self.file_button = ttk.Button(
            button_frame,
            text="Select WAV File",
            command=self.load_audio_file,
            width=15
        )
        self.file_button.pack(side=tk.LEFT, padx=5)

        # Send button
        self.send_button = ttk.Button(
            button_frame,
            text="Send",
            command=self.send_text,
            width=15
        )
        self.send_button.pack(side=tk.RIGHT, padx=5)

    def init_model(self):
        def load_model():
            self.model = pipeline(
                "automatic-speech-recognition",
                model="facebook/wav2vec2-base-960h",
                device="cpu"
            )
            self.model_queue.put(True)

        threading.Thread(target=load_model, daemon=True).start()

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav")]
        )
        if not file_path:
            return

        try:
            # Read audio file
            data, sr = sf.read(file_path)

            # Check sample rate
            if sr != self.sample_rate:
                self.result_queue.put(f"Error: Sample rate {sr}Hz does not meet the requirement (16000Hz needed)")
                return

            # Convert audio format
            if len(data.shape) > 1:  # If multi-channel, take the average
                data = data.mean(axis=1)
            self.audio_data = [data.astype(np.float32)]

            # Start processing thread
            threading.Thread(target=self.process_audio, daemon=True).start()

        except Exception as e:
            self.result_queue.put(f"Failed to read file: {str(e)}")

    def process_audio(self):
        if self.model_queue.empty():
            self.result_queue.put("Model is loading, please wait...")
            return

        try:
            audio = np.concatenate(self.audio_data, axis=0)
            result = self.model(audio)
            self.result_queue.put(result["text"].lower())
        except Exception as e:
            self.result_queue.put(f"Recognition error: {str(e)}")

    def check_queue(self):
        while not self.result_queue.empty():
            text = self.result_queue.get()
            self.text_input.insert(tk.END, text + "\n")
            self.text_input.see(tk.END)
        self.root.after(100, self.check_queue)

    def send_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            print("Sending text:", text)
            self.text_input.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    app = VoiceRecognitionApp(root)
    root.mainloop()
