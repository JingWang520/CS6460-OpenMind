import tkinter as tk
from tkinter import ttk, scrolledtext
import sounddevice as sd
import numpy as np
from transformers import pipeline
import threading
import queue


class VoiceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Input System")

        # Initialize recording parameters
        self.sample_rate = 16000  # Wav2Vec2 standard sample rate
        self.is_recording = False
        self.audio_data = []

        # Create UI components
        self.create_widgets()

        # Initialize speech recognition model
        self.model_queue = queue.Queue()
        self.init_model()

        # Initialize thread communication queue
        self.result_queue = queue.Queue()

        # Check queue periodically
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

        # Voice button
        self.voice_button = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self.toggle_recording,
            width=15
        )
        self.voice_button.pack(side=tk.LEFT, padx=5)

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

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.voice_button.config(text="Stop Recording")

        # Start recording
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop()
        self.voice_button.config(text="Start Recording")

        # Start recognition thread
        threading.Thread(target=self.process_audio, daemon=True).start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("Audio input error:", status)
        self.audio_data.append(indata.copy())

    def process_audio(self):
        # Wait for model to load
        if self.model_queue.empty():
            self.result_queue.put("Model is loading, please wait...")
            return

        # Merge audio data
        audio = np.concatenate(self.audio_data, axis=0)
        audio = audio.squeeze().astype(np.float32)

        # Perform speech recognition
        try:
            result = self.model(audio)
            self.result_queue.put(result["text"].lower())
        except Exception as e:
            self.result_queue.put(f"Recognition error: {str(e)}")

    def check_queue(self):
        while not self.result_queue.empty():
            text = self.result_queue.get()
            self.text_input.insert(tk.END, text + "\n")
            self.text_input.see(tk.END)  # Scroll to the bottom
        self.root.after(100, self.check_queue)

    def send_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            print("Sending text:", text)
            # Add sending logic here
            self.text_input.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    app = VoiceRecognitionApp(root)
    root.mainloop()
