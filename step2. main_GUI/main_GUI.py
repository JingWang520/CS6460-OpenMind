import os
import time
import threading
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import customtkinter as ctk
from PIL import Image, ImageTk
import requests
import tkinter as tk
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set theme and appearance
ctk.set_appearance_mode("System")  # System theme mode
ctk.set_default_color_theme("blue")  # Blue theme


class VoiceChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Voice Chat Assistant")
        self.geometry("800x600")
        self.minsize(600, 500)

        # Store image references
        self.image_references = []

        # Load speech recognition model
        self.load_speech_recognition_model()

        # Application state variables
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.current_user = "User"  # Default username

        # Create UI
        self.create_ui()

    def load_speech_recognition_model(self):
        print("Loading speech recognition model...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
        print("Speech recognition model loaded")

    def create_ui(self):
        # Create main layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create title bar
        self.title_frame = ctk.CTkFrame(self, height=40, fg_color=["#F0F0F0", "#333333"])
        self.title_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.title_frame.grid_propagate(False)
        self.title_frame.grid_columnconfigure(0, weight=1)

        # Add application title
        self.title_label = ctk.CTkLabel(
            self.title_frame,
            text="AI Voice Assistant",
            font=("Helvetica", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Add clear conversation button
        self.clear_button = ctk.CTkButton(
            self.title_frame,
            text="Clear Chat",
            width=100,
            height=30,
            command=self.clear_chat,
            fg_color=["#E74C3C", "#C0392B"],
            hover_color=["#C0392B", "#A93226"]
        )
        self.clear_button.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="e")

        # Create chat area frame
        self.chat_frame = ctk.CTkFrame(self)
        self.chat_frame.grid(row=1, column=0, padx=10, pady=(5, 5), sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        # Create chat display area
        self.chat_display = ctk.CTkTextbox(self.chat_frame, font=("Helvetica", 12))
        self.chat_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.chat_display.configure(state="disabled")

        # Get underlying tkinter Text widget to use tag_configure
        self._textbox = self.chat_display._textbox

        # Configure tag styles
        self._textbox.tag_configure("timestamp", foreground="#888888")
        self._textbox.tag_configure("user", foreground="#3498DB", font=("Helvetica", 12, "bold"))
        self._textbox.tag_configure("user_message", foreground="#000000")
        self._textbox.tag_configure("bot", foreground="#E74C3C", font=("Helvetica", 12, "bold"))
        self._textbox.tag_configure("bot_message", foreground="#333333")
        self._textbox.tag_configure("typing", foreground="#888888", font=("Helvetica", 11, "italic"))
        self._textbox.tag_configure("system", foreground="#7D3C98", font=("Helvetica", 11, "italic"))

        # Create input area frame
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Create message input box
        self.message_input = ctk.CTkTextbox(self.input_frame, height=60, font=("Helvetica", 12))
        self.message_input.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        # Create button frame
        self.button_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.button_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")

        # Create voice button
        self.voice_button = ctk.CTkButton(
            self.button_frame,
            text="🎤",
            width=40,
            height=30,
            font=("Helvetica", 16),
            command=self.toggle_recording
        )
        self.voice_button.grid(row=0, column=0, padx=(0, 5), pady=(0, 5))

        # Create send button
        self.send_button = ctk.CTkButton(
            self.button_frame,
            text="Send",
            width=60,
            height=30,
            command=self.send_message
        )
        self.send_button.grid(row=1, column=0, padx=0, pady=(0, 0))

        # Configure layout weights
        self.grid_rowconfigure(0, weight=0)  # Title bar fixed height
        self.grid_rowconfigure(1, weight=1)  # Chat area stretchable
        self.grid_rowconfigure(2, weight=0)  # Input area fixed height

        # Bind Enter key to send message
        self.message_input.bind("<Return>", lambda event: self.send_message())

        # Initial welcome message
        self.after(100, lambda: self.add_bot_message("Hello! I'm your AI assistant. How can I help you today?"))

    def clear_chat(self):
        """Clear all conversation content"""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("0.0", "end")
        self.chat_display.configure(state="disabled")
        self.image_references.clear()  # Clear image references

        # Add system message
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "System: Conversation cleared\n", "system")
        self.chat_display.configure(state="disabled")

        # Add welcome message
        self.add_bot_message("How can I help you today?")

    def toggle_recording(self):
        if not self.recording:
            # Start recording
            self.recording = True
            self.voice_button.configure(text="⏹️", fg_color="#E74C3C")
            self.audio_data = []

            # Start recording in a new thread
            threading.Thread(target=self.record_audio, daemon=True).start()
        else:
            # Stop recording
            self.recording = False
            self.voice_button.configure(text="🎤", fg_color=["#3B8ED0", "#1F6AA5"])

            # Process recording
            self.process_audio()

    def record_audio(self):
        """Function to record audio"""

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.append(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)

    def process_audio(self):
        """Process recorded audio"""
        if not self.audio_data:
            return

        # Convert to numpy array
        audio = np.concatenate(self.audio_data, axis=0)
        audio = audio.flatten()

        # Save as WAV file
        temp_file = "temp_recording.wav"
        wav.write(temp_file, self.sample_rate, audio)

        # Show processing message
        self.message_input.delete("0.0", "end")
        self.message_input.insert("0.0", "Recognizing speech...")

        # Perform speech recognition in a new thread
        threading.Thread(target=self.recognize_speech, args=(audio,), daemon=True).start()

    def recognize_speech(self, audio):
        """Use Wav2Vec2 model for speech recognition"""
        try:
            # Preprocess audio
            input_values = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_values

            # Get logits
            with torch.no_grad():
                logits = self.model(input_values).logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            # Update input box
            self.message_input.delete("0.0", "end")
            self.message_input.insert("0.0", transcription)

        except Exception as e:
            print(f"Speech recognition error: {e}")
            self.message_input.delete("0.0", "end")
            self.message_input.insert("0.0", "Speech recognition failed, please try again")

    def send_message(self):
        """Send message to chat interface and get AI response"""
        message = self.message_input.get("0.0", "end-1c").strip()
        if not message:
            return

        # Clear input box
        self.message_input.delete("0.0", "end")

        # Add user message to chat interface
        self.add_user_message(message)

        # Get AI response in a new thread
        threading.Thread(target=self.get_ai_response, args=(message,), daemon=True).start()

    def get_ai_response(self, message):
        """Get AI response from Ollama"""
        try:
            # Show "typing" status
            self.add_bot_typing_indicator()

            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "qwen2.5",  # Use your Ollama model
                    "messages": [{"role": "user", "content": message}],
                    "stream": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                ai_message = result["message"]["content"]

                # Remove "typing" indicator
                self.remove_typing_indicator()

                # Add AI response to chat interface
                self.add_bot_message(ai_message)
            else:
                self.remove_typing_indicator()
                self.add_bot_message(f"Error: Could not get response (Status code: {response.status_code})")

        except Exception as e:
            self.remove_typing_indicator()
            self.add_bot_message(f"Error: {str(e)}")

    def add_user_message(self, message, image_path=None):
        """Add user message to chat interface"""
        self.chat_display.configure(state="normal")

        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")

        # Add user identifier
        self.chat_display.insert("end", f"{self.current_user}: ", "user")

        # Add message content
        self.chat_display.insert("end", f"{message}\n", "user_message")

        # Add image if provided
        if image_path and os.path.exists(image_path):
            self.insert_image(image_path)

        # Scroll to bottom
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def add_bot_message(self, message, image_path=None):
        """Add bot message to chat interface"""
        self.chat_display.configure(state="normal")

        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")

        # Add bot identifier
        self.chat_display.insert("end", "AI Assistant: ", "bot")

        # Add message content
        self.chat_display.insert("end", f"{message}\n", "bot_message")

        # Add image if provided
        if image_path and os.path.exists(image_path):
            self.insert_image(image_path)

        # Scroll to bottom
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def add_bot_typing_indicator(self):
        """Add bot typing indicator"""
        self.chat_display.configure(state="normal")

        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")

        # Add bot identifier
        self.chat_display.insert("end", "AI Assistant: ", "bot")

        # Add "typing" indicator
        self.chat_display.insert("end", "Typing...", "typing")

        # Scroll to bottom
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def remove_typing_indicator(self):
        """Remove typing indicator"""
        self.chat_display.configure(state="normal")

        # Find and delete the last line
        last_line_start = self.chat_display.index("end-1c linestart")
        last_line_text = self.chat_display.get(last_line_start, "end-1c")
        if "Typing" in last_line_text:
            self.chat_display.delete(last_line_start, "end-1c")

        self.chat_display.configure(state="disabled")

    def insert_image(self, image_path):
        """Insert image into chat interface"""
        try:
            self.chat_display.configure(state="normal")

            # Open and resize image
            image = Image.open(image_path)
            max_width = 300  # Maximum width

            # Calculate adjusted dimensions
            width, height = image.size
            if width > max_width:
                ratio = max_width / width
                width = max_width
                height = int(height * ratio)
                image = image.resize((width, height), Image.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Save reference to prevent garbage collection
            self.image_references.append(photo)

            # Get underlying tkinter Text widget and insert image
            self.chat_display.insert("end", "\n")
            self._textbox.image_create("end-1c", image=photo)
            self.chat_display.insert("end", "\n")

            self.chat_display.configure(state="disabled")

        except Exception as e:
            print(f"Error inserting image: {e}")

    def set_user(self, username):
        """Set current username"""
        self.current_user = username


if __name__ == "__main__":
    # Create application instance
    app = VoiceChatApp()
    app.mainloop()
