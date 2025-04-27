import base64
import os
import re
import time
import threading
import uuid

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from googlesearch import search
import requests
from readability import Document

from mind_module.generate_mind import MindmapGenerator

import io
import wave


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
mg = MindmapGenerator()

class ZoomPanImageViewer(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.image = None
        self.photo_image = None
        self.image_id = None

        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0

        self.pan_start = None
        self.offset_x = 0
        self.offset_y = 0

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

        self.canvas.bind("<Configure>", self.on_resize)

    def on_button_press(self, event):
        self.pan_start = (event.x, event.y)

    def on_drag(self, event):
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.pan_start = (event.x, event.y)
        self.offset_x += dx
        self.offset_y += dy
        self.redraw_image()

    def on_mousewheel(self, event):
        old_scale = self.scale
        if event.delta:
            if event.delta > 0:
                self.scale *= 1.1
            else:
                self.scale /= 1.1
        else:
            if event.num == 4:
                self.scale *= 1.1
            elif event.num == 5:
                self.scale /= 1.1

        self.scale = max(self.min_scale, min(self.max_scale, self.scale))

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        mouse_x = event.x
        mouse_y = event.y

        self.offset_x = (self.offset_x - mouse_x) * (self.scale / old_scale) + mouse_x
        self.offset_y = (self.offset_y - mouse_y) * (self.scale / old_scale) + mouse_y

        self.redraw_image()

    def on_resize(self, event):
        self.redraw_image()

    def set_image(self, pil_image):
        self.image = pil_image
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.redraw_image()

    def clear_image(self):
        self.image = None
        self.photo_image = None
        self.canvas.delete("all")

    def redraw_image(self):
        self.canvas.delete("all")
        if self.image is None:
            return

        w, h = self.image.size
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        resized_image = self.image.resize((new_w, new_h), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized_image)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if self.offset_x == 0 and self.offset_y == 0:
            x = max((canvas_width - new_w) // 2, 0)
            y = max((canvas_height - new_h) // 2, 0)
        else:
            x = self.offset_x
            y = self.offset_y

        self.image_id = self.canvas.create_image(x, y, anchor="nw", image=self.photo_image)


class VoiceChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Voice Chat Assistant")
        self.geometry("1200x700")
        self.minsize(900, 600)

        self.image_references = []

        self.load_speech_recognition_model()

        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.current_user = "User"
        self.chat_history = [
            {"role": "system", "content": (
                """# Task Requirements
You need to first generate a `mindmap` code block covering the content of the detailed answer, using Markdown syntax (e.g., ```mindmap ... ```). Then, provide the detailed answer.

# Mindmap Requirements
- The mindmap must include a theme, main branches, and sub-branches. It should have a clear structure with distinct levels, preferably not exceeding four levels.
- Content must cover all key points of the detailed answer and show logical relationships.
- Use concise words or phrases for node content, avoiding lengthy sentences.
- Keep the map content complete to facilitate the detailed answer expansion.
- The mindmap code block format must be correct for proper rendering.

# Detailed Answer Requirements
- In the detailed answer, use a combination of text and images, alternating narration. Insert image descriptions using ```image ... ``` code blocks where relevant. Place image descriptions at appropriate points in the answer.
- Image descriptions within ```image ... ``` code blocks must be in English, with elements separated by commas.
- Do not mention the mindmap itself in the detailed answer.
- The detailed answer should use paragraph-style language, not Markdown syntax (except for image blocks). The language should be fluent and the information complete.
- The answer content should be rich and easy to understand, covering all points from the mindmap.

# General Configuration
*   **Language:** The entire response **must** be in **English**.

# Example
## Question  
What are shoes?  

## Answer(The answer follow the format of the example)
```mindmap
- Shoes
  - Definition
    - Items worn on the feet
  - Functions
    - Protect the feet
    - Provide comfort
    - Support the steps
    - Anti-slip and waterproof
  - Types
    - Sports shoes
    - Formal shoes
    - Sandals
    - Boots
  - Materials
    - Leather
    - Fabric
    - Rubber
    - Synthetic materials
```

Shoes are items worn on the feet, mainly used to protect the feet, provide comfort and support, and prevent slipping or injury.

According to their use and style, shoes come in various types, such as sports shoes, formal shoes, sandals, and boots.  
Sports shoes are usually lightweight and breathable, suitable for running and sports.  
Here is a picture of a pair of red running shoes:

```image  
A pair of red sports shoes with a breathable upper and anti-slip patterned sole, high definition, 4K.  
```  
Here is a picture of a pair of white running shoes:

```image  
A pair of white lightweight running shoes with a mesh upper, high definition.  
```  
The materials of shoes usually include leather, fabric, rubber, and various synthetic materials, which determine the comfort and functionality of the shoes..."""
            )}
        ]

        self.create_ui()

    def load_speech_recognition_model(self):
        print("Loading speech recognition model...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
        print("Speech recognition model loaded")

    def create_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        self.title_frame = ctk.CTkFrame(self, height=40, fg_color=["#F0F0F0", "#333333"])
        self.title_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="ew")
        self.title_frame.grid_propagate(False)
        self.title_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.title_frame, text="AI Voice Assistant", font=("Helvetica", 16, "bold"))
        self.title_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.clear_button = ctk.CTkButton(
            self.title_frame, text="Clear Chat", width=100, height=30,
            command=self.clear_chat,
            fg_color=["#E74C3C", "#C0392B"],
            hover_color=["#C0392B", "#A93226"]
        )
        self.clear_button.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="e")

        self.chat_frame = ctk.CTkFrame(self)
        self.chat_frame.grid(row=1, column=0, padx=10, pady=(5, 5), sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(self.chat_frame, font=("Helvetica", 12))
        self.chat_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.chat_display.configure(state="disabled")
        self._textbox = self.chat_display._textbox

        self._textbox.tag_configure("timestamp", foreground="#888888")
        self._textbox.tag_configure("user", foreground="#3498DB", font=("Helvetica", 12, "bold"))
        self._textbox.tag_configure("user_message", foreground="#000000")
        self._textbox.tag_configure("bot", foreground="#E74C3C", font=("Helvetica", 12, "bold"))
        self._textbox.tag_configure("bot_message", foreground="#333333")
        self._textbox.tag_configure("typing", foreground="#888888", font=("Helvetica", 11, "italic"))
        self._textbox.tag_configure("system", foreground="#7D3C98", font=("Helvetica", 11, "italic"))

        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.message_input = ctk.CTkTextbox(self.input_frame, height=60, font=("Helvetica", 12))
        self.message_input.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        self.button_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.button_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")

        self.voice_button = ctk.CTkButton(
            self.button_frame, text="🎤", width=40, height=30, font=("Helvetica", 16),
            command=self.toggle_recording
        )
        self.voice_button.grid(row=0, column=0, padx=(0, 5), pady=(0, 5))

        self.send_button = ctk.CTkButton(
            self.button_frame, text="Send", width=60, height=30,
            command=self.send_message
        )
        self.send_button.grid(row=1, column=0, padx=0, pady=(0, 0))

        self.search_var = tk.BooleanVar(value=False)
        self.search_checkbox = ctk.CTkCheckBox(
            self.input_frame, text="Enable Online Search",
            variable=self.search_var
        )
        self.search_checkbox.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="w")

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self.message_input.bind("<Return>", lambda event: self.send_message())

        self.mindmap_frame = ctk.CTkFrame(self, fg_color="white")
        self.mindmap_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")
        self.mindmap_viewer = ZoomPanImageViewer(self.mindmap_frame)
        self.mindmap_viewer.pack(expand=True, fill="both")

        self.after(100, lambda: self.add_bot_message("Hello! I am your AI assistant. How can I help you today?"))

    def remove_first_mindmap_block(self, text):
        pattern = re.compile(r"```mindmap\s*[\s\S]+?```", re.MULTILINE)
        return pattern.sub("", text, count=1).strip()

    def google_search(self, query):
        try:
            urls = list(search(query, num_results=3))
            snippets = []
            for url in urls:
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    doc = Document(resp.text)
                    content = doc.summary()
                    title = doc.title()
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator='\n', strip=True)
                    snippet = f"{title}\n{text[:500]}"
                    snippets.append(snippet)
                except Exception as e:
                    snippets.append(f"Failed to get webpage content ({url}): {e}")
            return "\n\n".join(snippets)
        except Exception as e:
            print(f"Google search failed: {e}")
            return ""

    def clear_chat(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("0.0", "end")
        self.chat_display.configure(state="disabled")
        self.image_references.clear()

        self.chat_history.clear()
        self.chat_history.append(
            {"role": "system", "content": (
                """# Task Requirements
You need to first generate a `mindmap` code block covering the content of the detailed answer, using Markdown syntax (e.g., ```mindmap ... ```). Then, provide the detailed answer.

# Mindmap Requirements
- The mindmap must include a theme, main branches, and sub-branches. It should have a clear structure with distinct levels, preferably not exceeding four levels.
- Content must cover all key points of the detailed answer and show logical relationships.
- Use concise words or phrases for node content, avoiding lengthy sentences.
- Keep the map content complete to facilitate the detailed answer expansion.
- The mindmap code block format must be correct for proper rendering.

# Detailed Answer Requirements
- In the detailed answer, use a combination of text and images, alternating narration. Insert image descriptions using ```image ... ``` code blocks where relevant. Place image descriptions at appropriate points in the answer.
- Image descriptions within ```image ... ``` code blocks must be in English, with elements separated by commas.
- Do not mention the mindmap itself in the detailed answer.
- The detailed answer should use paragraph-style language, not Markdown syntax (except for image blocks). The language should be fluent and the information complete.
- The answer content should be rich and easy to understand, covering all points from the mindmap.

# General Configuration
*   **Language:** The entire response **must** be in **English**.


# Example
## Question  
What are shoes?  

## Answer(The answer follow the format of the example)
```mindmap
- Shoes
  - Definition
    - Items worn on the feet
  - Functions
    - Protect the feet
    - Provide comfort
    - Support the steps
    - Anti-slip and waterproof
  - Types
    - Sports shoes
    - Formal shoes
    - Sandals
    - Boots
  - Materials
    - Leather
    - Fabric
    - Rubber
    - Synthetic materials
```

Shoes are items worn on the feet, mainly used to protect the feet, provide comfort and support, and prevent slipping or injury.

According to their use and style, shoes come in various types, such as sports shoes, formal shoes, sandals, and boots.  
Sports shoes are usually lightweight and breathable, suitable for running and sports.  
Here is a picture of a pair of red running shoes:

```image  
A pair of red sports shoes with a breathable upper and anti-slip patterned sole, high definition, 4K.  
```  
Here is a picture of a pair of white running shoes:

```image  
A pair of white lightweight running shoes with a mesh upper, high definition.  
```  
The materials of shoes usually include leather, fabric, rubber, and various synthetic materials, which determine the comfort and functionality of the shoes..."""
            )}
        )

        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "System: Chat cleared\n", "system")
        self.chat_display.configure(state="disabled")

        self.mindmap_viewer.clear_image()
        self.mindmap_viewer.canvas.create_text(
            self.mindmap_viewer.canvas.winfo_width() // 2,
            self.mindmap_viewer.canvas.winfo_height() // 2,
            text="Mindmap will be displayed here",
            fill="#888888",
            font=("Helvetica", 14),
            tags="placeholder"
        )

        self.add_bot_message("How can I help you today?")

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.voice_button.configure(text="⏹️", fg_color="#E74C3C")
            self.audio_data = []
            threading.Thread(target=self.record_audio, daemon=True).start()
        else:
            self.recording = False
            self.voice_button.configure(text="🎤", fg_color=["#3B8ED0", "#1F6AA5"])
            self.process_audio()

    def record_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.append(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)

    def generate_image_from_text(self, prompt):
        try:
            response = requests.post(
                "http://10.25.10.144:8008/generate-image/",
                json={"prompt": prompt}
            )

            if response.status_code == 200:
                data = response.json()
                image_data = base64.b64decode(data["image"])

                tmp_dir = "tmp_img"
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)

                filename = f"{uuid.uuid4().hex}.png"
                file_path = os.path.join(tmp_dir, filename)

                with open(file_path, "wb") as f:
                    f.write(image_data)

                return file_path
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Image generation error: {e}")
            return None

    def process_audio(self):
        if not self.audio_data:
            return
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        temp_file = "temp_recording.wav"
        wav.write(temp_file, self.sample_rate, audio)

        self.message_input.delete("0.0", "end")
        self.message_input.insert("0.0", "Recognizing speech...")
        threading.Thread(target=self.recognize_speech, args=(audio,), daemon=True).start()

    def split_text_and_images(self, text):
        pattern = re.compile(r"```image\s*([\s\S]+?)\s*```", re.MULTILINE)
        parts = []
        last_end = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > last_end:
                text_block = text[last_end:start]
                paragraphs = [p.strip() for p in text_block.split('\n\n') if p.strip()]
                for para in paragraphs:
                    parts.append({"type": "text", "content": para})
            image_desc = match.group(1).strip()
            parts.append({"type": "image", "content": image_desc})
            last_end = end
        if last_end < len(text):
            text_block = text[last_end:]
            paragraphs = [p.strip() for p in text_block.split('\n\n') if p.strip()]
            for para in paragraphs:
                parts.append({"type": "text", "content": para})
        return parts

    def recognize_speech(self, audio):
        try:
            input_values = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_values
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            self.message_input.delete("0.0", "end")
            self.message_input.insert("0.0", transcription)
        except Exception as e:
            print(f"Speech recognition error: {e}")
            self.message_input.delete("0.0", "end")
            self.message_input.insert("0.0", "Speech recognition failed, please try again")

    def send_message(self):
        message = self.message_input.get("0.0", "end-1c").strip()
        if not message:
            return
        self.message_input.delete("0.0", "end")

        # Disable input controls while sending
        self.message_input.configure(state="disabled")
        self.voice_button.configure(state="disabled")
        self.send_button.configure(state="disabled")

        if self.search_var.get():
            search_results = self.google_search(message)
            if search_results:
                message_to_send = f"[Online search data]:\n{search_results}\n\n{message}\n\n"
            else:
                message_to_send = message
        else:
            message_to_send = message

        self.add_user_message(message)
        threading.Thread(target=self.get_ai_response, args=(message_to_send,), daemon=True).start()

    def get_ai_response(self, message):
        try:
            self.add_bot_typing_indicator()

            messages = self.chat_history.copy()
            messages.append({"role": "user", "content": message})

            response = requests.post(
                "http://10.25.10.144:8008/generate-text/",
                json={"messages": messages}
            )

            self.remove_typing_indicator()

            if response.status_code != 200:
                self.add_bot_message(f"Error: Unable to get response (status code: {response.status_code})")
                self.enable_input_controls()
                return

            result = response.json()
            ai_message = result.get("response", "")
            print(ai_message)

            mindmap_md = self.extract_mindmap_md(ai_message)

            # Generate mindmap image
            png_path = None
            if mindmap_md:

                md_path = mg.save_markdown(mindmap_md)
                png_path = mg.convert_md_to_png(md_path)
                if png_path:
                    self.after(0, lambda: self.update_mindmap_image(png_path))

            # Remove mindmap code block
            ai_message_no_mindmap = self.remove_first_mindmap_block(ai_message)

            # Split text and image descriptions
            parts = self.split_text_and_images(ai_message_no_mindmap)

            # Store image generation results, key: image description, value: image path or None
            image_results = {}

            # Start all image generation threads first, save results to image_results
            def generate_image_worker(desc):
                img_path = self.generate_image_from_text(desc)
                image_results[desc] = img_path

            threads = []
            for part in parts:
                if part["type"] == "image":
                    desc = part["content"]
                    image_results[desc] = None  # Initialize as None
                    t = threading.Thread(target=generate_image_worker, args=(desc,), daemon=True)
                    t.start()
                    threads.append(t)

            # Play text sentence by sentence, when encounter image description wait for image generation to finish then insert
            self.chat_display.configure(state="normal")
            timestamp = time.strftime("%H:%M:%S")
            self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")
            self.chat_display.insert("end", "AI Assistant:\n", "bot")
            self.chat_display.configure(state="disabled")

            for part in parts:
                if part["type"] == "text":
                    sentences = self.split_into_sentences(part["content"])
                    for sentence in sentences:
                        self.append_bot_text(sentence)
                        self.tts_play_sentence(sentence)
                elif part["type"] == "image":
                    desc = part["content"]
                    # self.append_bot_text(f"[Image description]: {desc}")

                    # Wait for image generation to complete, timeout e.g. 10 seconds
                    wait_time = 0
                    while image_results.get(desc) is None and wait_time < 10:
                        time.sleep(0.5)
                        wait_time += 0.5

                    img_path = image_results.get(desc)
                    if img_path:
                        self.after(0, lambda p=img_path: self.insert_image(p))
                    else:
                        self.append_bot_text("[Image generation failed]")

            # Optionally wait for all image threads to finish
            for t in threads:
                t.join(timeout=0)

            # Update chat history
            self.chat_history.append({"role": "assistant", "content": ai_message})

            self.enable_input_controls()

        except Exception as e:
            self.remove_typing_indicator()
            self.add_bot_message(f"Error: {str(e)}")
            self.enable_input_controls()

    def extract_mindmap_md(self, text):
        pattern = re.compile(r"```mindmap\s*([\s\S]+?)\s*```", re.MULTILINE)
        match = pattern.search(text)
        if match:
            return match.group(1)
        return None

    def add_user_message(self, message, image_path=None):
        self.chat_display.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")
        self.chat_display.insert("end", f"{self.current_user}:\n", "user")
        self.chat_display.insert("end", f"{message}\n", "user_message")

        if image_path and os.path.exists(image_path):
            self.insert_image(image_path)

        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        self.chat_history.append({"role": "user", "content": message})

    def add_bot_message(self, message, image_path=None):
        self.chat_display.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")
        self.chat_display.insert("end", "AI Assistant:\n", "bot")

        message_without_mindmap = self.remove_first_mindmap_block(message)
        parts = self.split_text_and_images(message_without_mindmap)
        for part in parts:
            if part["type"] == "text":
                self.chat_display.insert("end", part["content"] + "\n", "bot_message")
            elif part["type"] == "image":
                desc = part["content"]
                self.chat_display.insert("end", f"\n[Image description]: {desc}\n", "bot_message")
                img_path = self.generate_image_from_text(desc)
                if img_path:
                    self.insert_image(img_path)
                else:
                    self.chat_display.insert("end", "[Image generation failed]\n", "bot_message")

        self.chat_display.insert("end", "\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

        self.chat_history.append({"role": "assistant", "content": message})

    def add_bot_typing_indicator(self):
        self.chat_display.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.chat_display.insert("end", f"\n[{timestamp}] ", "timestamp")
        self.chat_display.insert("end", "AI Assistant: ", "bot")
        self.chat_display.insert("end", "Typing...", "typing")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def remove_typing_indicator(self):
        self.chat_display.configure(state="normal")
        last_line_start = self.chat_display.index("end-1c linestart")
        last_line_end = self.chat_display.index(f"{last_line_start} +1 line linestart")
        last_line_text = self.chat_display.get(last_line_start, last_line_end)
        if "Typing" in last_line_text:
            self.chat_display.delete(last_line_start, last_line_end)
        self.chat_display.configure(state="disabled")

    def insert_image(self, image_path):
        try:
            self.chat_display.configure(state="normal")
            image = Image.open(image_path)
            max_width = 300
            width, height = image.size
            if width > max_width:
                ratio = max_width / width
                width = max_width
                height = int(height * ratio)
                image = image.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_references.append(photo)
            self._textbox.image_create("end-1c", image=photo)
            self.chat_display.insert("end", "\n")
            self.chat_display.see("end")
            self.chat_display.configure(state="disabled")
        except Exception as e:
            print(f"Insert image error: {e}")

    def update_mindmap_image(self, image_path):
        try:
            image = Image.open(image_path)
            max_width, max_height = 500, 600
            width, height = image.size
            ratio = min(max_width / width, max_height / height, 1.0)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            self.mindmap_viewer.set_image(image)
        except Exception as e:
            print(f"Failed to load mindmap image: {e}")
            self.mindmap_viewer.clear_image()

    def set_user(self, username):
        self.current_user = username

    def enable_input_controls(self):
        self.message_input.configure(state="normal")
        self.voice_button.configure(state="normal")
        self.send_button.configure(state="normal")

    def split_into_sentences(self, text):
        import re
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def append_bot_text(self, text):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text + "\n", "bot_message")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def tts_play_sentence(self, sentence):
        try:
            resp = requests.post("http://10.25.10.144:8008/tts/", json={"text": sentence})
            if resp.status_code == 200:
                data = resp.json()
                audio_b64 = data["audio_base64"]
                audio_bytes = base64.b64decode(audio_b64)

                with io.BytesIO(audio_bytes) as audio_io:
                    with wave.open(audio_io, 'rb') as wf:
                        rate = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                        audio_np = np.frombuffer(frames, dtype=np.int16)
                        sd.play(audio_np, rate)
                        sd.wait()
        except Exception as e:
            print(f"TTS playback failed: {e}")

    def async_generate_and_insert_image(self, desc):
        img_path = self.generate_image_from_text(desc)
        if img_path:
            self.after(0, lambda: self.insert_image(img_path))
        else:
            self.after(0, lambda: self.append_bot_text("[Image generation failed]"))


if __name__ == "__main__":
    app = VoiceChatApp()
    app.mainloop()
