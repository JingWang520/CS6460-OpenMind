# backend.py
import os
import time
import base64
from io import BytesIO
from typing import Literal, List

import PIL
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

# TTS library
from TTS.api import TTS

# Ollama library
from ollama import chat

# Create output directories
os.makedirs('generated_images', exist_ok=True)
os.makedirs('tts_audio', exist_ok=True)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model definitions
class ImageGenerationRequest(BaseModel):
    prompt: str

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class TextRequest(BaseModel):
    messages: List[Message]

class TTSRequest(BaseModel):
    text: str

# --- Load Janus model ---
print("=== Loading Janus model ===")
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
print("Janus model loaded successfully!")

# --- Initialize TTS model ---
print("=== Loading TTS model ===")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
tts_model_name = "tts_models/en/ljspeech/fast_pitch"  # Replace with the model you need

tts = TTS(model_name=tts_model_name, progress_bar=False).to(device)
print("TTS model loaded.")

# --- Initialize Ollama client and Qwen model ---
print("=== Initializing Ollama client ===")
qwen_model_name = "qwen2.5"
print("Ollama client ready.")

# Janus image generation function (your original code)
@torch.inference_mode()
def generate_image(
        mmgpt,
        vl_chat_processor,
        prompt,
        temperature=0.7,
        parallel_size=1,
        cfg_weight=10,
        image_token_num_per_image=576,
        img_size=384,
        patch_size=16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for j in range(parallel_size * 2):
        tokens[j, :] = input_ids
        if j % 2 != 0:
            tokens[j, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    outputs = None
    for j in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                             past_key_values=outputs.past_key_values if j != 0 else None)
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, j] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                             shape=[parallel_size, 8, img_size // patch_size,
                                                    img_size // patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    image = PIL.Image.fromarray(visual_img[0])
    return image

# Janus image generation endpoint
@app.post("/generate-image/")
async def generate_image_endpoint(request: ImageGenerationRequest):
    try:
        conversation = [
            {"role": "<|User|>", "content": request.prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_format + vl_chat_processor.image_start_tag

        image = generate_image(vl_gpt, vl_chat_processor, prompt_text)

        timestamp = int(time.time())
        save_path = os.path.join('generated_images', f"janus_image_{timestamp}.png")
        image.save(save_path)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image": img_str, "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New: Qwen text generation endpoint

@app.post("/generate-text/")
async def generate_text_endpoint(request: TextRequest):
    try:
        response = chat(
            model=qwen_model_name,
            messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
        )
        content = response.message.content

        return {"response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New: TTS endpoint
@app.post("/tts/")
async def tts_endpoint(request: TTSRequest):
    try:
        text = request.text
        timestamp = int(time.time())
        wav_path = os.path.join("tts_audio", f"tts_{timestamp}.wav")

        # Use TTS to synthesize speech and save
        tts.tts_to_file(text=text, file_path=wav_path)

        # Read wav file and convert to base64
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        return {"audio_base64": wav_base64, "path": wav_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
