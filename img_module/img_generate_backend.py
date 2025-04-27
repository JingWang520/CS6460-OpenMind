# backend.py
import os
import PIL.Image
import torch
import numpy as np
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from huggingface_hub import login

# Create output directory
os.makedirs('generated_images', exist_ok=True)

# Login to Hugging Face

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request model
class ImageGenerationRequest(BaseModel):
    prompt: str


# Load the Janus model at startup
print("=== Loading Janus model ===")
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
print("Janus model loaded successfully!")


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


@app.post("/generate-image/")
async def generate_image_endpoint(request: ImageGenerationRequest):
    try:
        # Format the prompt for Janus
        conversation = [
            {
                "role": "<|User|>",
                "content": request.prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_format + vl_chat_processor.image_start_tag

        # Generate image
        image = generate_image(vl_gpt, vl_chat_processor, prompt_text)

        # Save image
        timestamp = int(time.time())
        save_path = os.path.join('generated_images', f"janus_image_{timestamp}.png")
        image.save(save_path)

        # Convert image to base64 for response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image": img_str, "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import time

    uvicorn.run(app, host="0.0.0.0", port=8008)
