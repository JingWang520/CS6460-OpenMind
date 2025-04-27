import os
import PIL.Image
import torch
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import openai
from openai import OpenAI
import gc
from huggingface_hub import login
from tqdm import tqdm

# Login to Hugging Face


# Create output directory
os.makedirs('comparison_results', exist_ok=True)

# Set your OpenAI API key
openai.base_url = "https://www.dmxapi.com/v1"
openai.api_key = "sk-JWU9jpEGAv2kHo3YBNB1TPIuhVjT4Nnf60GT55n7iz5GY3g7"

prompts = [
    "A serene lake at sunset, mountains in background, birds flying, orange sky",
    "Futuristic cityscape, flying cars, neon lights, tall skyscrapers, cloudy weather",
    "Enchanted forest, glowing mushrooms, fairy lights, misty atmosphere, ancient trees",
    "Underwater scene, colorful coral reef, tropical fish, sunbeams, blue depths",
    "Cozy cabin interior, fireplace, wooden furniture, snow outside, warm lighting",
    "Desert landscape, red sand dunes, single cactus, clear blue sky, distant mountains",
    "Space station orbiting Earth, stars in background, astronaut, satellite, space debris",
    "Medieval castle, stone walls, knights in armor, flying banners, foggy moat",
    "Japanese garden, cherry blossoms, stone lanterns, koi pond, wooden bridge, maple trees",
    "Steampunk laboratory, brass instruments, gears, steam pipes, vintage equipment, inventor at work"
]


# Process all prompts with Stable Diffusion
def process_stable_diffusion():
    print("\n=== Loading Stable Diffusion model ===")
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    sd_images = []

    print("Generating images with Stable Diffusion...")
    for i, prompt in enumerate(tqdm(prompts)):
        image = pipe(prompt).images[0]
        save_path = os.path.join('comparison_results', f"prompt_{i + 1}_stable_diffusion.png")
        image.save(save_path)
        sd_images.append(image)

    # Clear GPU memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    print("Stable Diffusion model unloaded from memory")

    return sd_images


# Process all prompts with Janus
def process_janus():
    print("\n=== Loading Janus model ===")
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    janus_images = []

    print("Generating images with Janus...")
    for i, prompt in enumerate(tqdm(prompts)):
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_format + vl_chat_processor.image_start_tag

        @torch.inference_mode()
        def generate_image(
                mmgpt,
                vl_chat_processor,
                prompt,
                temperature=1,
                parallel_size=1,
                cfg_weight=5,
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

        image = generate_image(vl_gpt, vl_chat_processor, prompt_text)
        save_path = os.path.join('comparison_results', f"prompt_{i + 1}_janus.png")
        image.save(save_path)
        janus_images.append(image)

    # Clear GPU memory
    del vl_gpt
    del vl_chat_processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Janus model unloaded from memory")

    return janus_images


# Process all prompts with DALL-E 3
def process_dalle():
    print("\n=== Generating images with DALL-E 3 ===")
    dalle_images = []
    client = OpenAI(base_url="https://www.dmxapi.com/v1",api_key="sk-JWU9jpEGAv2kHo3YBNB1TPIuhVjT4Nnf60GT55n7iz5GY3g7")

    for i, prompt in enumerate(tqdm(prompts)):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        response = requests.get(image_url)
        image = PIL.Image.open(BytesIO(response.content))
        save_path = os.path.join('comparison_results', f"prompt_{i + 1}_dalle.png")
        image.save(save_path)
        dalle_images.append(image)

    return dalle_images


# Create comparison images for all prompts
def create_comparisons(sd_images, janus_images, dalle_images):
    print("\n=== Creating comparison images ===")
    for i, prompt in enumerate(tqdm(prompts)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(np.array(sd_images[i]))
        axes[0].set_title("Stable Diffusion", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(np.array(janus_images[i]))
        axes[1].set_title("DeepSeek-Janus-Pro-7B", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(np.array(dalle_images[i]))
        axes[2].set_title("DALL-E 3", fontsize=14)
        axes[2].axis("off")

        plt.suptitle(f"Prompt {i + 1}: {prompt}", fontsize=16)
        plt.tight_layout()

        comparison_path = os.path.join('comparison_results', f"prompt_{i + 1}_comparison.png")
        plt.savefig(comparison_path, dpi=300)
        plt.close()


# Main process
def main():
    print("Starting image generation comparison...")

    # Process all prompts with each model sequentially

    # sd_images = process_stable_diffusion()
    janus_images = process_janus()
    # dalle_images = process_dalle()

    # Create comparison images
    create_comparisons(sd_images, janus_images, dalle_images)

    print("\nAll comparisons completed!")


if __name__ == "__main__":
    main()
