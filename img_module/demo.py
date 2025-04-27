import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Create output directory
os.makedirs('shoes_images', exist_ok=True)

# Initialize Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.cpu)
pipe = pipe.to("cpu")

# Shoe categories and prompts
shoe_types = {
    "Casual Shoes": "High-quality photo of stylish casual shoes, comfortable design, modern look, clean white background, product photography",
    "Athletic Shoes": "Professional photo of running shoes, sporty design, breathable mesh, rubber soles, studio lighting, white background",
    "Formal Shoes": "Elegant leather dress shoes, polished finish, classic design, isolated on white background, luxury footwear",
    "Boots": "Fashionable leather boots, ankle height, detailed stitching, on white background, product shot",
    "Sandals": "Summer sandals, comfortable footbed, leather straps, isolated on white background, lifestyle product photography"
}

# Generate images for each shoe type
print("Generating shoe images with Stable Diffusion...")
for shoe_type, prompt in shoe_types.items():
    # Generate the image
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Save the image
    filename = f"{shoe_type.lower().replace(' ', '_')}.png"
    save_path = os.path.join('shoes_images', filename)
    image.save(save_path)
    print(f"Saved {shoe_type} image to {save_path}")

# Clean up
del pipe
torch.cpu.empty_cache()
print("Image generation completed!")
