import os
import torch
from .flux_pano_pipeline import FluxPipeline

def generate_panorama(
        prompt, 
        output_path=None, 
        seed=42, 
        lora_path="./checkpoints/pano_lora_720*1440_v1.safetensors",
        guidance_scale=7.0, 
        num_inference_steps=50, 
        height=720, 
        width=1440, 
        blend_extend=6
    ):
    """Generates a panorama image using FLUX.1-dev and a LoRA."""
    # Assuming pipeline_flux.py exists in the same directory or is installable
    if not os.path.isfile(lora_path):
         print(f"Error: LoRA path '{lora_path}' not found or is not a file.")
         print("Please provide a valid path using --lora_path.")
         return

    print("Loading panorama pipeline (FLUX.1-dev)...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device="cuda")
    print(f"Loading LoRA weights from: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()       # Required for panorama generation

    generator = torch.Generator("cpu").manual_seed(seed)
    print(f"Generating panorama image with seed: {seed}")
    image = pipe(
        prompt,
        height=height,
        width=width,
        generator=generator,
        num_inference_steps=num_inference_steps,
        blend_extend=blend_extend,
        guidance_scale=guidance_scale
    ).images[0]
    
    if output_path is not None:
        image.save(output_path)
        print(f"Panorama image saved to {output_path}")
    return image
