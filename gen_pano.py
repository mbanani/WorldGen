import argparse
import torch
import os
import random
from worldgen.pano_gen import generate_panorama

def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX pipelines.")

    # Core arguments
    parser.add_argument("prompt", type=str, help="Text prompt for image generation.")
    parser.add_argument("--output_folder", type=str, default=None, help="Directory to save the generated image (defaults based on image_type).")
    parser.add_argument("--filename", type=str, default="image.png", help="Output filename (default: generated_image.png).")
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1), help="Random seed for generation (default: random).")

    # Panorama generation specific arguments
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale for panorama generation (default: 7.0).")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for panorama generation (default: 50).")
    parser.add_argument("--height", type=int, default=720, help="Height for panorama generation (default: 720).")
    parser.add_argument("--width", type=int, default=1440, help="Width for panorama generation (default: 1440).")
    parser.add_argument("--blend_extend", type=int, default=6, help="Blend extend value for panorama generation (default: 6).")
    parser.add_argument("--lora_path", type=str, default="./checkpoints/pano_lora_720*1440_v1.safetensors", help="Path to the panorama LoRA file.")

    args = parser.parse_args()

    # Determine output folder based on image_type if not provided
    output_folder = args.output_folder
    output_folder = './data/background/'

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, args.filename)

    print(f"--- Starting Generation ---")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Output Path: {output_path}")

    print("Selected Panorama Generation Parameters:")
    print(f"  Guidance Scale: {args.guidance_scale}")
    print(f"  Inference Steps: {args.num_inference_steps}")
    print(f"  Height: {args.height}")
    print(f"  Width: {args.width}")
    print(f"  Blend Extend: {args.blend_extend}")
    print(f"  LoRA Path: {args.lora_path}")
    generate_panorama(
        args.prompt,
        output_path,
        args.seed,
        args.lora_path,
        args.guidance_scale,
        args.num_inference_steps,
        args.height,
        args.width,
        args.blend_extend
    )
    print(f"--- Generation Complete ---")

if __name__ == "__main__":
    main()
