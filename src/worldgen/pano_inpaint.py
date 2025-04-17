from PIL import Image
import cv2
import torch
import numpy as np
from .inpaint_model import LaMa
from .utils import pano_to_cube, cube_to_pano

def build_inpaint_model(device: torch.device = 'cuda'):
    model = LaMa(device=device)
    return model

def inpaint_image(model, image: Image.Image, mask: Image.Image) -> Image.Image:
    original_size = image.size
    inpainted_image = model.infer(np.array(image), np.array(mask))
    inpainted_image = Image.fromarray(inpainted_image)
    inpainted_image = inpainted_image.resize(original_size)
    return inpainted_image

@torch.inference_mode()
def inpaint_pano(model, image: Image.Image, mask: np.ndarray):
    H, W = image.height, image.width
    assert (H / W == 0.5),  "Input image aspect ratio is not 2:1. Is it a panorama?"
    cube_face_res = H // 2
    mask = Image.fromarray(mask * 255).convert("L")

    print(f"Processing as panorama. Converting to cubemap (calculated face res: {cube_face_res}px)...")
    cube_faces = pano_to_cube(image, face_w=cube_face_res)
    cube_masks = pano_to_cube(mask, face_w=cube_face_res, mode='nearest')

    cube_inpainted_faces = []
    for face, mask in zip(cube_faces, cube_masks):
        inpainted_face = inpaint_image(model, face, mask)
        cube_inpainted_faces.append(inpainted_face)

    pano_inpainted_image = cube_to_pano(cube_inpainted_faces, h=H, w=W, mode='bilinear')
    pano_inpainted_image.save("pano_inpainted_image.png")
    return pano_inpainted_image


# from diffusers import FluxFillPipeline
# def build_inpaint_model(device: torch.device = 'cuda'):
#     pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, device=device)
#     pipe.enable_model_cpu_offload()
#     pipe.enable_vae_tiling()
#     return pipe

# @torch.inference_mode()
# def inpaint_image(pipe, image: Image.Image, mask: np.ndarray, prompt: str = "Background of the image, no objects, plain scene, photorealistic, high quality"):
#     """Segments instances in a single image using OneFormer and returns a binary mask."""
#     original_size = image.size
#     mask = Image.fromarray(mask * 255).convert("RGB")
#     inpainted_image = pipe(
#         image=image,
#         mask_image=mask,
#         guidance_scale=30,
#         num_inference_steps=50,
#         height=original_size[1],
#         width=original_size[0],
#         prompt=prompt,
#         max_sequence_length=512,
#         generator=torch.Generator("cpu").manual_seed(0)
#     ).images[0]
#     return inpainted_image