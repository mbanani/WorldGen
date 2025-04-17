import py360convert
from PIL import Image
import numpy as np
import torch
from typing import TypedDict
import numpy.typing as npt

class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""

def pano_to_cube(pano_img: Image.Image, face_w: int, mode: str = 'bilinear') -> list[Image.Image]:
    """Converts a panoramic PIL Image to a list of 6 cubemap face PIL Images."""
    pano_np = np.array(pano_img)
    cube_faces = py360convert.e2c(pano_np, face_w=face_w, mode='bilinear', cube_format='list')
    cube_pil_faces = [Image.fromarray(face) for face in cube_faces]
    return cube_pil_faces

def cube_to_pano(cube_faces: list[Image.Image], h: int, w: int, mode: str = 'bilinear') -> Image.Image:
    """Converts a list of 6 cubemap face PIL Images (masks) back to a panoramic PIL Image."""
    cube_np_faces = []
    for face in cube_faces:
        face_np = np.array(face)
        if face_np.ndim == 2:
             pass
        elif face_np.ndim == 3 and face_np.shape[2] == 1:
             face_np = face_np.squeeze(axis=2)
        cube_np_faces.append(face_np)

    pano_np = py360convert.c2e(cube_np_faces, h=h, w=w, mode=mode, cube_format='list')
    pano_pil = Image.fromarray(pano_np.astype(np.uint8))
    return pano_pil

def convert_rgbd_to_gs(rgb, distance, rays, dis_threshold=0., epsilon=1e-3) -> SplatFile:
    """
    Given an equirectangular RGB-D image, back-project each pixel to a 3D point
    and compute the corresponding 3D Gaussian covariance so that the projection covers 1 pixel.

    Parameters:
        rgb (H x W x 3): RGB image as torch.Tensor, uint8
        distance (H x W): Distance map (in meters) as torch.Tensor, float32
        rays (H x W x 3): Ray directions as torch.Tensor, float32
        epsilon (float): Small Z-scale for the splat in ray direction

    Returns:
        centers (N x 3): 3D positions of splats
        covariances (N x 3 x 3): 3D Gaussian covariances of splats
        colors (N x 3): RGB values of splats
        opacities (N x 1): Opacities of splats
    """
    H, W = rgb.shape[:2]
    device = rgb.device

    valid_mask = distance > dis_threshold
    rays_flat = rays.view(-1, 3)
    distance_flat = distance.view(-1)
    valid_rays = rays_flat[valid_mask.view(-1)]
    valid_distance = distance_flat[valid_mask.view(-1)]
    centers = valid_rays * valid_distance[:, None]

    delta_phi = 2 * torch.pi / W
    delta_theta = torch.pi / H
    sigma_x = valid_distance * delta_phi 
    sigma_y = valid_distance * delta_theta 
    sigma_z = torch.ones_like(valid_distance) * epsilon

    S = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)
    covariances = torch.einsum('ni,nj->nij', S, S)  # Sigma = S @ S.T        # (N, 3, 3)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    x_axis = torch.nn.functional.normalize(torch.cross(up, valid_rays), dim=1)
    fallback_up = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    degenerate_mask = torch.isnan(x_axis).any(dim=1)
    x_axis[degenerate_mask] = torch.nn.functional.normalize(torch.cross(fallback_up[degenerate_mask], valid_rays[degenerate_mask]), dim=1)
    y_axis = torch.nn.functional.normalize(torch.cross(valid_rays, x_axis), dim=1)
    z_axis = valid_rays

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    # Step 5: apply covariance transformation: Sigma = R S S^T R^T
    S_matrices = torch.zeros((S.shape[0], 3, 3), device=device)
    S_matrices[:, 0, 0] = S[:, 0]
    S_matrices[:, 1, 1] = S[:, 1]
    S_matrices[:, 2, 2] = S[:, 2]

    covariances = R @ S_matrices @ S_matrices.transpose(1, 2) @ R.transpose(1, 2)
    colors = rgb.view(-1, 3).float() / 255.0
    opacities = torch.ones((centers.shape[0], 1))

    return SplatFile(
        centers=centers.cpu().numpy(),
        covariances=covariances.cpu().numpy(),
        rgbs=colors.cpu().numpy(),
        opacities=opacities.cpu().numpy(),
    )

def visualize_mask(mask: Image.Image, output_path: str):
    print("Visualizing panoramic mask with different colors...")
    mask_np = np.array(mask)
    unique_ids = np.unique(mask_np)
    unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
    
    # Create a colormap with random colors for each category
    colormap = np.zeros((np.max(unique_ids) + 1, 3), dtype=np.uint8)
    np.random.seed(42)  # For reproducible colors
    for id in unique_ids:
        colormap[id] = np.random.randint(0, 256, 3)
    
    vis_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for id in unique_ids:
        vis_mask[mask_np == id] = colormap[id]
    
    Image.fromarray(vis_mask).save(output_path)
    print(f"Visualization saved to {output_path}")