import torch 
import numpy as np
import cv2
from PIL import Image
from .pano_depth import build_depth_model, pred_pano_depth
from .pano_seg import build_segment_model, seg_pano_fg
from .pano_inpaint import build_inpaint_model, inpaint_pano, inpaint_image

from .utils import convert_rgbd_to_gs, SplatFile

class WorldGen:
    def __init__(self, device: torch.device = 'cuda', inpaint_bg: bool = False):
        self.device = device
        self.depth_model = build_depth_model(device)
        self.seg_processor, self.seg_model = build_segment_model(device)
        
        self.inpaint_bg = inpaint_bg
        if inpaint_bg:
            self.inpaint_pipe = build_inpaint_model(device)

    def depth2gs(self, predictions) -> SplatFile:
        rgb = predictions["rgb"]
        distance = predictions["distance"]
        rays = predictions["rays"]
        splat = convert_rgbd_to_gs(rgb, distance, rays)
        return splat
    
    def mask_splat(self, splat: SplatFile, mask: np.ndarray) -> SplatFile:
        H, W = mask.shape
        valid_mask = mask>0
        centers = splat["centers"]
        covariances = splat["covariances"]
        rgbs = splat["rgbs"]
        opacity = splat["opacities"]

        centers = centers.reshape(H, W, 3)[valid_mask]
        covariances = covariances.reshape(H, W, 3, 3)[valid_mask]
        rgbs = rgbs.reshape(H, W, 3)[valid_mask]
        opacity = opacity.reshape(H, W, 1)[valid_mask]

        splat = {
            "centers": centers,
            "covariances": covariances,
            "rgbs": rgbs,
            "opacities": opacity
        }
        return splat
    
    def merge_splats(self, splat1: SplatFile, splat2: SplatFile) -> SplatFile:
        return SplatFile(
            centers=np.concatenate([splat1["centers"], splat2["centers"]], axis=0),
            covariances=np.concatenate([splat1["covariances"], splat2["covariances"]], axis=0),
            rgbs=np.concatenate([splat1["rgbs"], splat2["rgbs"]], axis=0),
            opacities=np.concatenate([splat1["opacities"], splat2["opacities"]], axis=0)
        )
    
    def depth_match(self, init_pred: dict, bg_pred: dict, mask: np.ndarray) -> dict:
        valid_mask = (mask > 0)
        init_distance = init_pred["distance"][valid_mask]
        bg_distance = bg_pred["distance"][valid_mask]

        init_mask = init_distance < torch.quantile(init_distance, 0.3)
        bg_mask = bg_distance < torch.quantile(bg_distance, 0.3)
        scale = init_distance[init_mask].median() / bg_distance[bg_mask].median()
        bg_pred["distance"] *= scale
        return bg_pred

    def _generate_world(self, pano_image: Image.Image) -> SplatFile:
        init_pred = pred_pano_depth(self.depth_model, pano_image)
        init_splat = self.depth2gs(init_pred)
        
        fg_mask = seg_pano_fg(self.seg_processor, self.seg_model, pano_image, init_pred["distance"])
        edge_mask = cv2.dilate(fg_mask, np.ones((3,3), np.uint8), iterations=1) - cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        init_splat = self.mask_splat(init_splat, (1-edge_mask))

        if not self.inpaint_bg:
            return init_splat
        
        dilated_fg_mask = cv2.dilate(fg_mask, np.ones((5,5), np.uint8), iterations=10)
        pano_bg = inpaint_image(self.inpaint_pipe, pano_image, dilated_fg_mask)
        bg_pred = pred_pano_depth(self.depth_model, pano_bg)
        bg_pred = self.depth_match(init_pred, bg_pred, (1-dilated_fg_mask))
        pano_bg_splat = self.depth2gs(bg_pred)
        occ_bg_splat = self.mask_splat(pano_bg_splat, dilated_fg_mask)
        merged_splat = self.merge_splats(init_splat, occ_bg_splat)
        return merged_splat
    
    def generate_world(self, text: str, image: Image.Image = None) -> SplatFile:
        pass


if __name__ == "__main__":
    worldgen = WorldGen()
    image = Image.open("data/background/timeless_desert.png")
    worldgen._generate_world(image)