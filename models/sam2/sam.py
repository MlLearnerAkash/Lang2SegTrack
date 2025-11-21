import hydra
import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from models.sam2.sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_MODELS = {
    "sam2.1_hiera_tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "configs/samurai/sam2.1_hiera_t.yaml",
    },
    "sam2.1_hiera_small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "configs/samurai/sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "configs/samurai/sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "configs/samurai/sam2.1_hiera_l.yaml",
    },
}

class SAM:
    def build_model(self, sam_type: str, ckpt_path: str | None = None, predictor_type="img", device=torch.device('cuda:0'), use_txt_prompt=False):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        if predictor_type == "img":
            self.model = build_sam2(config_file=SAM_MODELS[self.sam_type]["config"], ckpt_path=self.ckpt_path, device=device)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
            self.img_predictor = SAM2ImagePredictor(self.model)
        elif predictor_type == "video" or predictor_type == "realtime":
            self.video_predictor = build_sam2_video_predictor(SAM_MODELS[self.sam_type]["config"], self.ckpt_path, device=device)
            # if use_txt_prompt:
            #     self.model = build_sam2(config_file=SAM_MODELS[self.sam_type]["config"], ckpt_path=self.ckpt_path, device=device)
            #     self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
            #     self.img_predictor = SAM2ImagePredictor(self.model)
            print("Building SAM2 image predictor for mask conversion...")
            self.model = build_sam2(
                config_file=SAM_MODELS[self.sam_type]["config"], 
                ckpt_path=self.ckpt_path, 
                device=device
            )
            # self.img_predictor = SAM2ImagePredictor(self.model)
            # print("SAM2 image predictor ready for bbox->mask conversion")

    def _load_checkpoint(self, model: torch.nn.Module):
        if self.ckpt_path is None:
            checkpoint_url = SAM_MODELS[self.sam_type]["url"]
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]
        else:
            checkpoint_url = self.ckpt_path  # Ensure checkpoint_url is defined
            state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)["model"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(
                f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                re-downloading it. Error: {e}"
            )

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Output format
        SAM2AutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information
        about the mask:

        segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        area - [int] - the area of the mask in pixels
        bbox - [List[int]] - the boundary box of the mask in xywh format
        predicted_iou - [float] - the model's own prediction for the quality of the mask
        point_coords - [List[List[float]]] - the sampled input point that generated this mask
        stability_score - [float] - an additional measure of mask quality
        crop_box - List[int] - the crop of the image used to generate this mask in xywh format
        """

        sam2_result = self.mask_generator.generate(image_rgb)
        return sam2_result

    def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.img_predictor.set_image(image_rgb)
        masks, scores, logits = self.img_predictor.predict(box=xyxy, multimask_output=False)
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks, scores, logits

    def predict_batch(
        self,
        images_rgb: list[np.ndarray],
        xyxy: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        self.img_predictor.set_image_batch(images_rgb)

        masks, scores, logits = self.img_predictor.predict_batch(box_batch=xyxy, multimask_output=False)

        masks = [np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks]
        scores = [np.squeeze(score) for score in scores]
        logits = [np.squeeze(logit, axis=1) if len(logit.shape) > 3 else logit for logit in logits]
        return masks, scores, logits

# import hydra
# import numpy as np
# import torch
# from hydra import compose
# from hydra.utils import instantiate
# from omegaconf import OmegaConf

# from models.sam2.sam2.sam2_video_predictor import SAM2VideoPredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from sam2.build_sam import build_sam2, build_sam2_video_predictor
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# SAM_MODELS = {
#     "sam2.1_hiera_tiny": {
#         "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
#         "config": "configs/samurai/sam2.1_hiera_t.yaml",
#     },
#     "sam2.1_hiera_small": {
#         "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
#         "config": "configs/samurai/sam2.1_hiera_s.yaml",
#     },
#     "sam2.1_hiera_base_plus": {
#         "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
#         "config": "configs/samurai/sam2.1_hiera_b+.yaml",
#     },
#     "sam2.1_hiera_large": {
#         "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
#         "config": "configs/samurai/sam2.1_hiera_l.yaml",
#     },
# }

# class SAM:
#     def build_model(self, sam_type: str, ckpt_path: str | None = None, predictor_type="img", 
#                     device=torch.device('cuda:0'), use_txt_prompt=False, conservativeness="high"):
#         self.sam_type = sam_type
#         self.ckpt_path = ckpt_path
        
#         # BALANCED: Prevent expansion while maintaining stability
#         conservative_overrides = [
#             # === PREVENT MASK MERGING (MOST IMPORTANT) ===
#             # CRITICAL: These keep objects separate
#             "++model.non_overlap_masks=true",
#             "++model.non_overlap_masks_for_mem_enc=true",
            
#             # === MAINTAIN STABILITY ===
#             # DON'T change sigmoid values - they destabilize tracking
#             # Default values work best: sigmoid_scale=1.0, sigmoid_bias=0.0
            
#             # Keep memory features for stable tracking
#             "++model.use_mask_input_as_output_without_sam=true",  # Use memory for stability
            
#             # DON'T binarize - keep soft probabilities for smooth tracking
#             "++model.binarize_mask_from_pts_for_mem_enc=false",
            
#             # === CONSERVATIVE POSTPROCESSING ===
#             # Disable aggressive hole filling
#             "++model.fill_hole_area=0",
            
#             # Disable dynamic multimask fallback
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=false",
            
#             # === MEMORY CONTROL ===
#             # Use temporal stride to reduce memory influence without breaking model
#             "++model.memory_temporal_stride_for_eval=2",
#         ]
        
#         if conservativeness == "ultra":
#             # For surgical precision - focus on NON-OVERLAP only
#             conservative_overrides = [
#                 "++model.non_overlap_masks=true",
#                 "++model.non_overlap_masks_for_mem_enc=true",
#                 "++model.memory_temporal_stride_for_eval=3",
#                 "++model.fill_hole_area=0",
#                 "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=false",
#             ]
#         elif conservativeness == "medium":
#             # Balanced approach
#             conservative_overrides = [
#                 "++model.non_overlap_masks=true",
#                 "++model.non_overlap_masks_for_mem_enc=true",
#                 "++model.memory_temporal_stride_for_eval=1",
#                 "++model.fill_hole_area=2",
#             ]
        
#         if predictor_type == "img":
#             self.model = build_sam2(
#                 config_file=SAM_MODELS[self.sam_type]["config"], 
#                 ckpt_path=self.ckpt_path, 
#                 device=device,
#                 hydra_overrides_extra=conservative_overrides,
#                 apply_postprocessing=False
#             )
#             self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
#             self.img_predictor = SAM2ImagePredictor(self.model)
            
#         elif predictor_type == "video" or predictor_type == "realtime":
#             self.video_predictor = build_sam2_video_predictor(
#                 SAM_MODELS[self.sam_type]["config"], 
#                 self.ckpt_path, 
#                 device=device,
#                 hydra_overrides_extra=conservative_overrides,
#                 apply_postprocessing=False
#             )

#     def _load_checkpoint(self, model: torch.nn.Module):
#         if self.ckpt_path is None:
#             checkpoint_url = SAM_MODELS[self.sam_type]["url"]
#             state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]
#         else:
#             checkpoint_url = self.ckpt_path  # Ensure checkpoint_url is defined
#             state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)["model"]
#         try:
#             model.load_state_dict(state_dict, strict=True)
#         except Exception as e:
#             raise ValueError(
#                 f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
#                 and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
#                 re-downloading it. Error: {e}"
#             )

#     def generate(self, image_rgb: np.ndarray) -> list[dict]:
#         """
#         Output format
#         SAM2AutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information
#         about the mask:

#         segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
#         area - [int] - the area of the mask in pixels
#         bbox - [List[int]] - the boundary box of the mask in xywh format
#         predicted_iou - [float] - the model's own prediction for the quality of the mask
#         point_coords - [List[List[float]]] - the sampled input point that generated this mask
#         stability_score - [float] - an additional measure of mask quality
#         crop_box - List[int] - the crop of the image used to generate this mask in xywh format
#         """

#         sam2_result = self.mask_generator.generate(image_rgb)
#         return sam2_result

#     def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         self.img_predictor.set_image(image_rgb)
#         masks, scores, logits = self.img_predictor.predict(box=xyxy, multimask_output=False)
#         if len(masks.shape) > 3:
#             masks = np.squeeze(masks, axis=1)
#         return masks, scores, logits

#     def predict_batch(
#         self,
#         images_rgb: list[np.ndarray],
#         xyxy: list[np.ndarray],
#     ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
#         self.img_predictor.set_image_batch(images_rgb)

#         masks, scores, logits = self.img_predictor.predict_batch(box_batch=xyxy, multimask_output=False)

#         masks = [np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks]
#         scores = [np.squeeze(score) for score in scores]
#         logits = [np.squeeze(logit, axis=1) if len(logit.shape) > 3 else logit for logit in logits]
#         return masks, scores, logits