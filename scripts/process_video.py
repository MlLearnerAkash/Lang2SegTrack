import argparse
import os
import os.path as osp

import imageio
import numpy as np
import cv2
import torch
import gc
import sys

from utils.color import COLOR
from utils.utils import determine_model_cfg, bbox_process, prepare_frames_or_path, save_frames_to_temp_dir
from models.sam2.sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
import imageio.v3 as iio

def process_video_in_chunks(args, initial_bbox_list: list[list[float]], chunk_seconds: int = 2):
    cap = cv2.VideoCapture(args.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    device = args.device

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device)

    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    if mask_dir:
        mask_dir.mkdir(exist_ok=True, parents=True)

    if args.save_to_video:
        writer = imageio.get_writer(args.video_output_path, fps=fps)

    chunk_size = chunk_seconds * fps
    current_frame_idx = 0
    bbox_list = initial_bbox_list

    while current_frame_idx < total_frames:
        # Step 1: 读取 chunk 的帧
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        for _ in range(min(chunk_size, total_frames - current_frame_idx)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) == 0:
            break
        frames_folder = save_frames_to_temp_dir(frames)
        # Step 2: 准备和初始化 predictor
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            state = predictor.init_state(frames_folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
            prompts = bbox_process(bbox_list)
            for idx, (bbox, _) in enumerate(prompts.values()):
                _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=idx)

            for i, object_ids, masks in predictor.propagate_in_video(state, disable_display=False):
                mask_to_vis = {}
                bbox_to_vis = {}
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    binary_mask = mask > 0.0
                    non_zero_indices = np.argwhere(binary_mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    mask_to_vis[obj_id] = binary_mask
                    bbox_to_vis[obj_id] = bbox

                    if mask_dir:
                        mask_path = mask_dir / f'OBJ{obj_id:02}_{current_frame_idx+i:04}.png'
                        iio.imwrite(mask_path, binary_mask.astype(np.uint8) * 255)

                if args.save_to_video:
                    img = frames[i]
                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = COLOR[obj_id % len(COLOR)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)

                    for obj_id, bbox in bbox_to_vis.items():
                        label = f"obj_{obj_id}"
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), COLOR[obj_id % len(COLOR)], 2)
                        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.append_data(img)

            # Step 3: 更新 bbox_list（取最后一帧的检测框）
            bbox_list = list(bbox_to_vis.values())

            del state
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            gc.collect()

        current_frame_idx += len(frames)

    cap.release()
    if args.save_to_video:
        writer.close()
    del predictor

def main(args, bbox_list:list[list[float]]):
    device = args.device
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device,)
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = bbox_process(bbox_list)

    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(".jpg")])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    if args.save_to_video:
        writer = imageio.get_writer(args.video_output_path, fps=30)

    mask_dir = args.mask_dir
    if args.mask_dir is not None:
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(exist_ok=True, parents=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
        all_masks = []
        for idx, (bbox, track_label) in enumerate(prompts.values()):
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=idx)
            all_masks.append(masks)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state, disable_display=False):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask
                if mask_dir is not None:
                    mask_path = mask_dir / f'OBJ{obj_id:02}_{frame_idx:04}.png'
                    iio.imwrite(mask_path, mask.astype(np.uint8) * 255)

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = COLOR[obj_id % len(COLOR)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.4, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    label = f"obj_{obj_id}"
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), COLOR[obj_id % len(COLOR)], 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                COLOR[obj_id % len(COLOR)], 2)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                writer.append_data(img)
        if (state["num_frames"] - 1) % 60 and len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
            predictor.append_frame_as_cond_frame(state, state["num_frames"] - 1)
        predictor.release_old_frames(state, state["num_frames"] - 1, 60, 0, release_images=True)

        if args.save_to_video:
            writer.close()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="06.mp4")
    parser.add_argument("--model_path", default="models/sam2/checkpoints/sam2.1_hiera_tiny.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="processed_video.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--mask_dir", help="If provided, save mask images to the given directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    process_video_in_chunks(args, [[607.75244140625, 126.3901596069336, 791.4397583007812, 356.09332275390625],
                          [612.8888549804688, 284.82171630859375, 692.8299560546875, 362.6283874511719],
                          [638.74560546875, 598.923095703125, 687.992431640625, 663.7514038085938],
                          [733.8016357421875, 612.997314453125, 782.0617065429688, 671.3180541992188],
                          [639.07861328125, 345.004638671875, 775.4666748046875, 615.2918701171875],
                          [797.278564453125, 261.87451171875, 818.1863403320312, 283.78070068359375]])
    # process_video_in_chunks(args, initial_bbox_list=[[364.53216552734375, 426.52178955078125, 437.9630126953125, 500.3838195800781],
    #                                                  [260.3464050292969, 0.1987624168395996, 445.94500732421875, 198.92283630371094],
    #                                                  [829.1287841796875, 292.64874267578125, 999.095703125, 544.427978515625],
    #                                                  [568.9965209960938, 291.2940673828125, 738.1513671875, 544.0554809570312]])