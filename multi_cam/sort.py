import base64
import os
import sys

import shutil
import threading
import queue
import time
from io import BytesIO
from collections import defaultdict

import cv2
import torch
import gc
import numpy as np
import imageio
from PIL import Image
from triton.language import dtype
from icecream import ic
from scipy.spatial.distance import cosine
import torchvision.ops as ops

from models.gdino.models.gdino import GDINO
from models.sam2.sam import SAM
from models.yolo.detection import YOLODetector

from utilities.color import COLOR
import pyrealsense2 as rs
from utilities.utils import save_frames_to_temp_dir, get_object_iou, batch_mask_iou, batch_box_iou, \
    visualize_selected_masks_as_video, filter_mask_outliers
from utilities.ObjectInfoManager import ObjectInfoManager

class Sort(object):
    def __init__(self, sam_type:str="sam2.1_hiera_tiny", model_path:str="./../models/sam2/checkpoints/sam2.1_hiera_large.pt",
                 video_path:str="", output_path:str="", use_txt_prompt:bool=False, max_frames:int=60,
                 first_prompts: list | None = None, save_video=True, device="cuda:0", mode="realtime",
                 yolo_path= "/data/dataset/weights/base_weight/weights/best_wo_specialised_training.pt",
                 conservativeness="high", shared_yolo=None, shared_sam=None, shared_gdino=None):
        self.sam_type = sam_type
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.max_frames = max_frames
        self.first_prompts = first_prompts
        self.save_video = save_video
        self.device = device
        self.mode = mode
        self.yolo_path = yolo_path
        if self.mode == 'img' and not use_txt_prompt:
            raise ValueError("In 'img' mode, use_txt_prompt must be True")

        # Use shared models if provided, otherwise initialize new ones
        if shared_yolo is not None:
            self.yolo = shared_yolo
        else:
            self.yolo = YOLODetector(self.yolo_path, conf_thres=0.45)
        
        if shared_sam is not None:
            self.sam = shared_sam
        else:
            self.sam = SAM()
            self.sam.build_model(self.sam_type, self.model_path, predictor_type=mode, device=device, use_txt_prompt=use_txt_prompt)
        
        if use_txt_prompt:
            if shared_gdino is not None:
                self.gdino = shared_gdino
            else:
                self.gdino = GDINO()
                self.gdino_16 = False
                if not self.gdino_16:
                    print("Building GroundingDINO model...")
                    # self.gdino.build_model(device=device)
        else:
            self.gdino = None

        self.history_frames = []
        self.all_forward_masks = {}
        self.all_final_masks = {}
        self.object_start_frame_idx = {}
        self.object_start_prompts = {}
        self.existing_obj_outputs = []
        self.current_text_prompt = None
        self.last_text_prompt = None
        if self.first_prompts is not None:
            self.prompts = {'prompts': self.first_prompts, 'labels': [None] * len(self.first_prompts), 'scores': [None] * len(self.first_prompts)}
            self.add_new = True
        else:
            self.prompts = {'prompts': [], 'labels': [], 'scores': []}
        self.iou_threshold = 0.3
        self.detection_frequency =10
        self.object_labels = {}
        self.last_known_bboxes = {}

        self.incision_area = None
        self.object_track_history = {}
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})

        self.input_queue = queue.Queue()
        self.drawing = False
        self.add_new = False
        self.ix, self.iy = -1, -1
        self.frame_display = None
        self.height, self.width = None, None
        self.prev_time = 0

        self.lost_obj_ids = set()
        self.prev_obj_ids = set()

        self.roi_align_size = (2, 2)
        self.feature_match_threshold = 0.25
        self.feature_maps = None
        self.existing_features = {}
        
        self._setup_feature_extraction()

    def _setup_feature_extraction(self):
        target_layer_index = -2
        target_layer = self.yolo.model.model.model[target_layer_index]
        
        def hook_fn(module, input, output):
            self.feature_maps = output
        
        target_layer.register_forward_hook(hook_fn)

    def extract_features_from_boxes(self, frame, boxes):
        if len(boxes) == 0:
            return np.array([])
        
        with torch.no_grad():
            results = self.yolo.model(frame)
        
        feature_maps = self.feature_maps.cpu().to(torch.float32)
        fmap_h, fmap_w = feature_maps.shape[2], feature_maps.shape[3]
        
        orig_h, orig_w = frame.shape[:2]
        
        scaled_bboxes = []
        for x1, y1, x2, y2 in boxes:
            x1_f = max(0, int(x1 / orig_w * fmap_w))
            x2_f = max(0, int(x2 / orig_w * fmap_w))
            y1_f = max(0, int(y1 / orig_h * fmap_h))
            y2_f = max(0, int(y2 / orig_h * fmap_h))
            scaled_bboxes.append([x1_f, y1_f, x2_f, y2_f])
        
        scaled_bboxes = torch.tensor(scaled_bboxes, dtype=torch.float32)
        pooled_features = ops.roi_align(feature_maps, [scaled_bboxes], output_size=self.roi_align_size)
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        
        return pooled_features.numpy()

    def match_features(self, new_features, new_boxes, new_labels):
        if len(self.existing_features) == 0 or len(new_features) == 0:
            return new_boxes, new_labels
        
        existing_ids = list(self.existing_features.keys())
        existing_feats = np.array([self.existing_features[oid] for oid in existing_ids])
        
        matched_boxes = []
        matched_labels = []
        used_indices = set()
        
        for new_feat, new_box, new_label in zip(new_features, new_boxes, new_labels):
            best_match_id = None
            best_similarity = -1
            
            for idx, (obj_id, exist_feat) in enumerate(zip(existing_ids, existing_feats)):
                if idx in used_indices:
                    continue
                
                similarity = 1 - cosine(new_feat, exist_feat)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = idx
            
            if best_match_id is None or best_similarity < self.feature_match_threshold:
                matched_boxes.append(new_box)
                matched_labels.append(new_label)
            else:
                used_indices.add(best_match_id)
        
        return matched_boxes, matched_labels
    
    def input_thread(self):
        while True:
            user_input = input()
            self.input_queue.put(user_input)
    
    def set_incision_area(self, polygon_points):
        if len(polygon_points) == 4 and not isinstance(polygon_points[0], (list, tuple)):
            x1, y1, x2, y2 = polygon_points
            self.incision_area = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        else:
            self.incision_area = np.array(polygon_points, dtype=np.int32)
        print(f"Incision area set: {self.incision_area}")
    
    def get_mask_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask
    
    def add_to_state(self, predictor, state, prompts, start_with_0=False):
        frame_idx = 0 if start_with_0 else state["num_frames"]-1
        for id, item in enumerate(prompts['prompts']):
            if isinstance(item, np.ndarray) and item.dtype == bool:
                predictor.add_new_mask(state, mask=item, frame_idx=frame_idx, obj_id=id)
            elif len(item) == 4:
                x1, y1, x2, y2 = item
                cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                predictor.add_new_points_or_box(state, box=item, frame_idx=frame_idx, obj_id=id)
            elif len(item) == 2:
                x, y = item
                cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
                pt = torch.tensor([[x, y]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points_or_box(state, points=pt, labels=lbl, frame_idx=frame_idx, obj_id=id)
    def draw_mask_and_bbox(self, frame, mask, bbox, obj_id):
        mask_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_img[mask] = COLOR[obj_id % len(COLOR)]
        frame[:] = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        
        label_text = f"obj_{obj_id}"
        if obj_id in self.object_labels:
            label_text = f"obj_{obj_id}_{self.object_labels[obj_id]}"
        
        # if self.is_inside_incision(bbox):
        #     label_text += " [IN]"
        
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)

    def track_and_visualize(self, predictor, state, frame, writer):
        if (any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"]))) or
            any(len(state["mask_inputs_per_obj"][i]) > 0 for i in range(len(state["mask_inputs_per_obj"])))):
            for frame_idx, obj_ids, masks in predictor.propagate_in_frame(state, state["num_frames"] - 1):
                self.existing_obj_outputs = []
                self.current_frame_masks = []
                current_obj_boxes = []
                for obj_id, mask in zip(obj_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0
                    mask = filter_mask_outliers(mask)
                    self.current_frame_masks.append(mask)
                    nonzero = np.argwhere(mask)
                    if nonzero.size == 0:
                        continue
                    else:
                        y_min, x_min = nonzero.min(axis=0)
                        y_max, x_max = nonzero.max(axis=0)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        self.last_known_bboxes[obj_id] = bbox
                        current_obj_boxes.append([x_min, y_min, x_max, y_max])
                    self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
                    category_name = self.object_labels.get(obj_id, "unknown")
                    self.existing_obj_outputs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                
                if len(current_obj_boxes) > 0:
                    current_features = self.extract_features_from_boxes(frame, current_obj_boxes)
                    for obj_id, feat in zip(obj_ids, current_features):
                        self.existing_features[obj_id] = feat
                
                self.prompts['prompts'] = self.existing_obj_outputs.copy()
        
        
        frame_text = f"Frame: {state['num_frames']}"
        # print("=="*20)
        # print("STATS FOR FRAME: #",state['num_frames'])
        # print("=="*20)
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = self.width - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, frame_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if writer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)
    def convert_boxes_to_masks(self, frame, boxes):
        if len(boxes) == 0:
            return [], []
        
        if not hasattr(self.sam, 'img_predictor'):
            print("Initializing SAM2 image predictor for mask conversion...")
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam.img_predictor = SAM2ImagePredictor(self.sam.model)
        
        self.sam.img_predictor.set_image(frame)
        
        masks = []
        scores = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            box_xyxy = np.array([x1, y1, x2, y2])
            
            mask, score, _ = self.sam.img_predictor.predict(
                box=box_xyxy,
                multimask_output=False
            )
            
            mask = mask[0].astype(bool)
            
            constrained_mask = np.zeros_like(mask, dtype=bool)
            constrained_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            
            masks.append(constrained_mask)
            scores.append(float(score[0]))
        
        return masks, scores
    
    def track(self):

        predictor = self.sam.video_predictor

        if self.mode == "realtime":
            print("Start with realtime mode.")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            get_frame = lambda: np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
        elif self.mode == "video":
            print("Start with video mode.")
            cap = cv2.VideoCapture(self.video_path)
            ret, color_image = cap.read()
            get_frame = lambda: cap.read()
        else:
            raise ValueError("The mode is not supported in this method.")

        self.height, self.width = color_image.shape[:2]

        if self.save_video:
            writer = imageio.get_writer(self.output_path, fps=5)
        else:
            writer = None

        threading.Thread(target=self.input_thread, daemon=True).start()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state_from_numpy_frames([color_image], offload_state_to_cpu=False, offload_video_to_cpu=False)
            while True:
                if self.mode == "realtime":
                    frame = get_frame()
                else:
                    ret, frame = get_frame()
                    if not ret:
                        break
                self.frame_display = frame.copy()

                if not self.input_queue.empty():
                    self.current_text_prompt = self.input_queue.get()

                if self.current_text_prompt is not None:
                    if (state['num_frames']-1) % self.detection_frequency == 0 or self.last_text_prompt is None:
                        detection= self.yolo.detect([frame], classes= [14])[0] #7,13,
                        
                        scores = detection['scores'].cpu().numpy()
                        labels = detection['labels']
                        boxes = detection['boxes'].cpu().numpy().tolist()

                        boxes_np = np.array(boxes, dtype=np.int32)
                        labels_np = np.array(labels)
                        scores_np = np.array(scores)
                        filter_mask = scores > 0.3
                        valid_boxes = boxes_np[filter_mask]
                        valid_labels = labels_np[filter_mask]
                        valid_scores = scores_np[filter_mask]

                        if self.last_text_prompt != self.current_text_prompt:
                            self.prompts['prompts'].extend(valid_boxes)
                            self.prompts['labels'].extend(valid_labels)
                            self.prompts['scores'].extend(valid_scores)
                            self.add_new = True
                        elif len(valid_boxes) > 0:
                            print(f"Checking {len(valid_boxes)} YOLO detections with feature matching...")
                            
                            valid_masks, mask_scores = self.convert_boxes_to_masks(frame, valid_boxes)
                            new_features = self.extract_features_from_boxes(frame, valid_boxes)
                            
                            matched_boxes, matched_labels = self.match_features(
                                new_features, valid_masks, valid_labels
                            )
                            
                            if len(matched_boxes) > 0:
                                print(f"  Adding {len(matched_boxes)} new detections after feature matching")
                                self.prompts['prompts'].extend(matched_boxes)
                                self.prompts['labels'].extend(matched_labels)
                                self.prompts['scores'].extend([None] * len(matched_boxes))
                                self.add_new = True
                            else:
                                print(f"  No new detections to add - all {len(valid_boxes)} detections matched existing objects")
    
                    self.last_text_prompt = self.current_text_prompt

                if self.add_new:
                    existing_obj_ids = set(state["obj_ids"])
                    predictor.reset_state(state)
                    self.add_to_state(predictor, state, self.prompts)
                    current_obj_ids = set(state["obj_ids"])
                    newly_added_ids = current_obj_ids - existing_obj_ids
                predictor.append_frame_to_inference_state(state, frame)
                self.track_and_visualize(predictor, state, frame, writer)
                if self.add_new:
                    for idx, obj_id in enumerate(newly_added_ids):
                        self.object_start_frame_idx[obj_id] = state['num_frames'] - 1
                        
                        prompt_idx = len(self.prompts['prompts']) - len(newly_added_ids) + idx
                        if prompt_idx < len(self.prompts['labels']) and self.prompts['labels'][prompt_idx] is not None:
                            class_id = self.prompts['labels'][prompt_idx]
                            class_name = self.yolo.model.names.get(class_id, f"class_{class_id}")
                            self.object_labels[obj_id] = class_name
                            print(f"  Object {obj_id} assigned class: {class_name}")
                        else:
                            self.object_labels[obj_id] = "unknown"
                    
                    self.add_new = False

                if state["num_frames"] % self.max_frames == 0:
                    if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                        predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                    predictor.release_old_frames(state)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        if self.mode == "realtime":
            pipeline.stop()
        else:
            cap.release()
        
        self.print_final_statistics()
        
        if writer:
            writer.close()
        cv2.destroyAllWindows()
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

    def initialize_tracking(self):
        """Initialize tracking state without starting the main loop"""
        self.predictor = self.sam.video_predictor
        
        if self.mode == "video":
            self.cap = cv2.VideoCapture(self.video_path)
            ret, self.first_frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Could not read from {self.video_path}")
        else:
            raise ValueError("Only 'video' mode supported for frame-by-frame processing")
        
        self.height, self.width = self.first_frame.shape[:2]
        
        if self.save_video:
            self.writer = imageio.get_writer(self.output_path, fps=5)
        else:
            self.writer = None
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.state = self.predictor.init_state_from_numpy_frames(
                [self.first_frame], 
                offload_state_to_cpu=False, 
                offload_video_to_cpu=False
            )
        
        self.initialized = True
    
    def process_frame(self):
        """Process a single frame. Returns False if video ended."""
        if not self.initialized:
            self.initialize_tracking()
        
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        self.frame_display = frame.copy()
        predictor= self.predictor
        state= self.state
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if not self.input_queue.empty():
                    self.current_text_prompt = self.input_queue.get()

            if self.current_text_prompt is not None:
                # state = predictor.init_state_from_numpy_frames([frame], offload_state_to_cpu=False, offload_video_to_cpu=False)

                if (state['num_frames']-1) % self.detection_frequency == 0 or self.last_text_prompt is None:
                    detection= self.yolo.detect([frame], classes= [14])[0] #7,13,

                    scores = detection['scores'].cpu().numpy()
                    labels = detection['labels']
                    boxes = detection['boxes'].cpu().numpy().tolist()

                    boxes_np = np.array(boxes, dtype=np.int32)
                    labels_np = np.array(labels)
                    scores_np = np.array(scores)
                    filter_mask = scores > 0.3
                    valid_boxes = boxes_np[filter_mask]
                    valid_labels = labels_np[filter_mask]
                    valid_scores = scores_np[filter_mask]

                    if self.last_text_prompt != self.current_text_prompt:
                        self.prompts['prompts'].extend(valid_boxes)
                        self.prompts['labels'].extend(valid_labels)
                        self.prompts['scores'].extend(valid_scores)
                        self.add_new = True
                    elif len(valid_boxes) > 0:
                        print(f"Checking {len(valid_boxes)} YOLO detections with feature matching...")
                        
                        valid_masks, mask_scores = self.convert_boxes_to_masks(frame, valid_boxes)
                        new_features = self.extract_features_from_boxes(frame, valid_boxes)
                        
                        matched_boxes, matched_labels = self.match_features(
                            new_features, valid_masks, valid_labels
                        )
                        
                        if len(matched_boxes) > 0:
                            print(f"  Adding {len(matched_boxes)} new detections after feature matching")
                            self.prompts['prompts'].extend(matched_boxes)
                            self.prompts['labels'].extend(matched_labels)
                            self.prompts['scores'].extend([None] * len(matched_boxes))
                            self.add_new = True
                        else:
                            print(f"  No new detections to add - all {len(valid_boxes)} detections matched existing objects")

                self.last_text_prompt = self.current_text_prompt

            if self.add_new:
                existing_obj_ids = set(state["obj_ids"])
                predictor.reset_state(state)
                self.add_to_state(predictor, state, self.prompts)
                current_obj_ids = set(state["obj_ids"])
                newly_added_ids = current_obj_ids - existing_obj_ids
            predictor.append_frame_to_inference_state(state, frame)
            self.track_and_visualize(predictor, state, frame, self.writer)
            if self.add_new:
                for idx, obj_id in enumerate(newly_added_ids):
                    self.object_start_frame_idx[obj_id] = state['num_frames'] - 1
                    
                    prompt_idx = len(self.prompts['prompts']) - len(newly_added_ids) + idx
                    if prompt_idx < len(self.prompts['labels']) and self.prompts['labels'][prompt_idx] is not None:
                        class_id = self.prompts['labels'][prompt_idx]
                        class_name = self.yolo.model.names.get(class_id, f"class_{class_id}")
                        self.object_labels[obj_id] = class_name
                        print(f"  Object {obj_id} assigned class: {class_name}")
                    else:
                        self.object_labels[obj_id] = "unknown"
                
                self.add_new = False

            if state["num_frames"] % self.max_frames == 0:
                if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                    predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                predictor.release_old_frames(state)
        return True


    def cleanup_tracking(self):
        """Clean up resources after tracking"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()
        if hasattr(self, 'predictor') and hasattr(self, 'state'):
            del self.predictor, self.state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()



    
if __name__ == "__main__":
    sam_type= "sam2.1_hiera_large"
    model_path= "./../models/sam2/checkpoints/sam2.1_hiera_large.pt"
    video_path= "/data/dataset/demo_video/output.mp4"
    output_path= "forward_tracked_video.mp4"
    video_mode= "video"

    tracker = Sort(sam_type="sam2.1_hiera_large",
                            model_path="./../models/sam2/checkpoints/sam2.1_hiera_large.pt",
                            video_path="/data/dataset/demo_video/output.mp4",
                            output_path="forward_tracked_video.mp4",
                            mode="video",
                            save_video=True,
                            use_txt_prompt=True)
    
    tracker.set_incision_area([700, 1040+200, 1490-200, 1790])
    
    tracker.current_text_prompt = 'car'
    tracker.track()