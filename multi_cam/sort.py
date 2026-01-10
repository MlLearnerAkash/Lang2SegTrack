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


from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import linear_sum_assignment

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
        self.detection_frequency =30
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


        self.active_yolo_track_ids = set()  # Track IDs seen across all frames
        self.tracking_lookback_frames = 5 

        
        self._setup_feature_extraction()

        #for kalman filter
        self.kalman_filters = {}
        self.object_bboxes = {}

    def _setup_feature_extraction(self):
        target_layer_index = -2
        target_layer = self.yolo.model.model.model[target_layer_index]
        
        def hook_fn(module, input, output):
            self.feature_maps = output
        
        target_layer.register_forward_hook(hook_fn)

    def _init_kalman_filter(self, bbox):
        """Initialize Kalman filter for bbox tracking"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [center_x, center_y, width, height]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.x = np.array([[cx], [cy], [w], [h]], dtype=float)
        kf.F = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        kf.P *= 1000.0
        kf.R = np.eye(2) * 10.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1, block_size=2)
        
        return kf

    def _predict_bbox(self, obj_id):
        """Predict bbox from Kalman filter"""
        if obj_id not in self.kalman_filters:
            return self.object_bboxes.get(obj_id)
        
        kf = self.kalman_filters[obj_id]
        kf.predict()
        
        cx, cy, w, h = kf.x.flatten()
        x1 = max(0, cx - w / 2.0)
        y1 = max(0, cy - h / 2.0)
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        return np.array([x1, y1, x2, y2])

    def _update_kalman_filter(self, obj_id, bbox):
        """Update Kalman filter with new measurement"""
        if obj_id not in self.kalman_filters:
            self.kalman_filters[obj_id] = self._init_kalman_filter(bbox)
        
        kf = self.kalman_filters[obj_id]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        z = np.array([[cx], [cy]])
        
        kf.update(z)

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _bbox_from_mask(self, mask):
        """Convert boolean mask to bbox [x1, y1, x2, y2]"""
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            nonzero = np.argwhere(mask)
            if nonzero.size == 0:
                return None
            y_min, x_min = nonzero.min(axis=0)
            y_max, x_max = nonzero.max(axis=0)
            return np.array([x_min, y_min, x_max, y_max])
        else:
            # Already a bbox [x1, y1, x2, y2]
            return np.array(mask, dtype=float)


    def _calculate_combined_cost(self, obj_id, detection_bbox, detection_feature,
                                 w_iou=0.4, w_feature=0.3, w_motion=0.3,
                                 iou_threshold=0.1):
        """
        Calculate combined cost for associating detection to track
        
        Args:
            obj_id: Tracking object ID
            detection_bbox: Detection bbox [x1, y1, x2, y2]
            detection_feature: Feature vector for detection
            w_iou: Weight for IoU cost
            w_feature: Weight for feature cost
            w_motion: Weight for motion cost
            iou_threshold: Minimum IoU for gating
        
        Returns:
            cost: Combined cost (lower is better), or np.inf if invalid
        """
        # 1. Predict track position using Kalman filter
        predicted_bbox = self._predict_bbox(obj_id)
        if predicted_bbox is None:
            if obj_id in self.object_bboxes:
                predicted_bbox = self.object_bboxes[obj_id]
            else:
                # No prediction and no history - can only use feature matching
                # This happens on first frame detection
                if obj_id in self.existing_features and detection_feature is not None:
                    feature_cost = cosine(self.existing_features[obj_id], detection_feature)
                    feature_cost = min(1.0, feature_cost / 2.0)
                    return 0.5 * feature_cost  # Use only feature cost
                else:
                    return np.inf  # Truly can't match
        # 2. IoU-based gating (filter out unlikely matches early)
        iou = self._calculate_iou(predicted_bbox, detection_bbox)
        # IMPORTANT: Use stricter gating only if we have good prediction
        # If using fallback bbox, be more lenient
        effective_iou_threshold = iou_threshold
        if obj_id not in self.kalman_filters:
            effective_iou_threshold = iou_threshold * 0.5  # Be more lenient with fallback
        
        if iou < effective_iou_threshold:
            return np.inf  # Don't consider this association

        iou_cost = 1.0 - iou  # Cost: 0 (perfect) to 1 (no overlap)
        
        # 3. Feature matching cost (cosine distance)
        if obj_id in self.existing_features and detection_feature is not None:
            feature_cost = cosine(self.existing_features[obj_id], detection_feature)
        else:
            feature_cost = 0.5  # Neutral cost if features unavailable
        
        # Clamp feature cost to [0, 1]
        feature_cost = min(1.0, feature_cost / 2.0)
        
        # 4. Motion consistency cost (center distance normalized by track size)
        pred_cx = (predicted_bbox[0] + predicted_bbox[2]) / 2.0
        pred_cy = (predicted_bbox[1] + predicted_bbox[3]) / 2.0
        det_cx = (detection_bbox[0] + detection_bbox[2]) / 2.0
        det_cy = (detection_bbox[1] + detection_bbox[3]) / 2.0
        
        center_distance = np.sqrt((pred_cx - det_cx)**2 + (pred_cy - det_cy)**2)
        
        # Normalize by track size
        track_w = predicted_bbox[2] - predicted_bbox[0]
        track_h = predicted_bbox[3] - predicted_bbox[1]
        track_size = np.sqrt(track_w * track_h)
        
        motion_cost = min(1.0, center_distance / (track_size + 1e-6))
        # 5. Weighted combination
        total_cost = (w_iou * iou_cost + 
                     w_feature * feature_cost + 
                     w_motion * motion_cost)
        
        return total_cost
   
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

    def extract_features_from_masks(self, frame, masks, force_yolo_run=True):
        """Extract features consistently from masks using same YOLO forward pass"""
        if len(masks) == 0:
            return np.array([])
        
        # Force a fresh YOLO inference to get current feature maps
        if force_yolo_run:
            with torch.no_grad():
                _ = self.yolo.model(frame)  # This updates self.feature_maps
        
        # Convert masks to bounding boxes
        boxes = []
        for mask in masks:
            nonzero = np.argwhere(mask)
            if nonzero.size > 0:
                y_min, x_min = nonzero.min(axis=0)
                y_max, x_max = nonzero.max(axis=0)
                boxes.append([x_min, y_min, x_max, y_max])
        
        if len(boxes) == 0:
            return np.array([])
        
        return self.extract_features_from_boxes(frame, boxes)

    def match_features(self, new_features, new_boxes, new_labels):
        if len(self.existing_features) == 0 or len(new_features) == 0:
            return new_boxes, new_labels
        
        existing_ids = list(self.existing_features.keys())
        
        matched_boxes = []
        matched_labels = []
        used_indices = set()
        
        for new_feat, new_box, new_label in zip(new_features, new_boxes, new_labels):
            # Normalize features to unit length
            new_feat_norm = new_feat / (np.linalg.norm(new_feat) + 1e-6)
            
            best_match_id = None
            best_distance = float('inf')
            
            for idx, obj_id in enumerate(existing_ids):
                if idx in used_indices:
                    continue
                
                exist_feat = self.existing_features[obj_id]
                exist_feat_norm = exist_feat / (np.linalg.norm(exist_feat) + 1e-6)
                
                # Euclidean on normalized features
                distance = np.linalg.norm(new_feat_norm - exist_feat_norm)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = idx
            
            # For normalized features, threshold ~0.3-0.5 works well
            if best_match_id is None or best_distance < self.feature_match_threshold:
                matched_boxes.append(new_box)
                matched_labels.append(new_label)
            else:
                used_indices.add(best_match_id)
        
        return matched_boxes, matched_labels
    def match_features_combined(self, new_features, new_boxes, new_labels):
        """
        Enhanced matching using combined cost function with fallbacks
        """
        if len(self.existing_features) == 0 or len(new_features) == 0:
            return new_boxes, new_labels
        
        existing_ids = list(self.existing_features.keys())
        n_tracks = len(existing_ids)
        n_detections = len(new_features)
        
        # Step 1: Convert all detections to bboxes
        detection_bboxes = []
        valid_det_indices = []
        
        for d_idx, det_box in enumerate(new_boxes):
            bbox = self._bbox_from_mask(det_box)
            if bbox is not None:
                detection_bboxes.append(bbox)
                valid_det_indices.append(d_idx)
        
        # If no valid detections, all are new
        if len(detection_bboxes) == 0:
            print(f"  No valid detection bboxes - treating all {len(new_boxes)} as new tracks")
            return new_boxes, new_labels
        
        n_detections_valid = len(detection_bboxes)
        
        # Step 2: Build cost matrix using combined cost
        cost_matrix = np.full((n_tracks, n_detections_valid), np.inf)
        
        for t_idx, obj_id in enumerate(existing_ids):
            for d_idx, det_bbox in enumerate(detection_bboxes):
                cost = self._calculate_combined_cost(
                    obj_id,
                    det_bbox,
                    new_features[valid_det_indices[d_idx]],
                    w_iou=0.2,
                    w_feature=0.6,
                    w_motion=0.2,
                    iou_threshold=0.1
                )
                cost_matrix[t_idx, d_idx] = cost
        
        print(f"  Cost matrix shape: {cost_matrix.shape}")
        print(f"  Inf count: {np.sum(np.isinf(cost_matrix))}/{cost_matrix.size}")
        
        # Step 3: Check if cost matrix is feasible
        if np.all(np.isinf(cost_matrix)):
            print(f"  ⚠ Cost matrix all inf - no spatial overlap detected")
            print(f"    Using feature-only matching as fallback...")
            
            # Fallback: Use ONLY feature similarity
            cost_matrix = np.full((n_tracks, n_detections_valid), np.inf)
            
            for t_idx, obj_id in enumerate(existing_ids):
                if obj_id not in self.existing_features:
                    continue
                
                track_feat = self.existing_features[obj_id]
                
                for d_idx, det_feat in enumerate(new_features[valid_det_indices]):
                    feature_cost = cosine(track_feat, det_feat)
                    feature_cost = min(1.0, feature_cost / 2.0)
                    cost_matrix[t_idx, d_idx] = feature_cost
            
            # If still all inf, all detections are new
            if np.all(np.isinf(cost_matrix)):
                print(f"  ⚠ Feature matching also failed - treating all as new tracks")
                return new_boxes, new_labels
            
            print(f"  ✓ Using feature-only cost matrix")
        
        # Step 4: Hungarian algorithm for optimal assignment
        try:
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            print(f"  ✗ Hungarian algorithm failed: {e}")
            return new_boxes, new_labels
        
        # Step 5: Extract matches
        matched_detection_indices = set()
        max_cost_threshold = 0.7  # Tunable: higher = more permissive
        
        for t_idx, d_idx in zip(track_indices, detection_indices):
            cost = cost_matrix[t_idx, d_idx]
            
            # Accept only if cost is below threshold and not infinity
            if cost < max_cost_threshold and not np.isinf(cost):
                obj_id = existing_ids[t_idx]
                det_bbox = detection_bboxes[d_idx]
                det_feat = new_features[valid_det_indices[d_idx]]
                
                # Update track
                self.existing_features[obj_id] = det_feat
                self.object_bboxes[obj_id] = det_bbox
                self._update_kalman_filter(obj_id, det_bbox)
                
                matched_detection_indices.add(d_idx)
        
        # Step 6: Unmatched detections become new tracks
        unmatched_boxes = []
        unmatched_labels = []
        
        for orig_idx in range(n_detections):
            # Check if this original detection was matched
            matched = False
            for valid_idx, orig_det_idx in enumerate(valid_det_indices):
                if orig_det_idx == orig_idx and valid_idx in matched_detection_indices:
                    matched = True
                    break
            
            if not matched:
                unmatched_boxes.append(new_boxes[orig_idx])
                unmatched_labels.append(new_labels[orig_idx])
        
        print(f"  Matched: {len(matched_detection_indices)} | New: {len(unmatched_boxes)}")
        return unmatched_boxes, unmatched_labels
    
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

                    
                    category_name = self.object_labels.get(obj_id, "unknown")
                    self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
                    self.update_counting(obj_id, bbox, category_name)
                    self.existing_obj_outputs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                
                if len(current_obj_boxes) > 0:
                    current_features = self.extract_features_from_boxes(frame, current_obj_boxes)
                    # current_features= self.extract_features_from_masks(
                    #                                                 frame, self.current_frame_masks, force_yolo_run=True
                    #                                             )
                    obj_ids_list = list(obj_ids) if not isinstance(obj_ids, list) else obj_ids
                    for idx, obj_id in enumerate(obj_ids_list):
                        if idx < len(current_obj_boxes):
                            self.existing_features[obj_id] = current_features[idx]
                            # NOTE: Updating Kalman filter
                            self._update_kalman_filter(obj_id, np.array(current_obj_boxes[idx]))
               
                
                self.prompts['prompts'] = self.existing_obj_outputs.copy()
        
        self.draw_incision_area(frame)
        class_count= self.draw_counting_stats(frame)

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
        return class_count

    
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
    
    def is_inside_incision(self, bbox):
        if self.incision_area is None:
            return False
        
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        result = cv2.pointPolygonTest(self.incision_area, (int(center_x), int(center_y)), False)
        return result >= 0
    
    def update_counting(self, obj_id, bbox, category_name):
            if self.incision_area is None:
                return
            
            x, y, w, h = bbox
            current_centroid = (int(x + w // 2), int(y + h // 2))
            
            if obj_id not in self.object_track_history:
                self.object_track_history[obj_id] = []
            
            self.object_track_history[obj_id].append(current_centroid)
            
            if len(self.object_track_history[obj_id]) > 30:
                self.object_track_history[obj_id].pop(0)
            
            if len(self.object_track_history[obj_id]) < 2:
                return
            
            prev_centroid = self.object_track_history[obj_id][-2]
            
            is_inside_now = self.is_centroid_inside_incision(current_centroid)
            was_inside = self.is_centroid_inside_incision(prev_centroid)
            
            if is_inside_now and not was_inside:
                self.classwise_count[category_name]["IN"] += 1
                print(f"  {category_name} obj_{obj_id} ENTERED incision (IN count: {self.classwise_count[category_name]['IN']})")
                
            elif not is_inside_now and was_inside:
                self.classwise_count[category_name]["OUT"] += 1
                print(f"  {category_name} obj_{obj_id} EXITED incision (OUT count: {self.classwise_count[category_name]['OUT']})")

   
    def is_centroid_inside_incision(self, centroid):
        if self.incision_area is None:
            return False
        
        result = cv2.pointPolygonTest(self.incision_area, centroid, False)
        return result >= 0
    
    def draw_incision_area(self, frame):
        if self.incision_area is not None:
            cv2.polylines(frame, [self.incision_area], True, (0, 255, 255), 3)
            cv2.putText(frame, "INCISION AREA", 
                       tuple(self.incision_area[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_counting_stats(self, frame):
        if self.incision_area is None:
            return
        
        y_offset = 70
        for category_name in sorted(self.classwise_count.keys()):
            in_count = self.classwise_count[category_name]["IN"]
            out_count = self.classwise_count[category_name]["OUT"]
            
            stats_text = f"{category_name}: IN={in_count} OUT={out_count}"
            cv2.putText(frame, stats_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30
        return self.classwise_count


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
        #NOTE: Tracking inference.
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        #     if not self.input_queue.empty():
        #             self.current_text_prompt = self.input_queue.get()

        #     if self.current_text_prompt is not None:
        #         current_frame_num = state['num_frames'] - 1
                
        #         # Only process at checkpoint intervals
        #         if current_frame_num % self.detection_frequency == 0 or self.last_text_prompt is None:
        #             print(f"\n🎯 YOLO Tracking Checkpoint at Frame {current_frame_num}")
        #             print(f"   Tracking last {self.tracking_lookback_frames} frames...")
                    
        #             # Step 1: Calculate lookback frame range
        #             start_frame = max(0, current_frame_num - self.tracking_lookback_frames + 1)
                    
        #             # Save current video position
        #             current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    
        #             # Step 2: Reset tracker to start fresh for this interval
        #             print(f"   🔄 Resetting YOLO tracker for fresh interval")
        #             _ = self.yolo.model.track(frame, persist=False, classes=[14], conf=0.3)
                    
        #             # Step 3: Track objects across the lookback window
        #             frame_track_ids = {}  # {frame_idx: set(track_ids)}
        #             all_track_detections = {}  # {track_id: {'bbox': [...], 'class_id': ..., 'last_frame': ...}}
                    
        #             for frame_idx in range(start_frame, current_frame_num + 1):
        #                 # Seek to specific frame
        #                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        #                 ret, lookback_frame = self.cap.read()
                        
        #                 if not ret:
        #                     print(f"  ⚠️ Could not read frame {frame_idx}")
        #                     continue
                        
        #                 # Run tracking with persist=True to maintain IDs across this 5-frame window
        #                 track_results = self.yolo.model.track(
        #                     lookback_frame,
        #                     persist=True,
        #                     classes=[14],
        #                     conf=0.85,
        #                     tracker="/data/opervu/ws/ultralytics/ultralytics/cfg/trackers/botsort.yaml"
        #                 )
                        
        #                 # Extract track IDs from this frame
        #                 current_frame_tracks = set()
                        
        #                 if track_results[0].boxes is not None and len(track_results[0].boxes) > 0:
        #                     for box in track_results[0].boxes:
        #                         if box.id is None:
        #                             continue
                                
        #                         track_id = int(box.id)
        #                         current_frame_tracks.add(track_id)
                                
        #                         # Store/update track info with LATEST position
        #                         bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
        #                         class_id = int(box.cls)
        #                         conf = float(box.conf)
                                
        #                         all_track_detections[track_id] = {
        #                             'bbox': bbox,
        #                             'class_id': class_id,
        #                             'conf': conf,
        #                             'last_frame': frame_idx  # Update to latest frame
        #                         }
                        
        #                 # Track IDs in each frame
        #                 frame_track_ids[frame_idx] = current_frame_tracks
                    
        #             # Step 4: Detect NEW tracks (track IDs that appear in later frames but not earlier)
        #             all_track_ids_in_interval = set()
        #             new_track_ids = set()
                    
        #             # Collect all track IDs seen
        #             for track_ids in frame_track_ids.values():
        #                 all_track_ids_in_interval.update(track_ids)
                    
        #             # Find tracks that are NEW (not in first frame or earlier frames)
        #             first_frame_tracks = frame_track_ids.get(start_frame, set())
                    
        #             for frame_idx in sorted(frame_track_ids.keys()):
        #                 current_tracks = frame_track_ids[frame_idx]
        #                 tracks_before = set()
                        
        #                 # Get all tracks seen BEFORE this frame
        #                 for prev_frame_idx in frame_track_ids.keys():
        #                     if prev_frame_idx < frame_idx:
        #                         tracks_before.update(frame_track_ids[prev_frame_idx])
                        
        #                 # NEW tracks = in current frame but NOT in any previous frame
        #                 newly_appeared = current_tracks - tracks_before
        #                 new_track_ids.update(newly_appeared)
                    
        #             # Step 5: Restore video position to current frame
        #             self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    
        #             # Step 6: Prepare new tracks for SAM2 (with LAST KNOWN position)
        #             new_track_boxes = []
        #             new_track_labels = []
                    
        #             print(f"   📊 Interval Analysis:")
        #             print(f"      Frames scanned: {start_frame} to {current_frame_num}")
        #             print(f"      First frame tracks: {len(first_frame_tracks)}")
        #             print(f"      Total unique tracks: {len(all_track_ids_in_interval)}")
        #             print(f"      NEW tracks detected: {len(new_track_ids)}")
                    
        #             if len(new_track_ids) > 0:
        #                 print(f"   📝 New Track Details (using LAST position):")
        #                 for track_id in sorted(new_track_ids):
        #                     track_info = all_track_detections[track_id]
        #                     new_track_boxes.append(track_info['bbox'])
        #                     new_track_labels.append(track_info['class_id'])
                            
        #                     print(f"      🆕 Track {track_id}: last seen@frame{track_info['last_frame']}, "
        #                         f"bbox={track_info['bbox']}, conf={track_info['conf']:.2f}")
                    
        #             # Step 7: Add new tracks to SAM2 prompts
        #             if len(new_track_boxes) > 0:
        #                 print(f"  ✅ Adding {len(new_track_boxes)} new tracks to SAM2")
        #                 self.prompts['prompts'].extend(new_track_boxes)
        #                 self.prompts['labels'].extend(new_track_labels)
        #                 self.prompts['scores'].extend([None] * len(new_track_boxes))
        #                 self.add_new = True
        #             else:
        #                 print(f"  ℹ️ No new tracks detected in interval")
        #                 print(f"     All tracks: {sorted(list(all_track_ids_in_interval))}")

        #         self.last_text_prompt = self.current_text_prompt


        # --------------- start of Feature matching ----------------------
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        #     if not self.input_queue.empty():
        #             self.current_text_prompt = self.input_queue.get()

        #     if self.current_text_prompt is not None:
        #         # state = predictor.init_state_from_numpy_frames([frame], offload_state_to_cpu=False, offload_video_to_cpu=False)

        #         if (state['num_frames']-1) % self.detection_frequency == 0 or self.last_text_prompt is None:
        #             detection= self.yolo.detect([frame], classes= [14])[0] #7,13,

        #             scores = detection['scores'].cpu().numpy()
        #             labels = detection['labels']
        #             boxes = detection['boxes'].cpu().numpy().tolist()

        #             boxes_np = np.array(boxes, dtype=np.int32)
        #             labels_np = np.array(labels)
        #             scores_np = np.array(scores)
        #             filter_mask = scores > 0.3
        #             valid_boxes = boxes_np[filter_mask]
        #             valid_labels = labels_np[filter_mask]
        #             valid_scores = scores_np[filter_mask]

        #             if self.last_text_prompt != self.current_text_prompt:
        #                 self.prompts['prompts'].extend(valid_boxes)
        #                 self.prompts['labels'].extend(valid_labels)
        #                 self.prompts['scores'].extend(valid_scores)
        #                 self.add_new = True
        #             elif len(valid_boxes) > 0:
        #                 print(f"Checking {len(valid_boxes)} YOLO detections with feature matching...")
                        
        #                 valid_masks, mask_scores = self.convert_boxes_to_masks(frame, valid_boxes)
        #                 new_features = self.extract_features_from_boxes(frame, valid_boxes)
                        
        #                 matched_boxes, matched_labels = self.match_features(
        #                     new_features, valid_masks, valid_labels
        #                 )
                        
        #                 if len(matched_boxes) > 0:
        #                     print(f"  Adding {len(matched_boxes)} new detections after feature matching")
        #                     self.prompts['prompts'].extend(matched_boxes)
        #                     self.prompts['labels'].extend(matched_labels)
        #                     self.prompts['scores'].extend([None] * len(matched_boxes))
        #                     self.add_new = True
        #                 else:
        #                     print(f"  No new detections to add - all {len(valid_boxes)} detections matched existing objects")

        #         self.last_text_prompt = self.current_text_prompt

        #---------- Start of <Feature match with mode numner of detection>------------
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if not self.input_queue.empty():
                    self.current_text_prompt = self.input_queue.get()

            if self.current_text_prompt is not None:
                current_frame_num = state['num_frames'] - 1
                
                if current_frame_num % self.detection_frequency == 0 or self.last_text_prompt is None:
                    print(f"\n🔍 Running detection on last 5 frames (current: {current_frame_num})")
                    
                    # Step 1: Get detections from last 5 frames
                    lookback_frames = min(5, current_frame_num + 1)
                    start_frame = max(0, current_frame_num - lookback_frames + 1)
                    
                    detection_counts = []  # List of detection counts per frame
                    frame_detections = {}  # Store detections by frame
                    
                    # Save current position
                    current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    
                    for frame_idx in range(start_frame, current_frame_num + 1):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, lookback_frame = self.cap.read()
                        
                        if not ret:
                            continue
                        
                        detection = self.yolo.detect([lookback_frame], classes=[14])[0]
                        
                        scores = detection['scores'].cpu().numpy()
                        labels = detection['labels']
                        boxes = detection['boxes'].cpu().numpy().tolist()
                        
                        filter_mask = scores > 0.3
                        valid_boxes = np.array(boxes, dtype=np.int32)[filter_mask]
                        valid_labels = np.array(labels)[filter_mask]
                        
                        detection_count = len(valid_boxes)
                        detection_counts.append(detection_count)
                        
                        frame_detections[frame_idx] = {
                            'boxes': valid_boxes,
                            'labels': valid_labels,
                            'count': detection_count
                        }
                    
                    # Restore position
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    
                    # Step 2: Calculate MODE (most frequent detection count)
                    if len(detection_counts) > 0:
                        from collections import Counter
                        count_frequency = Counter(detection_counts)
                        mode_detection_count = count_frequency.most_common(1)[0][0]
                        mode_frequency = count_frequency.most_common(1)[0][1]
                        
                        # Find a frame with the mode count (prefer latest frame)
                        frame_with_mode = None
                        for frame_idx in sorted(frame_detections.keys(), reverse=True):
                            if frame_detections[frame_idx]['count'] == mode_detection_count:
                                frame_with_mode = frame_idx
                                break
                    else:
                        mode_detection_count = 0
                        mode_frequency = 0
                        frame_with_mode = None
                    
                    # Step 3: Count existing SAM2 tracks
                    existing_track_count = len(state["obj_ids"])
                    
                    print(f"  📊 Detection Analysis:")
                    print(f"     Frames scanned: {start_frame} to {current_frame_num}")
                    print(f"     Detection counts: {detection_counts}")
                    print(f"     Mode (most frequent): {mode_detection_count} (appears in {mode_frequency}/{len(detection_counts)} frames)")
                    print(f"     Frame with mode: {frame_with_mode}")
                    print(f"     Existing SAM2 tracks: {existing_track_count}")
                    
                    # Step 4: Check if we need to add new tracks
                    if self.last_text_prompt != self.current_text_prompt:
                        # First time detection - use current frame
                        detection = self.yolo.detect([frame], classes=[14])[0]
                        scores = detection['scores'].cpu().numpy()
                        labels = detection['labels']
                        boxes = detection['boxes'].cpu().numpy().tolist()
                        
                        filter_mask = scores > 0.3
                        valid_boxes = np.array(boxes, dtype=np.int32)[filter_mask]
                        valid_labels = np.array(labels)[filter_mask]
                        
                        self.prompts['prompts'].extend(valid_boxes)
                        self.prompts['labels'].extend(valid_labels)
                        self.prompts['scores'].extend([None] * len(valid_boxes))
                        self.add_new = True
                        print(f"  ✅ First detection: Adding {len(valid_boxes)} objects")
                        
                    elif mode_detection_count > existing_track_count:
                        print(f"  ⚠️ Mismatch! Mode count ({mode_detection_count}) > SAM2 tracks ({existing_track_count})")
                        
                        if frame_with_mode is not None:
                            mode_boxes = frame_detections[frame_with_mode]['boxes']
                            mode_labels = frame_detections[frame_with_mode]['labels']
                            
                            if len(mode_boxes) > 0:
                                print(f"  🔍 Feature matching {len(mode_boxes)} detections from frame {frame_with_mode}...")
                                
                                # Re-read the frame with mode count for feature extraction
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_with_mode)
                                ret, mode_frame = self.cap.read()
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)  # Restore immediately
                                
                                if ret:
                                    mode_masks, mask_scores = self.convert_boxes_to_masks(mode_frame, mode_boxes)
                                    new_features = self.extract_features_from_boxes(mode_frame, mode_boxes)
                                    
                                    matched_boxes, matched_labels = self.match_features(
                                        new_features, mode_masks, mode_labels
                                    )
                                    
                                    if len(matched_boxes) > 0:
                                        print(f"  ✅ Adding {len(matched_boxes)} new objects after feature matching")
                                        self.prompts['prompts'].extend(matched_boxes)
                                        self.prompts['labels'].extend(matched_labels)
                                        self.prompts['scores'].extend([None] * len(matched_boxes))
                                        self.add_new = True
                                    else:
                                        print(f"  ℹ️ All detections matched existing tracks")
                    else:
                        print(f"  ✓ Track count OK: {existing_track_count} tracks >= {mode_detection_count} mode detections")

                self.last_text_prompt = self.current_text_prompt

            if self.add_new:
                existing_obj_ids = set(state["obj_ids"])
                predictor.reset_state(state)
                self.add_to_state(predictor, state, self.prompts)
                current_obj_ids = set(state["obj_ids"])
                newly_added_ids = current_obj_ids - existing_obj_ids
            predictor.append_frame_to_inference_state(state, frame)
            class_count= self.track_and_visualize(predictor, state, frame, self.writer)
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
        return frame, class_count, True


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