import numpy as np
from filterpy.kalman import KalmanFilter


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, label):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.label = label

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def extract_features_from_boxes(self, frame, boxes):
    """Extract feature vectors from bounding boxes using YOLO"""
    if len(boxes) == 0:
        return np.array([])
    
    from torchvision.ops import roi_align
    import torch
    
    with torch.no_grad():
        _ = self.yolo_model(frame)  # Forward pass to trigger hook
    
    feature_maps = self.feature_maps.cpu().to(torch.float32)
    fmap_h, fmap_w = feature_maps.shape[2], feature_maps.shape[3]
    orig_h, orig_w = frame.shape[:2]
    
    scaled_bboxes = []
    for x1, y1, x2, y2 in boxes:
        x1_f = max(0, int(x1 / orig_w * fmap_w))
        x2_f = min(fmap_w, int(x2 / orig_w * fmap_w))
        y1_f = max(0, int(y1 / orig_h * fmap_h))
        y2_f = min(fmap_h, int(y2 / orig_h * fmap_h))
        scaled_bboxes.append([x1_f, y1_f, x2_f, y2_f])
    
    scaled_bboxes = torch.tensor(scaled_bboxes, dtype=torch.float32)
    pooled_features = roi_align(feature_maps, [scaled_bboxes], output_size=self.roi_align_size)
    pooled_features = pooled_features.view(pooled_features.shape[0], -1)
    
    return pooled_features.numpy()

def match_features(self, new_features, new_iou_scores):
    """Match detections using feature similarity + IOU"""
    from scipy.spatial.distance import cosine
    
    if len(self.existing_features) == 0 or len(new_features) == 0:
        return new_iou_scores  # Return original IOU scores
    
    enhanced_scores = new_iou_scores.copy()
    
    for det_idx, new_feat in enumerate(new_features):
        for tracker_id, exist_feat in self.existing_features.items():
            # Compute cosine similarity
            similarity = 1 - cosine(new_feat, exist_feat)
            # Weighted combination: IOU + feature similarity
            enhanced_scores[det_idx, tracker_id] = (
                self.combined_match_weight * similarity + 
                (1 - self.combined_match_weight) * new_iou_scores[det_idx, tracker_id]
            )
    
    return enhanced_scores

# class Sort(object):
#     def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0

#         self.yolo_model = None  # Reference to YOLO model
#         self.feature_maps = None  # Captured feature maps from forward hook
#         self.roi_align_size = (2, 2)  # Spatial size for ROI pooling
#         self.feature_match_threshold = 0.25  # Cosine similarity threshold
#         self.existing_features = {}  # Maps tracker_id -> feature vector
#         self.combined_match_weight = 0.4  # Weight for combining IOU + feature similarity

#     def _setup_feature_extraction(self, yolo_model):
#         """Register forward hook on YOLO model to capture feature maps"""
#         self.yolo_model = yolo_model
#         target_layer_index = -2  # Second-to-last layer
#         target_layer = yolo_model.model.model[target_layer_index]
        
#         def hook_fn(module, input, output):
#             self.feature_maps = output
        
#         target_layer.register_forward_hook(hook_fn)

#     def update(self, frame, dets=np.empty((0, 5)), labels=np.empty((0, 1))):
#         """
#         Params:
#         frame - current video frame (numpy array)
#         dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.

#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         self.frame_count += 1
        
#         # Extract features from current detections
#         new_features = None
#         if len(dets) > 0 and self.yolo_model is not None:
#             det_boxes = dets[:, :4]
#             new_features = self.extract_features_from_boxes(frame, det_boxes)
        
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             tracker_id = self.trackers[t].id
#             if tracker_id in self.existing_features:
#                 del self.existing_features[tracker_id]
#             self.trackers.pop(t)
        
#         # Compute cost matrix using IOU
#         iou_matrix = iou_batch(dets[:, :4], trks[:, :4]) if len(dets) > 0 and len(trks) > 0 else np.zeros((len(dets), len(trks)))
        
#         # Enhance cost matrix with feature similarity if features are available
#         if new_features is not None and len(self.existing_features) > 0:
#             iou_matrix = self.match_features(new_features, iou_matrix)
        
#         # Use enhanced matrix for association
#         matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
#             dets, trks, self.iou_threshold
#         )

#         # update matched trackers with assigned detections and store their features
#         for m in matched:
#             tracker_idx = m[1]
#             det_idx = m[0]
#             self.trackers[tracker_idx].update(dets[det_idx, :])
            
#             # Store feature vector for this tracker
#             if new_features is not None and len(new_features) > det_idx:
#                 self.existing_features[self.trackers[tracker_idx].id] = new_features[det_idx]

#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(dets[i, :], labels[i])
#             self.trackers.append(trk)
            
#             # Store feature for newly created tracker
#             if new_features is not None and len(new_features) > i:
#                 self.existing_features[trk.id] = new_features[i]
        
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             if (trk.time_since_update < 1) and (
#                 trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
#             ):
#                 ret.append(
#                     np.concatenate((d, [trk.id + 1, trk.label])).reshape(1, -1)
#                 )  # +1 as MOT benchmark requires positive
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 dead_tracker_id = self.trackers[i].id
#                 if dead_tracker_id in self.existing_features:
#                     del self.existing_features[dead_tracker_id]
#                 self.trackers.pop(i)
        
#         if len(ret) > 0:
#             return np.int0(np.concatenate(ret))

#         return np.empty((0, 5))
################### End of old Sort algorithm ######################
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
    def __init__(self, sam_type:str="sam2.1_hiera_tiny", model_path:str="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                 video_path:str="", output_path:str="", use_txt_prompt:bool=False, max_frames:int=60,
                 first_prompts: list | None = None, save_video=True, device="cuda:0", mode="realtime",
                 yolo_path= "/Users/akashmanna/ws/opervu/unobstructed_view_gen/yolo_projection/best_wo_specialised_training.pt",
                 conservativeness="high"):
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

        self.sam = SAM()
        self.yolo= YOLODetector(self.yolo_path, conf_thres= 0.45)
        self.sam.build_model(self.sam_type, self.model_path, predictor_type=mode, device=device, use_txt_prompt=use_txt_prompt)
        if use_txt_prompt:
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
                        detection= self.yolo.detect([frame], classes= [7,13,14])[0]
                        
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


    
