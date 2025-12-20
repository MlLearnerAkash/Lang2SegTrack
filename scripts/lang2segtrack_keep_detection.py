# import base64
# import os
# import sys

# import shutil
# import threading
# import queue
# import time
# from io import BytesIO
# from collections import defaultdict

# import cv2
# import torch
# import gc
# import numpy as np
# import imageio
# from PIL import Image
# from triton.language import dtype
# from icecream import ic

# from models.gdino.models.gdino import GDINO
# from models.sam2.sam import SAM
# from models.yolo.detection import YOLODetector

# from utilities.color import COLOR
# import pyrealsense2 as rs
# from utilities.utils import save_frames_to_temp_dir, get_object_iou, batch_mask_iou, batch_box_iou, \
#     visualize_selected_masks_as_video, filter_mask_outliers
# from utilities.ObjectInfoManager import ObjectInfoManager

# class Lang2SegTrack:
#     def __init__(self, sam_type:str="sam2.1_hiera_tiny", model_path:str="models/sam2/checkpoints/sam2.1_hiera_large.pt",
#                  video_path:str="", output_path:str="", use_txt_prompt:bool=False, max_frames:int=60,
#                  first_prompts: list | None = None, save_video=True, device="cuda:0", mode="realtime",
#                  yolo_path= "/data/dataset/weights/opervu_seg_46SIs_211125/opervu_46SIs_21112025_2/weights/best.pt",
#                  conservativeness="high"):
#         self.sam_type = sam_type # the type of SAM model to use
#         self.model_path = model_path # the path to the SAM model checkpoint
#         self.video_path = video_path # the path to the video to track. If mode="video", this param is required.
#         self.output_path = output_path # the path to save the output video. If save_video=False, this param is ignored.
#         self.max_frames = max_frames # The maximum number of frames to be retained, beyond which the oldest frames are deleted,
#         # so that the memory footprint does not grow indefinitely
#         # If the number of tracked objects is large and likely to be occluded, set it to a larger value(such as 120) to enhance tracking
#         self.first_prompts = first_prompts  # the initial bounding boxes ,points or masks to track. If not None, the tracker will use the first frame to detect objects.
#         # [mask, point, bbox], mask: np.ndarray[H, W], point: list[int], bbox: list[int]
#         self.save_video = save_video # whether to save the output video
#         self.device = device
#         self.mode = mode # the mode to run the tracker. "video" or "realtime"
#         self.yolo_path = yolo_path
#         if self.mode == 'img' and not use_txt_prompt:
#             raise ValueError("In 'img' mode, use_txt_prompt must be True")

#         self.sam = SAM()
#         self.yolo= YOLODetector(self.yolo_path, conf_thres= 0.45)
#         self.sam.build_model(self.sam_type, self.model_path, predictor_type=mode, device=device, use_txt_prompt=use_txt_prompt)
#         # self.sam.build_model(self.sam_type, self.model_path, predictor_type=mode, 
#         #             device=device, use_txt_prompt=use_txt_prompt,
#         #             conservativeness=conservativeness)
#         if use_txt_prompt:
#             self.gdino = GDINO()
            
#             self.gdino_16 = False
#             if not self.gdino_16:
#                 print("Building GroundingDINO model...")
#                 self.gdino.build_model(device=device)
#         else:
#             self.gdino = None



#         # data management
#         self.history_frames = []
#         self.all_forward_masks = {}
#         self.all_final_masks = {}
#         self.object_start_frame_idx = {}
#         self.object_start_prompts = {}
#         self.existing_obj_outputs = []
#         self.current_text_prompt = None
#         self.last_text_prompt = None
#         if self.first_prompts is not None:
#             self.prompts = {'prompts': self.first_prompts, 'labels': [None] * len(self.first_prompts), 'scores': [None] * len(self.first_prompts)}
#             self.add_new = True
#         else:
#             self.prompts = {'prompts': [], 'labels': [], 'scores': []}
#         self.iou_threshold = 0.3
#         self.detection_frequency =10
#         self.object_labels = {}  # Maps obj_id to class name
#         self.last_known_bboxes = {}

#         # Counting-related attributes
#         self.incision_area = None  # Will store the incision polygon/bbox
#         self.object_track_history = {}  # Store centroid positions for each object
#         self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})  # Per-class counts

#         self.input_queue = queue.Queue()
#         self.drawing = False
#         self.add_new = False
#         self.ix, self.iy = -1, -1
#         self.frame_display = None
#         self.height, self.width = None, None
#         self.prev_time = 0

#         self.lost_obj_ids = set()  # Track which objects were lost
#         self.prev_obj_ids = set()  # Track previous frame's object IDs


#     def input_thread(self):
#         while True:
#             user_input = input()
#             self.input_queue.put(user_input)

#     def draw_bbox(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if flags & cv2.EVENT_FLAG_CTRLKEY:
#                 self.prompts['prompts'].append((x, y))
#                 self.prompts['labels'].append(None)
#                 self.prompts['scores'].append(None)
#                 self.add_new = True
#                 cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
#             else:
#                 self.drawing = True
#                 self.ix, self.iy = x, y
#         elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
#             img = param.copy()
#             cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
#             # cv2.imshow("Video Tracking", img)
#         elif event == cv2.EVENT_LBUTTONUP and self.drawing:
#             if abs(x - self.ix) > 2 and abs(y - self.iy) > 2:
#                 bbox = [self.ix, self.iy, x, y]
#                 self.prompts['prompts'].append(bbox)
#                 self.prompts['labels'].append(None)
#                 self.prompts['scores'].append(None)
#                 self.add_new = True
#                 cv2.rectangle(param, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
#             self.drawing = False

#     def set_incision_area(self, polygon_points):
#         """
#         Set the incision area for counting.
        
#         Args:
#             polygon_points: List of (x, y) tuples defining the incision area polygon
#                            or [x1, y1, x2, y2] for rectangular area
#         """
#         if len(polygon_points) == 4 and not isinstance(polygon_points[0], (list, tuple)):
#             # Convert bbox to polygon
#             x1, y1, x2, y2 = polygon_points
#             self.incision_area = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
#         else:
#             self.incision_area = np.array(polygon_points, dtype=np.int32)
#         print(f"Incision area set: {self.incision_area}")
    
#     def is_inside_incision(self, bbox):
#         """
#         Check if object's center point is inside the incision area.
        
#         Args:
#             bbox: [x, y, w, h] format
            
#         Returns:
#             bool: True if center is inside incision area
#         """
#         if self.incision_area is None:
#             return False
        
#         x, y, w, h = bbox
#         center_x = x + w // 2
#         center_y = y + h // 2
        
#         # Use cv2.pointPolygonTest for polygon containment
#         result = cv2.pointPolygonTest(self.incision_area, (int(center_x), int(center_y)), False)
#         return result >= 0
    
#     def update_counting(self, obj_id, bbox, category_name):
#         """
#         Update counting statistics based on object trajectory crossing the incision boundary.
#         Tracks multiple crossings per object.
        
#         Args:
#             obj_id: Object ID
#             bbox: Bounding box [x, y, w, h]
#             category_name: Category name of the object
#         """
#         if self.incision_area is None:
#             return
        
#         # Calculate current centroid
#         x, y, w, h = bbox
#         current_centroid = (int(x + w // 2), int(y + h // 2))
        
#         # Initialize tracking history for this object
#         if obj_id not in self.object_track_history:
#             self.object_track_history[obj_id] = []
        
#         # Store current centroid
#         self.object_track_history[obj_id].append(current_centroid)
        
#         # Keep only last 30 positions to save memory
#         if len(self.object_track_history[obj_id]) > 30:
#             self.object_track_history[obj_id].pop(0)
        
#         # Need at least 2 positions to determine crossing
#         if len(self.object_track_history[obj_id]) < 2:
#             return
        
#         prev_centroid = self.object_track_history[obj_id][-2]
        
#         #NOTE: to be thought of
#         # if obj_id in self.counted_ids:
#         #     return

#         # Check if the trajectory crosses the incision boundary
#         is_inside_now = self.is_centroid_inside_incision(current_centroid)
#         was_inside = self.is_centroid_inside_incision(prev_centroid)
        
#         # Detect boundary crossing - allow multiple crossings per object
#         if is_inside_now and not was_inside:
#             # Crossed from outside to inside
#             self.classwise_count[category_name]["IN"] += 1
#             print(f"  {category_name} obj_{obj_id} ENTERED incision (IN count: {self.classwise_count[category_name]['IN']})")
            
#         elif not is_inside_now and was_inside:
#             # Crossed from inside to outside
#             self.classwise_count[category_name]["OUT"] += 1
#             print(f"  {category_name} obj_{obj_id} EXITED incision (OUT count: {self.classwise_count[category_name]['OUT']})")

#         if obj_id ==1:
#             print(">>>>", is_inside_now)
#             print(">>>centroid location>>>", current_centroid)


#     def is_centroid_inside_incision(self, centroid):
#         """
#         Check if a centroid point is inside the incision area.
        
#         Args:
#             centroid: (x, y) tuple
            
#         Returns:
#             bool: True if centroid is inside incision area
#         """
#         if self.incision_area is None:
#             return False
        
#         result = cv2.pointPolygonTest(self.incision_area, centroid, False)
#         return result >= 0
    
#     def draw_incision_area(self, frame):
#         """Draw the incision area on the frame."""
#         if self.incision_area is not None:
#             cv2.polylines(frame, [self.incision_area], True, (0, 255, 255), 3)
#             cv2.putText(frame, "INCISION AREA", 
#                        tuple(self.incision_area[0]), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
#     def draw_counting_stats(self, frame):
#         """Draw counting statistics on the frame."""
#         if self.incision_area is None:
#             return
        
#         y_offset = 70
#         for category_name in sorted(self.classwise_count.keys()):
#             in_count = self.classwise_count[category_name]["IN"]
#             out_count = self.classwise_count[category_name]["OUT"]
            
#             stats_text = f"{category_name}: IN={in_count} OUT={out_count}"
#             cv2.putText(frame, stats_text, (10, y_offset), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
#             y_offset += 30
    
#     def print_final_statistics(self):
#         """Print final counting statistics."""
#         print("\n" + "="*60)
#         print("FINAL COUNTING STATISTICS")
#         print("="*60)
        
#         total_in = 0
#         total_out = 0
        
#         for category_name in sorted(self.classwise_count.keys()):
#             in_count = self.classwise_count[category_name]["IN"]
#             out_count = self.classwise_count[category_name]["OUT"]
#             total_in += in_count
#             total_out += out_count
            
#             print(f"\n{category_name.upper()}:")
#             print(f"  Objects entered (IN): {in_count}")
#             print(f"  Objects exited (OUT): {out_count}")
#             print(f"  Net count (IN - OUT): {in_count - out_count}")
        
#         print(f"\nTOTAL:")
#         print(f"  Total IN: {total_in}")
#         print(f"  Total OUT: {total_out}")
#         print(f"  Net objects inside: {total_in - total_out}")
#         print(f"  Total unique objects tracked: {len(self.object_track_history)}")
#         print("="*60 + "\n")

#     def convert_boxes_to_masks(self, frame, boxes):
#         """
#         Convert YOLO bounding boxes to SAM2 segmentation masks.
        
#         Args:
#             frame: Current video frame (numpy array)
#             boxes: List of bounding boxes in [x1, y1, x2, y2] format
            
#         Returns:
#             masks: List of boolean masks
#             scores: List of confidence scores for each mask
#         """
#         if len(boxes) == 0:
#             return [], []
        
#         # Lazy initialization of img_predictor
#         if not hasattr(self.sam, 'img_predictor'):
#             print("Initializing SAM2 image predictor for mask conversion...")
#             from sam2.sam2_image_predictor import SAM2ImagePredictor
#             self.sam.img_predictor = SAM2ImagePredictor(self.sam.model)
        
#         # Set the image for SAM2
#         self.sam.img_predictor.set_image(frame)
        
#         masks = []
#         scores = []
        
#         # Convert each bounding box to a mask
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             # SAM2 expects box in xyxy format
#             box_xyxy = np.array([x1, y1, x2, y2])
            
#             # Get mask from SAM2
#             mask, score, _ = self.sam.img_predictor.predict(
#                 box=box_xyxy,
#                 multimask_output=False
#             )
            
#             # mask shape is (1, H, W), we need (H, W)
#             mask = mask[0].astype(bool)
            
#             # Constrain the mask to stay within the bounding box region
#             constrained_mask = np.zeros_like(mask, dtype=bool)
#             constrained_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            
#             masks.append(constrained_mask)
#             scores.append(float(score[0]))
        
#         return masks, scores


#     def get_mask_from_bbox(self, bbox):
#         """
#         Helper function to create a simple mask from a bounding box.
#         Used for existing tracked objects.
        
#         Args:
#             bbox: [x1, y1, x2, y2] format
            
#         Returns:
#             mask: Boolean mask
#         """
#         x1, y1, x2, y2 = bbox
#         mask = np.zeros((self.height, self.width), dtype=bool)
#         mask[y1:y2, x1:x2] = True
#         return mask
    
#     def add_to_state(self, predictor, state, prompts, start_with_0=False):
#         frame_idx = 0 if start_with_0 else state["num_frames"]-1
#         for id, item in enumerate(prompts['prompts']):
#             # Check if item is a mask (numpy array with bool dtype)
#             if isinstance(item, np.ndarray) and item.dtype == bool:
#                 # It's a mask prompt
#                 predictor.add_new_mask(state, mask=item, frame_idx=frame_idx, obj_id=id)
#             elif len(item) == 4:
#                 # It's a bounding box [x1, y1, x2, y2]
#                 x1, y1, x2, y2 = item
#                 cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 predictor.add_new_points_or_box(state, box=item, frame_idx=frame_idx, obj_id=id)
#             elif len(item) == 2:
#                 # It's a point [x, y]
#                 x, y = item
#                 cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
#                 pt = torch.tensor([[x, y]], dtype=torch.float32)
#                 lbl = torch.tensor([1], dtype=torch.int32)
#                 predictor.add_new_points_or_box(state, points=pt, labels=lbl, frame_idx=frame_idx, obj_id=id)
    
#     # def add_to_state(self, predictor, state, prompts, start_with_0=False):
#     #     frame_idx = 0 if start_with_0 else state["num_frames"]-1
#     #     for id, item in enumerate(prompts['prompts']):
#     #         if len(item) == 4:
#     #             x1, y1, x2, y2 = item
#     #             cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     #             predictor.add_new_points_or_box(state, box=item, frame_idx=frame_idx, obj_id=id)
                
#     #             # Add negative points around the bounding box to prevent extension
#     #             center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#     #             neg_points = []
#     #             neg_labels = []
                
#     #             # Add negative points outside the box
#     #             for dx, dy in [(-10, 0), (10, 0), (0, -10), (0, 10)]:
#     #                 neg_x, neg_y = center_x + dx, center_y + dy
#     #                 if 0 <= neg_x < self.width and 0 <= neg_y < self.height:
#     #                     neg_points.append([neg_x, neg_y])
#     #                     neg_labels.append(0)  # 0 for negative
                
#     #             if neg_points:
#     #                 neg_pts = torch.tensor(neg_points, dtype=torch.float32)
#     #                 neg_lbls = torch.tensor(neg_labels, dtype=torch.int32)
#     #                 predictor.add_new_points_or_box(state, points=neg_pts, labels=neg_lbls, 
#     #                                             frame_idx=frame_idx, obj_id=id)
#     #             for (nx, ny) in neg_points:
#     #                 cv2.circle(self.frame_display, (int(nx), int(ny)), 3, (0, 0, 255), -1)

#     def track_and_visualize(self, predictor, state, frame, writer):
#         if (any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"]))) or
#             any(len(state["mask_inputs_per_obj"][i]) > 0 for i in range(len(state["mask_inputs_per_obj"])))):
#             for frame_idx, obj_ids, masks in predictor.propagate_in_frame(state, state["num_frames"] - 1):
#                 self.existing_obj_outputs = []
#                 # self.prompts['prompts'] = []
#                 self.current_frame_masks = []
#                 for obj_id, mask in zip(obj_ids, masks):
#                     mask = mask[0].cpu().numpy() > 0.0
#                     mask = filter_mask_outliers(mask)
#                     # Store the actual mask for IOU comparison
#                     self.current_frame_masks.append(mask)
#                     nonzero = np.argwhere(mask)
#                     if nonzero.size == 0:
#                         continue
#                         bbox = [0, 0, 0, 0]
#                         # bbox = self.last_known_bboxes.get(obj_id, [0, 0, 0, 0])
#                     else:
#                         y_min, x_min = nonzero.min(axis=0)
#                         y_max, x_max = nonzero.max(axis=0)
#                         bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
#                         self.last_known_bboxes[obj_id] = bbox
                    
#                     # Get category name for counting
#                     category_name = self.object_labels.get(obj_id, "unknown")
                    
#                     # Update counting statistics
#                     self.update_counting(obj_id, bbox, category_name)
                    
#                     self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
#                     # self.existing_masks.append(mask)
#                     self.existing_obj_outputs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
#                     # self.prompts['prompts'].append(mask)
#                     # self.all_forward_masks.setdefault(obj_id, []).append(mask)
#                 self.prompts['prompts'] = self.existing_obj_outputs.copy()
                
#         # Draw incision area and statistics
#         self.draw_incision_area(frame)
#         self.draw_counting_stats(frame)
        
#         frame_dis = self.show_fps(frame)
#         # cv2.imshow("Video Tracking", frame_dis)
#         # Add frame index at top-right corner
#         frame_text = f"Frame: {state['num_frames']}"
#         print("=="*20)
#         print("STATS FOR FRAME: #",state['num_frames'])
#         print("=="*20)
#         text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
#         text_x = self.width - text_size[0] - 10  # 10 pixels from right edge
#         text_y = 30  # 30 pixels from top
#         cv2.putText(frame, frame_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#         if writer:
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             writer.append_data(rgb)


#     def draw_mask_and_bbox(self, frame, mask, bbox, obj_id):
#         mask_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
#         mask_img[mask] = COLOR[obj_id % len(COLOR)]
#         frame[:] = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
#         x, y, w, h = bbox
#         cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        
#         # Get category name for this object
#         label_text = f"obj_{obj_id}"
#         if obj_id in self.object_labels:
#             label_text = f"obj_{obj_id}_{self.object_labels[obj_id]}"
        
#         # Add indicator if object is currently inside incision
#         if self.is_inside_incision(bbox):
#             label_text += " [IN]"
        
#         cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)


#     def show_fps(self, frame):
#         frame = frame.copy()
#         curr_time = time.time()
#         fps = 1 / (curr_time - self.prev_time)
#         self.prev_time = curr_time
#         fps_str = f"FPS: {fps:.2f}"
#         cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         return frame

#     def visualize_final_masks(self, output_path="final_tracked_video.mp4", fps=25):
#         if not hasattr(self, "all_final_masks") or not self.all_final_masks:
#             print("No final masks found. Please run `track()` and `track_backward()` first.")
#             return

#         print("Visualizing final tracking results...")
#         num_frames = len(self.all_final_masks[0])
#         assert len(self.history_frames)== num_frames
#         writer = imageio.get_writer(output_path, fps=fps)

#         for frame_idx in range(num_frames):
#             base_frame = self.history_frames[frame_idx].copy()
#             for obj_id, mask_list in self.all_final_masks.items():
#                 mask = mask_list[frame_idx]
#                 nonzero = np.argwhere(mask)
#                 if nonzero.size == 0:
#                     bbox = [0, 0, 0, 0]
#                 else:
#                     y_min, x_min = nonzero.min(axis=0)
#                     y_max, x_max = nonzero.max(axis=0)
#                     bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
#                 self.draw_mask_and_bbox(base_frame, mask, bbox, obj_id)

#             writer.append_data(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB))
#             # cv2.imshow("Final Tracking Visualization", base_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         writer.close()
#         print(f"Final visualization saved to {output_path}")
#         cv2.destroyAllWindows()

#     def predict_img(
#             self,
#             images_pil: list[Image.Image],
#             texts_prompt: list[str],
#             box_threshold: float = 0.3,
#             text_threshold: float = 0.25,
#     ):
#         """
#         Parameters:
#             images_pil (list[Image.Image]): List of input images.
#             texts_prompt (list[str]): List of text prompts corresponding to the images.
#             box_threshold (float): Threshold for box predictions.
#             text_threshold (float): Threshold for text predictions.
#         Returns:
#             list[dict]: List of results containing masks and other outputs for each image.
#             Output format:
#             [{
#                 "boxes": np.ndarray,
#                 "scores": np.ndarray,
#                 "masks": np.ndarray,
#                 "mask_scores": np.ndarray,
#             }, ...]
#         """
#         if self.yolo:
#             yolo_results= self.yolo.predict(images_pil)
#             # ic(yolo_results)
#         if self.gdino_16:
#             if len(images_pil) > 1:
#                 raise ValueError("GroundingDINO_16 only support single image")
#             byte_io = BytesIO()
#             images_pil[0].save(byte_io, format='PNG')
#             image_bytes = byte_io.getvalue()
#             base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
#             texts_prompt = texts_prompt[0]
#             gdino_results = self.gdino.predict_dino_1_6_pro(base64_encoded, texts_prompt, box_threshold, text_threshold)
#         else:
#             gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        
#         all_results = []
#         sam_images = []
#         sam_boxes = []
#         sam_indices = []
#         for idx, result in enumerate(gdino_results):
#             result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
#             processed_result = {
#                 **result,
#                 "masks": [],
#                 "mask_scores": [],
#             }

#             if result["labels"]:
#                 sam_images.append(np.asarray(images_pil[idx]))
#                 sam_boxes.append(processed_result["boxes"])
#                 sam_indices.append(idx)

#             all_results.append(processed_result)
#         if sam_images:
#             # print(f"Predicting {len(sam_boxes)} masks")
#             masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
#             for idx, mask, score in zip(sam_indices, masks, mask_scores):
#                 all_results[idx].update(
#                     {
#                         "masks": mask,
#                         "mask_scores": score,
#                     }
#                 )
#         return all_results

#     def track_backward(self):
#         predictor = self.sam.video_predictor

#         print("Starting backward tracking for each object...")
#         with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#             for obj_id in range(max(self.object_start_frame_idx)+1):

#                 start_idx = self.object_start_frame_idx[obj_id]
#                 if start_idx == 0:
#                     full_masks = self.all_forward_masks[obj_id]
#                 else:
#                     print('\n')
#                     print(f"\033[92mINFO: Object_{obj_id} is being tracked backward in time......\033[0m")
#                     history_frames = self.history_frames[:start_idx]
#                     history_frames = history_frames[::-1]
#                     frames = save_frames_to_temp_dir(history_frames)
#                     prompt = self.object_start_prompts[obj_id]
#                     reverse_state = predictor.init_state(
#                         frames, offload_state_to_cpu=False, offload_video_to_cpu=False
#                     )
#                     self.add_to_state(predictor, reverse_state, [prompt], start_with_0=True)
#                     backward_masks = []
#                     for frame_idx, obj_ids, masks in predictor.propagate_in_video(reverse_state):
#                         for mid, mask in zip(obj_ids, masks):
#                             mask_np = mask[0].cpu().numpy() > 0.0
#                             backward_masks.append(mask_np)

#                     backward_masks = backward_masks[::-1]
#                     forward_masks = self.all_forward_masks.get(obj_id, [])
#                     full_masks = backward_masks + forward_masks[1:] if len(forward_masks) > 1 else backward_masks
#                     #predictor.reset_state(reverse_state)
#                 self.all_final_masks[obj_id] = full_masks

#         print("Backward tracking completed. Merged object trajectories are ready.")

#         # save mask img
#         output_dir = "mask_outputs"
#         if os.path.exists(output_dir) and os.path.isdir(output_dir):
#             shutil.rmtree(output_dir)
#         os.makedirs(output_dir, exist_ok=True)
#         for obj_id, masks in self.all_final_masks.items():
#             obj_dir = os.path.join(output_dir, f"obj_{obj_id}")
#             os.makedirs(obj_dir, exist_ok=True)
#             for frame_idx, mask in enumerate(masks):
#                 mask_image = (mask * 255).astype(np.uint8)
#                 mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.png")
#                 cv2.imwrite(mask_path, mask_image)
#         print(f"Masks saved to {output_dir}")
#         visualize_selected_masks_as_video()


#     def track(self):

#         predictor = self.sam.video_predictor

#         if self.mode == "realtime":
#             print("Start with realtime mode.")
#             pipeline = rs.pipeline()
#             config = rs.config()
#             config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#             pipeline.start(config)
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             color_image = np.asanyarray(color_frame.get_data())
#             get_frame = lambda: np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
#         elif self.mode == "video":
#             print("Start with video mode.")
#             cap = cv2.VideoCapture(self.video_path)
#             ret, color_image = cap.read()
#             get_frame = lambda: cap.read()
#         else:
#             raise ValueError("The mode is not supported in this method.")

#         self.height, self.width = color_image.shape[:2]

#         if self.save_video:
#             writer = imageio.get_writer(self.output_path, fps=5)
#         else:
#             writer = None

#         # cv2.namedWindow("Video Tracking")

#         threading.Thread(target=self.input_thread, daemon=True).start()

#         with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#             state = predictor.init_state_from_numpy_frames([color_image], offload_state_to_cpu=False, offload_video_to_cpu=False)
#             while True:
#                 if self.mode == "realtime":
#                     frame = get_frame()
#                 else:
#                     ret, frame = get_frame()
#                     if not ret:
#                         break
#                 self.frame_display = frame.copy()
#                 # self.history_frames.append(frame)
#                 # cv2.setMouseCallback("Video Tracking", self.draw_bbox, param=self.frame_display)

#                 if not self.input_queue.empty():
#                     self.current_text_prompt = self.input_queue.get()
#                     # self.persistent_text_prompts.add(text)

#                 if self.current_text_prompt is not None:
#                     if (state['num_frames']-1) % self.detection_frequency == 0 or self.last_text_prompt is None:
#                         detection= self.yolo.detect([frame], classes= [7,13,14])[0]
                        
#                         scores = detection['scores'].cpu().numpy()
#                         labels = detection['labels']
#                         boxes = detection['boxes'].cpu().numpy().tolist()

#                         boxes_np = np.array(boxes, dtype=np.int32)
#                         labels_np = np.array(labels)
#                         scores_np = np.array(scores)
#                         filter_mask = scores > 0.3
#                         valid_boxes = boxes_np[filter_mask]
#                         valid_labels = labels_np[filter_mask]
#                         valid_scores = scores_np[filter_mask]

#                         if self.last_text_prompt != self.current_text_prompt:
#                             # New prompt - add all valid detections
#                             self.prompts['prompts'].extend(valid_boxes)
#                             self.prompts['labels'].extend(valid_labels)
#                             self.prompts['scores'].extend(valid_scores)
#                             self.add_new = True
#                         elif self.existing_obj_outputs:
#                             # Check both spatial overlap (IOU) AND class labels
#                             if len(valid_boxes) > 0:
#                                 print(f"Checking {len(valid_boxes)} YOLO detections against existing objects...")
                                
#                                 # Convert YOLO boxes to SAM2 masks
#                                 valid_masks, mask_scores = self.convert_boxes_to_masks(frame, valid_boxes)
#                                 #NOTE: turned off to load xurrent mask
#                                 # Get existing masks from tracked objects
#                                 # existing_masks = [self.get_mask_from_bbox(bbox) for bbox in self.existing_obj_outputs]
                                
#                                 # Use CURRENT frame's tracked masks instead of bbox-based masks
#                                 if hasattr(self, 'current_frame_masks') and len(self.current_frame_masks) > 0:
#                                     existing_masks = self.current_frame_masks
#                                 else:
#                                     # Fallback to bbox-based masks
#                                     existing_masks = [self.get_mask_from_bbox(bbox) for bbox in self.existing_obj_outputs]

                                
#                                 # Calculate spatial IOU
#                                 iou_matrix = batch_mask_iou(np.array(valid_masks), np.array(existing_masks))
                                
#                                 # Check class labels - build existing labels list from prompts
#                                 existing_labels = []
#                                 for i, prompt in enumerate(self.prompts['prompts']):
#                                     if i < len(self.prompts['labels']) and self.prompts['labels'][i] is not None:
#                                         existing_labels.append(self.prompts['labels'][i])
#                                     else:
#                                         existing_labels.append(-1)  # Unknown class for manually added objects
                
#                                 # Filter out detections that have:
#                                 # 1. High IOU with existing object AND
#                                 # 2. Same class label as that existing object
#                                 new_detections = []
#                                 for i in range(len(valid_masks)):
#                                     is_new = True
#                                     max_iou_idx = np.argmax(iou_matrix[i])
#                                     max_iou = iou_matrix[i, max_iou_idx]
                                    
#                                     # If high spatial overlap with an existing object
#                                     if max_iou >= self.iou_threshold:
#                                         # Check if it's the same class
#                                         if max_iou_idx < len(existing_labels):
#                                             existing_class = existing_labels[max_iou_idx]
#                                             detected_class = valid_labels[i]
                                            
#                                             # If same class and same location, skip it
#                                             if existing_class == detected_class:
#                                                 is_new = False
#                                                 print(f"  Skipping detection: class {detected_class} already tracked at this location (IOU: {max_iou:.2f})")
                                    
#                                     if is_new:
#                                         new_detections.append(i)
                                
#                                 # Add only new detections
#                                 if new_detections:
#                                     valid_masks_filtered = [valid_masks[i] for i in new_detections]
#                                     valid_labels_filtered = valid_labels[new_detections]
#                                     valid_scores_filtered = valid_scores[new_detections]
                                    
#                                     print(f"  Adding {len(new_detections)} new detections (filtered {len(valid_masks) - len(new_detections)} duplicates)")
                                    
#                                     self.prompts['prompts'].extend(valid_masks_filtered)
#                                     self.prompts['labels'].extend(valid_labels_filtered)
#                                     self.prompts['scores'].extend(valid_scores_filtered)
#                                     self.add_new = True
#                                 else:
#                                     print(f"  No new detections to add - all {len(valid_boxes)} detections are duplicates")
    
#                     self.last_text_prompt = self.current_text_prompt

#                 if self.add_new:
#                     existing_obj_ids = set(state["obj_ids"])
#                     predictor.reset_state(state)
#                     self.add_to_state(predictor, state, self.prompts)
#                     current_obj_ids = set(state["obj_ids"])
#                     newly_added_ids = current_obj_ids - existing_obj_ids
#                 predictor.append_frame_to_inference_state(state, frame)
#                 self.track_and_visualize(predictor, state, frame, writer)
#                 if self.add_new:
#                     for idx, obj_id in enumerate(newly_added_ids):
#                         self.object_start_frame_idx[obj_id] = state['num_frames'] - 1
                        
#                         # Associate the object with its category name
#                         # Find the corresponding label from prompts
#                         prompt_idx = len(self.prompts['prompts']) - len(newly_added_ids) + idx
#                         if prompt_idx < len(self.prompts['labels']) and self.prompts['labels'][prompt_idx] is not None:
#                             class_id = self.prompts['labels'][prompt_idx]
#                             # Get class name from YOLO detector
#                             class_name = self.yolo.model.names.get(class_id, f"class_{class_id}")
#                             self.object_labels[obj_id] = class_name
#                             print(f"  Object {obj_id} assigned class: {class_name}")
#                         else:
#                             self.object_labels[obj_id] = "unknown"
                    
#                     self.add_new = False

#                 if state["num_frames"] % self.max_frames == 0:
#                     if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
#                         predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
#                     predictor.release_old_frames(state)

#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     break

#         if self.mode == "realtime":
#             pipeline.stop()
#         else:
#             cap.release()
        
#         # Print final statistics
#         self.print_final_statistics()
        
#         # self.track_backward()
#         # self.visualize_final_masks()
#         if writer:
#             writer.close()
#         cv2.destroyAllWindows()
#         del predictor, state
#         gc.collect()
#         torch.clear_autocast_cache()
#         torch.cuda.empty_cache()


# if __name__ == "__main__":
#     tracker = Lang2SegTrack(sam_type="sam2.1_hiera_large",
#                             model_path="models/sam2/checkpoints/sam2.1_hiera_large.pt",
#                             video_path="/data/dataset/demo_video/output.mp4",
#                             # video_path="assets/05_default_juggle.mp4",
#                             output_path="forward_tracked_video.mp4",
#                             mode="video",
#                             save_video=True,
#                             use_txt_prompt=True)
    
#     # Define incision area (example: center rectangle)
#     # Adjust coordinates based on your video dimensions
#     # Format: [x1, y1, x2, y2] or [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] for polygon
#     tracker.set_incision_area([700, 1040+200, 1490-200, 1790])
    
#     tracker.current_text_prompt = 'car'
#     tracker.track()



####################Previous code######################
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

class Lang2SegTrack:
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

    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.prompts['prompts'].append((x, y))
                self.prompts['labels'].append(None)
                self.prompts['scores'].append(None)
                self.add_new = True
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            else:
                self.drawing = True
                self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = param.copy()
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            if abs(x - self.ix) > 2 and abs(y - self.iy) > 2:
                bbox = [self.ix, self.iy, x, y]
                self.prompts['prompts'].append(bbox)
                self.prompts['labels'].append(None)
                self.prompts['scores'].append(None)
                self.add_new = True
                cv2.rectangle(param, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.drawing = False

    def set_incision_area(self, polygon_points):
        if len(polygon_points) == 4 and not isinstance(polygon_points[0], (list, tuple)):
            x1, y1, x2, y2 = polygon_points
            self.incision_area = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        else:
            self.incision_area = np.array(polygon_points, dtype=np.int32)
        print(f"Incision area set: {self.incision_area}")
    
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
    
    def print_final_statistics(self):
        print("\n" + "="*60)
        print("FINAL COUNTING STATISTICS")
        print("="*60)
        
        total_in = 0
        total_out = 0
        
        for category_name in sorted(self.classwise_count.keys()):
            in_count = self.classwise_count[category_name]["IN"]
            out_count = self.classwise_count[category_name]["OUT"]
            total_in += in_count
            total_out += out_count
            
            print(f"\n{category_name.upper()}:")
            print(f"  Objects entered (IN): {in_count}")
            print(f"  Objects exited (OUT): {out_count}")
            print(f"  Net count (IN - OUT): {in_count - out_count}")
        
        print(f"\nTOTAL:")
        print(f"  Total IN: {total_in}")
        print(f"  Total OUT: {total_out}")
        print(f"  Net objects inside: {total_in - total_out}")
        print(f"  Total unique objects tracked: {len(self.object_track_history)}")
        print("="*60 + "\n")

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
                    self.update_counting(obj_id, bbox, category_name)
                    self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
                    self.existing_obj_outputs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                
                if len(current_obj_boxes) > 0:
                    current_features = self.extract_features_from_boxes(frame, current_obj_boxes)
                    for obj_id, feat in zip(obj_ids, current_features):
                        self.existing_features[obj_id] = feat
                
                self.prompts['prompts'] = self.existing_obj_outputs.copy()
        
        self.draw_incision_area(frame)
        self.draw_counting_stats(frame)
        
        frame_dis = self.show_fps(frame)
        frame_text = f"Frame: {state['num_frames']}"
        print("=="*20)
        print("STATS FOR FRAME: #",state['num_frames'])
        print("=="*20)
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = self.width - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, frame_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if writer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)

    def draw_mask_and_bbox(self, frame, mask, bbox, obj_id):
        mask_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_img[mask] = COLOR[obj_id % len(COLOR)]
        frame[:] = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        
        label_text = f"obj_{obj_id}"
        if obj_id in self.object_labels:
            label_text = f"obj_{obj_id}_{self.object_labels[obj_id]}"
        
        if self.is_inside_incision(bbox):
            label_text += " [IN]"
        
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)

    def show_fps(self, frame):
        frame = frame.copy()
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        fps_str = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def visualize_final_masks(self, output_path="final_tracked_video.mp4", fps=25):
        if not hasattr(self, "all_final_masks") or not self.all_final_masks:
            print("No final masks found. Please run `track()` and `track_backward()` first.")
            return

        print("Visualizing final tracking results...")
        num_frames = len(self.all_final_masks[0])
        assert len(self.history_frames)== num_frames
        writer = imageio.get_writer(output_path, fps=fps)

        for frame_idx in range(num_frames):
            base_frame = self.history_frames[frame_idx].copy()
            for obj_id, mask_list in self.all_final_masks.items():
                mask = mask_list[frame_idx]
                nonzero = np.argwhere(mask)
                if nonzero.size == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = nonzero.min(axis=0)
                    y_max, x_max = nonzero.max(axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                self.draw_mask_and_bbox(base_frame, mask, bbox, obj_id)

            writer.append_data(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        writer.close()
        print(f"Final visualization saved to {output_path}")
        cv2.destroyAllWindows()

    def predict_img(
            self,
            images_pil: list[Image.Image],
            texts_prompt: list[str],
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
    ):
        if self.yolo:
            yolo_results= self.yolo.predict(images_pil)
        if self.gdino_16:
            if len(images_pil) > 1:
                raise ValueError("GroundingDINO_16 only support single image")
            byte_io = BytesIO()
            images_pil[0].save(byte_io, format='PNG')
            image_bytes = byte_io.getvalue()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            texts_prompt = texts_prompt[0]
            gdino_results = self.gdino.predict_dino_1_6_pro(base64_encoded, texts_prompt, box_threshold, text_threshold)
        else:
            gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
        return all_results

    def track_backward(self):
        predictor = self.sam.video_predictor

        print("Starting backward tracking for each object...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for obj_id in range(max(self.object_start_frame_idx)+1):

                start_idx = self.object_start_frame_idx[obj_id]
                if start_idx == 0:
                    full_masks = self.all_forward_masks[obj_id]
                else:
                    print('\n')
                    print(f"\033[92mINFO: Object_{obj_id} is being tracked backward in time......\033[0m")
                    history_frames = self.history_frames[:start_idx]
                    history_frames = history_frames[::-1]
                    frames = save_frames_to_temp_dir(history_frames)
                    prompt = self.object_start_prompts[obj_id]
                    reverse_state = predictor.init_state(
                        frames, offload_state_to_cpu=False, offload_video_to_cpu=False
                    )
                    self.add_to_state(predictor, reverse_state, [prompt], start_with_0=True)
                    backward_masks = []
                    for frame_idx, obj_ids, masks in predictor.propagate_in_video(reverse_state):
                        for mid, mask in zip(obj_ids, masks):
                            mask_np = mask[0].cpu().numpy() > 0.0
                            backward_masks.append(mask_np)

                    backward_masks = backward_masks[::-1]
                    forward_masks = self.all_forward_masks.get(obj_id, [])
                    full_masks = backward_masks + forward_masks[1:] if len(forward_masks) > 1 else backward_masks
                self.all_final_masks[obj_id] = full_masks

        print("Backward tracking completed. Merged object trajectories are ready.")

        output_dir = "mask_outputs"
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for obj_id, masks in self.all_final_masks.items():
            obj_dir = os.path.join(output_dir, f"obj_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)
            for frame_idx, mask in enumerate(masks):
                mask_image = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(mask_path, mask_image)
        print(f"Masks saved to {output_dir}")
        visualize_selected_masks_as_video()

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


if __name__ == "__main__":
    tracker = Lang2SegTrack(sam_type="sam2.1_hiera_large",
                            model_path="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                            video_path="/data/dataset/demo_video/output.mp4",
                            output_path="forward_tracked_video.mp4",
                            mode="video",
                            save_video=True,
                            use_txt_prompt=True)
    
    tracker.set_incision_area([700, 1040+200, 1490-200, 1790])
    
    tracker.current_text_prompt = 'car'
    tracker.track()