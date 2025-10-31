import torch
import numpy as np
from pathlib import Path
import cv2
import time
from ultralytics import YOLO
from icecream import ic

class YOLODetector:
    """
    YOLO object detector class that loads models and performs inference.
    """
    def __init__(self, model_path, conf_thres=0.45, iou_thres=0.25, device='cuda:0', ):
        """
        Initialize the YOLO detector with a model.

        Args:
            model_path (str): Path to the YOLO model
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            device (str): Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        
        # Load the model
        self.model = self._load_model(model_path)
        self.names = self.model.names if hasattr(self.model, 'names') else {}
        # self.classes= classes
        
       
        
        print(f"YOLODetector initialized with model: {Path(model_path).name}")

    def _load_model(self, model_path):
        """Load YOLO model from path."""
        try:
            model= YOLO(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")
    
    
    
    def detect(self, image, classes: list= [], return_raw=False):
        """
        Run detection on an image.
        
        Args:
            image (numpy.ndarray or list): RGB image(s) (HWC format)
            return_raw (bool): Whether to return raw model output
            
        Returns:
            list: List of detections, each a dict with keys:
                - boxes (numpy.ndarray): Bounding boxes in [x1, y1, x2, y2] format
                - scores (numpy.ndarray): Confidence scores
                - classes (numpy.ndarray): Class indices
            or raw model output if return_raw is True
        """
        # Handle batch input
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        batch_results = []

        for img in images:
            if not isinstance(img, np.ndarray):
                raise ValueError("Input image must be a numpy array")
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Input image must have shape (H, W, 3)")
            
            # Convert BGR to RGB if needed
            if img.dtype == np.uint8:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            start_time = time.time()
            
            # Perform inference
            results = self.model(source=img, conf=self.conf_thres, iou=self.iou_thres, device=self.device, classes= classes)
            
            if return_raw:
                batch_results.append(results)
                continue
            
            # Parse results
            boxes = []
            scores = []
            classes = []
            if results is not None and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]#.cpu().numpy()
                            conf = box.conf[0]#.cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            boxes.append([x1, y1, x2, y2])
                            scores.append(conf)
                            classes.append(cls)
                            
                    results = {
                        "boxes": torch.tensor(boxes, device=self.device) if boxes else torch.empty((0, 4), device=self.device),
                        "scores": torch.tensor(scores, device=self.device) if scores else torch.empty(0, device=self.device),
                        "labels": classes,
                        "names": [self.names[c] for c in classes],
                        "time": time.time() - start_time
                        }
            else:
                results = {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "labels": np.array([]),
                    "names": [],
                    "time": time.time() - start_time
                }
            
        batch_results.append(results)
        ic(batch_results)
        return batch_results if is_batch else batch_results[0]
    
    