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
    def __init__(self, model_path, conf_thres=0.45, iou_thres=0.25, device='cuda:0', imgsz= 2480):
        """
        Initialize the YOLO detector with a model.

        Args:
            model_path (str): Path to the YOLO model
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            device (str): Device to run inference on ('cuda:0', 'cpu', etc.)
            imgsz (int): Image size for inference
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.imgsz = imgsz
        
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
        """
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        batch_results = []

        for img in images:
            if not isinstance(img, np.ndarray):
                raise ValueError("Input image must be a numpy array")
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Input image must have shape (H, W, 3)")
            
            start_time = time.time()
            
            # USE .predict() INSTEAD OF __call__()
            results = self.model.predict(
                source=img, 
                conf=self.conf_thres, 
                iou=self.iou_thres, 
                device=self.device, 
                classes=classes,
                verbose=False,
                imgsz= self.imgsz
            )
            
            if return_raw:
                batch_results.append(results)
                continue
            
            # Parse results
            boxes = []
            scores = []
            class_ids = []
            
            if results is not None and len(results) > 0:
                result = results[0]  # Get first result
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if len(result.boxes) > 0:
                        boxes_data = result.boxes.xyxy.cpu().numpy()
                        scores_data = result.boxes.conf.cpu().numpy()
                        classes_data = result.boxes.cls.cpu().numpy().astype(int)
                        
                        boxes = boxes_data.tolist()
                        scores = scores_data.tolist()
                        class_ids = classes_data.tolist()
            
            parsed_results = {
                "boxes": torch.tensor(boxes, device=self.device) if boxes else torch.empty((0, 4), device=self.device),
                "scores": torch.tensor(scores, device=self.device) if scores else torch.empty(0, device=self.device),
                "labels": class_ids,
                "names": [self.names[c] for c in class_ids],
                "time": time.time() - start_time
            }
            
            batch_results.append(parsed_results)
        
        return batch_results if is_batch else batch_results[0]

    def predict_raw(self, image, **kwargs):
        return self.model.predict(image, **kwargs)
    
    