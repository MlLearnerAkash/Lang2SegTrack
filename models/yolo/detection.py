import torch
import numpy as np
from pathlib import Path
import cv2
import time


class YOLODetector:
    """
    YOLO object detector class that loads models and performs inference.
    """
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, device='cuda:0'):
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
        
        # Get model info
        self.stride = int(self.model.stride.max()) if hasattr(self.model, 'stride') else 32
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.input_size = self.model.module.input_size if hasattr(self.model, 'module') else getattr(self.model, 'input_size', [640, 640])
        
        # Warm up the model
        self._warmup()
        
        print(f"YOLODetector initialized with model: {Path(model_path).name}")

    def _load_model(self, model_path):
        """Load YOLO model from path."""
        try:
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                model = torch.load(model_path, map_location=self.device)['model']
                model = model.float().to(self.device)
                model.eval()
                return model
            else:
                # Attempt to load as a TorchScript model
                return torch.jit.load(model_path).to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")
    
    def _warmup(self):
        """Warm up the model with a dummy input."""
        dummy_input = torch.zeros((1, 3, *self.input_size), device=self.device)
        for _ in range(3):
            _ = self.model(dummy_input)
    
    def preprocess(self, image):
        """
        Preprocess image for YOLO model input.
        
        Args:
            image (numpy.ndarray): RGB image in HWC format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Resize and pad image
        img = cv2.resize(image, tuple(self.input_size))
        
        # Convert to torch tensor
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
    
    def detect(self, image, return_raw=False):
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
        
        # Process each image
        for img in images:
            # Preprocess
            input_tensor = self.preprocess(img)
            
            # Get original image dimensions
            orig_h, orig_w = img.shape[:2]
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                pred = self.model(input_tensor)
            
            # Process predictions
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # Apply NMS
            if hasattr(self.model, 'non_max_suppression'):
                pred = self.model.non_max_suppression(
                    pred, self.conf_thres, self.iou_thres
                )[0]
            else:
                # Manual NMS if model doesn't have it built-in
                pred = self._non_max_suppression(pred[0], self.conf_thres, self.iou_thres)
            
            if return_raw:
                batch_results.append(pred)
                continue
                
            # Convert to numpy and scale to original image dimensions
            if len(pred):
                boxes = pred[:, :4].cpu().numpy()
                
                # Rescale boxes to original image dimensions
                scale_w = orig_w / self.input_size[0]
                scale_h = orig_h / self.input_size[1]
                
                boxes[:, 0] *= scale_w  # x1
                boxes[:, 2] *= scale_w  # x2
                boxes[:, 1] *= scale_h  # y1
                boxes[:, 3] *= scale_h  # y2
                
                scores = pred[:, 4].cpu().numpy()
                classes = pred[:, 5].cpu().numpy().astype(int)
                
                results = {
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes,
                    "names": [self.names[c] for c in classes],
                    "time": time.time() - start_time
                }
            else:
                results = {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "names": [],
                    "time": time.time() - start_time
                }
                
            batch_results.append(results)
        
        return batch_results if is_batch else batch_results[0]
    
    def _non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """
        Performs Non-Maximum Suppression (NMS) on inference results
        Returns detections with shape:
            (x1, y1, x2, y2, confidence, class)
        """
        max_det = 300  # Maximum number of detections
        
        # Class-specific NMS
        nc = prediction.shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres
        
        # Settings
        min_wh, max_wh = 2, 4096  # min and max box width and height
        
        # Filter by confidence
        prediction = prediction[xc]
        
        # If none remain, process next image
        if not prediction.shape[0]:
            return prediction
            
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = self._xywh2xyxy(prediction[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = prediction[:, 4:5], prediction[:, 5:6].long()
        x = torch.cat((box, conf, j.float()), 1)
        
        # Apply NMS
        boxes, scores = x[:, :4], x[:, 4]  # boxes, scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        
        return x[i]
    
    def _xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y