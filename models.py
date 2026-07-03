import time
import torch
import numpy as np
from PIL import Image
import supervision as sv

# Attempt to import RF-DETR
try:
    import rfdetr
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    from rfdetr.util.coco_classes import COCO_CLASSES as RF_COCO_CLASSES
    RF_DETR_AVAILABLE = True
except ImportError:
    RF_DETR_AVAILABLE = False
    RF_COCO_CLASSES = []

# Attempt to import Ultralytics
try:
    from ultralytics import RTDETR as UltralyticsRTDETR
    RT_DETR_AVAILABLE = True
except ImportError:
    RT_DETR_AVAILABLE = False

# Attempt to import Hugging Face Transformers
try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    HF_DETR_AVAILABLE = True
except ImportError:
    HF_DETR_AVAILABLE = False


class ModelManager:
    """Manages loading and inference of multiple DETR-family backends."""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def is_backend_available(self, backend: str) -> bool:
        """Check if the libraries for a specific backend are installed."""
        if backend == "RF-DETR (Roboflow)":
            return RF_DETR_AVAILABLE
        elif backend == "RT-DETR (Ultralytics)":
            return RT_DETR_AVAILABLE
        elif backend == "Original DETR (HuggingFace)":
            return HF_DETR_AVAILABLE
        return False

    def load_model(self, backend: str, model_size: str = "Medium"):
        """Load a model with a specific backend and size, caching it for subsequent calls."""
        model_key = f"{backend}_{model_size}"
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        print(f"Loading {backend} model ({model_size}) on {self.device}...")

        if backend == "RF-DETR (Roboflow)":
            if not RF_DETR_AVAILABLE:
                raise ImportError("rfdetr package is not installed.")
            
            # Map size to correct class, passing device parameter directly
            if model_size == "Nano":
                model = RFDETRNano(device=self.device)
            elif model_size == "Small":
                model = RFDETRSmall(device=self.device)
            elif model_size == "Large":
                model = RFDETRLarge(device=self.device)
            else:
                model = RFDETRMedium(device=self.device)  # Default to Medium
            
            self.loaded_models[model_key] = model
            return model

        elif backend == "RT-DETR (Ultralytics)":
            if not RT_DETR_AVAILABLE:
                raise ImportError("ultralytics package is not installed.")
            
            # Map size to Ultralytics RT-DETR weight files
            # Size mapping: Small -> rtdetr-l.pt, Medium -> rtdetr-l.pt, Large -> rtdetr-x.pt
            if model_size == "Large":
                weights_path = "rtdetr-x.pt"
            else:
                weights_path = "rtdetr-l.pt"  # L is default for Ultralytics RT-DETR
                
            model = UltralyticsRTDETR(weights_path)
            # Move to device
            model.to(self.device)
            self.loaded_models[model_key] = model
            return model

        elif backend == "Original DETR (HuggingFace)":
            if not HF_DETR_AVAILABLE:
                raise ImportError("transformers and timm packages are not installed.")
            
            # facebook/detr-resnet-50 is the standard COCO baseline
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
            
            self.loaded_models[model_key] = (processor, model)
            return (processor, model)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(self, backend: str, model_size: str, image: Image.Image, threshold: float = 0.3):
        """Run object detection on an image, measuring execution time and returning predictions."""
        start_time = time.time()
        model_instance = self.load_model(backend, model_size)
        
        # Convert image to RGB PIL if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        width, height = image.size

        if backend == "RF-DETR (Roboflow)":
            # rfdetr predict handles PIL images, returns supervision.Detections
            # Note: We pass the threshold to predict or filter post-inference
            detections = model_instance.predict(image, threshold=threshold)
            
            # Map class names
            class_names = [RF_COCO_CLASSES[class_id] for class_id in detections.class_id]
            
        elif backend == "RT-DETR (Ultralytics)":
            # ultralytics predict accepts PIL images
            results = model_instance.predict(image, conf=threshold, device=self.device, verbose=False)
            result = results[0]
            
            # Convert to supervision.Detections
            detections = sv.Detections.from_ultralytics(result)
            
            # Map class names
            class_names = [result.names[class_id] for class_id in detections.class_id]
            
        elif backend == "Original DETR (HuggingFace)":
            processor, model = model_instance
            
            # Preprocess image
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Postprocess predictions
            # target_sizes is height, width
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=[(height, width)], 
                threshold=threshold
            )[0]
            
            # Extract coordinates, scores, and labels
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            
            if len(boxes) > 0:
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=scores,
                    class_id=labels
                )
                class_names = [model.config.id2label[class_id] for class_id in detections.class_id]
            else:
                detections = sv.Detections.empty()
                class_names = []

        inference_time_ms = (time.time() - start_time) * 1000
        
        return detections, class_names, inference_time_ms
