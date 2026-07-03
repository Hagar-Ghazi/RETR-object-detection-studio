import os
from PIL import Image
from models import ModelManager

def main():
    print("=== Object Detection Project Test Script ===")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Check what backends are available
    print("\n--- Package Availability ---")
    for backend in ["RF-DETR (Roboflow)", "RT-DETR (Ultralytics)", "Original DETR (HuggingFace)"]:
        available = manager.is_backend_available(backend)
        print(f"{backend}: {'Available' if available else 'NOT Available'}")
        
    # Create a blank image for testing (640x640)
    print("\nCreating test dummy image...")
    dummy_image = Image.new("RGB", (640, 640), color="white")
    
    # Test loading and predicting on the available backends
    print("\n--- Testing Model Inference ---")
    
    # We will test Original DETR first, as it installs very reliably on Hugging Face
    if manager.is_backend_available("Original DETR (HuggingFace)"):
        try:
            print("Testing Original DETR (HuggingFace)...")
            detections, class_names, latency = manager.predict(
                backend="Original DETR (HuggingFace)",
                model_size="Medium",
                image=dummy_image,
                threshold=0.1
            )
            print(f"Success! Detected {len(detections)} objects. Latency: {latency:.2f} ms")
        except Exception as e:
            print(f"Error testing Original DETR: {e}")
            
    # Test RT-DETR
    if manager.is_backend_available("RT-DETR (Ultralytics)"):
        try:
            print("Testing RT-DETR (Ultralytics)...")
            detections, class_names, latency = manager.predict(
                backend="RT-DETR (Ultralytics)",
                model_size="Medium",
                image=dummy_image,
                threshold=0.1
            )
            print(f"Success! Detected {len(detections)} objects. Latency: {latency:.2f} ms")
        except Exception as e:
            print(f"Error testing RT-DETR: {e}")
            
    # Test RF-DETR
    if manager.is_backend_available("RF-DETR (Roboflow)"):
        try:
            print("Testing RF-DETR (Roboflow)...")
            detections, class_names, latency = manager.predict(
                backend="RF-DETR (Roboflow)",
                model_size="Medium",
                image=dummy_image,
                threshold=0.1
            )
            print(f"Success! Detected {len(detections)} objects. Latency: {latency:.2f} ms")
        except Exception as e:
            print(f"Error testing RF-DETR: {e}")

if __name__ == "__main__":
    main()
