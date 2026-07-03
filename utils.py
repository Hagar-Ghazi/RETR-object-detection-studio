import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def get_color_palette(class_id: int):
    """Generate a soft, professional color palette based on class ID."""
    # List of 20 distinct soft, modern colors (RGBA)
    palette = [
        (52, 152, 219, 255),    # Soft Blue
        (46, 204, 113, 255),    # Soft Green
        (155, 89, 182, 255),    # Soft Purple
        (230, 126, 34, 255),    # Soft Orange
        (231, 76, 60, 255),     # Soft Red
        (26, 188, 156, 255),    # Turquoise
        (241, 196, 15, 255),    # Soft Yellow
        (52, 73, 94, 255),      # Navy Gray
        (243, 156, 18, 255),    # Amber
        (211, 84, 0, 255),      # Rust Orange
        (192, 57, 43, 255),     # Dark Red
        (142, 68, 173, 255),    # Dark Purple
        (41, 128, 185, 255),    # Dark Blue
        (39, 174, 96, 255),     # Dark Green
        (22, 160, 133, 255),    # Dark Turquoise
        (39, 60, 117, 255),     # Slate Blue
        (232, 65, 24, 255),     # Crimson
        (156, 136, 255, 255),   # Lavender
        (251, 197, 49, 255),    # Gold
        (76, 209, 224, 255)     # Cyan
    ]
    return palette[class_id % len(palette)]

def annotate_image(
    image: Image.Image,
    detections,
    class_names,
    box_thickness=2,
    fill_alpha=0.15,
    show_labels=True,
    font_size=14
) -> Image.Image:
    """Draw custom semi-transparent bounding boxes and labels onto the image."""
    if len(detections) == 0:
        return image

    # Convert source image to RGBA to support transparency
    source_img = image.convert("RGBA")
    
    # Create overlay layers for boxes and text labels
    box_layer = Image.new("RGBA", source_img.size, (0, 0, 0, 0))
    label_layer = Image.new("RGBA", source_img.size, (0, 0, 0, 0))
    
    draw_box = ImageDraw.Draw(box_layer)
    draw_label = ImageDraw.Draw(label_layer)
    
    # Try to load a nice font, fallback to default if not found
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    for i in range(len(detections.xyxy)):
        # Coordinates
        xmin, ymin, xmax, ymax = detections.xyxy[i]
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]
        class_name = class_names[i]
        
        # Color matching
        color = get_color_palette(class_id)
        border_color = (color[0], color[1], color[2], 255)
        fill_color = (color[0], color[1], color[2], int(255 * fill_alpha))
        
        # Draw bounding box
        draw_box.rectangle(
            [xmin, ymin, xmax, ymax],
            fill=fill_color,
            outline=border_color,
            width=box_thickness
        )
        
        if show_labels:
            # Setup text label: "Class Conf%"
            label_text = f"{class_name} {confidence:.0%}"
            
            # Get text size
            try:
                # PIL version >= 10.0.0 uses getbbox
                text_bbox = draw_label.textbbox((0, 0), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # Old PIL compatibility
                text_w, text_h = draw_label.textsize(label_text, font=font)
                
            # Place label slightly above box, or inside box if too close to top
            padding = 4
            text_xmin = xmin
            text_ymin = ymin - text_h - (padding * 2)
            
            if text_ymin < 0:
                text_ymin = ymin + padding
                
            text_xmax = text_xmin + text_w + (padding * 2)
            text_ymax = text_ymin + text_h + (padding * 2)
            
            # Draw label background pill (rounded rectangle if supported, otherwise rectangle)
            try:
                draw_label.rounded_rectangle(
                    [text_xmin, text_ymin, text_xmax, text_ymax],
                    radius=3,
                    fill=border_color
                )
            except AttributeError:
                draw_label.rectangle(
                    [text_xmin, text_ymin, text_xmax, text_ymax],
                    fill=border_color
                )
                
            # Draw label text
            # Calculate offset to center text vertically in the pill
            text_y_offset = (text_ymax - text_ymin - text_h) // 2 - 1
            draw_label.text(
                (text_xmin + padding, text_ymin + text_y_offset),
                label_text,
                fill=(255, 255, 255, 255),
                font=font
            )
            
    # Composite layers
    out_img = Image.alpha_composite(source_img, box_layer)
    out_img = Image.alpha_composite(out_img, label_layer)
    
    return out_img.convert("RGB")

def get_detections_df(detections, class_names) -> pd.DataFrame:
    """Convert detection results into a clean, formatable Pandas DataFrame."""
    if len(detections) == 0:
        return pd.DataFrame(columns=["Class", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax", "Width", "Height"])
        
    records = []
    for i in range(len(detections.xyxy)):
        xmin, ymin, xmax, ymax = detections.xyxy[i]
        class_name = class_names[i]
        confidence = float(detections.confidence[i])
        
        width = xmax - xmin
        height = ymax - ymin
        
        records.append({
            "Class": class_name,
            "Confidence": confidence,
            "Xmin": int(xmin),
            "Ymin": int(ymin),
            "Xmax": int(xmax),
            "Ymax": int(ymax),
            "Width": int(width),
            "Height": int(height)
        })
        
    df = pd.DataFrame(records)
    # Sort by confidence descending
    return df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)

def get_crop_images(image: Image.Image, detections, class_names, margin=5):
    """Crop each detected object from the original image, returning crops with labels."""
    crops = []
    if len(detections) == 0:
        return crops
        
    width, height = image.size
    
    for i in range(len(detections.xyxy)):
        xmin, ymin, xmax, ymax = detections.xyxy[i]
        class_name = class_names[i]
        confidence = float(detections.confidence[i])
        class_id = int(detections.class_id[i])
        
        # Add margin to crop, keeping within image bounds
        xmin_m = max(0, int(xmin) - margin)
        ymin_m = max(0, int(ymin) - margin)
        xmax_m = min(width, int(xmax) + margin)
        ymax_m = min(height, int(ymax) + margin)
        
        # Avoid cropping 0-width/0-height regions
        if xmax_m <= xmin_m or ymax_m <= ymin_m:
            continue
            
        crop_img = image.crop((xmin_m, ymin_m, xmax_m, ymax_m))
        
        crops.append({
            "image": crop_img,
            "label": class_name,
            "confidence": confidence,
            "color": get_color_palette(class_id)
        })
        
    # Sort crops by confidence descending
    return sorted(crops, key=lambda x: x["confidence"], reverse=True)

def convert_image_to_bytes(image: Image.Image) -> bytes:
    """Helper to convert a PIL Image to bytes for file downloads."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()
