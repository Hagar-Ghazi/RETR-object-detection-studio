import streamlit as st
from PIL import Image
import os

# Base directory of the script to resolve file paths robustly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import plotly.express as px
import json

# Import our custom modules
from models import ModelManager
from utils import annotate_image, get_detections_df, get_crop_images, convert_image_to_bytes

# Initialize model manager (cached across reruns)
@st.cache_resource
def get_model_manager():
    return ModelManager()

model_manager = get_model_manager()

# Set page config
st.set_page_config(
    page_title="RF-DETR Detection Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium UI
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">

<style>
    /* Theme fonts */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    .stCodeBlock, code {
        font-family: 'Space Grotesk', monospace !important;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-card h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0 0 0.5rem 0;
        color: #ffffff;
    }
    .header-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Stat Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #2a5298;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #2a5298;
        margin-bottom: 0.2rem;
    }
    .dark .metric-val {
        color: #5d9cec;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }
    
    /* Crop card styling */
    .crop-card {
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-card">
    <h1>🎯 RF-DETR Object Detection Studio</h1>
    <p>Compare state-of-the-art DEtection TRansformer models (RF-DETR, RT-DETR, and DETR) with custom visual rendering, real-time metrics, and object-specific inspection.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("## ⚙️ Configuration")

# Model Engine Selection
backend_options = [
    "RF-DETR (Roboflow)",
    "RT-DETR (Ultralytics)",
    "Original DETR (HuggingFace)"
]
selected_backend = st.sidebar.selectbox(
    "Select Model Engine",
    backend_options,
    help="RF-DETR runs Roboflow's model, RT-DETR runs Baidu's real-time model, and Original DETR runs the Facebook ResNet-50 baseline."
)

# Size selection (Not applicable to HF original DETR)
size_disabled = selected_backend == "Original DETR (HuggingFace)"
size_options = ["Nano", "Small", "Medium", "Large"]
selected_size = st.sidebar.selectbox(
    "Select Model Size",
    size_options,
    index=2, # Default to Medium
    disabled=size_disabled,
    help="Different model dimensions. Larger models have higher accuracy but are slower. Nano/Small are optimized for speed."
)

# Threshold sliders
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.30,
    step=0.05,
    help="Minimum confidence score required to display a detection box."
)

# Stylization parameters
st.sidebar.markdown("### 🎨 Box Styling")
box_thickness = st.sidebar.slider("Box Thickness (px)", 1, 5, 2)
fill_alpha = st.sidebar.slider("Fill Opacity", 0.00, 0.50, 0.15, step=0.05)
font_size = st.sidebar.slider("Label Font Size (pt)", 8, 24, 12)
show_labels = st.sidebar.checkbox("Show Labels", value=True)

# Check backend availability and show warnings
backend_available = model_manager.is_backend_available(selected_backend)
if not backend_available:
    st.sidebar.warning(f"⚠️ {selected_backend} is not installed locally. The app will run, but selecting this will throw an import error. Please install required libraries to use it.")

# ----------------- IMAGE INPUT -----------------
st.markdown("### 🖼️ Input Image")

# Session state key to reset the file uploader programmatically
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Use radio selector to choose the active source exclusively
input_source = st.radio(
    "Select Input Source",
    ["Upload Custom Image", "Try Sample Images"],
    horizontal=True,
    help="Toggle between uploading your own photo or selecting one of our high-quality testing images."
)

image_to_process = None

if input_source == "Upload Custom Image":
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"uploader_{st.session_state['uploader_key']}",
        help="Upload a JPG, PNG or WebP image to execute object detection."
    )
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
        
        # Add a clear button for quick reset
        if st.button("❌ Clear Uploaded Image", use_container_width=True):
            st.session_state["uploader_key"] += 1
            st.rerun()

else:
    # Preloaded sample images
    sample_images = {
        "City Street (Traffic & Pedestrians)": os.path.join(SCRIPT_DIR, "sample_images", "street.png"),
        "Cozy Living Room (Furniture & Decor)": os.path.join(SCRIPT_DIR, "sample_images", "living_room.png"),
        "Office Desk (Electronics & Workspace)": os.path.join(SCRIPT_DIR, "sample_images", "office.png")
    }
    
    selected_sample = st.selectbox(
        "Select a sample image:",
        list(sample_images.keys())
    )
    
    sample_path = sample_images[selected_sample]
    if os.path.exists(sample_path):
        try:
            image_to_process = Image.open(sample_path)
        except Exception as e:
            st.error(f"Error loading sample image: {e}")
            
    # Small preview grid of samples
    cols = st.columns(3)
    for idx, (name, path) in enumerate(sample_images.items()):
        with cols[idx]:
            if os.path.exists(path):
                img = Image.open(path)
                st.image(img, caption=name, use_container_width=True)

# ----------------- INFERENCE & VISUALIZATION -----------------
if image_to_process is not None:
    # Trigger model run
    with st.spinner(f"Running inference with {selected_backend} ({selected_size})..."):
        try:
            detections, class_names, latency = model_manager.predict(
                backend=selected_backend,
                model_size=selected_size,
                image=image_to_process,
                threshold=conf_threshold
            )
            model_error = None
        except Exception as e:
            detections = None
            class_names = []
            latency = 0.0
            model_error = str(e)
            
    # Show error details or results
    if model_error:
        st.error(f"### ❌ Error Executing Model backend: {selected_backend}")
        st.write(f"**Details:** `{model_error}`")
        st.markdown(f"""
        This is likely because the package dependencies are not compiled on this environment.
        
        #### How to Fix:
        *   **For RF-DETR**: Install CMake and build dependencies:
            `pip install rfdetr`
        *   **For RT-DETR**: Install Ultralytics:
            `pip install ultralytics`
        *   **For Original DETR**: Install Transformers and timm:
            `pip install transformers timm`
        
        **Alternative:** Switch to another model engine in the sidebar (e.g. **RT-DETR (Ultralytics)** or **Original DETR (HuggingFace)** which install very easily).
        """)
    else:
        num_detections = len(detections)
        
        # Display performance metric cards
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{selected_backend.split(' ')[0]}</div>
                <div class="metric-label">Active Architecture</div>
            </div>
            """, unsafe_allow_html=True)
        with met_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{latency:.1f} ms</div>
                <div class="metric-label">Inference Time (Latency)</div>
            </div>
            """, unsafe_allow_html=True)
        with met_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{num_detections}</div>
                <div class="metric-label">Objects Detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("") # spacing

        # --- TABS FOR RESULTS ---
        tab_dashboard, tab_crops = st.tabs([
            "📊 Detection Dashboard", 
            "🔍 Crop & Inspect"
        ])

        with tab_dashboard:
            col_img, col_metrics = st.columns([3, 2])
            
            with col_img:
                st.markdown("#### Annotated Output")
                annotated_img = annotate_image(
                    image=image_to_process,
                    detections=detections,
                    class_names=class_names,
                    box_thickness=box_thickness,
                    fill_alpha=fill_alpha,
                    show_labels=show_labels,
                    font_size=font_size
                )
                st.image(annotated_img, use_container_width=True)
                
                # Download Annotated Image
                img_bytes = convert_image_to_bytes(annotated_img)
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_bytes,
                    file_name=f"detected_{selected_backend.split(' ')[0].lower()}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
            with col_metrics:
                st.markdown("#### Detections Analysis")
                
                # Get dataframe of results
                df_detections = get_detections_df(detections, class_names)
                
                if not df_detections.empty:
                    # Class Frequency bar chart
                    class_counts = df_detections["Class"].value_counts().reset_index()
                    class_counts.columns = ["Class", "Count"]
                    
                    fig = px.bar(
                        class_counts, 
                        x="Class", 
                        y="Count", 
                        color="Class",
                        title="Detected Object Frequencies",
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display Table
                    st.markdown("##### Detailed Coordinate Table")
                    # Format float columns
                    st.dataframe(
                        df_detections.style.format({"Confidence": "{:.2%}"}),
                        use_container_width=True,
                        height=250
                    )
                    
                    # Export options
                    st.markdown("##### Export Bounding Box Data")
                    csv_data = df_detections.to_csv(index=False)
                    json_data = json.dumps(df_detections.to_dict(orient="records"), indent=2)
                    
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv_data,
                            file_name="detections.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with exp_col2:
                        st.download_button(
                            label="👾 Download JSON",
                            data=json_data,
                            file_name="detections.json",
                            mime="application/json",
                            use_container_width=True
                        )
                else:
                    st.info("No objects detected above the selected confidence threshold.")
                    
        with tab_crops:
            st.markdown("#### Crop & Inspect Detected Regions")
            st.write("Click and view zoomed crop patches of the detected objects:")
            
            crops = get_crop_images(image_to_process, detections, class_names)
            
            if crops:
                # Display in a grid of 4 columns
                cols_per_row = 4
                rows = [crops[i:i + cols_per_row] for i in range(0, len(crops), cols_per_row)]
                
                for row_idx, row_crops in enumerate(rows):
                    grid_cols = st.columns(cols_per_row)
                    for col_idx, crop_data in enumerate(row_crops):
                        with grid_cols[col_idx]:
                            # Draw border colored corresponding to the class
                            c = crop_data['color']
                            border_css = f"rgba({c[0]}, {c[1]}, {c[2]}, 0.8)"
                            
                            st.markdown(f"""
                            <div class="crop-card" style="border-top: 4px solid {border_css};">
                                <div style="font-weight:600; font-size:1.1rem; color:#2c3e50;">{crop_data['label']}</div>
                                <div style="font-size:0.9rem; color:#7f8c8d; margin-bottom:0.5rem;">Confidence: {crop_data['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(crop_data['image'], use_container_width=True)
            else:
                st.info("No objects cropped. Adjust the confidence threshold in the sidebar if you expect detections.")
                

            
else:
    st.info("💡 Please upload an image or select one of the sample images above to execute the model and display results.")
