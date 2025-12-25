"""
Streamlit web application for face mask detection
Upload images, process webcam, and visualize results
"""
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.inference import MaskDetectionInference

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-with-mask {
        color: #28a745;
        font-weight: bold;
    }
    .status-without-mask {
        color: #dc3545;
        font-weight: bold;
    }
    .status-incorrect {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">😷 Face Mask Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_path = st.text_input("Model Path", value="models/saved_model")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This system uses a CNN-based deep learning model to detect face masks in images.
    
    **Classes:**
    - ✅ With Mask
    - ❌ Without Mask
    - ⚠️ Mask Worn Incorrectly
    
    **Model:** MobileNetV2 + Dual-Head
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📊 Model Info", "📖 How to Use"])

with tab1:
    st.header("Upload and Analyze Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            st.error("Please upload a color image")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction")
            
            with st.spinner("Processing..."):
                try:
                    # Load model and make prediction
                    @st.cache_resource
                    def load_inference_model(path):
                        return MaskDetectionInference(path)
                    
                    inference = load_inference_model(model_path)
                    prediction = inference.predict(image_bgr)
                    result = inference.draw_prediction(image_bgr, prediction)
                    
                    # Convert BGR to RGB for display
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    
                    st.image(result_rgb, use_container_width=True)
                    
                    # Display metrics
                    st.markdown("### Detection Results")
                    
                    # Status indicator
                    class_id = prediction['class_id']
                    if class_id == 0:
                        status_class = "status-with-mask"
                        status_icon = "✅"
                    elif class_id == 1:
                        status_class = "status-without-mask"
                        status_icon = "❌"
                    else:
                        status_class = "status-incorrect"
                        status_icon = "⚠️"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{status_icon} Status: <span class="{status_class}">{prediction['class_name']}</span></h3>
                        <p><strong>Confidence:</strong> {prediction['confidence']:.2%}</p>
                        <p><strong>Bounding Box:</strong> [{', '.join(f'{x:.3f}' for x in prediction['bbox'])}]</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Warning if no mask detected
                    if class_id == 1:
                        st.error("⚠️ WARNING: No face mask detected!")
                    elif class_id == 2:
                        st.warning("⚠️ CAUTION: Face mask worn incorrectly!")
                    else:
                        st.success("✅ Face mask detected correctly!")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.error("Make sure the model is trained and saved correctly.")

with tab2:
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Architecture", "MobileNetV2")
        st.metric("Input Size", "416×416")
    
    with col2:
        st.metric("Number of Classes", "3")
        st.metric("Output Type", "Dual-Head")
    
    with col3:
        st.metric("Base Model", "ImageNet")
        st.metric("Loss Function", "MSE + CCE")
    
    st.markdown("---")
    
    st.subheader("Model Architecture")
    st.code("""
    Input (416, 416, 3)
            ↓
    MobileNetV2 (ImageNet pretrained)
            ↓
    Global Average Pooling
            ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
Bbox Head                      Class Head
    ↓                               ↓
Dense(128) → Dense(64)        Dense(128) → Dense(64)
    ↓                               ↓
Output(4) [bbox]              Output(3) [classes]
    """)
    
    st.subheader("Performance Metrics")
    st.info("""
    **Target Metrics:**
    - mAP@IoU=0.5 ≥ 0.75
    - Per-class F1 Score ≥ 0.70
    - Real-time Inference ≥ 15 FPS
    """)

with tab3:
    st.header("How to Use the System")
    
    st.markdown("""
    ### 📤 Uploading Images
    1. Go to the "Upload Image" tab
    2. Click "Browse files" or drag and drop an image
    3. Wait for the model to process the image
    4. View the prediction results and confidence scores
    
    ### 🎯 Understanding Results
    
    - **✅ With Mask (Green)**: Person is wearing a face mask correctly
    - **❌ Without Mask (Red)**: Person is not wearing a face mask
    - **⚠️ Incorrect Mask (Yellow)**: Person is wearing a mask, but incorrectly
    
    ### 🔧 Adjusting Settings
    
    Use the sidebar to:
    - Change the model path (if you have multiple trained models)
    - Adjust the confidence threshold
    
    ### 📊 Interpreting Confidence Scores
    
    - **> 90%**: Very confident prediction
    - **70-90%**: Confident prediction
    - **50-70%**: Moderately confident
    - **< 50%**: Low confidence (use with caution)
    
    ### 💡 Tips for Best Results
    
    - Use clear, well-lit images
    - Ensure faces are visible and not too small
    - Avoid very blurry or low-quality images
    - The system works best with frontal or near-frontal face views
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Face Mask Detection System | Powered by TensorFlow & Streamlit</p>
    <p>Deep Learning Internship Project | Smart Surveillance Company</p>
</div>
""", unsafe_allow_html=True)
