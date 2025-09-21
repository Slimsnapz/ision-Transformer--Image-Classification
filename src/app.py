%%writefile app.py

"""
app.py
Streamlit app for demoing the Vision Transformer image classification model.
How to run:
streamlit run app.py
"""
import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

# Set page config first
st.set_page_config(
    page_title="ViT Image Classification",
    page_icon="🖼️",
    layout="wide",
)

# Add some styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .main-prediction {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        margin: 20px 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_dir):
    """Load the model with caching to avoid reloading on every interaction"""
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return image_processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict(image, image_processor, model, device, top_k=3):
    """Make a prediction using the loaded model"""
    try:
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        ids = probs.argsort()[::-1][:top_k]
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(len(probs))}
        return [(id2label.get(int(i), str(i)), float(probs[int(i)])) for i in ids]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return []

st.markdown('<div class="main-header">Vision Transformer — Image Classification Demo</div>', unsafe_allow_html=True)
st.write("Upload an image and the fine-tuned ViT model will predict the label.")

# Initialize session state variables
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None

# Set default model path for your local system
default_model_path = r"C:\Users\001\Documents\workbench\indian_food_classification"  # Your current path

# Sidebar configuration
st.sidebar.header("Model Configuration")
model_dir = st.sidebar.text_input(
    "Model directory path", 
    value=default_model_path,
    help="Path to your trained Vision Transformer model directory"
)

# Load model button
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        st.session_state.image_processor, st.session_state.model, st.session_state.device = load_model(model_dir)
        if st.session_state.model is not None:
            st.session_state.model_loaded = True
            st.sidebar.success(f"Model loaded successfully on {st.session_state.device}")
        else:
            st.session_state.model_loaded = False
            st.sidebar.error("Failed to load model. Please check the path.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            # Fixed: Replaced use_column_width with use_container_width
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Only show predict button if model is loaded
            if st.session_state.model_loaded:
                if st.button("Predict", key="predict_button"):
                    with st.spinner("Predicting..."):
                        results = predict(img, st.session_state.image_processor, 
                                         st.session_state.model, st.session_state.device, top_k=5)
                        
                        if results:
                            # Display the top prediction prominently
                            top_label, top_prob = results[0]
                            st.markdown(f'<div class="main-prediction">Prediction: {top_label}</div>', 
                                       unsafe_allow_html=True)
                            st.markdown(f'**Confidence: {top_prob:.2%}**')
                            
                            # Progress bar for the top prediction
                            st.progress(top_prob)
                            
                            # Show all predictions in an expandable section
                            with st.expander("Show all predictions"):
                                st.markdown("### All Predictions:")
                                for i, (label, prob) in enumerate(results):
                                    st.markdown(f"**{i+1}. {label}**")
                                    st.markdown(f"Confidence: {prob:.4f}")
                                    st.markdown("---")
            else:
                st.warning("Please load the model first using the sidebar button.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with col2:
    if not uploaded_file:
        st.info("Upload an image to see predictions here")
    elif not st.session_state.model_loaded:
        st.warning("Model not loaded. Please load the model first.")

# Add some instructions in the sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Instructions:**
    1. Set the model directory path (or use the default)
    2. Click 'Load Model'
    3. Upload an image
    4. Click 'Predict'
    """
)
