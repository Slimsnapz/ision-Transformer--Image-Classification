"""
app.py
Streamlit app for demoing the Vision Transformer image classification model.
How to run:
streamlit run app.py -- --model_dir models/exp1
"""
import streamlit as st
from PIL import Image
import tempfile
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_model(model_dir):
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return image_processor, model, device

def predict(image, image_processor, model, device, top_k=3):
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    ids = probs.argsort()[::-1][:top_k]
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {i:str(i) for i in range(len(probs))}
    return [(id2label.get(int(i), str(i)), float(probs[int(i)])) for i in ids]

st.title("Vision Transformer — Image Classification Demo")
st.write("Upload an image and the fine-tuned ViT model will predict the label.")

model_dir = st.sidebar.text_input("Model directory", value="models/exp1")
if st.sidebar.button("Load model"):
    with st.spinner("Loading model..."):
        image_processor, model, device = load_model(model_dir)
    st.success("Model loaded.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        try:
            image_processor, model, device
        except NameError:
            with st.spinner("Loading model..."):
                image_processor, model, device = load_model(model_dir)
        with st.spinner("Predicting..."):
            results = predict(img, image_processor, model, device, top_k=5)
        st.write("Predictions:")
        for label, prob in results:
            st.write(f"- {label}: {prob:.4f}")
