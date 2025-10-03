import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

# Title and description
st.title("🫁 Lung Cancer Detection")
st.markdown("Upload a lung CT scan image to classify it into one of the following categories:")
st.markdown("- **Normal** - **Adenocarcinoma** - **Large Cell Carcinoma** - **Squamous Cell Carcinoma**")

# Function to load model
@st.cache_resource
def load_cancer_model():
    model_path = "trained_lung_cancer_model.h5"
    
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload or place 'trained_lung_cancer_model.h5' in the app directory.")
        return None
    
    try:
        model = load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_cancer_model()

# Class labels
CLASS_NAMES = ["normal", "adenocarcinoma", "large_cell_carcinoma", "squamous_cell_carcinoma"]

# Function to preprocess image
def preprocess_image(img):
    try:
        # ✅ Resize to match model's expected input (350x350)
        img = img.resize((350, 350))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# File uploader
uploaded_file = st.file_uploader("Choose a lung CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("Model is not available. Please try refreshing the page.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            pil_image = Image.open(uploaded_file).convert("RGB")
            # Updated per-streamlit deprecation warning:
            st.image(pil_image, caption="Uploaded CT Scan", use_container_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            processed_image = preprocess_image(pil_image)
            
            if processed_image is not None:
                with st.spinner("Analyzing image..."):
                    try:
                        predictions = model.predict(processed_image)
                        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                        confidence = np.max(predictions[0])
                        
                        st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
                        st.info(f"**Confidence:** {confidence:.2%}")
                        
                        st.subheader("Confidence Scores:")
                        for i, class_name in enumerate(CLASS_NAMES):
                            score = predictions[0][i]
                            st.write(f"- {class_name.replace('_', ' ').title()}: {score:.2%}")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

# Instructions section
st.markdown("---")
st.subheader("📋 Instructions")
st.markdown("""
1. Upload a clear CT scan image of the lung  
2. Supported formats: JPG, JPEG, PNG  
3. Wait for the model to analyze the image  
4. Review the prediction and confidence scores  
""")

# Disclaimer
st.markdown("---")
st.markdown("""
<div style='background-color: #ffcccc; padding: 10px; border-radius: 5px;'>
<small><strong>Disclaimer:</strong> This tool is for educational and research purposes only. 
Always consult with healthcare professionals for medical diagnoses.</small>
</div>
""", unsafe_allow_html=True)
