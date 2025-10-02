import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import gdown

# Set page config
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

# Title and description
st.title("🫁 Lung Cancer Detection")
st.markdown("Upload a lung CT scan image to classify it into one of the following categories:")
st.markdown("- **Normal** - **Adenocarcinoma** - **Large Cell Carcinoma** - **Squamous Cell Carcinoma**")

# Function to download model if not exists
@st.cache_resource
def load_cancer_model():
    model_path = 'trained_lung_cancer_model.h5'
    
    # If model doesn't exist locally, download it
    if not os.path.exists(model_path):
        st.warning("Downloading model file... This may take a few minutes.")
        try:
            url = 'https://drive.google.com/uc?id=1Q9lcSbBvRqwkcfSoNpsn0lPEaNwyDhNl'
            gdown.download(url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return None
    
    try:
        # Try loading with different methods for compatibility
        model = load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        try:
            # Alternative loading method
            model = load_model(model_path)
            st.success("Model loaded successfully with alternative method!")
            return model
        except Exception as e2:
            st.error(f"All loading methods failed: {str(e2)}")
            return None

# Load model
model = load_cancer_model()

# Class labels
CLASS_NAMES = ['normal', 'adenocarcinoma', 'large_cell_carcinoma', 'squamous_cell_carcinoma']

# Function to preprocess image
def preprocess_image(img):
    try:
        # Resize image to match model's expected sizing
        img = img.resize((256, 256))
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        # Add batch dimension
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
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            # Convert to PIL Image
            pil_image = image.load_img(uploaded_file)
            st.image(pil_image, caption="Uploaded CT Scan", use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Preprocess the image
            processed_image = preprocess_image(pil_image)
            
            if processed_image is not None:
                # Make prediction
                with st.spinner('Analyzing image...'):
                    try:
                        predictions = model.predict(processed_image)
                        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                        confidence = np.max(predictions[0])
                        
                        # Display results
                        st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
                        st.info(f"**Confidence:** {confidence:.2%}")
                        
                        # Show confidence scores for all classes
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