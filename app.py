import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = load_model('trained_lung_cancer_model.h5')

# Class labels
CLASS_NAMES = ['normal', 'adenocarcinoma', 'large_cell_carcinoma', 'squamous_cell_carcinoma']

st.title("Lung Cancer Classification System")
st.write("Upload a lung CT scan image for classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = image.load_img(uploaded_file, target_size=(350, 350))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Show results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Class", CLASS_NAMES[predicted_class])
    
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    # Show probabilities
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Probability': predictions[0]
    })
    st.bar_chart(prob_df.set_index('Class'))