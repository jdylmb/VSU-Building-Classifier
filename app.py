import streamlit as st
import os
from PIL import Image
import numpy as np
from utils import load_model, preprocess_image

# Set page configuration
st.set_page_config(
    page_title="Building Type Classifier",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS with forest green palette
st.markdown("""
    <style>
    /* Color palette */
    :root {
        --dark-green: #2C5530;
        --medium-green: #4A7856;
        --light-green: #95B8A2;
        --pale-green: #E8F3E9;
    }
    
    .main {
        background-color: white;
    }
    
    h1, h2, h3 {
        color: var(--dark-green) !important;
    }
    
    .stButton>button {
        background-color: var(--dark-green) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--medium-green) !important;
        transform: translateY(-2px);
    }
    
    /* File uploader */
    .stUploadedFile {
        border: 2px dashed var(--medium-green) !important;
    }
    
    /* Prediction box */
    .prediction-box {
        background-color: var(--pale-green);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid var(--light-green);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--medium-green);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: var(--pale-green);
        color: var(--dark-green);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🏢 Building Type Classifier")
    st.write("Upload an image of a building to classify its type.")
    
    # Load the model and class names
    if os.path.exists('building_classifier.h5'):
        model = load_model('building_classifier.h5')
        with open('class_names.txt', 'r') as f:
            class_names = f.read().splitlines()
    else:
        st.error("Model file not found. Please train the model first.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=None, width=600)
        
        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display results with styled container
        st.markdown("""
            <div class='prediction-box'>
                <h3>Prediction Results:</h3>
                <p style='color: var(--dark-green); font-size: 1.2em; font-weight: bold;'>
                    Building Type: {}</p>
                <p style='color: var(--medium-green); font-size: 1.1em;'>
                    Confidence: {:.2f}%</p>
            </div>
        """.format(predicted_class, confidence), unsafe_allow_html=True)
        
        # Display all class probabilities with styled container
        st.markdown("<h3>All Class Probabilities:</h3>", unsafe_allow_html=True)
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        for class_name, prob in zip(class_names, prediction[0]):
            prob_percentage = prob * 100
            st.markdown(
                f"<p style='color: var(--dark-green);'>{class_name}: "
                f"<span style='color: var(--medium-green); font-weight: bold;'>{prob_percentage:.2f}%</span></p>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()