import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import keras
import gdown
import os 
import tempfile

# Setup
st.set_page_config(page_title="SkinAI", layout="centered")
class_names = ["chickenpox", "hfmd", "measles", "unknown"]

# Load model
file_id = "1pRUGLcLattWs4MI2U9YFq8ltbbSF7p1_"
tmp_model_path = None

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
        tmp_model_path = tmp_file.name
        gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_model_path, quiet=False)
    model = keras.models.load_model(tmp_model_path)
    st.success("✅ VGG19 model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
finally:
    try:
        os.remove(tmp_model_path)
    except OSError:
        pass

# UI Styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://i0.wp.com/post.healthline.com/wp-content/uploads/2022/04/hand-foot-and-mouth-disease-body8.jpg?w=1155&h=1528");
        background-size: cover;
        background-position: center;
        font-family: 'Segoe UI', sans-serif;
    }
 
    .title {
        font-size: 48px; /* Increased font size */
        font-weight: 800;
        color: #111;
    }
    .subtitle {
        font-size: 24px; /* Increased font size */
        color: #333;
    }
    .button-custom {
        display: inline-block;
        padding: 12px 24px;
        font-size: 18px; /* Slightly larger font for buttons */
        font-weight: bold;
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        margin-top: 10px;
    }
    .button-custom:hover {
        background-color: #0056b3;
    }
    
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Skin<span style='color:#4F9CDA'>AI</span></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Child Skin Disease Detection</p>", unsafe_allow_html=True)

# Upload and camera
option = st.radio("Choose input method:", ["📤 Upload Image", "📷 Take Picture"], horizontal=True)

image_data = None
if option == "📤 Upload Image":
    image_data = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
elif option == "📷 Take Picture":
    image_data = st.camera_input("Capture Skin Area")

# Process and Predict
if image_data:
    # Check if the image is from camera or file upload
    if isinstance(image_data, bytes):
        img = Image.open(image_data).convert("RGB")
    else:
        img = Image.open(image_data).convert("RGB")

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_input)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    st.image(img.resize((300, 300)), caption="Uploaded Image", use_column_width=False)
    st.markdown(f"""
        <div style='background-color:#FFFFFF;padding:20px;border-radius:15px;text-align:center;margin-top:20px'>
            <h2 style='color:#FF4444;'>Disease: {predicted_class.upper()}</h2>
            <p style='font-size:20px; color: black;'>Confidence: {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
