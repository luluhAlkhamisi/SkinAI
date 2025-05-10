import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import keras
import gdown
import os 
import tempfile

# Setup
st.set_page_config(page_title="SkinAI", layout="wide")

# Class names
class_names = ["chickenpox", "hfmd", "measles", "unknown"]

# Load model from Google Drive
file_id = "1pRUGLcLattWs4MI2U9YFq8ltbbSF7p1_"
tmp_model_path = None

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
        tmp_model_path = tmp_file.name
    gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_model_path, quiet=False)
    model = keras.models.load_model(tmp_model_path)
    st.success("VGG19 model loaded successfully!")
except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    try:
        os.remove(tmp_model_path)
    except OSError as e:
        print(f"Error removing temporary file: {e}")

# Custom CSS styling
css = f"""
    <style>
    .stApp {{
        background-image: url("https://i0.wp.com/post.healthline.com/wp-content/uploads/2022/04/hand-foot-and-mouth-disease-body8.jpg?w=1155&h=1528");
        background-size: cover;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }}
    .custom-box {{
        background-color: #FFFFFF;
        border-radius: 30px;
        padding: 40px;
        max-width: 500px;
        margin: auto;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
        text-align: center;
    }}
    .TAKE-PICTURE {{
        background-color: white;
        color: #0D0D1C;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        margin: 10px;
    }}
    .title {{
        color: black;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 20px;
    }}
    .subtitle {{
        font-size: 18px;
        color: #0D0D1C;
    }}
    .logo {{
        position: absolute;
        top: 20px;
        right: 20px;
        max-height: 60px;
    }}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="position: absolute; top: -75px; left: -50px; color: white;">
        <h1 style="color: black;"><strong>Skin<span style='color:#4F9CDA'>AI</span></strong></h1>
        <p style="font-size:20px; color:black;">AI-POWERED CHILD<br>SKIN DISEASE DETECTION</p>
    </div>
""", unsafe_allow_html=True)

# Central title box
st.markdown("""
    <div class="custom-box">
        <div class="title">CHECK SKIN</div>
    </div>
""", unsafe_allow_html=True)

# State to control camera visibility
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Take Picture button
col1, col2 = st.columns([1, 1])
with col1:
    if not st.session_state.show_camera:
        if st.button("📷 Take Picture", key="take_pic_btn"):
            st.session_state.show_camera = True
with col2:
    uploaded_file = st.file_uploader("Or upload a skin image", type=["jpg", "jpeg", "png"])

# Show camera input only after clicking the button
image_data = None
if st.session_state.show_camera:
    image_data = st.camera_input("Take a picture")

# Use uploaded image if available
if uploaded_file:
    image_data = uploaded_file

# Prediction & Result
if image_data is not None:
    img = Image.open(image_data).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_input)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    # Show image and result
    img_display = img.resize((300, 300))
    st.image(img_display, use_column_width=False)
    st.markdown(f"""
        <div style='background-color:#FFFFFF;padding:10px;border-radius:15px;text-align:center'>
            <h2 style='color:#FF4444;'>Disease: {predicted_class.upper()}</h2>
            <p style='font-size:25px; color: black;'>Confidence: {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
