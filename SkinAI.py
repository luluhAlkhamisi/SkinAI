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

# Load model once
# @st.cache(allow_output_mutation=True)
# def load_model():
#     return keras.models.load_model("VGG19-96.keras")

# model = load_model()

# load model from google drive
file_id = "1pRUGLcLattWs4MI2U9YFq8ltbbSF7p1_"
tmp_model_path = None  # Initialize tmp_model_path outside the try block

try:
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
        tmp_model_path = tmp_file.name
        print(f"Temporary model file will be saved to: {tmp_model_path}")

    # Download the model from Google Drive using gdown
    print(f"Downloading model from Google Drive ID: {file_id} to c:/Users/emanm/OneDrive/Desktop/python/New folder/task2")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_model_path, quiet=False)
    print("Download complete.")

    # Load the model
    print(f"Loading model from: {tmp_model_path}")
    model = keras.models.load_model(tmp_model_path)
    print("Model loaded successfully!")
    st.success("VGG19 model loaded successfully!")

except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    # Clean up the temporary file
    try:
        os.remove(tmp_model_path)
        print(f"Temporary file {tmp_model_path} removed.")
    except OSError as e:
        print(f"Error removing temporary file {tmp_model_path}: {e}")


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
    .upload{{
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


# Logo and title on left
st.markdown("""
    <div style="position: absolute; top: -75px; left: -50px; color: white;">
        <h1 style="color: black;"><strong>Skin<span style='color:#4F9CDA'>AI</span></strong></h1>
        <p style="font-size:20px; color:black;">AI-POWERED CHILD<br>SKIN DISEASE DETECTION</p>
    </div>
""", unsafe_allow_html=True)

# Center UI
# st.markdown('<div class="custom-box"><div class="title">CHECK SKIN</div> <button class="TAKE-PICTURE" onclick="image_data = takePicture()">Take Picture </button> <button class="upload" onclick="uploadPicture()">UPLOAD PICTURE </button>', unsafe_allow_html=True)
# Function to handle picture taking (could be connected to a camera API)

st.markdown("""
    <style>
        .custom-box {
            background-color: #f2f2f2;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }
        .custom-box .title {
            font-size: 25px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .custom-box button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .custom-box button:hover {
            background-color: #0056b3;
        }
    </style>
    <div class="custom-box">
        <div class="title">CHECK SKIN</div>
    </div>
""", unsafe_allow_html=True)


# Upload or take image
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or take a picture" )

# Use uploaded image or camera input
image_data = uploaded_file if uploaded_file else camera_file

if image_data is not None:
    img = Image.open(image_data).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_input)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    # Show result screen
    #Image.open(image_data)
    img_resized2 = img.resize((300, 300))
    st.image( img_resized2  , use_column_width=False)
    st.markdown(f"""
        <div style='background-color:#FFFFFF;padding:10px;border-radius:15px;text-align:center'>
            <h2 style='color:#FF4444;'>Disease: {predicted_class.upper()}</h2>
            <p style='font-size:25px; color: black;'>Confidence: {confidence:.2f}%</p>

        </div>
        st.markdown("""
    <style>
    [data-testid="stCameraInput"] video {
        max-height: 300px;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)
