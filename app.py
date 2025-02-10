import os
os.system("pip install opencv-python-headless")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("waste_detection_best (1).pt")  # Ensure the model file is in the same directory

st.title("Waste Detection System")
st.write("Upload an image and detect waste using YOLOv8.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to a format compatible with OpenCV
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert to numpy array

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO prediction
    results = model.predict(source=image_bgr, conf=0.25, save=False)

    # Get the annotated image
    annotated_image = results[0].plot()

    # Convert BGR to RGB for proper display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display the prediction output
    st.image(annotated_image_rgb, caption="Detected Waste", use_column_width=True)

    # Show details
    st.write("Detection Details:")
    st.write(results[0].boxes)
