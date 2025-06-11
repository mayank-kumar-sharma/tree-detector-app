import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import gdown

# Google Drive link (replace this if your link ever changes)
GDRIVE_URL = "https://drive.google.com/uc?id=17YL268guczOGcRvbUF_gjU-FMVuhietc"
MODEL_PATH = "best.pt"

# Download best.pt from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load model. Please make sure 'best.pt' is a valid YOLOv8 model file.")
    st.stop()

st.title("🌲 Tree Counter App")
st.markdown("Upload an image to count the number of trees detected using YOLOv8.")

uploaded_file = st.file_uploader("Upload Tree Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model.predict(image, conf=0.3)
    count = len(results[0].boxes)

    st.success(f"✅ Detected Trees: {count}")

    results[0].save(filename="output.jpg")
    st.image("output.jpg", caption="Detection Result", use_column_width=True)
