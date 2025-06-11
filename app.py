import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import gdown

MODEL_PATH = "best.pt"
GDRIVE_URL = "https://drive.google.com/uc?id=17YL268guczOGcRvbUF_gjU-FMVuhietc"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load the model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load best.pt. Make sure it's a valid YOLOv8 model.")
    st.stop()

# UI
st.title("🌲 Tree Counter App")
uploaded_file = st.file_uploader("📸 Upload Tree Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model.predict(image, conf=0.3)
    count = len(results[0].boxes)
    st.success(f"✅ Trees Detected: {count}")

    results[0].save(filename="output.jpg")
    st.image("output.jpg", caption="Detection Result", use_column_width=True)
