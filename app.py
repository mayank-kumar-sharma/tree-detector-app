import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model from local file
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error("❌ Failed to load best.pt. Make sure it's a valid YOLOv8 model file.")
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
