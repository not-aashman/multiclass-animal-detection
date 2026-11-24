"""Streamlit demo for the Multi-Class Animal Detection model (YOLOv8n, 20 classes).

Run with:  streamlit run Detection.py
Expects best.pt (the fine-tuned model) in the same folder as this file.
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "best.pt"


@st.cache_resource
def load_model(path: Path) -> YOLO:
    # Cached so the model loads once, not on every click.
    return YOLO(str(path))


def detect(model: YOLO, image: Image.Image, conf: float):
    # Pass the PIL image straight to YOLO. It resizes internally and returns
    # box coordinates in the ORIGINAL image's pixels, so the boxes line up.
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # plot() returns BGR; flip to RGB for display
    detections = [
        (model.names[int(c)], float(s))
        for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    return annotated, detections


st.sidebar.title("Animal Detection System")
page = st.sidebar.selectbox("Select Page", ["Home", "About", "Animal Detection"])

if page == "Home":
    st.header("Animal Species Detection System")
    st.markdown("Upload a photo and the model draws a box around each animal it recognises.")

elif page == "About":
    st.header("About")
    st.markdown(
        "- **Model:** YOLOv8n (COCO-pretrained), fine-tuned on 20 animal classes\n"
        "- **Dataset:** 2,000 images, 100 per class, labelled by the team in Roboflow; "
        "split 1,400 / 300 / 300 (train / validation / test)\n"
        "- **Classes:** Buffalo, Camel, Cat, Cheetah, Cow, Deer, Dog, Elephant, Goat, Gorilla, "
        "Hippo, Horse, Lion, Monkeys, Panda, Rat, Rhino, Tiger, Wolf, Zebra\n"
        "- Semester V minor project, LNCT (2024), team of four"
    )

elif page == "Animal Detection":
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}. Put the fine-tuned best.pt next to this script.")
        st.stop()
    model = load_model(MODEL_PATH)

    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    upload = st.file_uploader("Upload Animal Image:", type=["jpg", "jpeg", "png"])

    if upload:
        image = Image.open(upload).convert("RGB")  # handles PNGs with transparency and greyscale images
        st.image(image, caption="Uploaded Image", width="stretch")

        if st.button("Predict"):
            with st.spinner("Processing..."):
                annotated, detections = detect(model, image, conf)

            if detections:
                st.image(annotated, caption="Detected Animals", width="stretch")
                st.write("Detected Animals:")
                for name, score in detections:
                    st.write(f"{name}: {score:.2f}")
            else:
                st.warning("No animals detected above the confidence threshold.")
