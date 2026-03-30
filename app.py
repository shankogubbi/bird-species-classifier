import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Bird Species Identifier",
    page_icon="🐦",
    layout="centered"
)

@st.cache_resource
def load_model_and_labels():
    interpreter = tf.lite.Interpreter(model_path="bird_model.tflite")
    interpreter.allocate_tensors()
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return interpreter, labels

interpreter, labels = load_model_and_labels()

st.title("🐦 Bird Species Identifier")
st.write("Upload a bird photo and the model will identify the species.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Identifying..."):
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx   = int(np.argmax(preds))
    top_label = labels[top_idx]
    top_conf  = float(preds[top_idx]) * 100

    st.markdown("---")
    if top_conf >= 60:
        st.success(f"**{top_label}** — {top_conf:.1f}% confidence")
    elif top_conf >= 35:
        st.warning(f"**{top_label}** — {top_conf:.1f}% confidence (uncertain)")
    else:
        st.error(f"**{top_label}** — {top_conf:.1f}% confidence (very uncertain)")
