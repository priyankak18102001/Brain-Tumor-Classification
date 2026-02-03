import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")

#  class names load
@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

class_names = load_classes()

# Build model architecture again (same as training)
@st.cache_resource
def build_model():
    num_classes = 4

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # IMPORTANT compile for loading weights safely
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model()

#  Load weights
model.load_weights("brain_tumor_weights.weights.h5")


st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to predict tumor type with confidence score.")

uploaded_file = st.file_uploader(" Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # preprocess_input already inside model pipeline (we built it before base model),
    # but here we kept preprocess in model itself, so DON'T apply it again outside.

    preds = model.predict(img_array)
    probs = preds[0]
    pred_index = int(np.argmax(probs))

    st.subheader("Prediction Result")
    st.success(f"**Tumor Type:** {class_names[pred_index].upper()}")
    st.info(f"**Confidence:** {probs[pred_index]*100:.2f}%")

    st.subheader(" Probabilities")
    for i, p in enumerate(probs):
        st.write(f"{class_names[i]} : **{p*100:.2f}%**")
