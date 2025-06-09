import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image

# Cache the model load (for older Streamlit, change to @st.cache(allow_output_mutation=True))
@st.cache_resource(show_spinner=True)
def load_trained_model():
    from tensorflow.keras.models import load_model
    model = load_model("mobilenetv2_best_finetuned.keras")
    return model

st.title("Skin Lesion Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array

    with st.spinner("Predicting..."):
        model = load_trained_model()
        prediction = model.predict(img_array)[0][0]

    class_names = ['benign', 'malignant']
    label = class_names[1] if prediction >= 0.5 else class_names[0]

    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability (malignant):** {prediction:.4f}")

