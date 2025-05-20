import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import os
import pandas as pd

# --- Configuración ----------------------------------------------------------
IMG_SIZE = 224
CATEGORIES = ["Anthracnose", "Bacterial Blight", "Citrus Canker", "Curl Virus", "Deficiency Leaf", "Dry Leaf", "Healthy Leaf", "Sooty Mould", "Spider Mites"]
MODEL_PATH = "../old_model.h5"
NEW_MODEL_PATH = "../new_model.h5"

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Hojas",
    page_icon="🍃",
    layout="wide"
)

# Título y descripción
st.title("Clasificador de Hojas")
st.write("Sube una imagen de una hoja para clasificarla")

# Cargar modelo una sola vez
@st.cache_resource
def load_ml_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_new_ml_model():
    return load_model(NEW_MODEL_PATH)

model = load_ml_model()
new_model = load_new_ml_model()

def preprocess_image_pil(pil_img: Image.Image) -> np.ndarray:
    # Convertir PIL → OpenCV BGR
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    final = cv2.merge([blurred, blurred, blurred])

    final = final.astype("float32") / 255.0
    return np.expand_dims(final, axis=0)  # shape (1, 224, 224, 3)

# Widget para subir archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    _, col_img, _ = st.columns([0.35, 0.3, 0.35])
    with col_img:
        st.image(image, caption="Imagen subida", use_container_width=True)
    
    # Preprocesar y predecir
    img_batch = preprocess_image_pil(image)
    preds = model.predict(img_batch)
    
    # Obtener resultados
    class_idx = int(np.argmax(preds))
    class_name = CATEGORIES[class_idx]
    confidence = float(preds[0][class_idx])
    
    # Mostrar resultados
    st.subheader("Resultados")
    st.write(f"Clase predicha: {class_name}")
    st.write(f"Confianza: {confidence:.2%}")
    
    # Crear DataFrame para el gráfico
    probs_df = pd.DataFrame({
        'Clase': CATEGORIES,
        'Probabilidad': preds[0] * 100
    })
    
    # Crear layout horizontal con columnas de ancho completo
    col1, col2 = st.columns([1, 1])  # Columnas de igual ancho
    
    # Columna izquierda: Probabilidades con barras de progreso
    with col1:
        st.subheader("Probabilidades por clase")
        for i, category in enumerate(CATEGORIES):
            prob = float(preds[0][i])
            st.progress(prob)
            st.write(f"{category}: {prob:.2%}")
    
    # Columna derecha: Gráfico de barras
    with col2:
        st.subheader("Visualización")
        st.bar_chart(probs_df.set_index('Clase'), use_container_width=True)