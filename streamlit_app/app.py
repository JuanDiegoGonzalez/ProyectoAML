import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import os
import pandas as pd
import glob
from streamlit_option_menu import option_menu
import altair as alt

# --- Configuración ----------------------------------------------------------
IMG_SIZE = 224
CATEGORIES = ["Anthracnose", "Bacterial Blight", "Citrus Canker", "Curl Virus", "Deficiency Leaf", "Dry Leaf", "Healthy Leaf", "Sooty Mould", "Spider Mites"]
CATEGORIES_TRANSLATION = {
    "Anthracnose": "Antracnosis",
    "Bacterial Blight": "Tizón Bacteriano",
    "Citrus Canker": "Cáncer Cítrico",
    "Curl Virus": "Virus del Enrollamiento",
    "Deficiency Leaf": "Hoja con Deficiencia",
    "Dry Leaf": "Hoja Seca",
    "Healthy Leaf": "Hoja Sana",
    "Sooty Mould": "Moho Negro",
    "Spider Mites": "Ácaros"
}
MODEL_PATH = "../old_model.h5"
NEW_MODEL_PATH = "../new_model.h5"

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Enfermedades en Hojas de Limón",
    page_icon="🍃",
    layout="wide"
)

# Sidebar navigation
with st.sidebar:
    st.title("🍃 Menú")
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,
        options=["Clasificador", "Galería"],
        icons=["camera", "images"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Clasificador de enfermedades en hojas usando Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)

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

# Página del Clasificador
if selected == "Clasificador":
    st.title("Clasificador de Enfermedades en Hojas de Limón")
    st.write("Sube una imagen de una hoja para clasificarla")

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
        class_name_translated = CATEGORIES_TRANSLATION[class_name]
        confidence = float(preds[0][class_idx])
        
        # Mostrar resultados
        st.subheader("Resultados")
        st.write(f"Clase predicha: {class_name_translated}")
        st.write(f"Confianza: {confidence:.2%}")
        
        # Crear DataFrame para el gráfico
        probs_df = pd.DataFrame({
            'Clase': [CATEGORIES_TRANSLATION[cat] for cat in CATEGORIES],
            'Probabilidad': preds[0] * 100
        })
        
        # Ordenar por probabilidad de mayor a menor
        probs_df = probs_df.sort_values('Probabilidad', ascending=False)
        
        # Crear layout horizontal con columnas de ancho completo
        col1, col2 = st.columns([1, 1])  # Columnas de igual ancho
        
        # Columna izquierda: Probabilidades con barras de progreso
        with col1:
            st.subheader("Probabilidades por clase")
            for i, category in enumerate(CATEGORIES):
                prob = float(preds[0][i])
                st.write(f"{CATEGORIES_TRANSLATION[category]}: {prob:.2%}")
                st.progress(prob)
        
        # Columna derecha: Gráfico de barras
        with col2:
            st.subheader("Visualización")
            st.markdown("#### Distribución de Probabilidades")
            # Crear un nuevo DataFrame con el índice ordenado
            chart_df = probs_df.set_index('Clase')
            
            # Crear gráfico con Altair
            chart = alt.Chart(chart_df.reset_index()).mark_bar().encode(
                x=alt.X('Clase:N', sort='-y', title='Enfermedad'),
                y=alt.Y('Probabilidad:Q', title='Probabilidad (%)'),
                color=alt.Color('Probabilidad:Q', scale=alt.Scale(scheme='greenblue')),
                tooltip=['Clase', 'Probabilidad']
            ).properties(
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)

# Página de la Galería
elif selected == "Galería":
    st.title("Galería de Hojas por Enfermedad")
    
    # Obtener la lista de carpetas de enfermedades
    disease_folders = [d for d in os.listdir("images") if os.path.isdir(os.path.join("images", d))]
    
    # Crear pestañas para cada enfermedad
    tabs = st.tabs([CATEGORIES_TRANSLATION[disease] for disease in disease_folders])
    
    # Para cada pestaña, mostrar las imágenes de esa enfermedad
    for tab, disease in zip(tabs, disease_folders):
        with tab:
            st.subheader(CATEGORIES_TRANSLATION[disease])
            
            # Obtener todas las imágenes en la carpeta de la enfermedad
            image_paths = glob.glob(os.path.join("images", disease, "*.*"))
            
            # Crear columnas para mostrar las imágenes en una cuadrícula
            cols = st.columns(3)  # 3 imágenes por fila
            
            for idx, img_path in enumerate(image_paths):
                try:
                    # Cargar y mostrar la imagen
                    img = Image.open(img_path)
                    with cols[idx % 3]:
                        st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Error al cargar la imagen {img_path}: {str(e)}")