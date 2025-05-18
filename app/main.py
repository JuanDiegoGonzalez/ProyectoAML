from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
import os

# --- Configuración ----------------------------------------------------------
IMG_SIZE = 224
DATA_DIR = "../Dataset"  # ruta original con subcarpetas por clase
CATEGORIES = os.listdir(DATA_DIR)          # ['Healthy', 'Scab', ...] por ejemplo
MODEL_PATH = "../cnn_hojas_hiperparametros.h5"

app = FastAPI(title="Clasificador de Hojas – FastAPI")

# Cargar modelo una sola vez al arrancar
model = load_model(MODEL_PATH)

# Montar archivos estáticos y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Función de preprocesamiento (misma que en tu notebook) -----------------
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

# --- Rutas ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer bytes → PIL
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocesar
    img_batch = preprocess_image_pil(pil_image)

    # Predecir
    preds = model.predict(img_batch)
    class_idx = int(np.argmax(preds))
    class_name = CATEGORIES[class_idx]
    confidence = float(preds[0][class_idx])

    # Crear diccionario nombre_clase: probabilidad
    probabilities_dict = {
        CATEGORIES[i]: round(float(prob), 6)
        for i, prob in enumerate(preds[0])
    }

    return JSONResponse({
        "class_index": class_idx,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "probabilities": probabilities_dict
    })
