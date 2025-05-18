# Clasificador de Enfermedades en Hojas

Esta aplicación web permite subir una imagen de una hoja y predecir automáticamente la enfermedad que podría estar afectándola. Utiliza un modelo de deep learning entrenado previamente, un backend en FastAPI y una interfaz web simple hecha con HTML, CSS y JavaScript.

---

## Tecnologías utilizadas

- **Python 3.12.9**
- **FastAPI** (backend)
- **TensorFlow / Keras** (modelo)
- **HTML + CSS + JavaScript** (frontend)
- **Uvicorn** (servidor ASGI)

---

## Instalación y ejecución

1. Abrir una terminal en la carpeta `/app`.

2. Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

3. Ejecutar el servidor local:

```bash
uvicorn main:app --reload
```

4. Abre tu navegador y visita:

```
http://127.0.0.1:8000/
```

---

## Cómo usar la aplicación

1. Cargar una imagen de una hoja en el formulario.
2. Hacer clic en **Predecir**.
3. El modelo preprocesará la imagen y clasificará su enfermedad.
4. Verás en pantalla:
   - La clase predicha
   - La probabilidad/confianza del modelo
   - Una lista de todas las clases con sus probabilidades, ordenadas de mayor a menor

---

## Sobre el modelo

- Entrenado para detectar distintas enfermedades en hojas a partir de imágenes.
- Utiliza una red neuronal convolucional (CNN).
- El modelo espera imágenes de entrada en formato RGB y con un tamaño específico (definido en el preprocesamiento).

---

## Realizado por

Diego Felipe Carvajal Lombo - 201911910 

María Alejandra Pérez Petro - 201923972 

Juan Diego González Gómez - 201911031 

Daniel Esteban Aguilera Figueroa - 202010592 
