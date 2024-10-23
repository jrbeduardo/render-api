# API
from fastapi import FastAPI,UploadFile, File, Query, HTTPException
from models.images import ImageData

# Conversion Base64
import base64
from io import BytesIO

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


import os
from PIL import Image, ImageOps
import io
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

###### CARGA DE MODELO

model = tf.keras.models.load_model('primer_red_densa.h5', compile = False)


class_names = [
  'T-shirt/top', 
  'Trouser', 
  'Pullover', 
  'Dress', 
  'Coat', 
  'Sandal', 
  'Shirt', 
  'Sneaker', 
  'Bag', 
  'Ankle boot'
]

class_names_es = [
    'Camiseta/top',
    'Pantalón',
    'Suéter',
    'Vestido',
    'Abrigo',
    'Sandalia',
    'Camisa',
    'Zapatillas deportivas',
    'Bolso',
    'Bota de caña baja'
]

## INSTANCIA DE FASTAPI
app = FastAPI(title='Clasificadores con Tensorflow', description="""
## ¿Qué es un Clasificador?

Un clasificador es un algoritmo de aprendizaje automático que asigna una etiqueta o categoría a un conjunto de datos de entrada. En otras palabras, "clasifica" los datos en grupos predefinidos.

## TensorFlow y la Clasificación

TensorFlow, una biblioteca de código abierto desarrollada por Google, es una herramienta poderosa para construir y entrenar modelos de aprendizaje automático, incluyendo clasificadores. Ofrece una interfaz flexible y una gran comunidad, lo que lo convierte en una elección popular para una amplia gama de aplicaciones.
""")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto para limitar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/get_clases', tags=["Clasificador de prendas"])
def get_clases_name(idioma:str = Query(default='en', max_length=2, description='Idioma de las clases de prendas')):
    if idioma =='es':
        return {
        'data': dict(enumerate(class_names_es)) 
    }
    return {
        'data': dict(enumerate(class_names)) 
    }

@app.post("/clasification_image", tags=["Clasificador de prendas"])
def upload_image(image_data: ImageData):
    try:
        # Decodificar la imagen base64
        image_data_bytes = base64.b64decode(image_data.img_base64)
        image = Image.open(BytesIO(image_data_bytes)).convert('L')
        image = image.resize((28,28))
        image = ImageOps.invert(image)

        # Preprocesar la imagen para el modelo
        image = img_to_array(image) / 255.0
        image = image.reshape(1, 28, 28)

        # Realizar la predicción
        data = {}
        data["prediction"] = model.predict(image).tolist()[0]
        data["max_prediction"] = data["prediction"][int(np.argmax(data['prediction']))]
        data["class_name"] = class_names_es[int(np.argmax(data['prediction']))]

        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")



