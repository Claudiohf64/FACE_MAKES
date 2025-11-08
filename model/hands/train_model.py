import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
import os
import numpy as np

# 1 Configuracion de Parametros

ruta_datos = "gesture_data" # Ruta relativa a la carpeta de datos
ancho_imagen = 64
alto_imagen = 64
canales_imagen = 3 
tamano_lote = 32 
epocas = 50

# Gestos actualizados para coincidir con tus carpetas
gestos = ["abierta", "like", "nada", "cerrada"] # 4 gestos

# 2 Cargar y Preparar el Dataset

try:
    dataset_entrenamiento = image_dataset_from_directory(
        ruta_datos,
        validation_split=0.2, 
        subset="training",
        seed=123,
        image_size=(alto_imagen, ancho_imagen),
        batch_size=tamano_lote,
        label_mode='categorical'
    )

    dataset_validacion = image_dataset_from_directory(
        ruta_datos,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(alto_imagen, ancho_imagen),
        batch_size=tamano_lote,
        label_mode='categorical'
    )
except FileNotFoundError:
    print(f"ERROR: No se encontro la carpeta '{ruta_datos}'")
    print("Asegurate de que la ruta sea correcta")
    exit()


nombres_clases = dataset_entrenamiento.class_names
print(f"Clases encontradas: {nombres_clases}")

# Verificar si las clases encontradas coinciden con la lista gestos
if set(nombres_clases) != set(gestos):
    print(f"ERROR: Desajuste de clases")
    print(f"Script esperaba: {gestos}")
    print(f"Carpetas encontradas: {nombres_clases}")
    print("Asegurate que la lista GESTOS coincida exactamente con las carpetas")
    exit()

autotune = tf.data.AUTOTUNE
dataset_entrenamiento = dataset_entrenamiento.cache().prefetch(buffer_size=autotune)
dataset_validacion = dataset_validacion.cache().prefetch(buffer_size=autotune)

# 3 Definir la Arquitectura de la CNN
modelo = Sequential([
    # Capa de aumento de datos y normalizacion
    Rescaling(1./255, input_shape=(alto_imagen, ancho_imagen, canales_imagen)),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    
    # Bloques Convolucionales
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Capas Densas (Clasificacion)
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout para reducir el sobreajuste
    Dense(len(gestos), activation='softmax') # Capa de salida (4 neuronas)
])

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelo.summary()

# 4 Entrenar el Modelo
print("\n Iniciando Entrenamiento ")
historial = modelo.fit(
    dataset_entrenamiento,
    validation_data=dataset_validacion,
    epochs=epocas 
)
print("\n Entrenamiento Finalizado ")

# 5 Guardar el Modelo
nombre_modelo = "gesture_model.h5"
modelo.save(nombre_modelo)
print(f"Modelo guardado como: {nombre_modelo}")