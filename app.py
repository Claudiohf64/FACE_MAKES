import cv2
import mediapipe as mp
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import base64
import os

# Cargar TensorFlow y el modelo entrenado
import tensorflow as tf
import numpy as np

print("Cargando modelo de gestos (gesture_model.h5)...")
try:
    modelo_gestos = tf.keras.models.load_model("gesture_model.h5")
    
    clases_gestos = ["abierta", "cerrada", "like", "nada"]
    
    print("Modelo de gestos cargado exitosamente")
except Exception as e:
    print(f"ERROR FATAL: No se pudo cargar 'gesture_model.h5'")
    print(f"Asegurate de que el archivo esta en la misma carpeta que este script")
    print(f"Error: {e}")
    exit()

# Configuracion Inicial de Flask y SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mi_clave_secreta'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuracion de MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

camara = cv2.VideoCapture(0) # Usando la camara

# Cambio de resolucion 16:9 pero mas rapido
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camara.isOpened():
    print("ERROR FATAL: No se pudo abrir la webcam 0")
    exit()
else:
    print("Camara 0 iniciada correctamente")
    ancho = camara.get(cv2.CAP_PROP_FRAME_WIDTH)
    alto = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolucion de camara: {ancho}x{alto}")

# VARIABLES GLOBALES PARA CONTROLAR EL HILO
hilo = None
bloqueo_hilo = threading.Lock()

# Ruta Principal de Flask
@app.route('/')
def index():
    return render_template('index.html') 

# Evento de Conexion de SocketIO
@socketio.on('connect')
def handle_connect():
    global hilo
    print('Cliente conectado')
    with bloqueo_hilo:
        if hilo is None or not hilo.is_alive():
            print("Iniciando hilo de deteccion...")
            hilo = socketio.start_background_task(target=detectar_todo)
        else:
            print("El hilo de deteccion ya esta en ejecucion")

# Funcion de ayuda para convertir landmarks
def landmarks_a_lista(obj_landmarks):
    if obj_landmarks is None:
        return []
    if hasattr(obj_landmarks, 'landmark'):
        lista_landmarks = obj_landmarks.landmark
        return [{'x': l.x, 'y': l.y, 'z': l.z} for l in lista_landmarks]
    if isinstance(obj_landmarks, list):
         return [{'x': l.x, 'y': l.y, 'z': l.z} for l in obj_landmarks]
    return []

# Funcion Principal de Deteccion
def detectar_todo():
    global camara
    
    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands, \
         mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5) as face_mesh:
        
        print("Modelos (Pose + Manos + FaceMesh) cargados. Hilo iniciado")
        
        while True:
            exito, fotograma = camara.read()
            if not exito:
                socketio.sleep(0.1)
                continue

            fotograma = cv2.flip(fotograma, 1)
            fotograma_anotado = fotograma.copy() 
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            fotograma_rgb.flags.writeable = False
            
            resultados_pose = pose.process(fotograma_rgb)
            resultados_manos = hands.process(fotograma_rgb) 
            resultados_rostro = face_mesh.process(fotograma_rgb)

            fotograma_rgb.flags.writeable = True

            # Extraer ROSTRO
            lista_rostro = []
            if resultados_rostro.multi_face_landmarks:
                landmarks_rostro = resultados_rostro.multi_face_landmarks[0]
                lista_rostro = landmarks_a_lista(landmarks_rostro)

            # Extraer POSE (Pantalla y Mundo Real)
            lista_pose = []
            lista_pose_mundo = []
            if resultados_pose.pose_landmarks:
                lista_pose = landmarks_a_lista(resultados_pose.pose_landmarks)
                lista_pose_mundo = landmarks_a_lista(resultados_pose.pose_world_landmarks)
                mp_drawing.draw_landmarks(
                    fotograma_anotado,
                    resultados_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Extraer MANOS y Predecir Gestos
            lista_mano_izquierda = []
            lista_mano_derecha = []
            
            # Optimizacion: 'nada' es el valor por defecto
            gesto_actual = "nada" 
            
            if resultados_manos.multi_hand_landmarks:
                for i, landmarks_mano in enumerate(resultados_manos.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        fotograma_anotado,
                        landmarks_mano,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                    
                    lateralidad = resultados_manos.multi_handedness[i].classification[0].label
                    lista_mano = landmarks_a_lista(landmarks_mano)
                    
                    if lateralidad == 'Right':
                        lista_mano_izquierda = lista_mano 
                    elif lateralidad == 'Left':
                        lista_mano_derecha = lista_mano 

                    # Logica de Prediccion (SOLO PARA LA PRIMERA MANO)
                    # Optimizacion: esto solo corre SI se detecta una mano
                    if i == 0:
                        alto_f, ancho_f, _ = fotograma.shape
                        x_min, y_min, x_max, y_max = ancho_f, alto_f, 0, 0
                        for lm in landmarks_mano.landmark:
                            x, y = int(lm.x * ancho_f), int(lm.y * alto_f)
                            if x < x_min: x_min = x
                            if x > x_max: x_max = x
                            if y < y_min: y_min = y
                            if y > y_max: y_max = y
                        
                        relleno = 50 
                        x_min = max(0, x_min - relleno)
                        y_min = max(0, y_min - relleno)
                        x_max = min(ancho_f, x_max + relleno)
                        y_max = min(alto_f, y_max + relleno)

                        region_interes = fotograma[y_min:y_max, x_min:x_max]

                        if region_interes.size > 0:
                            try:
                                img_redimensionada = cv2.resize(region_interes, (64, 64))
                                img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
                                
                                # El modelo ya incluye Rescaling, no normalizar aqui
                                lote_img = np.expand_dims(img_rgb, axis=0) 

                                prediccion = modelo_gestos.predict(lote_img, verbose=0) 
                                indice_predicho = np.argmax(prediccion)
                                confianza = np.max(prediccion)
                                
                                if confianza > 0.5: # Umbral de confianza
                                    gesto_actual = clases_gestos[indice_predicho]
                                # (Si no, se queda en "nada" por defecto)

                                texto = f"GESTO: {gesto_actual.upper()} ({confianza*100:.0f}%)"
                                cv2.putText(fotograma_anotado, texto, (x_min, y_min - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                            except Exception as e:
                                pass
                        # Fin logica de prediccion
            
            # Codificar y Enviar
            _, buffer = cv2.imencode('.jpg', fotograma_anotado)
            jpg_como_texto = base64.b64encode(buffer).decode('utf-8')

            socketio.emit('update_data', {
                'face_mesh': lista_rostro,
                'pose_mesh': lista_pose,
                'pose_world_mesh': lista_pose_mundo, 
                'left_hand_mesh': lista_mano_izquierda,  
                'right_hand_mesh': lista_mano_derecha,
                'frame': jpg_como_texto,
                'gesture': gesto_actual
            })

            socketio.sleep(0.01)

# Iniciar el Servidor
if __name__ == '__main__':
    print("Iniciando servidor en http://localhost:5002")
    socketio.run(app, debug=True, port=5002, allow_unsafe_werkzeug=True)