import cv2
import mediapipe as mp
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import base64
import os
import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R

print("Cargando modelo de gestos (gesture_model.h5)...")
try:
    modelo_gestos = tf.keras.models.load_model("gesture_model_remake.h5")
    clases_gestos = ["abierta", "cerrada", "like", "nada"]
    print("Modelo de gestos cargado exitosamente")
except Exception as e:
    print(f"ERROR FATAL: No se pudo cargar 'gesture_model.h5'")
    print(f"Asegurate de que el archivo esta en la misma carpeta que este script")
    print(f"Error: {e}")
    exit()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mi_clave_secreta'
socketio = SocketIO(app, cors_allowed_origins="*")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec_rostro = mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
drawing_spec_pose = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
drawing_spec_manos = mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)

camara = cv2.VideoCapture(0)
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

hilo = None
bloqueo_hilo = threading.Lock()

# --- FUNCIONES DE CALCULO (Para Pose) ---

def get_landmarks_array(landmarks, num_points=33):
    """
    Convierte los landmarks a un array de NumPy [n, 4]
    y TRADUCE el sistema de coordenadas de MediaPipe al de VRM.
    """
    if landmarks is None:
        return None
    points = np.empty((num_points, 4), dtype=np.float32) 
    for i in range(num_points):
        # Eje X de MediaPipe = Eje X de VRM (lo manejamos con el espejo)
        points[i, 0] = landmarks[i].x
        # Eje Y de MediaPipe = Eje -Y de VRM (Arriba/Abajo)
        points[i, 1] = -landmarks[i].y
        # Eje Z de MediaPipe = Eje Z de VRM (Adelante/Atras)
        points[i, 2] = landmarks[i].z  # <-- ¡ESTA ES LA CORRECCION CLAVE!
        points[i, 3] = landmarks[i].visibility
    return points

def get_vector(p1, p2):
    v = p2[:3] - p1[:3] 
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([0, 0, 1], dtype=np.float32)
    return v / norm

def get_rotation_quat(v_from, v_to):
    try:
        rot, _ = R.align_vectors([v_to], [v_from])
        return rot.as_quat().tolist()
    except (ValueError, np.linalg.LinAlgError):
        return [0, 0, 0, 1]

# Vectores base del esqueleto T-Pose (como ahora los calculamos)
V_UP = np.array([0, 1, 0], dtype=np.float32)
V_RIGHT = np.array([1, 0, 0], dtype=np.float32)
V_FWD = np.array([0, 0, 1], dtype=np.float32)

@app.route('/')
def index():
    return render_template('index.html') 

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
    
    VISIBILITY_THRESHOLD = 0.5
    
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7) as hands, \
         mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        
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

            # Extraer ROSTRO y dibujar
            lista_rostro = []
            if resultados_rostro.multi_face_landmarks:
                landmarks_rostro = resultados_rostro.multi_face_landmarks[0]
                lista_rostro = landmarks_a_lista(landmarks_rostro) 
                mp_drawing.draw_landmarks(
                    fotograma_anotado, landmarks_rostro,
                    mp_face_mesh.FACEMESH_TESSELATION, 
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec_rostro)

            # Logica de POSE
            pose_rotations = {} 
            
            # Dibujar esqueleto 2D
            if resultados_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    fotograma_anotado, resultados_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec_pose)

            # Calcular rotaciones 3D
            if resultados_pose.pose_world_landmarks:
                # lm ahora tiene los ejes Y y Z corregidos
                lm = get_landmarks_array(resultados_pose.pose_world_landmarks.landmark)

                # 1. Columna (Spine)
                if (lm[23, 3] > VISIBILITY_THRESHOLD and 
                    lm[24, 3] > VISIBILITY_THRESHOLD and
                    lm[11, 3] > VISIBILITY_THRESHOLD and
                    lm[12, 3] > VISIBILITY_THRESHOLD):
                    
                    hip_avg = (lm[23] + lm[24]) / 2.0
                    shoulder_avg = (lm[11] + lm[12]) / 2.0
                    v_hips_live = get_vector(hip_avg, shoulder_avg)
                    pose_rotations['Spine'] = get_rotation_quat(V_UP, v_hips_live)

                # --- INICIO: LOGICA DE BRAZOS (MAPEADO EN ESPEJO) ---
                
                v_left_upper_arm_live = None
                v_left_lower_arm_live = None
                v_right_upper_arm_live = None
                v_right_lower_arm_live = None

                # Vector Brazo Izquierdo REAL (lm[11])
                if (lm[11, 3] > VISIBILITY_THRESHOLD and lm[13, 3] > VISIBILITY_THRESHOLD and lm[15, 3] > VISIBILITY_THRESHOLD):
                    v_left_upper_arm_live = get_vector(lm[11], lm[13])
                    v_left_lower_arm_live = get_vector(lm[13], lm[15])

                # Vector Brazo Derecho REAL (lm[12])
                if (lm[12, 3] > VISIBILITY_THRESHOLD and lm[14, 3] > VISIBILITY_THRESHOLD and lm[16, 3] > VISIBILITY_THRESHOLD):
                    v_right_upper_arm_live = get_vector(lm[12], lm[14])
                    v_right_lower_arm_live = get_vector(lm[14], lm[16])
                
                # --- MAPEO EN ESPEJO ---

                # 2. Brazo Izquierdo del AVATAR (controlado por tu brazo DERECHO real)
                if v_right_upper_arm_live is not None:
                    pose_rotations['LeftUpperArm'] = get_rotation_quat(-V_RIGHT, v_right_upper_arm_live)
                
                if v_right_lower_arm_live is not None:
                    pose_rotations['LeftLowerArm'] = get_rotation_quat(-V_RIGHT, v_right_lower_arm_live)

                # 3. Brazo Derecho del AVATAR (controlado por tu brazo IZQUIERDO real)
                if v_left_upper_arm_live is not None:
                    pose_rotations['RightUpperArm'] = get_rotation_quat(V_RIGHT, v_left_upper_arm_live)
                
                if v_left_lower_arm_live is not None:
                    pose_rotations['RightLowerArm'] = get_rotation_quat(V_RIGHT, v_left_lower_arm_live)

                # --- FIN: LOGICA DE BRAZOS ---


            # Extraer MANOS y Predecir Gestos
            lista_mano_izquierda = []
            lista_mano_derecha = []
            gesto_actual = "nada" 
            
            if resultados_manos.multi_hand_landmarks:
                for i, landmarks_mano in enumerate(resultados_manos.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        fotograma_anotado,
                        landmarks_mano,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec_manos)
                    
                    lateralidad = resultados_manos.multi_handedness[i].classification[0].label
                    lista_mano = landmarks_a_lista(landmarks_mano) 
                    
                    if lateralidad == 'Right':
                        lista_mano_izquierda = lista_mano 
                    elif lateralidad == 'Left':
                        lista_mano_derecha = lista_mano 

                    if i == 0: # Logica de prediccion
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
                                lote_img = np.expand_dims(img_rgb, axis=0) 

                                prediccion = modelo_gestos.predict(lote_img, verbose=0) 
                                indice_predicho = np.argmax(prediccion)
                                confianza = np.max(prediccion)
                                
                                if confianza > 0.5:
                                    gesto_actual = clases_gestos[indice_predicho]

                                texto = f"GESTO: {gesto_actual.upper()} ({confianza*100:.0f}%)"
                                cv2.putText(fotograma_anotado, texto, (x_min, y_min - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                            except Exception as e:
                                pass
            
            # Codificar y Enviar
            _, buffer = cv2.imencode('.jpg', fotograma_anotado)
            jpg_como_texto = base64.b64encode(buffer).decode('utf-8')

            socketio.emit('update_data', {
                'face_mesh': lista_rostro,
                'pose_rotations': pose_rotations,
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