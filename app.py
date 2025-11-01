import cv2
import mediapipe as mp
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mi_clave_secreta_super_segura!'
socketio = SocketIO(app, cors_allowed_origins="*")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_FILE = 'face_landmarker.task'

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False)

print("Iniciando cámara Iriun (índice 0)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("¡ERROR FATAL! No se pudo abrir la webcam 0.")
    exit()
else:
    print("Cámara 0 (Iriun) iniciada correctamente.")


thread = None
thread_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global thread
    print('¡Cliente conectado!')
    with thread_lock:
        if thread is None or not thread.is_alive():
            print("Iniciando hilo de detección por primera vez...")
            thread = socketio.start_background_task(target=detectar_rostro)
        else:
            print("El hilo de detección ya está en ejecución.")

def detectar_rostro():
    global cap
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Modelo cargado. Hilo de detección iniciado.")
        frame_timestamp_ms = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignorando fotograma vacío (Iriun?).")
                socketio.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            annotated_frame = frame.copy() 
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)

            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if face_landmarker_result and face_landmarker_result.face_landmarks:
                
                face_landmarks = face_landmarker_result.face_landmarks[0]
                
                landmarks_list = [{'x': l.x, 'y': l.y, 'z': l.z} for l in face_landmarks]
                
                h, w, _ = annotated_frame.shape
                for landmark in face_landmarks:
                    cv2.circle(annotated_frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

                _, buffer = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                socketio.emit('update_data', {
                    'mesh': landmarks_list,
                    'frame': jpg_as_text
                })

            socketio.sleep(0.01)

if __name__ == '__main__':
    print("Iniciando servidor en http://localhost:5002")
    socketio.run(app, debug=True, port=5002, allow_unsafe_werkzeug=True)