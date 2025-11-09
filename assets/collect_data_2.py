import cv2
import mediapipe as mp
import os
import time
import numpy as np

# --- 1. Configuracion Inicial ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# --- Establecer resolucion 16:9 (HD) ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: No se pudo abrir la camara.")
    exit()
else:
    # --- Verificar la resolucion ---
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camara iniciada. Solicitada: 1280x720. Obtenida: {width}x{height}")

# --- 2. Configuracion de Carpetas ---
DATA_PATH = "gesture_data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

gestures = ["open", "fist", "like", "none"]
num_images_per_gesture = 101

for gesture in gestures:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

print(f"Directorios creados en '{DATA_PATH}'.")
print(f"Vamos a recolectar {num_images_per_gesture} imagenes para: {gestures}")
print("¡IMPORTANTE! Graba 'open', 'fist', 'like' contra un fondo simple (pared).")
print("Para 'none', graba el fondo SIN tu mano.")
print(f"Presiona 'o', 'f', 'l', o 'n' para empezar...")
print("¡PREPARATE! La recoleccion comenzara en 5 segundos...")
time.sleep(5)

# --- 3. Bucle Principal de Recoleccion ---
START_INDEX = 300 # <--- ¡CAMBIO AQUI! El numero de archivo inicial
current_gesture_index = 0
image_count = 0 # Esto contara de 0 a 99 (total 100 imagenes)
capturing = False
last_capture_time = 0
CAPTURE_DELAY = 0.3

# --- NUEVO: Variables para la cuenta regresiva ---
is_counting_down = False
countdown_start_time = 0
COUNTDOWN_SECONDS = 2
# --- FIN NUEVO ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("WARN: No se pudo obtener el frame. Reintentando...")
        time.sleep(0.1)
        continue

    frame = cv2.flip(frame, 1)
    annotated_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = w, h, 0, 0
    hand_found = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
        
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        hand_found = True

    # --- 4. Logica de Captura y Cuenta Regresiva ---
    
    is_none_gesture = (gestures[current_gesture_index] == "none")

    # --- NUEVO: Logica de Cuenta Regresiva ---
    if is_counting_down:
        elapsed = time.time() - countdown_start_time
        countdown_value = COUNTDOWN_SECONDS - int(elapsed)
        
        if countdown_value > 0:
            # Dibuja el numero grande en el centro
            text_size = cv2.getTextSize(str(countdown_value), cv2.FONT_HERSHEY_SIMPLEX, 7, 20)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(annotated_frame, str(countdown_value), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 20)
        else:
            # ¡Se acabo el tiempo! Empezar a capturar
            print("¡YA!")
            is_counting_down = False
            capturing = True
            last_capture_time = time.time()
    # --- FIN NUEVO ---

    # Mostrar estado (solo si NO estamos en cuenta regresiva)
    elif capturing:
        # El contador (image_count) seguira mostrando 0/100, 1/100, etc. para saber cuantas llevas
        status_text = f"RECOLECTANDO: {gestures[current_gesture_index].upper()} ({image_count}/{num_images_per_gesture})"
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Muestra el gesto actual que se espera
        status_text = f"LISTO PARA: {gestures[current_gesture_index].upper()}. Presiona '{gestures[current_gesture_index][0]}'"
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    # Logica de guardado de fotos (solo se activa si capturing=True y paso el delay)
    if capturing and (time.time() - last_capture_time > CAPTURE_DELAY):
        if (hand_found and not is_none_gesture) or (not hand_found and is_none_gesture):
            
            if is_none_gesture:
                x_min, y_min = np.random.randint(0, w//2), np.random.randint(0, h//2)
                x_max, y_max = x_min + 100, y_min + 100
                x_max, y_max = min(w, x_max), min(h, y_max)
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            roi = frame[y_min:y_max, x_min:x_max]
            
            if roi.size > 0:
                gesture_folder = os.path.join(DATA_PATH, gestures[current_gesture_index])
                
                # --- ¡CAMBIO AQUI! Se calcula el nombre del archivo ---
                file_name = image_count + START_INDEX
                image_path = os.path.join(gesture_folder, f"{file_name}.jpg")
                # --- FIN DEL CAMBIO ---
                
                cv2.imwrite(image_path, roi)
                print(f"Guardada: {image_path}")
                
                image_count += 1
                last_capture_time = time.time()

                if image_count >= num_images_per_gesture:
                    image_count = 0 # El contador de progreso se resetea
                    current_gesture_index += 1
                    capturing = False
                    
                    if current_gesture_index >= len(gestures):
                        print("¡Recoleccion de datos completa!")
                        break
                    else:
                        print(f"\n¡Perfecto! Ahora prepara el gesto: {gestures[current_gesture_index].upper()}")
                        print(f"Presiona '{gestures[current_gesture_index][0]}' para empezar...")

    cv2.imshow('Colector de Datos - Presiona Q para salir', annotated_frame)

    # --- 5. Controles ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    
    key_char = chr(key).lower()
    
    # --- NUEVO: Logica de Controles actualizada ---
    # Solo escucha teclas si no estamos capturando O en cuenta regresiva
    if not capturing and not is_counting_down and current_gesture_index < len(gestures):
        expected_key = gestures[current_gesture_index][0] # 'o', 'f', 'l', 'n'
        
        if key_char == expected_key:
            if is_none_gesture:
                # El gesto 'none' empieza de inmediato
                print(f"¡Empezando captura de 'NONE'! (QUITA LA MANO DEL FRAME)")
                capturing = True
                last_capture_time = time.time()
            else:
                # Los otros gestos inician la cuenta regresiva
                print(f"¡Preparate para '{gestures[current_gesture_index].upper()}'! Empezando en {COUNTDOWN_SECONDS} segundos...")
                is_counting_down = True
                countdown_start_time = time.time()
    # --- FIN NUEVO ---

# --- 6. Limpieza ---
cap.release()
cv2.destroyAllWindows()
hands.close()