from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import threading
import time
import os
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob

app = Flask(__name__)

# Variables globales
predicted_sentence = ""
current_sign = ""
prediction_count = 0
threshold_frames = 15
last_prediction = ""
camera_active = False

# Crear directorio estático para archivos de audio si no existe
AUDIO_DIR = os.path.join('static', 'audio')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

model_path = "model/asl_model.joblib"
encoder_path = "model/label_encoder.joblib"

# Verificar que los archivos del modelo existen
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    print("❌ Error: Archivos del modelo no encontrados")
    print(f"Buscando: {model_path}")
    print(f"Buscando: {encoder_path}")
    print("\n🔧 Para entrenar tu propio modelo:")
    print("1. Ejecuta: python collect_data.py (para recolectar datos)")
    print("2. Ejecuta: python train_model.py (para entrenar el modelo)")
    print("3. Reinicia la aplicación")
    model_loaded = False
    model = None
    le = None
else:
    # Cargar el modelo y el codificador
    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        model_loaded = True
        print("✅ Modelo cargado exitosamente")
        print(f"📊 Clases disponibles: {list(le.classes_)}")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        model_loaded = False
        model = None
        le = None

# Configurar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Reducido para mejor rendimiento
    min_tracking_confidence=0.5    # Reducido para mejor estabilidad
)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise Exception("No se pudo abrir la cámara")
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)  # Establecer FPS
        
    def __del__(self):
        if hasattr(self, 'video') and self.video:
            self.video.release()
        
    def get_frame(self):
        global predicted_sentence, current_sign, prediction_count, last_prediction
        
        try:
            ret, frame = self.video.read()
            if not ret:
                return None
                
            # Voltear imagen para sensación natural
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convertir a RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            if result.multi_hand_landmarks and model_loaded:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Dibujar puntos de referencia
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    try:
                        # Extraer 63 características: x, y, z para cada punto
                        features = []
                        for lm in hand_landmarks.landmark:
                            features.extend([lm.x, lm.y, lm.z])
                        
                        # Convertir a numpy y reestructurar para el modelo
                        x_input = np.array(features).reshape(1, -1)
                        
                        # Predecir
                        y_pred = model.predict(x_input)
                        label = le.inverse_transform(y_pred)[0]
                        
                        # Mostrar la predicción actual igual que el código original
                        current_sign = label
                        
                        # Comprobación de estabilidad en varios cuadros
                        if label == last_prediction:
                            prediction_count += 1
                        else:
                            prediction_count = 0
                            last_prediction = label
                        
                        # Si es estable por suficientes cuadros, actualizar la oración
                        if prediction_count == threshold_frames:
                            if label == "space":
                                predicted_sentence += " "
                            elif label == "del":
                                predicted_sentence = predicted_sentence[:-1]
                            elif label != "nothing":
                                predicted_sentence += label
                            prediction_count = 0  # reiniciar para el siguiente signo
                    except Exception as e:
                        print(f"Error durante la predicción: {e}")
                        current_sign = "error"
            else:
                current_sign = "nothing"
            
            # Añadir texto sobre la imagen (como el código original de OpenCV)
            cv2.rectangle(frame, (10, 10), (630, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Signo: {current_sign}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Oración: {predicted_sentence}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Codificar el cuadro
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            return frame
            
        except Exception as e:
            print(f"Error en get_frame: {e}")
            # Retornar un cuadro negro con mensaje de error
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error de cámara", (250, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            return buffer.tobytes()

def gen_frames():
    camera = None
    try:
        camera = VideoCamera()
        while camera_active:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1/30)  # 30 FPS
    except Exception as e:
        print(f"Error en gen_frames: {e}")
    finally:
        if camera:
            del camera

def cleanup_old_audio_files():
    """Eliminar archivos de audio con más de 1 hora de antigüedad"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=1)
        audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        
        for file_path in audio_files:
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Archivo de audio antiguo eliminado: {file_path}")
    except Exception as e:
        print(f"Error limpiando archivos de audio: {e}")

def generate_audio_file(text):
    """Generar archivo MP3 a partir de texto usando gTTS"""
    try:
        if not text or text.strip() == "":
            return None
            
        # Limpiar archivos antiguos primero
        cleanup_old_audio_files()
        
        # Generar nombre de archivo único
        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        # Generar voz
        tts = gTTS(text=text, lang='es', slow=False)  # Cambiado a español
        tts.save(filepath)
        
        return filename
    except Exception as e:
        print(f"Error generando audio: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'status': 'Cámara iniciada'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'Cámara detenida'})

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    return jsonify({
        'sentence': predicted_sentence,
        'current_sign': current_sign,
        'prediction_count': prediction_count,
        'threshold_frames': threshold_frames
    })

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global predicted_sentence
    predicted_sentence = ""
    return jsonify({'status': 'Oración borrada'})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global predicted_sentence
    
    if not predicted_sentence or predicted_sentence.strip() == "":
        return jsonify({
            'status': 'error',
            'message': 'No hay oración para reproducir'
        }), 400
    
    # Generar archivo de audio
    audio_filename = generate_audio_file(predicted_sentence)
    
    if audio_filename:
        return jsonify({
            'status': 'success',
            'sentence': predicted_sentence,
            'audio_url': f'/static/audio/{audio_filename}',
            'message': 'Audio generado exitosamente'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No se pudo generar el audio'
        }), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Servir archivos de audio"""
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
