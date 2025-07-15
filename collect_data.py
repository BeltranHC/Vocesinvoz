"""
Script para capturar datos de entrenamiento desde la cámara web
Útil para crear tu propio dataset de señas ASL
"""

import cv2
import mediapipe as mp
import os
import time
import numpy as np
from collections import Counter

class DataCollector:
    def __init__(self, data_dir="data"):
        """
        Inicializar el colector de datos
        
        Args:
            data_dir: Directorio donde se guardarán las imágenes
        """
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Crear directorio principal si no existe
        os.makedirs(self.data_dir, exist_ok=True)
    
    def create_class_directory(self, class_name):
        """
        Crear directorio para una clase específica
        """
        class_dir = os.path.join(self.data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        return class_dir
    
    def get_next_filename(self, class_dir, class_name):
        """
        Obtener el siguiente nombre de archivo disponible
        """
        existing_files = os.listdir(class_dir)
        numbers = []
        for file in existing_files:
            if file.startswith(f"{class_name}_") and file.endswith('.jpg'):
                try:
                    num = int(file.split('_')[1].split('.')[0])
                    numbers.append(num)
                except:
                    continue
        
        next_num = max(numbers) + 1 if numbers else 1
        return f"{class_name}_{next_num:04d}.jpg"
    
    def collect_for_class(self, class_name, num_samples=100):
        """
        Recolectar muestras para una clase específica
        
        Args:
            class_name: Nombre de la clase (A, B, C, etc.)
            num_samples: Número de muestras a recolectar
        """
        print(f"\n🎯 Recolectando datos para la clase: {class_name}")
        print(f"📝 Objetivo: {num_samples} muestras")
        print("\n⚠️  INSTRUCCIONES:")
        print("- Haz la seña correspondiente frente a la cámara")
        print("- Mantén la mano dentro del recuadro verde")
        print("- Presiona ESPACIO para capturar una imagen")
        print("- Presiona 'q' para terminar esta clase")
        print("- Presiona 'n' para ir a la siguiente clase")
        print("\n🚀 Presiona cualquier tecla para comenzar...")
        
        input()
        
        # Crear directorio para la clase
        class_dir = self.create_class_directory(class_name)
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        samples_collected = 0
        last_capture_time = 0
        capture_cooldown = 0.5  # Segundos entre capturas
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Voltear imagen
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convertir a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Dibujar recuadro de captura
            cv2.rectangle(frame, (100, 100), (540, 380), (0, 255, 0), 2)
            
            # Información en pantalla
            cv2.putText(frame, f"Clase: {class_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Muestras: {samples_collected}/{num_samples}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Detectar manos
            hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    hand_detected = True
            
            # Mostrar estado de detección
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "Mano detectada" if hand_detected else "No hay mano"
            cv2.putText(frame, status_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Instrucciones
            cv2.putText(frame, "ESPACIO: Capturar | q: Terminar | n: Siguiente", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Captura de Datos ASL', frame)
            
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord(' ') and hand_detected and (current_time - last_capture_time) > capture_cooldown:
                # Capturar imagen
                filename = self.get_next_filename(class_dir, class_name)
                filepath = os.path.join(class_dir, filename)
                
                # Guardar imagen
                cv2.imwrite(filepath, frame)
                samples_collected += 1
                last_capture_time = current_time
                
                print(f"✅ Capturada: {filename} ({samples_collected}/{num_samples})")
                
                # Efecto visual de captura
                capture_frame = frame.copy()
                cv2.rectangle(capture_frame, (0, 0), (w, h), (0, 255, 0), 10)
                cv2.imshow('Captura de Datos ASL', capture_frame)
                cv2.waitKey(100)
            
            elif key == ord('q'):
                break
            elif key == ord('n'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Clase '{class_name}' completada: {samples_collected} muestras")
        return samples_collected
    
    def collect_full_dataset(self, classes=None, samples_per_class=100):
        """
        Recolectar dataset completo
        
        Args:
            classes: Lista de clases a recolectar. Si None, usa alfabeto completo
            samples_per_class: Número de muestras por clase
        """
        if classes is None:
            # Alfabeto completo + comandos especiales
            classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]  # A-Z
            classes.extend(['space', 'del', 'nothing'])
        
        print(f"🎯 Recolección de Dataset ASL")
        print(f"📊 Clases a recolectar: {len(classes)}")
        print(f"📝 Muestras por clase: {samples_per_class}")
        print(f"📈 Total estimado: {len(classes) * samples_per_class} muestras")
        
        total_collected = 0
        
        for i, class_name in enumerate(classes):
            print(f"\n📋 Progreso: {i+1}/{len(classes)} clases")
            collected = self.collect_for_class(class_name, samples_per_class)
            total_collected += collected
            
            if collected == 0:
                print(f"⚠️  No se recolectaron muestras para '{class_name}'")
        
        print(f"\n🎉 Recolección completada!")
        print(f"📊 Total recolectado: {total_collected} muestras")
        print(f"💾 Datos guardados en: {self.data_dir}/")
        
        # Mostrar resumen
        self.show_dataset_summary()
    
    def show_dataset_summary(self):
        """
        Mostrar resumen del dataset
        """
        print(f"\n📊 RESUMEN DEL DATASET")
        print(f"{'='*50}")
        
        if not os.path.exists(self.data_dir):
            print("No hay datos recolectados aún")
            return
        
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d))]
        
        total_samples = 0
        class_counts = {}
        
        for class_name in sorted(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            num_samples = len([f for f in os.listdir(class_dir) 
                             if f.endswith('.jpg')])
            class_counts[class_name] = num_samples
            total_samples += num_samples
            print(f"  {class_name}: {num_samples} muestras")
        
        print(f"\n📈 Total de clases: {len(classes)}")
        print(f"📈 Total de muestras: {total_samples}")
        
        if class_counts:
            print(f"📊 Promedio por clase: {total_samples/len(classes):.1f}")
            print(f"📊 Mínimo: {min(class_counts.values())}")
            print(f"📊 Máximo: {max(class_counts.values())}")

def main():
    """
    Función principal para recolectar datos
    """
    collector = DataCollector()
    
    print("🎯 RECOLECTOR DE DATOS ASL")
    print("=" * 50)
    print("Este script te ayudará a crear tu propio dataset de señas ASL")
    print("\nOpciones:")
    print("1. Recolectar dataset completo (A-Z + comandos)")
    print("2. Recolectar clases específicas")
    print("3. Continuar recolección existente")
    print("4. Ver resumen del dataset actual")
    
    while True:
        choice = input("\nSelecciona una opción (1-4): ").strip()
        
        if choice == '1':
            samples = int(input("Número de muestras por clase (default: 100): ") or "100")
            collector.collect_full_dataset(samples_per_class=samples)
            break
            
        elif choice == '2':
            classes_input = input("Ingresa las clases separadas por coma (ej: A,B,C): ").strip()
            classes = [c.strip().upper() for c in classes_input.split(',')]
            samples = int(input("Número de muestras por clase (default: 100): ") or "100")
            collector.collect_full_dataset(classes=classes, samples_per_class=samples)
            break
            
        elif choice == '3':
            # Mostrar clases existentes
            collector.show_dataset_summary()
            class_name = input("\nIngresa la clase a continuar: ").strip().upper()
            samples = int(input("Número adicional de muestras (default: 50): ") or "50")
            collector.collect_for_class(class_name, samples)
            break
            
        elif choice == '4':
            collector.show_dataset_summary()
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
