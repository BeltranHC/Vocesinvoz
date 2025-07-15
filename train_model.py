"""
Script para entrenar el modelo ASL desde datos propios
Genera características desde imágenes usando MediaPipe y entrena un RandomForestClassifier
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class ASLModelTrainer:
    def __init__(self, data_dir="data"):
        """
        Inicializar el entrenador de modelos ASL
        
        Args:
            data_dir: Directorio que contiene las carpetas de cada clase
                     Estructura esperada:
                     data/
                     ├── A/
                     │   ├── img1.jpg
                     │   ├── img2.jpg
                     │   └── ...
                     ├── B/
                     ├── C/
                     └── ...
        """
        self.data_dir = data_dir
        self.model = None
        self.label_encoder = None
        self.features = []
        self.labels = []
        
        # Configurar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Crear directorios si no existen
        os.makedirs("model", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
    
    def extract_features_from_image(self, image_path):
        """
        Extraer características de una imagen usando MediaPipe
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            features: Array numpy con 63 características (21 puntos * 3 coordenadas)
                     o None si no se detecta una mano
        """
        try:
            # Leer imagen
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convertir a RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Extraer características del primer (único) conjunto de puntos
                hand_landmarks = results.multi_hand_landmarks[0]
                
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                return np.array(features)
            else:
                return None
                
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None
    
    def load_data(self):
        """
        Cargar datos desde el directorio de datos
        """
        print("Cargando datos y extrayendo características...")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directorio de datos no encontrado: {self.data_dir}")
        
        # Obtener todas las clases (carpetas)
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"Clases encontradas: {classes}")
        
        # Procesar cada clase
        for class_name in tqdm(classes, desc="Procesando clases"):
            class_dir = os.path.join(self.data_dir, class_name)
            
            # Obtener todas las imágenes de la clase
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_dir, ext)))
            
            print(f"Clase '{class_name}': {len(image_files)} imágenes")
            
            # Procesar cada imagen
            for img_path in tqdm(image_files, desc=f"Procesando {class_name}", leave=False):
                features = self.extract_features_from_image(img_path)
                if features is not None:
                    self.features.append(features)
                    self.labels.append(class_name)
        
        print(f"Total de muestras cargadas: {len(self.features)}")
        
        if len(self.features) == 0:
            raise ValueError("No se pudieron extraer características de ninguna imagen")
        
        # Convertir a arrays numpy
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        print(f"Forma de características: {self.features.shape}")
        print(f"Clases únicas: {np.unique(self.labels)}")
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Entrenar el modelo RandomForestClassifier
        
        Args:
            test_size: Proporción de datos para testing
            random_state: Semilla para reproducibilidad
        """
        print("Entrenando modelo...")
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
        print(f"Datos de prueba: {X_test.shape[0]} muestras")
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {accuracy:.4f}")
        
        # Reporte detallado
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        print("\nReporte de clasificación:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Guardar reporte
        self.save_training_report(accuracy, report, cm, class_names)
        
        return accuracy, report, cm
    
    def save_training_report(self, accuracy, report, confusion_matrix, class_names):
        """
        Guardar reporte de entrenamiento
        """
        # Crear visualización de matriz de confusión
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Matriz de Confusión - Precisión: {accuracy:.4f}')
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Real')
        plt.tight_layout()
        plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar reporte en texto
        with open('reports/training_report.txt', 'w') as f:
            f.write(f"Reporte de Entrenamiento del Modelo ASL\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Precisión: {accuracy:.4f}\n\n")
            f.write(f"Número de clases: {len(class_names)}\n")
            f.write(f"Clases: {', '.join(class_names)}\n\n")
            f.write(f"Reporte de Clasificación:\n")
            f.write(report)
        
        # Guardar metadatos en JSON
        metadata = {
            'accuracy': float(accuracy),
            'num_classes': len(class_names),
            'classes': class_names.tolist(),
            'num_samples': len(self.features),
            'features_shape': self.features.shape,
            'model_type': 'RandomForestClassifier'
        }
        
        with open('reports/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Reporte guardado en: reports/")
    
    def save_model(self, model_path="model/asl_model.joblib", encoder_path="model/label_encoder.joblib"):
        """
        Guardar modelo y codificador entrenados
        """
        if self.model is None or self.label_encoder is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_model() primero.")
        
        # Guardar modelo
        joblib.dump(self.model, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        # Guardar codificador
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Codificador guardado en: {encoder_path}")
    
    def load_model(self, model_path="model/asl_model.joblib", encoder_path="model/label_encoder.joblib"):
        """
        Cargar modelo y codificador guardados
        """
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Modelo cargado exitosamente")
    
    def predict_from_image(self, image_path):
        """
        Predecir clase de una imagen
        """
        if self.model is None or self.label_encoder is None:
            raise ValueError("Modelo no cargado")
        
        features = self.extract_features_from_image(image_path)
        if features is None:
            return None, 0.0
        
        # Predecir
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = np.max(probabilities)
        
        # Decodificar etiqueta
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        return label, confidence

def main():
    """
    Función principal para entrenar el modelo
    """
    # Configuración
    DATA_DIR = "data/asl_alphabet_train"  # Usar tu dataset específico
    
    # Verificar que existe el directorio de datos
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directorio de datos '{DATA_DIR}' no encontrado.")
        print("Estructura esperada:")
        print("data/")
        print("├── asl_alphabet_train/")
        print("│   ├── A/")
        print("│   ├── B/")
        print("│   └── ...")
        return
    
    # Crear entrenador
    trainer = ASLModelTrainer(DATA_DIR)
    
    try:
        # Cargar datos
        trainer.load_data()
        
        # Entrenar modelo
        accuracy, report, cm = trainer.train_model()
        
        # Guardar modelo
        trainer.save_model()
        
        print(f"\n🎉 Entrenamiento completado exitosamente!")
        print(f"📊 Precisión final: {accuracy:.4f}")
        print(f"💾 Modelo guardado en: model/")
        print(f"📋 Reportes guardados en: reports/")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()
