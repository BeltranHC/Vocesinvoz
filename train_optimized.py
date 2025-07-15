"""
Script optimizado para entrenar modelo ASL con datasets grandes
Incluye técnicas de optimización y procesamiento eficiente
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class OptimizedASLTrainer:
    def __init__(self, data_dir="data/asl_alphabet_train", use_subset=False, subset_size=500):
        """
        Inicializar el entrenador optimizado
        
        Args:
            data_dir: Directorio con datos de entrenamiento
            use_subset: Si usar solo un subconjunto de datos para pruebas rápidas
            subset_size: Tamaño del subconjunto por clase
        """
        self.data_dir = data_dir
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.model = None
        self.label_encoder = None
        self.features = []
        self.labels = []
        
        # Configurar MediaPipe
        self.mp_hands = mp.solutions.hands
        
        # Crear directorios
        os.makedirs("model", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        print(f"🚀 Entrenador ASL Optimizado")
        print(f"📂 Directorio de datos: {data_dir}")
        print(f"🔬 Modo subconjunto: {'Sí' if use_subset else 'No'}")
        if use_subset:
            print(f"📊 Tamaño subconjunto: {subset_size} por clase")
    
    def extract_features_batch(self, image_paths_and_labels):
        """
        Extraer características de un lote de imágenes
        Función optimizada para procesamiento paralelo
        """
        # Configurar MediaPipe para este proceso
        hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        features_batch = []
        labels_batch = []
        
        for img_path, label in image_paths_and_labels:
            try:
                # Leer imagen
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convertir a RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Procesar con MediaPipe
                results = hands.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    # Extraer características
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                    
                    features_batch.append(features)
                    labels_batch.append(label)
                    
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue
        
        hands.close()
        return features_batch, labels_batch
    
    def load_data_parallel(self, num_workers=None):
        """
        Cargar datos usando procesamiento paralelo
        """
        if num_workers is None:
            num_workers = min(8, multiprocessing.cpu_count())
        
        print(f"🔄 Cargando datos con {num_workers} workers...")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directorio de datos no encontrado: {self.data_dir}")
        
        # Obtener todas las clases
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"📊 Clases encontradas: {len(classes)}")
        print(f"📋 Clases: {classes}")
        
        # Recolectar todos los paths e imágenes
        all_image_paths = []
        
        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            
            # Obtener imágenes de la clase
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_dir, ext)))
            
            # Usar subconjunto si está habilitado
            if self.use_subset and len(image_files) > self.subset_size:
                import random
                random.shuffle(image_files)
                image_files = image_files[:self.subset_size]
            
            # Agregar paths con etiquetas
            for img_path in image_files:
                all_image_paths.append((img_path, class_name))
            
            print(f"  {class_name}: {len(image_files)} imágenes")
        
        total_images = len(all_image_paths)
        print(f"📈 Total de imágenes a procesar: {total_images}")
        
        if total_images == 0:
            raise ValueError("No se encontraron imágenes para procesar")
        
        # Dividir en lotes para procesamiento paralelo
        batch_size = max(100, total_images // num_workers)
        batches = []
        
        for i in range(0, total_images, batch_size):
            batch = all_image_paths[i:i + batch_size]
            batches.append(batch)
        
        print(f"🔄 Procesando {len(batches)} lotes...")
        
        # Procesar lotes en paralelo
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Enviar trabajos
            futures = [executor.submit(self.extract_features_batch, batch) 
                      for batch in batches]
            
            # Recolectar resultados
            for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando lotes"):
                try:
                    features_batch, labels_batch = future.result()
                    self.features.extend(features_batch)
                    self.labels.extend(labels_batch)
                except Exception as e:
                    print(f"Error en lote: {e}")
        
        processing_time = time.time() - start_time
        
        print(f"✅ Procesamiento completado en {processing_time:.2f} segundos")
        print(f"📊 Características extraídas: {len(self.features)}")
        print(f"📈 Tasa de éxito: {len(self.features)/total_images*100:.1f}%")
        
        if len(self.features) == 0:
            raise ValueError("No se pudieron extraer características de ninguna imagen")
        
        # Convertir a arrays numpy
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        print(f"📐 Forma de características: {self.features.shape}")
        print(f"🏷️  Clases únicas: {np.unique(self.labels)}")
        
        # Guardar datos procesados para uso futuro
        self.save_processed_data()
    
    def save_processed_data(self):
        """
        Guardar datos procesados para evitar reprocesamiento
        """
        data_cache = {
            'features': self.features,
            'labels': self.labels,
            'use_subset': self.use_subset,
            'subset_size': self.subset_size if self.use_subset else None
        }
        
        cache_path = f"model/processed_data_{'subset' if self.use_subset else 'full'}.joblib"
        joblib.dump(data_cache, cache_path)
        print(f"💾 Datos procesados guardados en: {cache_path}")
    
    def load_processed_data(self):
        """
        Cargar datos procesados si existen
        """
        cache_path = f"model/processed_data_{'subset' if self.use_subset else 'full'}.joblib"
        
        if os.path.exists(cache_path):
            print(f"📥 Cargando datos procesados desde: {cache_path}")
            data_cache = joblib.load(cache_path)
            
            self.features = data_cache['features']
            self.labels = data_cache['labels']
            
            print(f"✅ Datos cargados: {len(self.features)} muestras")
            return True
        return False
    
    def train_optimized_model(self, test_size=0.2, random_state=42):
        """
        Entrenar modelo optimizado con validación cruzada
        """
        print("🤖 Entrenando modelo optimizado...")
        
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
        
        print(f"📊 Datos de entrenamiento: {X_train.shape[0]} muestras")
        print(f"📊 Datos de prueba: {X_test.shape[0]} muestras")
        
        # Configurar modelo optimizado
        self.model = RandomForestClassifier(
            n_estimators=300,        # Más árboles para mejor precisión
            max_depth=25,            # Profundidad ajustada
            min_samples_split=10,    # Evitar sobreajuste
            min_samples_leaf=4,      # Evitar sobreajuste
            max_features='sqrt',     # Características óptimas
            random_state=random_state,
            n_jobs=-1,               # Usar todos los cores
            class_weight='balanced'  # Balance de clases
        )
        
        # Entrenar modelo
        print("🔄 Entrenando RandomForestClassifier...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        # Evaluación en conjunto de prueba
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Precisión en conjunto de prueba: {accuracy:.4f}")
        
        # Validación cruzada
        print("🔄 Realizando validación cruzada...")
        cv_scores = cross_val_score(self.model, self.features, y_encoded, cv=5, scoring='accuracy')
        
        print(f"📊 Precisión promedio (CV): {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Reporte detallado
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        print("\n📋 Reporte de clasificación:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Guardar reportes
        self.save_training_report(accuracy, cv_scores, report, cm, class_names, training_time)
        
        return accuracy, cv_scores, report, cm
    
    def save_training_report(self, accuracy, cv_scores, report, cm, class_names, training_time):
        """
        Guardar reporte completo de entrenamiento
        """
        # Matriz de confusión
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Número de Predicciones'})
        plt.title(f'Matriz de Confusión - Precisión: {accuracy:.4f}')
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('reports/confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reporte de texto
        with open('reports/training_report_optimized.txt', 'w') as f:
            f.write(f"Reporte de Entrenamiento ASL - Modelo Optimizado\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Configuración:\n")
            f.write(f"- Directorio de datos: {self.data_dir}\n")
            f.write(f"- Modo subconjunto: {'Sí' if self.use_subset else 'No'}\n")
            if self.use_subset:
                f.write(f"- Tamaño subconjunto: {self.subset_size} por clase\n")
            f.write(f"- Tiempo de entrenamiento: {training_time:.2f} segundos\n\n")
            
            f.write(f"Resultados:\n")
            f.write(f"- Precisión: {accuracy:.4f}\n")
            f.write(f"- Precisión promedio (CV): {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})\n")
            f.write(f"- Número de clases: {len(class_names)}\n")
            f.write(f"- Clases: {', '.join(class_names)}\n")
            f.write(f"- Número de muestras: {len(self.features)}\n")
            f.write(f"- Características por muestra: {self.features.shape[1]}\n\n")
            
            f.write(f"Reporte de Clasificación:\n")
            f.write(report)
        
        # Metadatos JSON
        metadata = {
            'accuracy': float(accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'num_classes': len(class_names),
            'classes': class_names.tolist(),
            'num_samples': len(self.features),
            'features_shape': self.features.shape,
            'model_type': 'RandomForestClassifier',
            'training_time': training_time,
            'use_subset': self.use_subset,
            'subset_size': self.subset_size if self.use_subset else None
        }
        
        with open('reports/model_metadata_optimized.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Reportes guardados en: reports/")
    
    def save_model(self, model_path="model/asl_model.joblib", encoder_path="model/label_encoder.joblib"):
        """
        Guardar modelo entrenado
        """
        if self.model is None or self.label_encoder is None:
            raise ValueError("Modelo no entrenado")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"💾 Modelo guardado en: {model_path}")
        print(f"💾 Codificador guardado en: {encoder_path}")

def main():
    """
    Función principal optimizada
    """
    print("🚀 ENTRENADOR ASL OPTIMIZADO")
    print("=" * 50)
    
    # Configuración
    DATA_DIR = "data/asl_alphabet_train"
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Directorio de datos '{DATA_DIR}' no encontrado.")
        return
    
    # Opciones de entrenamiento
    print("Opciones de entrenamiento:")
    print("1. Entrenamiento completo (87,000 imágenes) - Recomendado")
    print("2. Entrenamiento rápido (subconjunto) - Para pruebas")
    
    choice = input("Selecciona opción (1-2): ").strip()
    
    if choice == '1':
        trainer = OptimizedASLTrainer(DATA_DIR, use_subset=False)
    elif choice == '2':
        subset_size = int(input("Tamaño del subconjunto por clase (default: 500): ") or "500")
        trainer = OptimizedASLTrainer(DATA_DIR, use_subset=True, subset_size=subset_size)
    else:
        print("❌ Opción inválida")
        return
    
    try:
        # Intentar cargar datos procesados
        if not trainer.load_processed_data():
            # Cargar y procesar datos
            trainer.load_data_parallel()
        
        # Entrenar modelo
        accuracy, cv_scores, report, cm = trainer.train_optimized_model()
        
        # Guardar modelo
        trainer.save_model()
        
        print(f"\n🎉 Entrenamiento completado exitosamente!")
        print(f"📊 Precisión final: {accuracy:.4f}")
        print(f"📊 Precisión promedio (CV): {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"💾 Modelo guardado en: model/")
        print(f"📋 Reportes guardados en: reports/")
        
        # Sugerir siguiente paso
        print(f"\n🔮 Próximos pasos:")
        print(f"1. Evaluar modelo: python evaluate_dataset.py")
        print(f"2. Probar aplicación: python app.py")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
