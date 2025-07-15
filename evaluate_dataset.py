"""
Script especializado para evaluar modelo ASL con el dataset específico
Usa las imágenes de test individuales para evaluación
"""

import os
import joblib
import numpy as np
import cv2
import mediapipe as mp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ASLModelEvaluator:
    def __init__(self, model_path="model/asl_model.joblib", encoder_path="model/label_encoder.joblib"):
        """
        Inicializar evaluador de modelo ASL
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        
        # Cargar modelo
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"✅ Modelo cargado: {model_path}")
            print(f"✅ Codificador cargado: {encoder_path}")
        else:
            raise FileNotFoundError("Archivos del modelo no encontrados")
        
        # Configurar MediaPipe con configuración más permisiva
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Más permisivo
            min_tracking_confidence=0.1    # Más permisivo
        )
    
    def extract_features_from_image(self, image_path):
        """
        Extraer características de una imagen usando MediaPipe
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
                # Extraer características del primer conjunto de puntos
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
    
    def predict_single_image(self, image_path):
        """
        Predecir la clase de una sola imagen
        """
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
    
    def evaluate_test_images(self, test_dir="data/asl_alphabet_test"):
        """
        Evaluar usando las imágenes de test individuales
        """
        print(f"🧪 Evaluando imágenes de test desde: {test_dir}")
        
        if not os.path.exists(test_dir):
            print(f"❌ Directorio de test no encontrado: {test_dir}")
            return
        
        # Obtener imágenes de test
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        
        if not test_images:
            print(f"❌ No se encontraron imágenes de test en {test_dir}")
            return
        
        print(f"📊 Evaluando {len(test_images)} imágenes de test...")
        
        results = []
        
        for img_file in test_images:
            # Extraer la etiqueta verdadera del nombre del archivo
            true_label = img_file.replace('_test.jpg', '')
            
            # Predecir
            img_path = os.path.join(test_dir, img_file)
            pred_label, confidence = self.predict_single_image(img_path)
            
            if pred_label is not None:
                is_correct = pred_label.lower() == true_label.lower()
                results.append({
                    'image': img_file,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {img_file}: {true_label} -> {pred_label} (conf: {confidence:.3f})")
            else:
                print(f"⚠️  {img_file}: No se pudo detectar mano")
                results.append({
                    'image': img_file,
                    'true_label': true_label,
                    'predicted_label': 'NO_HAND',
                    'confidence': 0.0,
                    'correct': False
                })
        
        # Calcular métricas
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len([r for r in results if r['predicted_label'] != 'NO_HAND'])
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"\n📊 RESULTADOS DE EVALUACIÓN")
            print(f"{'='*50}")
            print(f"✅ Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"📈 Correctas: {correct_predictions}/{total_predictions}")
            print(f"⚠️  Sin detección de mano: {len(results) - total_predictions}")
            
            # Crear DataFrame para análisis
            df_results = pd.DataFrame(results)
            
            # Mostrar errores
            errors = df_results[~df_results['correct'] & (df_results['predicted_label'] != 'NO_HAND')]
            if not errors.empty:
                print(f"\n❌ ERRORES ENCONTRADOS:")
                for _, row in errors.iterrows():
                    print(f"  {row['image']}: {row['true_label']} -> {row['predicted_label']} (conf: {row['confidence']:.3f})")
            
            # Guardar resultados
            self.save_test_results(df_results, accuracy)
            
            return accuracy, results
        else:
            print("❌ No se pudieron procesar las imágenes de test")
            return 0.0, []
    
    def evaluate_training_subset(self, train_dir="data/asl_alphabet_train", samples_per_class=100):
        """
        Evaluar usando un subconjunto del dataset de entrenamiento
        """
        print(f"🧪 Evaluando subconjunto de datos de entrenamiento...")
        print(f"📊 Muestras por clase: {samples_per_class}")
        
        if not os.path.exists(train_dir):
            print(f"❌ Directorio de entrenamiento no encontrado: {train_dir}")
            return
        
        # Obtener clases
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        all_features = []
        all_labels = []
        
        for class_name in classes:
            class_dir = os.path.join(train_dir, class_name)
            
            # Obtener imágenes de la clase
            images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            
            # Tomar muestra aleatoria
            import random
            sample_images = random.sample(images, min(samples_per_class, len(images)))
            
            for img_file in sample_images:
                img_path = os.path.join(class_dir, img_file)
                features = self.extract_features_from_image(img_path)
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(class_name)
        
        if not all_features:
            print("❌ No se pudieron extraer características")
            return
        
        # Convertir a arrays numpy
        X = np.array(all_features)
        y_true = np.array(all_labels)
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.transform(y_true)
        
        # Predecir
        y_pred = self.model.predict(X)
        
        # Calcular métricas
        accuracy = accuracy_score(y_encoded, y_pred)
        
        print(f"\n📊 RESULTADOS DE EVALUACIÓN (Subconjunto)")
        print(f"{'='*50}")
        print(f"✅ Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 Muestras evaluadas: {len(all_features)}")
        print(f"📊 Clases: {len(classes)}")
        
        # Reporte detallado
        class_names = self.label_encoder.classes_
        report = classification_report(y_encoded, y_pred, target_names=class_names)
        print(f"\n📋 Reporte detallado:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(y_encoded, y_pred)
        self.plot_confusion_matrix(cm, class_names, f"Evaluación Subconjunto - Acc: {accuracy:.4f}")
        
        return accuracy, report, cm
    
    def plot_confusion_matrix(self, cm, class_names, title):
        """
        Graficar matriz de confusión
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Número de Predicciones'})
        plt.title(title)
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrix_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_test_results(self, df_results, accuracy):
        """
        Guardar resultados de evaluación
        """
        os.makedirs('reports', exist_ok=True)
        
        # Guardar CSV con resultados detallados
        df_results.to_csv('reports/test_results.csv', index=False)
        
        # Guardar reporte de texto
        with open('reports/test_evaluation_report.txt', 'w') as f:
            f.write(f"Reporte de Evaluación - Imágenes de Test\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Total de imágenes: {len(df_results)}\n")
            f.write(f"Correctas: {sum(df_results['correct'])}\n\n")
            
            # Errores
            errors = df_results[~df_results['correct'] & (df_results['predicted_label'] != 'NO_HAND')]
            if not errors.empty:
                f.write("Errores encontrados:\n")
                for _, row in errors.iterrows():
                    f.write(f"  {row['image']}: {row['true_label']} -> {row['predicted_label']} (conf: {row['confidence']:.3f})\n")
        
        print(f"💾 Resultados guardados en: reports/test_results.csv")
        print(f"📋 Reporte guardado en: reports/test_evaluation_report.txt")

def main():
    """
    Función principal para evaluación
    """
    print("🔍 EVALUADOR ESPECIALIZADO ASL")
    print("=" * 50)
    
    try:
        evaluator = ASLModelEvaluator()
        
        print("Opciones de evaluación:")
        print("1. Evaluar con imágenes de test individuales")
        print("2. Evaluar con subconjunto de entrenamiento")
        print("3. Ambas evaluaciones")
        
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == '1':
            evaluator.evaluate_test_images()
        elif choice == '2':
            samples = int(input("Muestras por clase (default: 100): ") or "100")
            evaluator.evaluate_training_subset(samples_per_class=samples)
        elif choice == '3':
            print("\n🧪 Evaluando con imágenes de test...")
            evaluator.evaluate_test_images()
            
            print("\n🧪 Evaluando con subconjunto de entrenamiento...")
            samples = int(input("Muestras por clase (default: 100): ") or "100")
            evaluator.evaluate_training_subset(samples_per_class=samples)
        else:
            print("❌ Opción inválida")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de que el modelo esté entrenado primero con: python train_model.py")

if __name__ == "__main__":
    main()
