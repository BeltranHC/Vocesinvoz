"""
Script para diagnosticar y mejorar la detección de manos en imágenes de test
"""

import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class HandDetectionDiagnostic:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configuraciones diferentes para probar
        self.configs = {
            'strict': {
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.5
            },
            'medium': {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.3
            },
            'relaxed': {
                'min_detection_confidence': 0.3,
                'min_tracking_confidence': 0.1
            }
        }
    
    def test_image_with_configs(self, image_path):
        """
        Probar una imagen con diferentes configuraciones de MediaPipe
        """
        print(f"\n🔍 Diagnosticando: {os.path.basename(image_path)}")
        
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: No se pudo cargar la imagen")
            return
        
        # Convertir a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {}
        
        # Probar cada configuración
        for config_name, config_params in self.configs.items():
            hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                **config_params
            )
            
            result = hands.process(rgb_image)
            
            if result.multi_hand_landmarks:
                # Extraer características
                hand_landmarks = result.multi_hand_landmarks[0]
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                results[config_name] = {
                    'detected': True,
                    'landmarks': hand_landmarks,
                    'features': features
                }
                print(f"  ✅ {config_name}: Mano detectada")
            else:
                results[config_name] = {
                    'detected': False,
                    'landmarks': None,
                    'features': None
                }
                print(f"  ❌ {config_name}: No se detectó mano")
            
            hands.close()
        
        return results, image, rgb_image
    
    def create_diagnostic_visualization(self, image_path, results, image, rgb_image):
        """
        Crear visualización diagnóstica
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Diagnóstico: {os.path.basename(image_path)}', fontsize=14)
        
        # Imagen original
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # Configuraciones de detección
        config_names = ['strict', 'medium', 'relaxed']
        positions = [(0, 1), (1, 0), (1, 1)]
        
        for i, config_name in enumerate(config_names):
            row, col = positions[i]
            
            # Crear copia de la imagen
            vis_image = rgb_image.copy()
            
            if results[config_name]['detected']:
                # Dibujar landmarks
                annotated_image = vis_image.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_image, 
                    results[config_name]['landmarks'], 
                    self.mp_hands.HAND_CONNECTIONS
                )
                axes[row, col].imshow(annotated_image)
                axes[row, col].set_title(f'{config_name.upper()}: ✅ Detectado')
            else:
                axes[row, col].imshow(vis_image)
                axes[row, col].set_title(f'{config_name.upper()}: ❌ No detectado')
            
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Guardar
        os.makedirs('reports/diagnostics', exist_ok=True)
        filename = f"diagnostic_{os.path.basename(image_path).replace('.jpg', '')}.png"
        plt.savefig(f'reports/diagnostics/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def diagnose_test_images(self, test_dir="data/asl_alphabet_test"):
        """
        Diagnosticar todas las imágenes de test
        """
        print("🔍 DIAGNÓSTICO DE DETECCIÓN DE MANOS")
        print("=" * 50)
        
        if not os.path.exists(test_dir):
            print(f"❌ Directorio no encontrado: {test_dir}")
            return
        
        # Obtener imágenes de test
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        
        if not test_images:
            print(f"❌ No se encontraron imágenes en {test_dir}")
            return
        
        print(f"📊 Analizando {len(test_images)} imágenes...")
        
        # Estadísticas
        stats = {config: {'detected': 0, 'total': 0} for config in self.configs.keys()}
        
        problem_images = []
        
        for img_file in test_images:
            img_path = os.path.join(test_dir, img_file)
            
            try:
                results, image, rgb_image = self.test_image_with_configs(img_path)
                
                # Actualizar estadísticas
                all_failed = True
                for config_name, result in results.items():
                    stats[config_name]['total'] += 1
                    if result['detected']:
                        stats[config_name]['detected'] += 1
                        all_failed = False
                
                # Si ninguna configuración detectó la mano
                if all_failed:
                    problem_images.append(img_file)
                    # Crear visualización diagnóstica
                    diag_file = self.create_diagnostic_visualization(img_path, results, image, rgb_image)
                    print(f"  📊 Diagnóstico guardado: {diag_file}")
                
            except Exception as e:
                print(f"❌ Error procesando {img_file}: {e}")
        
        # Mostrar estadísticas finales
        print(f"\n📊 ESTADÍSTICAS DE DETECCIÓN")
        print(f"=" * 30)
        
        for config_name, stat in stats.items():
            if stat['total'] > 0:
                detection_rate = stat['detected'] / stat['total'] * 100
                print(f"{config_name.upper()}: {stat['detected']}/{stat['total']} ({detection_rate:.1f}%)")
        
        # Imágenes problemáticas
        if problem_images:
            print(f"\n❌ IMÁGENES PROBLEMÁTICAS ({len(problem_images)}):")
            for img in problem_images:
                print(f"  - {img}")
            
            print(f"\n💡 RECOMENDACIONES:")
            print(f"1. Revisar las imágenes en reports/diagnostics/")
            print(f"2. Verificar calidad/iluminación de las imágenes")
            print(f"3. Considerar preprocesamiento de imágenes")
            print(f"4. Usar configuración 'relaxed' en el evaluador")
        
        return stats, problem_images
    
    def suggest_optimal_config(self, stats):
        """
        Sugerir configuración óptima
        """
        print(f"\n🎯 CONFIGURACIÓN RECOMENDADA:")
        
        best_config = None
        best_rate = 0
        
        for config_name, stat in stats.items():
            if stat['total'] > 0:
                rate = stat['detected'] / stat['total']
                if rate > best_rate:
                    best_rate = rate
                    best_config = config_name
        
        if best_config:
            print(f"✅ Usar configuración '{best_config}' ({best_rate*100:.1f}% detección)")
            print(f"📋 Parámetros:")
            for param, value in self.configs[best_config].items():
                print(f"  - {param}: {value}")
        
        return best_config

def main():
    """
    Función principal para diagnóstico
    """
    diagnostic = HandDetectionDiagnostic()
    
    print("🔍 DIAGNÓSTICO DE DETECCIÓN DE MANOS")
    print("=" * 50)
    
    # Ejecutar diagnóstico
    stats, problem_images = diagnostic.diagnose_test_images()
    
    # Sugerir configuración óptima
    optimal_config = diagnostic.suggest_optimal_config(stats)
    
    # Crear script de evaluación mejorado
    if optimal_config:
        create_improved_evaluator(optimal_config, diagnostic.configs[optimal_config])
    
    print(f"\n💾 Resultados guardados en: reports/diagnostics/")

def create_improved_evaluator(config_name, config_params):
    """
    Crear evaluador mejorado con la configuración óptima
    """
    script_content = f'''"""
Evaluador mejorado con configuración optimizada de MediaPipe
Configuración: {config_name}
"""

import os
import joblib
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd

class ImprovedEvaluator:
    def __init__(self, model_path="model/asl_model.joblib", encoder_path="model/label_encoder.joblib"):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Configuración optimizada
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence={config_params['min_detection_confidence']},
            min_tracking_confidence={config_params['min_tracking_confidence']}
        )
        
        print(f"✅ Evaluador mejorado iniciado con configuración {config_name}")
    
    def predict_image(self, image_path):
        """Predecir imagen con configuración optimizada"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, 0.0
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                X = np.array(features).reshape(1, -1)
                prediction = self.model.predict(X)[0]
                probabilities = self.model.predict_proba(X)[0]
                confidence = np.max(probabilities)
                
                label = self.label_encoder.inverse_transform([prediction])[0]
                return label, confidence
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"Error procesando {{image_path}}: {{e}}")
            return None, 0.0
    
    def evaluate_test_set(self, test_dir="data/asl_alphabet_test"):
        """Evaluar conjunto de test con configuración mejorada"""
        print(f"🧪 Evaluando con configuración optimizada...")
        
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        results = []
        
        for img_file in test_images:
            true_label = img_file.replace('_test.jpg', '')
            img_path = os.path.join(test_dir, img_file)
            
            pred_label, confidence = self.predict_image(img_path)
            
            if pred_label is not None:
                is_correct = pred_label.lower() == true_label.lower()
                results.append({{
                    'image': img_file,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'correct': is_correct
                }})
                
                status = "✅" if is_correct else "❌"
                print(f"{{status}} {{img_file}}: {{true_label}} -> {{pred_label}} ({{confidence:.3f}})")
            else:
                print(f"⚠️  {{img_file}}: No se detectó mano")
                results.append({{
                    'image': img_file,
                    'true_label': true_label,
                    'predicted_label': 'NO_HAND',
                    'confidence': 0.0,
                    'correct': False
                }})
        
        # Calcular métricas
        df = pd.DataFrame(results)
        valid_predictions = len(df[df['predicted_label'] != 'NO_HAND'])
        correct_predictions = len(df[df['correct'] == True])
        
        print(f"\\n📊 RESULTADOS MEJORADOS:")
        print(f"✅ Predicciones válidas: {{valid_predictions}}/{{len(df)}} ({{valid_predictions/len(df)*100:.1f}}%)")
        if valid_predictions > 0:
            accuracy = correct_predictions / valid_predictions
            print(f"📊 Precisión: {{accuracy:.3f}} ({{accuracy*100:.1f}}%)")
        
        # Guardar resultados
        df.to_csv('reports/improved_test_results.csv', index=False)
        print(f"💾 Resultados guardados en: reports/improved_test_results.csv")
        
        return df

if __name__ == "__main__":
    evaluator = ImprovedEvaluator()
    evaluator.evaluate_test_set()
'''
    
    with open('evaluate_improved.py', 'w') as f:
        f.write(script_content)
    
    print(f"✅ Evaluador mejorado creado: evaluate_improved.py")
    print(f"🚀 Ejecutar con: python evaluate_improved.py")

if __name__ == "__main__":
    main()
