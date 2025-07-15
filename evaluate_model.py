"""
Script para evaluar un modelo ASL existente o comparar modelos
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from train_model import ASLModelTrainer

def evaluate_model(model_path="model/asl_model.joblib", 
                  encoder_path="model/label_encoder.joblib",
                  test_data_dir="data"):
    """
    Evaluar un modelo entrenado con datos de prueba
    
    Args:
        model_path: Ruta al modelo guardado
        encoder_path: Ruta al codificador guardado
        test_data_dir: Directorio con datos de prueba
    """
    
    print("🔍 Evaluando modelo ASL...")
    
    # Verificar que los archivos existen
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return
    
    if not os.path.exists(encoder_path):
        print(f"❌ Error: Codificador no encontrado en {encoder_path}")
        return
    
    # Cargar modelo
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    print(f"✅ Modelo cargado: {model_path}")
    print(f"✅ Codificador cargado: {encoder_path}")
    print(f"📊 Clases disponibles: {list(label_encoder.classes_)}")
    
    # Cargar datos de prueba
    if os.path.exists(test_data_dir):
        print(f"📂 Cargando datos de prueba desde: {test_data_dir}")
        trainer = ASLModelTrainer(test_data_dir)
        trainer.load_data()
        
        # Codificar etiquetas
        y_encoded = label_encoder.transform(trainer.labels)
        
        # Predicciones
        y_pred = model.predict(trainer.features)
        
        # Calcular métricas
        accuracy = accuracy_score(y_encoded, y_pred)
        
        print(f"\n📊 RESULTADOS DE EVALUACIÓN")
        print(f"{'='*50}")
        print(f"✅ Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Reporte detallado
        class_names = label_encoder.classes_
        report = classification_report(y_encoded, y_pred, target_names=class_names)
        print(f"\n📋 Reporte detallado:")
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(y_encoded, y_pred)
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Matriz de Confusión - Precisión: {accuracy:.4f}')
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Real')
        plt.tight_layout()
        
        # Guardar reporte
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        plt.savefig('reports/evaluation_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Guardar reporte en archivo
        with open('reports/evaluation_report.txt', 'w') as f:
            f.write(f"Reporte de Evaluación del Modelo ASL\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Modelo: {model_path}\n")
            f.write(f"Codificador: {encoder_path}\n")
            f.write(f"Datos de prueba: {test_data_dir}\n\n")
            f.write(f"Precisión: {accuracy:.4f}\n\n")
            f.write(f"Número de muestras: {len(trainer.features)}\n")
            f.write(f"Número de clases: {len(class_names)}\n\n")
            f.write(f"Reporte de Clasificación:\n")
            f.write(report)
        
        print(f"\n💾 Reporte guardado en: reports/evaluation_report.txt")
        print(f"📊 Gráfico guardado en: reports/evaluation_confusion_matrix.png")
        
        return accuracy, report, cm
    
    else:
        print(f"⚠️  Directorio de datos de prueba no encontrado: {test_data_dir}")
        print("Para evaluar completamente el modelo, necesitas datos de prueba.")
        
        # Mostrar información básica del modelo
        print(f"\n📊 INFORMACIÓN DEL MODELO")
        print(f"{'='*50}")
        print(f"Tipo de modelo: {type(model).__name__}")
        print(f"Número de clases: {len(label_encoder.classes_)}")
        print(f"Clases: {', '.join(label_encoder.classes_)}")
        
        if hasattr(model, 'n_estimators'):
            print(f"Número de árboles: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"Profundidad máxima: {model.max_depth}")
        
        return None, None, None

def compare_models(model1_path, model2_path, encoder_path, test_data_dir):
    """
    Comparar dos modelos diferentes
    """
    print("⚔️  Comparando modelos...")
    
    # Evaluar primer modelo
    print(f"\n🔍 Evaluando Modelo 1: {model1_path}")
    acc1, _, _ = evaluate_model(model1_path, encoder_path, test_data_dir)
    
    # Evaluar segundo modelo
    print(f"\n🔍 Evaluando Modelo 2: {model2_path}")
    acc2, _, _ = evaluate_model(model2_path, encoder_path, test_data_dir)
    
    if acc1 is not None and acc2 is not None:
        print(f"\n🏆 COMPARACIÓN FINAL")
        print(f"{'='*50}")
        print(f"Modelo 1: {acc1:.4f} ({acc1*100:.2f}%)")
        print(f"Modelo 2: {acc2:.4f} ({acc2*100:.2f}%)")
        
        if acc1 > acc2:
            print(f"🥇 Ganador: Modelo 1 (diferencia: {(acc1-acc2)*100:.2f}%)")
        elif acc2 > acc1:
            print(f"🥇 Ganador: Modelo 2 (diferencia: {(acc2-acc1)*100:.2f}%)")
        else:
            print(f"🤝 Empate!")

def test_live_predictions(model_path="model/asl_model.joblib", 
                         encoder_path="model/label_encoder.joblib",
                         test_images_dir="test_images"):
    """
    Probar predicciones en tiempo real con imágenes específicas
    """
    print("🎯 Probando predicciones en vivo...")
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Directorio de imágenes de prueba no encontrado: {test_images_dir}")
        return
    
    # Crear instancia del entrenador para usar sus métodos
    trainer = ASLModelTrainer()
    trainer.load_model(model_path, encoder_path)
    
    # Obtener imágenes de prueba
    import glob
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    test_images.extend(glob.glob(os.path.join(test_images_dir, "*.png")))
    
    if not test_images:
        print(f"❌ No se encontraron imágenes en {test_images_dir}")
        return
    
    print(f"📸 Probando {len(test_images)} imágenes...")
    
    correct_predictions = 0
    total_predictions = 0
    
    for img_path in test_images:
        filename = os.path.basename(img_path)
        # Intentar extraer la etiqueta verdadera del nombre del archivo
        true_label = filename.split('_')[0] if '_' in filename else 'unknown'
        
        # Predecir
        pred_label, confidence = trainer.predict_from_image(img_path)
        
        if pred_label is not None:
            total_predictions += 1
            is_correct = pred_label.lower() == true_label.lower()
            if is_correct:
                correct_predictions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {filename}: {pred_label} (conf: {confidence:.3f})")
        else:
            print(f"⚠️  {filename}: No se pudo detectar mano")
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n📊 Precisión en imágenes de prueba: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 Correctas: {correct_predictions}/{total_predictions}")

def main():
    """
    Función principal para evaluación
    """
    print("🔍 EVALUADOR DE MODELOS ASL")
    print("=" * 50)
    
    print("Opciones:")
    print("1. Evaluar modelo actual")
    print("2. Comparar dos modelos")
    print("3. Probar predicciones en imágenes específicas")
    print("4. Mostrar información del modelo")
    
    while True:
        choice = input("\nSelecciona una opción (1-4): ").strip()
        
        if choice == '1':
            test_dir = input("Directorio de datos de prueba (default: data/asl_alphabet_train): ").strip() or "data/asl_alphabet_train"
            evaluate_model(test_data_dir=test_dir)
            break
            
        elif choice == '2':
            model1 = input("Ruta al primer modelo: ").strip()
            model2 = input("Ruta al segundo modelo: ").strip()
            encoder = input("Ruta al codificador (default: model/label_encoder.joblib): ").strip() or "model/label_encoder.joblib"
            test_dir = input("Directorio de datos de prueba (default: data/asl_alphabet_train): ").strip() or "data/asl_alphabet_train"
            compare_models(model1, model2, encoder, test_dir)
            break
            
        elif choice == '3':
            test_images_dir = input("Directorio de imágenes de prueba (default: test_images): ").strip() or "test_images"
            test_live_predictions(test_images_dir=test_images_dir)
            break
            
        elif choice == '4':
            evaluate_model()
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
