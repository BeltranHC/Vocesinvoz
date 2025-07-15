#!/usr/bin/env python3
"""
Script simple para entrenar el modelo ASL con tu dataset
Optimizado para tu estructura de datos específica
"""

import os
import sys

def main():
    """
    Función principal para entrenar el modelo
    """
    print("🚀 ENTRENADOR ASL - DATASET PERSONALIZADO")
    print("=" * 60)
    
    # Verificar dataset
    train_dir = "data/asl_alphabet_train"
    if not os.path.exists(train_dir):
        print(f"❌ Error: Directorio de entrenamiento no encontrado: {train_dir}")
        return
    
    # Contar clases
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"📊 Dataset detectado: {len(classes)} clases")
    print(f"🏷️  Clases: {sorted(classes)}")
    
    # Verificar algunas clases
    sample_class = classes[0]
    sample_dir = os.path.join(train_dir, sample_class)
    sample_count = len([f for f in os.listdir(sample_dir) if f.endswith('.jpg')])
    print(f"📈 Ejemplo clase '{sample_class}': {sample_count:,} imágenes")
    
    print(f"\n🎯 Opciones de entrenamiento:")
    print(f"1. Entrenamiento completo (Recomendado)")
    print(f"2. Entrenamiento rápido (subconjunto de 500 por clase)")
    print(f"3. Entrenamiento ultra-rápido (subconjunto de 100 por clase)")
    
    choice = input("\nSelecciona opción (1-3): ").strip()
    
    if choice == '1':
        print("\n🚀 Iniciando entrenamiento completo...")
        print("⏱️  Esto puede tomar varios minutos con 87,000 imágenes")
        run_full_training()
    elif choice == '2':
        print("\n🚀 Iniciando entrenamiento rápido...")
        print("⏱️  Esto tomará unos minutos con ~14,500 imágenes")
        run_subset_training(500)
    elif choice == '3':
        print("\n🚀 Iniciando entrenamiento ultra-rápido...")
        print("⏱️  Esto tomará 1-2 minutos con ~2,900 imágenes")
        run_subset_training(100)
    else:
        print("❌ Opción inválida")

def run_full_training():
    """
    Ejecutar entrenamiento completo
    """
    try:
        # Intentar script optimizado primero
        if os.path.exists("train_optimized.py"):
            print("🔧 Usando entrenador optimizado...")
            os.system("python train_optimized.py")
        else:
            print("🔧 Usando entrenador estándar...")
            os.system("python train_model.py")
            
        print("\n✅ Entrenamiento completado!")
        suggest_next_steps()
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")

def run_subset_training(subset_size):
    """
    Ejecutar entrenamiento con subconjunto
    """
    try:
        # Crear archivo temporal de configuración
        config_script = f"""
import sys
sys.path.append('.')
from train_optimized import OptimizedASLTrainer

trainer = OptimizedASLTrainer("data/asl_alphabet_train", use_subset=True, subset_size={subset_size})

try:
    if not trainer.load_processed_data():
        trainer.load_data_parallel()
    
    accuracy, cv_scores, report, cm = trainer.train_optimized_model()
    trainer.save_model()
    
    print(f"\\n🎉 Entrenamiento completado!")
    print(f"📊 Precisión: {{accuracy:.4f}}")
    print(f"📊 Precisión CV: {{cv_scores.mean():.4f}} (±{{cv_scores.std()*2:.4f}})")
    
except Exception as e:
    print(f"❌ Error: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        with open("temp_train.py", "w") as f:
            f.write(config_script)
        
        os.system("python temp_train.py")
        
        # Limpiar archivo temporal
        if os.path.exists("temp_train.py"):
            os.remove("temp_train.py")
            
        print("\n✅ Entrenamiento completado!")
        suggest_next_steps()
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")

def suggest_next_steps():
    """
    Sugerir próximos pasos
    """
    print(f"\n🔮 PRÓXIMOS PASOS:")
    print(f"1. Evaluar modelo: python evaluate_dataset.py")
    print(f"2. Probar aplicación: python app.py")
    print(f"3. Ver reportes en: reports/")
    
    # Verificar si el modelo fue creado
    if os.path.exists("model/asl_model.joblib"):
        print(f"\n✅ Modelo creado exitosamente!")
        
        choice = input("\n¿Quieres probar la aplicación ahora? (y/n): ").lower().strip()
        if choice == 'y':
            print("🚀 Iniciando aplicación...")
            os.system("python app.py")
    else:
        print(f"\n❌ No se pudo crear el modelo")

if __name__ == "__main__":
    main()
