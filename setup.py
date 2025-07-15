#!/usr/bin/env python3
"""
Script de inicio rápido para el proyecto SignSpeak
Guía paso a paso para configurar y ejecutar el proyecto
"""

import os
import sys
import subprocess

def print_header(text):
    """Imprimir encabezado decorado"""
    print("\n" + "="*60)
    print(f"🚀 {text}")
    print("="*60)

def print_step(step_num, description):
    """Imprimir paso numerado"""
    print(f"\n📋 Paso {step_num}: {description}")

def check_file_exists(filepath):
    """Verificar si un archivo existe"""
    return os.path.exists(filepath)

def run_command(command, description):
    """Ejecutar comando con descripción"""
    print(f"⚡ Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"Salida del error: {e.stderr}")
        return False

def main():
    """Función principal del setup"""
    print_header("CONFIGURACIÓN RÁPIDA DE SIGNSPEAK")
    
    print("""
🧠 SignSpeak: Traductor de ASL a Voz en Tiempo Real
===================================================

Este script te guiará através de la configuración completa del proyecto.
""")
    
    # Verificar Python
    print_step(1, "Verificando Python")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 7:
        print(f"✅ Python {python_version.major}.{python_version.minor} detectado")
    else:
        print(f"❌ Python 3.7+ requerido. Versión actual: {python_version.major}.{python_version.minor}")
        return
    
    # Verificar dependencias
    print_step(2, "Verificando dependencias")
    if check_file_exists("requirements.txt"):
        print("✅ requirements.txt encontrado")
        install_deps = input("¿Instalar dependencias? (y/n): ").lower().strip()
        if install_deps == 'y':
            if run_command("pip install -r requirements.txt", "Instalación de dependencias"):
                print("✅ Todas las dependencias instaladas")
            else:
                print("❌ Error instalando dependencias")
                return
    else:
        print("❌ requirements.txt no encontrado")
        return
    
    # Verificar modelo
    print_step(3, "Verificando modelo")
    model_exists = check_file_exists("model/asl_model.joblib")
    encoder_exists = check_file_exists("model/label_encoder.joblib")
    
    if model_exists and encoder_exists:
        print("✅ Modelo encontrado")
        print("🎯 Puedes ejecutar directamente la aplicación")
        
        choice = input("\n¿Qué quieres hacer?\n1. Ejecutar aplicación\n2. Entrenar nuevo modelo\n3. Evaluar modelo actual\nOpción (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 Iniciando aplicación...")
            run_command("python app.py", "Aplicación web")
        elif choice == '2':
            print("\n🔄 Entrenando nuevo modelo...")
            setup_training()
        elif choice == '3':
            print("\n📊 Evaluando modelo...")
            run_command("python evaluate_model.py", "Evaluación del modelo")
        else:
            print("❌ Opción inválida")
            
    else:
        print("❌ Modelo no encontrado")
        print("🔧 Necesitas entrenar un modelo primero")
        setup_training()

def setup_training():
    """Configurar proceso de entrenamiento"""
    print_step(4, "Configurando entrenamiento")
    
    # Verificar directorio de datos
    if not os.path.exists("data"):
        print("📁 Creando directorio de datos...")
        os.makedirs("data", exist_ok=True)
    
    # Verificar si hay datos
    data_dirs = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
    
    if not data_dirs:
        print("📸 No hay datos de entrenamiento")
        print("🎯 Opciones disponibles:")
        print("1. Recolectar datos con la cámara (Recomendado)")
        print("2. Usar tus propias imágenes")
        print("3. Descargar dataset de ejemplo")
        
        choice = input("\nSelecciona opción (1-3): ").strip()
        
        if choice == '1':
            print("\n📷 Iniciando recolección de datos...")
            print("💡 Tip: Recolecta mínimo 100 imágenes por clase")
            if run_command("python collect_data.py", "Recolección de datos"):
                print("✅ Datos recolectados exitosamente")
                train_model()
            else:
                print("❌ Error en recolección de datos")
        
        elif choice == '2':
            print("\n📁 Organiza tus imágenes así:")
            print("data/")
            print("├── A/")
            print("│   ├── img1.jpg")
            print("│   └── img2.jpg")
            print("├── B/")
            print("└── ...")
            print("\nPresiona Enter cuando tengas listas las imágenes...")
            input()
            train_model()
        
        elif choice == '3':
            print("📥 Funcionalidad de descarga de dataset no implementada aún")
            print("🔧 Usa las opciones 1 o 2 por ahora")
        
        else:
            print("❌ Opción inválida")
    
    else:
        print(f"✅ Datos encontrados en: {data_dirs}")
        train_model()

def train_model():
    """Entrenar el modelo"""
    print_step(5, "Entrenando modelo")
    
    if run_command("python train_model.py", "Entrenamiento del modelo"):
        print("🎉 Modelo entrenado exitosamente!")
        
        # Preguntar si quiere evaluar
        evaluate = input("\n¿Evaluar el modelo? (y/n): ").lower().strip()
        if evaluate == 'y':
            run_command("python evaluate_model.py", "Evaluación del modelo")
        
        # Preguntar si quiere ejecutar la app
        run_app = input("\n¿Ejecutar la aplicación? (y/n): ").lower().strip()
        if run_app == 'y':
            print("\n🚀 Iniciando aplicación...")
            run_command("python app.py", "Aplicación web")
    
    else:
        print("❌ Error en entrenamiento")

def show_help():
    """Mostrar ayuda"""
    print_header("AYUDA RÁPIDA")
    print("""
🔧 Scripts disponibles:
- collect_data.py    : Recolectar datos con la cámara
- train_model.py     : Entrenar modelo de ML
- evaluate_model.py  : Evaluar modelo entrenado
- app.py            : Aplicación web principal

📁 Estructura del proyecto:
- data/             : Datos de entrenamiento
- model/            : Modelos entrenados
- reports/          : Reportes de entrenamiento
- static/           : Archivos estáticos web
- templates/        : Templates HTML

🚀 Flujo típico:
1. Recolectar datos: python collect_data.py
2. Entrenar modelo: python train_model.py
3. Evaluar modelo: python evaluate_model.py
4. Ejecutar app: python app.py
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        main()
