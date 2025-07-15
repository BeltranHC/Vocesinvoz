# 🧠 SignSpeak: Traductor de ASL a Voz en Tiempo Real

Una aplicación web en tiempo real que utiliza tu cámara web y un modelo de aprendizaje automático para reconocer el **alfabeto del Lenguaje de Señas Americano (ASL)** y convertirlo en **texto y voz**. Construida con **Flask**, **OpenCV**, **MediaPipe** y **gTTS**, con una interfaz **HTML/CSS/JS** limpia y responsiva.

---

## ✨ Características

- 📷 Detección de manos en tiempo real usando MediaPipe
- 🔠 Reconoce señas estáticas de ASL para 26 letras + "espacio", "nada" y "del"
- 🧠 Modelo de ML (entrenado con 63 características de puntos de MediaPipe)
- ✍️ Construye frases completas a partir de las señas en secuencia
- 🗣️ Convierte la frase construida en voz usando Google Text-to-Speech
- 🎨 Interfaz moderna y responsiva (creada desde cero con HTML y CSS)

---

## 📂 Estructura de Carpetas

```
project/
├── app.py                    # Backend principal en Flask
├── requirements.txt          # Dependencias de Python
├── model/
│   ├── asl_model.joblib      # Modelo de ML (descarga automática)
│   └── label_encoder.joblib  # Codificador de etiquetas (descarga automática)
├── static/
│   └── audio/                # Contiene archivos .mp3 generados
├── templates/
│   └── index.html            # Página web frontend
```

---

## 🧠 ¿Cómo Funciona?

1. OpenCV captura video desde la cámara web.
2. MediaPipe extrae 21 puntos de referencia de la mano (x, y, z = 63 características).
3. El modelo RandomForestClassifier predice la letra ASL.
4. Una letra solo se agrega a la frase si:
   - Es estable en varios fotogramas, y
   - La confianza del modelo es **85% o más**.
5. El usuario puede presionar:
   - ✅ **Hablar** → convierte la frase en un archivo de audio
   - ❌ **Limpiar** → reinicia la frase actual
6. El audio se guarda y se reproduce en el navegador.

---

## 🚀 Comenzando

### 1. Clona el Repositorio

```bash
git clone https://github.com/your-username/signspeak.git
cd signspeak
```

### 2. Instala los Requisitos de Python

```bash
pip install -r requirements.txt
```

### 3. Ejecuta la Aplicación

```bash
python app.py
```

Visita [http://localhost:5000](http://localhost:5000) en tu navegador.

---

## 🧠 Detalles del Modelo

El modelo utilizado es un clasificador tradicional entrenado con 63 características extraídas de los 21 puntos de referencia de la mano de MediaPipe (x, y, z por punto). El modelo está entrenado con más de 60,000 ejemplos que abarcan:

- 26 letras (A-Z)
- Espacio
- Nada (mano en reposo / fondo)
- Del (eliminar carácter)

### 🎯 Precisión Alcanzada

- Precisión general: **98.47%**
- Macro Promedio F1-Score: **0.98**

| Ejemplo de Informe de Clasificación |
|-------------------------------------|
| ![asl_model_report](screenshots/model_report.png) |

El modelo se guarda como `asl_model.joblib` junto con `label_encoder.joblib`. Ambos se descargan automáticamente vía `gdown` en la primera ejecución de la app.

---

## 📦 Archivos del Modelo

El modelo (`asl_model.joblib`) y el codificador (`label_encoder.joblib`) se descargarán automáticamente desde Google Drive en la primera ejecución.

> Esto evita que el repositorio se llene de archivos binarios grandes.

Puedes reemplazar manualmente el modelo dentro de la carpeta `model/` más adelante.

---

## 📦 Requisitos

- `Flask`
- `opencv-python`
- `mediapipe`
- `numpy`
- `joblib`
- `gdown`
- `gTTS`

Instala con:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Mejoras Futuras

- [ ] Agregar predicción a nivel de palabra o frase usando 3D CNNs
- [ ] Soportar señas dinámicas con LSTM o Transformers
- [ ] Integrar modelo de lenguaje (T5, GPT) para limpiar frases
- [ ] Desplegar como demo pública en Render o Hugging Face Spaces

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Eres libre de usarlo, compartirlo y modificarlo con atribución.

---

## 👤 Autor

**Somya Soneji**  
Construido con ❤️ para unir la comunicación entre el mundo Sordo y oyente.

> ¡Siéntete libre de hacer fork, dar estrella ⭐ y compartirlo con otros!
