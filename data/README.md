# Estructura de Datos para Entrenamiento ASL

Este directorio debe contener las imágenes organizadas por clases para entrenar el modelo.

## Estructura requerida:

```
data/
├── A/
│   ├── A_0001.jpg
│   ├── A_0002.jpg
│   └── ...
├── B/
│   ├── B_0001.jpg
│   ├── B_0002.jpg
│   └── ...
├── C/
├── D/
├── ...
├── Z/
├── space/
├── del/
└── nothing/
```

## Cómo obtener los datos:

### Opción 1: Capturar con la cámara (Recomendado)
```bash
python collect_data.py
```

### Opción 2: Usar tus propias imágenes
- Crea carpetas para cada clase (A, B, C, ..., Z, space, del, nothing)
- Coloca las imágenes correspondientes en cada carpeta
- Asegúrate de que las imágenes sean claras y muestren bien las manos

## Recomendaciones:

- **Mínimo 100 imágenes por clase** para buen rendimiento
- **Variedad**: Diferentes iluminaciones, fondos, posiciones de mano
- **Calidad**: Imágenes claras donde se vea bien la mano
- **Formato**: JPG, PNG, o BMP
- **Resolución**: No necesariamente alta, MediaPipe funciona bien con resoluciones moderadas

## Después de tener los datos:

1. Ejecuta `python train_model.py` para entrenar el modelo
2. Los archivos `asl_model.joblib` y `label_encoder.joblib` se guardarán en `model/`
3. Ejecuta `python app.py` para usar la aplicación web
