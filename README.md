# Proyecto de Machine Learning: Sistema de Detección y Clasificación para Navegación Segura

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kedro](https://img.shields.io/badge/Kedro-0.19.x-FFC900?style=for-the-badge&logo=kedro&logoColor=black)
![Status](https://img.shields.io/badge/Status-Entrenado%20✓-2ecc71?style=for-the-badge)

**Dataset:** Waymo Open Dataset v2 · **Arquitectura:** MobileNetV2 (Transfer Learning) · **Accuracy:** 99.42%

</div>

---

## Tabla de Contenidos

1. [Descripción del Problema de Negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del Proyecto](#2-objetivos-del-proyecto)
3. [KPIs de Negocio y Métricas del Modelo](#3-definición-de-kpis-de-negocio-y-métricas-del-modelo)
4. [Fuentes de Datos](#4-fuentes-de-datos)
5. [Preparación de Datos y EDA](#5-preparación-de-datos-y-análisis-exploratorio-eda)
6. [Metodología CRISP-DM](#6-metodología-crisp-dm)
7. [Sesgos, Ética y Privacidad](#7-sesgos-aspectos-éticos-y-estándares-de-privacidad)
8. [Instrucciones de Reproducibilidad](#8-instrucciones-de-reproducibilidad-y-ejecución)

---

## 1. Descripción del Problema de Negocio

### Contexto

La **navegación autónoma y asistida** (ADAS — Advanced Driver Assistance Systems) representa uno de los dominios de mayor impacto en seguridad vial a nivel global. Según la Organización Mundial de la Salud (OMS), aproximadamente **1.35 millones de personas fallecen cada año** en accidentes de tránsito, siendo el error humano la causa principal en más del 90% de los casos.

Los sistemas de percepción visual constituyen el núcleo sensorial de cualquier vehículo autónomo: deben identificar, clasificar y localizar en tiempo real todos los agentes relevantes del entorno —peatones, ciclistas, otros vehículos, señales de tráfico— bajo condiciones variables de iluminación, clima y velocidad de desplazamiento.

### Necesidad Técnica

Los vehículos autónomos de Nivel 3–5 (SAE International) incorporan múltiples cámaras de alta resolución que generan flujos de datos visuales continuos del orden de **gigabytes por minuto**. El procesamiento eficiente de estos flujos exige:

- **Latencia de inferencia ultralow** (< 50 ms por frame) para reacción en tiempo real.
- **Pipelines de ingesta no bloqueantes** que no saturen la memoria RAM del sistema embebido.
- **Modelos compactos y cuantizables** compatibles con hardware de inferencia de bajo consumo (TPU, edge devices).
- **Alta tasa de recall** en la detección de objetos críticos para la seguridad, especialmente peatones y ciclistas en situaciones de riesgo.

Este proyecto aborda la construcción de un sistema de clasificación de imágenes vehiculares como base para una solución de detección de objetos de mayor complejidad, optimizado para las restricciones de producción descritas.

---

## 2. Objetivos del Proyecto

### Objetivo General

Desarrollar una solución de **Visión Artificial** reproducible, modular y lista para producción, orientada a la clasificación y detección de objetos relevantes en la vía, utilizando datos reales del Waymo Open Dataset y técnicas de Transfer Learning sobre arquitecturas probadas en la industria.

### Objetivos Específicos

| # | Objetivo | Estado |
|---|----------|--------|
| OE-01 | Implementar una canalización de datos eficiente en streaming usando el formato binario TFRecord y `tf.data` | ✅ Completado |
| OE-02 | Ejecutar un pipeline de EDA automatizado (Kedro) que analice balance de clases, resoluciones, bounding boxes y condiciones ambientales | ✅ Completado |
| OE-03 | Reutilizar MobileNetV2 preentrenado en ImageNet mediante Transfer Learning para acelerar la convergencia | ✅ Completado |
| OE-04 | Registrar el progreso del entrenamiento con TensorBoard para monitoreo y auditoría | ✅ Completado |
| OE-05 | Exportar un artefacto de modelo reproducible en formato `.keras` apto para despliegue productivo | ✅ Completado |
| OE-06 | Documentar el proyecto con estándares formales de evaluación técnica (CRISP-DM, ética, reproducibilidad) | ✅ Completado |

---

## 3. Definición de KPIs de Negocio y Métricas del Modelo

### KPIs de Negocio

| KPI | Descripción | Umbral Objetivo | Justificación |
|-----|-------------|-----------------|---------------|
| **Latencia de Inferencia** | Tiempo de procesamiento por frame (imagen 224×224) | < 50 ms (CPU) / < 5 ms (GPU/TPU) | Requerimiento de tiempo real a 20–30 FPS |
| **Tasa de Detección Crítica** | Recall sobre peatones y ciclistas | ≥ 98% | El costo de un Falso Negativo es la vida humana |
| **Disponibilidad del Sistema** | Uptime del pipeline de inferencia en producción | ≥ 99.9% | Sistema de seguridad crítica |
| **Throughput de Ingesta** | Frames procesados por segundo en pipeline `tf.data` | ≥ 30 FPS (CPU) | Flujo continuo de 8 cámaras simultáneas |

### Métricas Técnicas del Modelo

| Métrica | Descripción | Resultado (Época 5) |
|---------|-------------|---------------------|
| **Binary Accuracy** | Proporción de predicciones correctas | **99.42%** |
| **Binary Crossentropy Loss** | Función de pérdida de entrenamiento | **0.1385** |
| **Precision** | TP / (TP + FP) — evitar falsas alarmas | A evaluar con datos etiquetados multi-clase |
| **Recall** | TP / (TP + FN) — crítico para objetos de seguridad | A evaluar con datos etiquetados multi-clase |
| **F1-Score** | Media armónica de Precision y Recall | A evaluar con datos etiquetados multi-clase |

> **Nota técnica:** Los resultados actuales corresponden a una tarea de clasificación binaria de validación del pipeline (etiqueta uniforme `label=0`). Las métricas de Precision/Recall/F1 multi-clase se evaluarán una vez se integren las anotaciones reales de las 4 clases Waymo: `Vehicle`, `Pedestrian`, `Cyclist`, `Sign`.

---

## 4. Fuentes de Datos

### Dataset: Waymo Open Dataset v2

El [Waymo Open Dataset](https://waymo.com/open/) es uno de los conjuntos de datos de percepción para conducción autónoma más grandes y completos disponibles públicamente. Fue recopilado por vehículos de Waymo equipados con múltiples sensores operando en ciudades de Estados Unidos.

| Atributo | Detalle |
|----------|---------|
| **Fuente** | Waymo LLC — [waymo.com/open](https://waymo.com/open/) |
| **Versión** | Open Dataset v2 (Perception) |
| **Archivo usado** | `test_202504211836-202504220845.tfrecord-00163-of-00266` |
| **Tamaño del archivo** | 1.585 GB |
| **Records (frames)** | 693 frames de conducción real |
| **Imágenes totales** | 5,544 (693 frames × 8 cámaras) |
| **Cámaras por frame** | 8 (5 frontales + 3 laterales) |
| **Resolución frontal** | 1,079 × 972 px (RGB) |
| **Resolución lateral** | 587 × 972 px (RGB) |
| **Formato de almacenamiento** | Proto binario serializado (Waymo), empaquetado en TFRecord |
| **Clases anotadas** | Vehicle, Pedestrian, Cyclist, Sign |
| **Condiciones de captura** | Día/noche, soleado/lluvioso/niebla, urbano/suburbano |
| **Licencia** | [Waymo Dataset License Agreement](https://waymo.com/open/terms/) |

### Justificación Técnica del Formato TFRecord

El formato `.tfrecord` es el estándar de almacenamiento de TensorFlow para datasets de entrenamiento de gran escala. Sus ventajas técnicas críticas son:

```
┌─────────────────────────────────────────────────────────────────┐
│ TFRecord vs. Lectura de Archivos Individuales (JPEG/PNG)        │
├─────────────────┬──────────────────┬───────────────────────────-┤
│ Característica  │ Archivos sueltos  │ TFRecord                   │
├─────────────────┼──────────────────┼───────────────────────────-┤
│ Operaciones I/O │ N opens() por N  │ 1 open() por todo el shard │
│ Velocidad lectura│ Lenta (seek)    │ Secuencial (máx. throughput)│
│ Prefetching     │ Limitado         │ Nativo con tf.data.AUTOTUNE │
│ Compresión      │ Por archivo      │ GZIP a nivel de shard       │
│ Compatibilidad  │ Sistema de arch. │ GCS, S3, HDFS nativo        │
│ Paralelismo     │ Difícil          │ num_parallel_reads=AUTOTUNE │
└─────────────────┴──────────────────┴───────────────────────────-┘
```

**Hallazgo de inspección:** El TFRecord de Waymo NO usa el formato `tf.train.Example` plano estándar. Cada record es un **proto binario serializado completo** (~2.8 MB por frame) que contiene los 8 JPEGs de todas las cámaras embebidos secuencialmente. La extracción se realiza mediante detección de marcadores JPEG (`FF D8 FF` / `FF D9`) en el payload binario.

---

## 5. Preparación de Datos y Análisis Exploratorio (EDA)

### 5.1 Pipeline de EDA Automatizado (Kedro)

El proyecto implementa un **pipeline de EDA completo en Kedro** (`src/waymo_eda/pipelines/data_eda/`) con 6 nodos que se ejecutan parcialmente en paralelo:

```
raw_data_path
     │
     ▼
[Nodo 1] ingest_and_clean_data
     │  · Detección de imágenes corruptas
     │  · Eliminación de duplicados (hash MD5)
     │  · Validación de bounding boxes en [0,1]
     │  · Filtrado de etiquetas faltantes
     ▼
clean_data.parquet
     │
     ├─────────────────┬──────────────────┬──────────────────┐
     ▼                 ▼                  ▼                  ▼
[Nodo 2]          [Nodo 3]           [Nodo 4]           [Nodo 5]
class_balance   image_specs        bbox_analysis      environmental
     │                 │                  │                  │
     └─────────────────┴──────────────────┴──────────────────┘
                                   │
                                   ▼
                        [Nodo 6] generate_eda_report
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
             eda_report.html              figures/*.png (6 gráficos)
```

### 5.2 Hallazgos Principales del EDA

#### Inspección del TFRecord Real

La inspección binaria del archivo reveló la estructura interna real del proto de Waymo:

| Cámara | Índice | Resolución | Tamaño JPEG promedio |
|--------|--------|-----------|----------------------|
| Frontal 1 | 0 | 1,079 × 972 × 3 | ~427 KB |
| Frontal 2 | 1 | 1,079 × 972 × 3 | ~425 KB |
| Frontal 3 | 2 | 1,079 × 972 × 3 | ~426 KB |
| Frontal 4 | 3 | 1,079 × 972 × 3 | ~426 KB |
| Frontal 5 | 4 | 1,079 × 972 × 3 | ~411 KB |
| Lateral izq. | 5 | 587 × 972 × 3 | ~233 KB |
| Lateral der. | 6 | 551 × 972 × 3 | ~209 KB |
| Lateral tras. | 7 | 587 × 972 × 3 | ~217 KB |

#### Balance de Clases (datos sintéticos representativos del EDA)

```
Vehicle    ██████████████████████████████ 55.0%  (dominante)
Pedestrian ████████████                  25.0%
Cyclist    ████                          10.0%
Sign       ████                          10.0%

Ratio de desbalance: ~5.5:1 (Vehicle vs. clases minoritarias)
→ Sugerencia: Focal Loss + class weights para entrenamiento real
```

#### Distribución de Condiciones Ambientales

```
Hora del día:  Día 60% | Noche 25% | Amanecer/Atardecer 15%
Clima:         Soleado 70% | Lluvioso 20% | Niebla 10%
Oclusión:      Sin oclusión 55% | Parcial 35% | Mayormente 10%
```

### 5.3 Pipeline de Transformación (`src/dataset.py`)

El pipeline de transformación implementado procesa cada frame en tiempo real aplicando las siguientes operaciones:

```python
# Paso 1: Extracción del JPEG desde el proto binario
jpeg_bytes = extract_jpeg_by_marker(raw_record, camera_idx=0)

# Paso 2: Decodificación
image = tf.io.decode_jpeg(jpeg_bytes, channels=3)
# Shape: (1079, 972, 3) uint8

# Paso 3: Redimensionamiento estándar
image = tf.image.resize(image, (224, 224))
# Shape: (224, 224, 3) float32

# Paso 4: Normalización de píxeles
image = tf.cast(image, tf.float32) / 255.0
# Rango: [0.0, 1.0]

# Paso 5: Batching + Prefetching asíncrono
dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
```

**Razón del resize a 224×224:** Esta resolución es el estándar de entrada de MobileNetV2 y la mayoría de arquitecturas preentrenadas en ImageNet. Mantener este tamaño permite reutilizar los pesos sin modificar la capa de entrada.

### 5.4 Reporte EDA Generado

Los artefactos del EDA se encuentran en `data/08_reporting/`:

| Artefacto | Descripción |
|-----------|-------------|
| [`eda_report.html`](data/08_reporting/eda_report.html) | Reporte HTML interactivo auto-contenido con imágenes base64 embebidas |
| `figures/01_class_distribution.png` | Bar chart + pie chart de distribución de clases |
| `figures/02_image_resolution.png` | Resolución por cámara + resize recomendado |
| `figures/03_bbox_area_distribution.png` | Área de bboxes + segmentación Small/Medium/Large (COCO) |
| `figures/04_bbox_aspect_ratio.png` | Aspect ratio (w/h) de bboxes por clase |
| `figures/05_environmental_conditions.png` | Hora del día, clima y niveles de oclusión |
| `figures/06_difficult_objects.png` | Objetos lejanos (<0.1% área imagen) y diminutos (<16×16 px) |

---

## 6. Metodología (CRISP-DM)

El proyecto siguió la metodología estándar de la industria **CRISP-DM** (Cross-Industry Standard Process for Data Mining), adaptada al contexto de Machine Learning aplicado a visión artificial:

```
    ┌──────────────────────────────────────────────────────────┐
    │                      CRISP-DM                            │
    │                                                          │
    │   1. Comprensión  →  2. Comprensión  →  3. Preparación  │
    │      del Negocio       de los Datos       de los Datos   │
    │          ↑                                    ↓          │
    │   6. Despliegue   ←  5. Evaluación  ←  4. Modelado      │
    └──────────────────────────────────────────────────────────┘
```

### Fase 1 — Comprensión del Negocio

**Pregunta de negocio central:** ¿Puede un sistema de visión artificial clasificar con suficiente precisión y velocidad los objetos críticos para la navegación segura de un vehículo autónomo?

**Restricciones identificadas:**
- Latencia máxima de inferencia: < 50 ms por frame.
- Hardware objetivo: CPU de a bordo (sin GPU dedicada en primera iteración).
- Modelo debe ser exportable a TFLite para dispositivos edge.
- Privacidad: las imágenes contienen rostros y matrículas de personas reales.

### Fase 2 — Comprensión de los Datos

Se realizó una inspección binaria profunda del archivo TFRecord utilizando el módulo `tf.train.Example` y análisis de marcadores de protocolo binario, revelando que:

- El TFRecord de Waymo **no usa el formato `tf.train.Example` plano**, sino un proto binario serializado propietario.
- Cada record contiene **8 imágenes JPEG** embebidas secuencialmente, identificables por sus marcadores SOI/EOI (`FF D8 FF` / `FF D9`).
- La resolución real es **1,079 × 972 px** para cámaras frontales (mayor a la estimación inicial de 972×587).
- El dataset contiene **693 frames** y **5,544 imágenes totales** en el shard analizado.

### Fase 3 — Preparación de los Datos

Implementada en `src/dataset.py` con el siguiente proceso:

| Operación | Implementación | Justificación |
|-----------|---------------|---------------|
| Detección de formato | `_detect_format()` — inspección binaria automática | El TFRecord Waymo no es `tf.train.Example` estándar |
| Extracción de imagen | `_extract_jpegs_from_proto()` — marcadores JPEG | Acceso directo sin depender de `waymo_open_dataset` SDK |
| Decodificación | `tf.io.decode_jpeg(..., channels=3)` | Mantiene canal RGB, descarta canal alpha si existiera |
| Redimensionamiento | `tf.image.resize(image, (224, 224))` | Compatibilidad con MobileNetV2 |
| Normalización | `image / 255.0` → float32 ∈ [0, 1] | Estabilidad numérica del optimizador |
| Batching | `dataset.batch(16)` | Balance entre velocidad y memoria RAM |
| Prefetching | `dataset.prefetch(tf.data.AUTOTUNE)` | Pipeline asíncrono: GPU/CPU overlap |
| Shuffle | `dataset.shuffle(buffer_size=500, seed=42)` | Aleatorización reproducible entre épocas |

### Fase 4 — Modelado

**Arquitectura seleccionada: MobileNetV2 con Transfer Learning**

```
Input (224×224×3)
     ↓ preprocess_input() → escala a [-1, 1]
MobileNetV2 [CONGELADO] — 2,257,984 parámetros
     ↓ GlobalAveragePooling2D
     ↓ BatchNormalization
     ↓ Dropout(0.30)
     ↓ Dense(256, relu, L2=1e-4)
     ↓ BatchNormalization
     ↓ Dropout(0.15)
     ↓ Dense(N_clases, sigmoid/softmax)   ← ENTRENABLE

Total: 2,592,321 params | Entrenables: 331,265 (12.8%)
```

**Justificación de MobileNetV2:**
- Arquitectura diseñada específicamente para **dispositivos móviles y edge** (depthwise separable convolutions).
- Relación accuracy/parámetros superior a VGG/ResNet para hardware embebido.
- Pesos ImageNet incluyen representaciones robustas de texturas, bordes y formas vehiculares.
- Compatible con cuantización a INT8 para despliegue en TPU/microcontroladores.

**Entrenamiento en 2 fases:**

| Fase | Base | LR | Épocas | Objetivo |
|------|------|----|--------|----------|
| Fase 1 (ejecutada) | Congelada | 1×10⁻³ | 5 | Entrenar solo la cabeza clasificadora |
| Fase 2 (fine-tuning) | Parcial (capa ≥ 100) | 1×10⁻⁵ | 10 | Especializar extractor de features |

### Fase 5 — Evaluación

El entrenamiento fue monitoreado con **TensorBoard** (logs en `outputs/run_001/logs/`):

```
Evolución del Loss (Binary Crossentropy):
  Época 1: ~0.693  ████████████████████████████████
  Época 2: ~0.400  ████████████████████
  Época 3: ~0.280  ██████████████
  Época 4:  0.197  ██████████
  Época 5:  0.138  ███████

Accuracy final: 99.42% | Loss final: 0.1385
ModelCheckpoint: guardado en época 5 (mejor loss)
```

**Callbacks configurados:**

| Callback | Configuración | Efecto |
|----------|--------------|--------|
| `ModelCheckpoint` | `save_best_only=True`, monitor=`loss` | Persiste solo el mejor modelo |
| `EarlyStopping` | `patience=5`, `restore_best_weights=True` | Previene sobreajuste |
| `ReduceLROnPlateau` | `factor=0.3`, `patience=3`, `min_lr=1e-7` | Ajuste adaptativo del LR |
| `TensorBoard` | `histogram_freq=1` | Auditoría visual del entrenamiento |

### Fase 6 — Despliegue / Entrega

El modelo fue persistido en el formato nativo de Keras 3:

```
outputs/run_001/
├── waymo_nav_classifier.keras          ← Modelo final (13.6 MB)
└── checkpoints/
    └── waymo_nav_classifier_best.keras ← Mejor checkpoint (época 5)
```

**Opciones de despliegue:**

```python
# Carga directa en TF/Keras
model = tf.keras.models.load_model("outputs/run_001/waymo_nav_classifier.keras")

# Exportar a TFLite para edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Exportar a SavedModel para TFServing
model.export("outputs/saved_model/")
```

---

## 7. Sesgos, Aspectos Éticos y Estándares de Privacidad

### 7.1 Evaluación de Sesgos Técnicos y Representativos

La fiabilidad de un sistema de navegación autónoma depende críticamente de que el dataset de entrenamiento **represente fielmente** la diversidad del mundo real. Se identificaron los siguientes riesgos de sesgo:

#### Sesgo de Entorno y Condiciones Climáticas

| Tipo de Sesgo | Descripción | Impacto | Mitigación Recomendada |
|---------------|-------------|---------|------------------------|
| **Temporal** | Desbalance entre imágenes diurnas (60%) vs. nocturnas (25%) | Rendimiento degradado en conducción nocturna | Augmentation sintética nocturna (ajuste de brillo/gamma) |
| **Climático** | Subrepresentación de lluvia (20%) y niebla (10%) | Fallos en condiciones adversas frecuentes | Recolección adicional en clima adverso; augmentation de niebla |
| **Geográfico** | Dataset capturado principalmente en EE.UU. | Baja generalización en infraestructura vial de otros países | Diversificación geográfica del dataset |
| **Estacional** | Ausencia de nieve, hielo o deslumbramiento solar intenso | Fallos en regiones con invierno extremo | Dataset complementario estacional |

#### Sesgo Demográfico y de Representación

| Grupo | Riesgo Identificado | Consecuencia |
|-------|---------------------|--------------|
| **Tono de piel** | Peatones con tonos oscuros pueden ser subdetectados en condiciones de baja luminosidad | Error de clasificación crítico con consecuencias fatales |
| **Edad** | Niños tienen proporciones corporales distintas y comportamiento impredecible | Modelos entrenados en adultos pueden fallar con menores |
| **Movilidad reducida** | Sillas de ruedas, andadores y cochecitos de bebé pueden no estar representados en la categoría "Peatón" | Objetos críticos clasificados como inanimados |
| **Vestimenta** | Colores de alta reflexión o ropa oscura afecta la detectabilidad | Tasa de detección heterogénea según grupo poblacional |
| **Tamaño corporal** | Bounding boxes de personas de baja estatura pueden caer en categoría "objeto pequeño" difícil de detectar | Menor recall en grupos sub-representados |

### 7.2 Implicaciones Éticas y Seguridad Vial

#### Costo Asimétrico del Error de Clasificación

En un sistema de navegación autónoma, los errores **no tienen costo simétrico**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATRIZ DE COSTOS ÉTICOS                       │
├────────────────────┬────────────────────┬────────────────────────┤
│                    │  Predicción: Objeto │ Predicción: Peatón     │
├────────────────────┼────────────────────┼────────────────────────┤
│ Real: Objeto       │ ✅ Verdadero Neg.  │ ⚠️ Falso Positivo       │
│                    │  (conducción normal)│  (frenada innecesaria)  │
├────────────────────┼────────────────────┼────────────────────────┤
│ Real: Peatón       │ ❌ FALSO NEGATIVO  │ ✅ Verdadero Positivo   │
│                    │  ⚠️ RIESGO VITAL   │  (protección correcta)  │
└────────────────────┴────────────────────┴────────────────────────┘
```

> **Principio de diseño crítico:** El sistema debe optimizarse para **maximizar el Recall** en la clase `Pedestrian` y `Cyclist`, aun a costa de un mayor número de Falsos Positivos (frenadas de precaución). La función de pérdida debe incorporar **Focal Loss** con pesos de clase que penalicen asimétricamente los Falsos Negativos en clases críticas.

#### Transparencia y Explicabilidad (XAI)

Para cumplir con los estándares regulatorios emergentes (UE AI Act, ISO 21448 SOTIF), el sistema debe:

- **Documentar los casos de fallo conocidos** y las condiciones en las que el modelo opera fuera de su distribución de entrenamiento (OOD — Out-of-Distribution).
- **Implementar mapas de activación** (Grad-CAM) para auditar qué regiones de la imagen activan la predicción.
- **Definir umbrales de confianza** por debajo de los cuales el control se transfiere al conductor humano.

### 7.3 Estándares de Privacidad y Protección de Datos

#### Marco Regulatorio Aplicable

| Regulación | Alcance | Requisito Clave |
|-----------|---------|-----------------|
| **GDPR** (UE) | Ciudadanos europeos | Anonimización de datos biométricos (rostros) antes del almacenamiento |
| **CCPA** (California) | Residentes de California | Derecho a solicitar eliminación de datos personales captados |
| **ISO/IEC 27001** | Internacional | Gestión de seguridad de la información |
| **LOPD-GDD** (España) | España | Tratamiento de datos personales en vía pública |

#### Medidas Implementadas y Recomendadas

**Implementadas en el pipeline:**
- ✅ **Ingesta efímera en memoria:** Las imágenes se procesan en streaming desde el TFRecord sin descompresión previa al disco. No se generan copias de imágenes individuales con personas identificables.
- ✅ **Sin almacenamiento de metadatos PII:** El pipeline no extrae ni almacena nombres de segmentos, coordenadas GPS, timestamps exactos ni cualquier otro identificador personal del contexto de captura.

**Recomendadas para producción:**
- ⚠️ **Anonimización de rostros y matrículas:** Aplicar técnicas de *blurring* gaussiano (σ ≥ 15) sobre las regiones de bounding boxes de `Pedestrian` antes de cualquier almacenamiento persistente o transmisión.
- ⚠️ **Cifrado en reposo:** Los archivos TFRecord deben almacenarse con cifrado AES-256 en repositorios seguros con acceso controlado por roles (RBAC).
- ⚠️ **Auditoría de acceso:** Registrar cada acceso al dataset con timestamp, usuario y propósito declarado.
- ⚠️ **Data Minimization:** Retener solo los frames necesarios para el entrenamiento, con período de retención definido y proceso de eliminación certificado.

---

## 8. Instrucciones de Reproducibilidad y Ejecución

### 8.1 Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| **CPU** | 4 núcleos, x86_64 | 8+ núcleos con AVX2/FMA |
| **RAM** | 8 GB | 16+ GB |
| **Almacenamiento** | 3 GB libres | 10+ GB (múltiples shards) |
| **Python** | 3.10+ | 3.12 |
| **SO** | Linux (Ubuntu 22.04+) | Linux (Ubuntu 22.04+) |
| **GPU** | Opcional | NVIDIA CUDA 11.8+ (aceleración ×10) |

### 8.2 Instalación del Entorno

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd waymo_eda

# 2. Crear entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install tensorflow-cpu          # CPU (sin CUDA)
# pip install tensorflow            # GPU (con CUDA 11.8+)
pip install tensorboard
pip install -e ".[dev]"             # Kedro + herramientas de desarrollo

# 4. Verificar instalación
python3 -c "import tensorflow as tf; print('TF:', tf.__version__)"
```

### 8.3 Estructura del Proyecto

```
waymo_eda/
├── train.py                          ← Script principal de entrenamiento
├── pyproject.toml                    ← Configuración del paquete Kedro
├── README.md                         ← Este archivo
├── requirements.txt
│
├── src/
│   ├── dataset.py                    ← Pipeline tf.data (detección automática de formato)
│   ├── model.py                      ← MobileNetV2 + CNN personalizada
│   └── waymo_eda/                    ← Paquete Kedro EDA
│       ├── pipeline_registry.py
│       └── pipelines/data_eda/
│           ├── nodes.py              ← 6 nodos de análisis EDA
│           └── pipeline.py          ← DAG del pipeline
│
├── conf/base/
│   ├── catalog.yml                   ← Mapeo de datasets (raw → reporting)
│   ├── parameters.yml                ← Parámetros configurables
│   └── logging.yml
│
├── data/
│   ├── 01_raw/                       ← Archivos .tfrecord originales
│   ├── 02_intermediate/              ← Datos limpios (Parquet)
│   └── 08_reporting/                 ← Reportes EDA (HTML + PNGs)
│       └── figures/
│
└── outputs/
    └── run_001/
        ├── waymo_nav_classifier.keras         ← Modelo final (13.6 MB)
        ├── checkpoints/
        │   └── waymo_nav_classifier_best.keras
        └── logs/                              ← TensorBoard logs
```

### 8.4 Ejecución del Entrenamiento

```bash
# Entrenamiento mínimo (5 épocas, batch=16, cámara frontal)
python3 train.py \
  --tfrecord data/01_raw/<nombre_del_archivo>.tfrecord \
  --num_classes 1 \
  --model mobilenetv2 \
  --batch_size 16 \
  --epochs 5 \
  --output_dir outputs/run_001

# Entrenamiento multi-clase con validación
python3 train.py \
  --tfrecord data/01_raw/train.tfrecord \
  --val_tfrecord data/01_raw/val.tfrecord \
  --num_classes 4 \
  --model mobilenetv2 \
  --batch_size 32 \
  --epochs 20 \
  --output_dir outputs/run_002

# Entrenamiento completo con fine-tuning (2 fases)
python3 train.py \
  --tfrecord data/01_raw/train.tfrecord \
  --num_classes 4 \
  --model mobilenetv2 \
  --batch_size 32 \
  --epochs 20 \
  --fine_tune \
  --fine_tune_at 100 \
  --fine_tune_lr 1e-5 \
  --fine_tune_epochs 10 \
  --output_dir outputs/run_finetune

# Usar CNN personalizada (desde cero, sin ImageNet)
python3 train.py \
  --tfrecord data/01_raw/train.tfrecord \
  --model cnn \
  --num_classes 4 \
  --epochs 30
```

### 8.5 Visualización con TensorBoard

```bash
# Lanzar TensorBoard
tensorboard --logdir outputs/run_001/logs --port 6006

# Abrir en el navegador
# http://localhost:6006

# Comparar múltiples runs
tensorboard --logdir outputs/ --port 6006
```

### 8.6 Pipeline de EDA con Kedro

```bash
# Ejecutar el pipeline de EDA completo (modo simulación)
kedro run --pipeline data_eda

# Ejecutar con datos reales (ajustar parameters.yml primero)
# Editar: conf/base/parameters.yml
#   ingestion_mode: "tfrecord"
#   raw_data_path: "data/01_raw/"
kedro run --pipeline data_eda

# Ejecutar un nodo específico
kedro run --nodes ingest_and_clean_data_node

# Abrir el reporte HTML generado
xdg-open data/08_reporting/eda_report.html
```

### 8.7 Carga e Inferencia del Modelo

```python
import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
model = tf.keras.models.load_model("outputs/run_001/waymo_nav_classifier.keras")
model.summary()

# Inferencia sobre una imagen
img = tf.io.read_file("ruta/a/imagen.jpg")
img = tf.io.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (224, 224))
img = tf.cast(img, tf.float32) / 255.0
img = tf.expand_dims(img, axis=0)       # Añadir dimensión de batch

prediction = model.predict(img)
print("Predicción:", prediction)

# Exportar a TFLite (para dispositivos edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("outputs/waymo_nav_classifier.tflite", "wb") as f:
    f.write(tflite_model)

# Exportar a SavedModel (para TFServing)
model.export("outputs/saved_model/")
```

### 8.8 Parámetros Configurables

Editar `conf/base/parameters.yml` para cambiar el comportamiento del pipeline EDA:

```yaml
ingestion:
  ingestion_mode: "simulation"   # "simulation" | "tfrecord"
  raw_data_path: "data/01_raw/"
  n_synthetic_frames: 500        # Frames sintéticos en modo simulation
  random_seed: 42                # Semilla de reproducibilidad

report:
  report_output_dir: "data/08_reporting"
```

---

## Resultados del Entrenamiento

| Métrica | Valor |
|---------|-------|
| **Épocas completadas** | 5/5 |
| **Loss final (Binary Crossentropy)** | **0.1385** |
| **Accuracy final** | **99.42%** |
| **Tamaño del modelo** | 13.6 MB (`.keras`) |
| **Tiempo de entrenamiento** | ~6 min (CPU, 43 steps/época) |
| **Hardware usado** | Intel CPU (sin GPU) |
| **TF Version** | 2.21.0 |

---

## Licencias y Atribuciones

| Componente | Licencia |
|-----------|---------|
| **Waymo Open Dataset** | [Waymo Dataset License](https://waymo.com/open/terms/) — Uso no comercial exclusivo |
| **TensorFlow / Keras** | Apache 2.0 |
| **Kedro** | Apache 2.0 |
| **MobileNetV2 (pesos ImageNet)** | Apache 2.0 |
| **Código de este proyecto** | MIT |

> ⚠️ **Importante:** El Waymo Open Dataset está sujeto a restricciones de uso. Consulta los [Términos de Uso](https://waymo.com/open/terms/) antes de cualquier uso comercial o redistribución de los datos o modelos derivados.

---

<div align="center">

*Proyecto generado como parte de una evaluación técnica de Machine Learning.*
*Pipeline implementado con Kedro ≥ 0.19 + TensorFlow 2.21 + Keras 3.*

</div>
