"""
Nodos del pipeline de EDA para el dataset de percepción de Waymo.

Cada función representa un nodo independiente del pipeline de Kedro,
diseñado para procesar y analizar datos de imágenes y anotaciones 3D/2D
del Waymo Open Dataset.

Nodos implementados:
    1. ingest_and_clean_data — Ingesta, limpieza y validación
    2. analyze_class_balance — Distribución de clases y desbalance
    3. analyze_image_specs — Resoluciones y recomendación de resize
    4. analyze_bounding_boxes — Área, aspect ratio, segmentación por tamaño
    5. analyze_environmental_conditions — Iluminación, clima, oclusión
    6. generate_eda_report — Reporte HTML interactivo + PNGs
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para generación en servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ─── Mapeo de clases del Waymo Open Dataset ──────────────────────────────────
WAYMO_CLASSES: Dict[int, str] = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Sign",
    4: "Cyclist",
}

# ─── Especificaciones de cámaras (Waymo Open Dataset v2) ─────────────────────
CAMERA_SPECS: Dict[str, Dict[str, int]] = {
    "FRONT":       {"width": 1920, "height": 1280},
    "FRONT_LEFT":  {"width": 1920, "height": 1280},
    "FRONT_RIGHT": {"width": 1920, "height": 1280},
    "SIDE_LEFT":   {"width": 1920, "height": 886},
    "SIDE_RIGHT":  {"width": 1920, "height": 886},
}


# ═════════════════════════════════════════════════════════════════════════════
#  NODO 1: INGESTA Y LIMPIEZA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def ingest_and_clean_data(parameters: Dict[str, Any]) -> pd.DataFrame:
    """Ingesta y limpieza de datos crudos del dataset Waymo.

    Soporta dos modos de operación:
    - ``simulation``: genera datos sintéticos representativos para desarrollo
      y pruebas rápidas sin necesidad de los ~1 TB de archivos reales.
    - ``tfrecord``: lee archivos ``.tfrecord`` reales (requiere
      ``waymo-open-dataset-tf`` y ``tensorflow``).

    El proceso de limpieza incluye:
    1. Detección y marcado de imágenes corruptas (dimensiones/canales inválidos).
    2. Eliminación de registros duplicados por hash de contenido.
    3. Validación de bounding boxes (coordenadas en [0, 1], dimensiones > 0).
    4. Marcado y filtrado de etiquetas faltantes (``class_label`` nulo).

    Args:
        parameters: Diccionario con claves ``ingestion_mode``,
            ``raw_data_path``, ``n_synthetic_frames``, ``random_seed``.

    Returns:
        DataFrame limpio con columnas: ``image_id``, ``frame_id``,
        ``camera_name``, ``image_width``, ``image_height``, ``channels``,
        ``class_id``, ``class_label``, ``bbox_cx_norm``, ``bbox_cy_norm``,
        ``bbox_w_norm``, ``bbox_h_norm``, ``time_of_day``, ``weather``,
        ``occlusion_level``, ``difficulty_level``.
    """
    mode = parameters.get("ingestion_mode", "simulation")

    logger.info("=" * 60)
    logger.info("NODO 1: Ingesta y Limpieza — Modo: %s", mode)
    logger.info("=" * 60)

    # ── Ingesta ──
    if mode == "simulation":
        df = _generate_synthetic_data(parameters)
    elif mode == "tfrecord":
        df = _read_tfrecords(parameters)
    else:
        raise ValueError(f"Modo de ingesta no soportado: {mode}")

    initial_count = len(df)
    logger.info("Registros crudos ingresados: %d", initial_count)

    # ── Paso 1: Imágenes corruptas ──
    df = _mark_corrupt_images(df)
    corrupt_count = int(df["is_corrupt"].sum())
    logger.info("Imágenes corruptas detectadas: %d", corrupt_count)

    # ── Paso 2: Duplicados ──
    df, dup_count = _remove_duplicates(df)
    logger.info("Duplicados eliminados: %d", dup_count)

    # ── Paso 3: Bounding boxes ──
    df = _validate_bounding_boxes(df)
    invalid_bbox = int((~df["bbox_valid"]).sum())
    logger.info("Bounding boxes inválidas marcadas: %d", invalid_bbox)

    # ── Paso 4: Etiquetas faltantes ──
    df = _filter_missing_labels(df)

    # ── Filtrado final ──
    clean_mask = (~df["is_corrupt"]) & (df["bbox_valid"]) & (df["has_label"])
    df_clean = df.loc[clean_mask].copy()
    df_clean.drop(columns=["is_corrupt", "bbox_valid", "has_label"], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    final_count = len(df_clean)
    removed = initial_count - final_count
    pct = removed / initial_count * 100 if initial_count > 0 else 0
    logger.info(
        "Registros tras limpieza: %d (eliminados: %d, %.1f%%)",
        final_count, removed, pct,
    )

    return df_clean


# ── Funciones auxiliares de ingesta ───────────────────────────────────────────

def _generate_synthetic_data(parameters: Dict[str, Any]) -> pd.DataFrame:
    """Genera datos sintéticos que imitan la distribución real de Waymo.

    Se inyecta intencionalmente un ~3 % de datos «sucios» (bounding boxes
    fuera de rango, etiquetas nulas y duplicados) para demostrar el proceso
    de limpieza.
    """
    rng = np.random.default_rng(parameters.get("random_seed", 42))
    n_frames = parameters.get("n_synthetic_frames", 500)
    cameras = list(CAMERA_SPECS.keys())

    records: list[dict] = []

    for frame_idx in range(n_frames):
        frame_id = f"frame_{frame_idx:06d}"

        # Condiciones ambientales compartidas por todas las cámaras del frame
        time_of_day = rng.choice(
            ["Day", "Night", "Dawn/Dusk"], p=[0.60, 0.25, 0.15],
        )
        weather = rng.choice(
            ["Sunny", "Rainy", "Foggy"], p=[0.70, 0.20, 0.10],
        )

        for cam in cameras:
            image_id = f"{frame_id}_{cam}"
            base_w = CAMERA_SPECS[cam]["width"]
            base_h = CAMERA_SPECS[cam]["height"]

            # Variación menor para simular capturas reales
            w_actual = int(base_w + rng.integers(-10, 11))
            h_actual = int(base_h + rng.integers(-10, 11))

            # Número de objetos por imagen (distribución de Poisson)
            n_objects = int(np.clip(rng.poisson(lam=8), 1, 50))

            for _ in range(n_objects):
                # Distribución de clases: Vehicle dominante
                class_id = int(rng.choice(
                    [1, 2, 4, 3], p=[0.55, 0.25, 0.10, 0.10],
                ))
                class_label = WAYMO_CLASSES[class_id]

                # Centro del bbox en coordenadas normalizadas
                cx = float(rng.uniform(0.05, 0.95))
                cy = float(rng.uniform(0.10, 0.90))

                # Dimensiones dependen de la clase (log-normal para realismo)
                size_params = {
                    "Vehicle":    (-2.5, 0.8, -2.5, 0.7),
                    "Pedestrian": (-3.2, 0.6, -2.0, 0.6),
                    "Cyclist":    (-3.0, 0.7, -2.3, 0.6),
                    "Sign":       (-3.5, 0.5, -3.5, 0.5),
                }
                mw, sw, mh, sh = size_params[class_label]
                bw = float(np.clip(rng.lognormal(mean=mw, sigma=sw), 0.005, 0.8))
                bh = float(np.clip(rng.lognormal(mean=mh, sigma=sh), 0.005, 0.8))

                # Oclusión y dificultad
                occlusion = rng.choice(
                    ["NOT_OCCLUDED", "PARTIALLY_OCCLUDED", "MOSTLY_OCCLUDED"],
                    p=[0.55, 0.35, 0.10],
                )
                difficulty = int(rng.choice([1, 2], p=[0.60, 0.40]))

                records.append({
                    "image_id": image_id,
                    "frame_id": frame_id,
                    "camera_name": cam,
                    "image_width": w_actual,
                    "image_height": h_actual,
                    "channels": 3,
                    "class_id": class_id,
                    "class_label": class_label,
                    "bbox_cx_norm": cx,
                    "bbox_cy_norm": cy,
                    "bbox_w_norm": bw,
                    "bbox_h_norm": bh,
                    "time_of_day": time_of_day,
                    "weather": weather,
                    "occlusion_level": occlusion,
                    "difficulty_level": difficulty,
                })

    df = pd.DataFrame(records)

    # ── Inyectar datos sucios para demo de limpieza ──
    n_dirty = max(10, int(len(df) * 0.03))
    dirty_idx = rng.choice(df.index, size=n_dirty, replace=False)
    third = n_dirty // 3

    # Bboxes fuera de rango [0, 1]
    df.loc[dirty_idx[:third], "bbox_cx_norm"] = rng.uniform(-0.5, -0.01, size=third)

    # Bboxes con ancho cero
    df.loc[dirty_idx[third: 2 * third], "bbox_w_norm"] = 0.0

    # Etiquetas faltantes
    df.loc[dirty_idx[2 * third:], "class_label"] = np.nan
    df.loc[dirty_idx[2 * third:], "class_id"] = np.nan

    # Duplicados
    n_dups = max(5, int(len(df) * 0.005))
    dup_rows = df.sample(n=n_dups, random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)

    logger.info(
        "Datos sintéticos generados: %d registros (%d frames, %d cámaras)",
        len(df), n_frames, len(cameras),
    )
    logger.info(
        "  → Sucios inyectados: %d (bbox inválidos + labels) + %d duplicados",
        n_dirty, n_dups,
    )

    return df


def _read_tfrecords(parameters: Dict[str, Any]) -> pd.DataFrame:
    """Lee archivos .tfrecord reales usando TensorFlow nativo.

    Utiliza exclusivamente la API nativa de TensorFlow
    (``tf.io.parse_single_example`` / ``tf.train.Example``) para
    deserializar los registros, sin depender de
    ``waymo_open_dataset.dataset_pb2`` ni de ningún paquete específico
    de Waymo.

    El formato Waymo Open Dataset v2 almacena cada frame como un
    ``tf.train.Example`` con features tipadas por cámara. Esta función
    extrae los campos necesarios para el EDA:

    **Features de metadatos del frame** (string / int64):
    - ``context/name``      — identificador del segmento
    - ``timestamp_micros``  — timestamp del frame
    - ``time_of_day``       — ``Day`` / ``Night`` / ``Dawn/Dusk``
    - ``weather``           — ``Sunny`` / ``Rainy`` / ``Foggy``

    **Features por cámara** (float32 / int64 VarLen):
    - ``<cam>/width``, ``<cam>/height``   — dimensiones de imagen
    - ``<cam>/bbox/cx``, ``cy``, ``w``, ``h`` — bbox normalizada
    - ``<cam>/bbox/class``               — class_id (1=Vehicle, …)
    - ``<cam>/bbox/difficulty``          — nivel de dificultad
    - ``<cam>/bbox/occlusion``           — 0/1/2 → NOT/PARTIALLY/MOSTLY

    Si alguna feature no está presente en un record concreto, se usa
    el valor por defecto (0, 0.0, o cadena vacía), de modo que el
    parseo nunca lanza excepción por features ausentes.

    Args:
        parameters: Diccionario con la clave ``raw_data_path`` apuntando
            al directorio que contiene los archivos ``.tfrecord``.

    Returns:
        DataFrame con una fila por anotación (bounding box), con las
        mismas columnas que produce ``_generate_synthetic_data``.

    Raises:
        ImportError: Si TensorFlow no está instalado.
        FileNotFoundError: Si no hay archivos ``.tfrecord`` en la ruta.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Para leer .tfrecord reales, instala TensorFlow:\n"
            "  pip install tensorflow>=2.12.0"
        ) from exc

    raw_data_path = parameters.get("raw_data_path", "data/01_raw/")
    tfrecord_files = sorted(Path(raw_data_path).glob("*.tfrecord"))

    if not tfrecord_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .tfrecord en: {raw_data_path}"
        )

    logger.info("Encontrados %d archivos .tfrecord", len(tfrecord_files))

    cameras = list(CAMERA_SPECS.keys())

    # ── Descriptor de features para tf.io.parse_single_example ──────────────
    # VarLenFeature tolera ausencia del campo (devuelve SparseTensor vacío).
    # FixedLenFeature usa default_value para campos escalares opcionales.
    feature_spec: Dict[str, Any] = {
        "context/name":     tf.io.FixedLenFeature([], tf.string, default_value=""),
        "timestamp_micros": tf.io.FixedLenFeature([], tf.int64,  default_value=0),
        "time_of_day":      tf.io.FixedLenFeature([], tf.string, default_value="Unknown"),
        "weather":          tf.io.FixedLenFeature([], tf.string, default_value="Unknown"),
    }
    for cam in cameras:
        k = cam.lower()
        feature_spec[f"{k}/width"]           = tf.io.FixedLenFeature([], tf.int64,   default_value=CAMERA_SPECS[cam]["width"])
        feature_spec[f"{k}/height"]          = tf.io.FixedLenFeature([], tf.int64,   default_value=CAMERA_SPECS[cam]["height"])
        feature_spec[f"{k}/bbox/cx"]         = tf.io.VarLenFeature(tf.float32)
        feature_spec[f"{k}/bbox/cy"]         = tf.io.VarLenFeature(tf.float32)
        feature_spec[f"{k}/bbox/w"]          = tf.io.VarLenFeature(tf.float32)
        feature_spec[f"{k}/bbox/h"]          = tf.io.VarLenFeature(tf.float32)
        feature_spec[f"{k}/bbox/class"]      = tf.io.VarLenFeature(tf.int64)
        feature_spec[f"{k}/bbox/difficulty"] = tf.io.VarLenFeature(tf.int64)
        feature_spec[f"{k}/bbox/occlusion"]  = tf.io.VarLenFeature(tf.int64)

    # Mapas locales: sin dependencias de protobuf de Waymo
    occlusion_map: Dict[int, str] = {
        0: "NOT_OCCLUDED",
        1: "PARTIALLY_OCCLUDED",
        2: "MOSTLY_OCCLUDED",
    }

    records: list[dict] = []

    for tf_path in tfrecord_files:
        logger.debug("Procesando: %s", tf_path.name)
        raw_dataset = tf.data.TFRecordDataset(str(tf_path), compression_type="")

        for raw_record in raw_dataset:
            # Parsear el record como tf.train.Example nativo
            try:
                parsed = tf.io.parse_single_example(raw_record, feature_spec)
            except tf.errors.InvalidArgumentError as exc:
                logger.warning(
                    "Record no parseable en %s, omitido: %s",
                    tf_path.name, exc,
                )
                continue

            context_name = parsed["context/name"].numpy().decode("utf-8")
            timestamp    = int(parsed["timestamp_micros"].numpy())
            time_of_day  = parsed["time_of_day"].numpy().decode("utf-8") or "Unknown"
            weather      = parsed["weather"].numpy().decode("utf-8") or "Unknown"
            frame_id     = f"{context_name}_{timestamp}"

            for cam in cameras:
                k = cam.lower()

                img_w = int(parsed[f"{k}/width"].numpy())
                img_h = int(parsed[f"{k}/height"].numpy())

                # VarLenFeature → SparseTensor → numpy array denso
                bx  = tf.sparse.to_dense(parsed[f"{k}/bbox/cx"]).numpy()
                by  = tf.sparse.to_dense(parsed[f"{k}/bbox/cy"]).numpy()
                bw  = tf.sparse.to_dense(parsed[f"{k}/bbox/w"]).numpy()
                bh  = tf.sparse.to_dense(parsed[f"{k}/bbox/h"]).numpy()
                cls = tf.sparse.to_dense(parsed[f"{k}/bbox/class"]).numpy()
                dif = tf.sparse.to_dense(parsed[f"{k}/bbox/difficulty"]).numpy()
                occ = tf.sparse.to_dense(parsed[f"{k}/bbox/occlusion"]).numpy()

                n_boxes = len(bx)
                if n_boxes == 0:
                    continue  # Cámara sin anotaciones en este frame

                image_id = f"{frame_id}_{cam}"

                for i in range(n_boxes):
                    class_id   = int(cls[i])               if i < len(cls) else 0
                    diff_level = int(dif[i])               if i < len(dif) else 1
                    occ_int    = int(occ[i])               if i < len(occ) else 0

                    records.append({
                        "image_id":        image_id,
                        "frame_id":        frame_id,
                        "camera_name":     cam,
                        "image_width":     img_w,
                        "image_height":    img_h,
                        "channels":        3,
                        "class_id":        class_id,
                        "class_label":     WAYMO_CLASSES.get(class_id, "Unknown"),
                        "bbox_cx_norm":    float(bx[i]),
                        "bbox_cy_norm":    float(by[i]),
                        "bbox_w_norm":     float(bw[i]) if i < len(bw) else 0.0,
                        "bbox_h_norm":     float(bh[i]) if i < len(bh) else 0.0,
                        "time_of_day":     time_of_day,
                        "weather":         weather,
                        "occlusion_level": occlusion_map.get(occ_int, "NOT_OCCLUDED"),
                        "difficulty_level": diff_level,
                    })

    logger.info("Registros leídos de tfrecord: %d", len(records))
    return pd.DataFrame(records)


def _mark_corrupt_images(df: pd.DataFrame) -> pd.DataFrame:
    """Marca imágenes con dimensiones inválidas o canales incorrectos."""
    df = df.copy()
    df["is_corrupt"] = (
        (df["image_width"] <= 0)
        | (df["image_height"] <= 0)
        | (df["channels"] != 3)
        | df["image_width"].isna()
        | df["image_height"].isna()
    )
    return df


def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Elimina filas duplicadas basado en image_id y coordenadas de bbox."""
    subset_cols = [
        "image_id", "class_id",
        "bbox_cx_norm", "bbox_cy_norm",
        "bbox_w_norm", "bbox_h_norm",
    ]
    initial = len(df)
    df = df.drop_duplicates(subset=subset_cols, keep="first")
    return df, initial - len(df)


def _validate_bounding_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Valida que las bounding boxes tengan coordenadas en [0, 1] y área > 0."""
    df = df.copy()
    df["bbox_valid"] = (
        df["bbox_cx_norm"].between(0, 1)
        & df["bbox_cy_norm"].between(0, 1)
        & df["bbox_w_norm"].between(0.001, 1)
        & df["bbox_h_norm"].between(0.001, 1)
        & ((df["bbox_cx_norm"] - df["bbox_w_norm"] / 2) >= -0.01)
        & ((df["bbox_cx_norm"] + df["bbox_w_norm"] / 2) <= 1.01)
        & ((df["bbox_cy_norm"] - df["bbox_h_norm"] / 2) >= -0.01)
        & ((df["bbox_cy_norm"] + df["bbox_h_norm"] / 2) <= 1.01)
    )
    return df


def _filter_missing_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Marca registros con etiquetas faltantes."""
    df = df.copy()
    df["has_label"] = df["class_label"].notna() & df["class_id"].notna()
    missing_count = int((~df["has_label"]).sum())
    logger.info("Etiquetas faltantes detectadas: %d", missing_count)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  NODO 2: ANÁLISIS DE BALANCE DE CLASES
# ═════════════════════════════════════════════════════════════════════════════

def analyze_class_balance(clean_data: pd.DataFrame) -> pd.DataFrame:
    """Analiza la distribución de clases y genera métricas de desbalance.

    Calcula:
    - Frecuencia absoluta y relativa por clase.
    - Ratio de desbalance ``max_count / min_count``.
    - Entropía de Shannon normalizada (1.0 = perfectamente balanceado).
    - Pesos sugeridos para ``class_weights`` en entrenamiento.
    - Recomendaciones automáticas de data augmentation.

    Args:
        clean_data: DataFrame limpio del nodo de ingesta.

    Returns:
        DataFrame con una fila por clase y columnas de métricas.
    """
    logger.info("=" * 60)
    logger.info("NODO 2: Análisis de Balance de Clases")
    logger.info("=" * 60)

    class_counts = clean_data["class_label"].value_counts()
    total = class_counts.sum()

    # ── Métricas por clase ──
    metrics: list[dict] = []
    for cls_name in class_counts.index:
        count = int(class_counts[cls_name])
        freq = count / total
        metrics.append({
            "class_label": cls_name,
            "count": count,
            "frequency": round(freq, 4),
            "percentage": round(freq * 100, 2),
        })

    df_metrics = pd.DataFrame(metrics)

    # ── Métricas globales ──
    max_count = int(class_counts.max())
    min_count = int(class_counts.min())
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # Entropía de Shannon normalizada
    probs = class_counts.values.astype(float) / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_entropy = float(np.log2(len(class_counts)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    df_metrics["imbalance_ratio_global"] = round(imbalance_ratio, 2)
    df_metrics["entropy"] = round(entropy, 4)
    df_metrics["normalized_entropy"] = round(normalized_entropy, 4)

    # Peso sugerido (inversamente proporcional a la frecuencia)
    median_count = float(class_counts.median())
    df_metrics["suggested_weight"] = round(median_count / df_metrics["count"], 4)

    # ── Sugerencias automáticas ──
    suggestions: list[str] = []
    if imbalance_ratio > 10:
        suggestions.append(
            "CRITICO: Desbalance severo (ratio > 10). "
            "Usar oversampling agresivo + class weights + focal loss."
        )
    elif imbalance_ratio > 5:
        suggestions.append(
            "ALTO: Desbalance significativo (ratio > 5). "
            "Considerar SMOTE, focal loss o copy-paste augmentation."
        )
    elif imbalance_ratio > 2:
        suggestions.append(
            "MODERADO: Aplicar data augmentation selectiva "
            "a clases sub-representadas (flip, rotate, mosaic)."
        )
    else:
        suggestions.append(
            "BAJO: Distribución relativamente balanceada. "
            "Data augmentation estándar es suficiente."
        )

    # Identificar clases minoritarias
    expected_equal_share = total / len(class_counts)
    minority = class_counts[class_counts < expected_equal_share * 0.5].index.tolist()
    if minority:
        suggestions.append(
            f"Clases minoritarias para augmentation prioritario: "
            f"{', '.join(minority)}"
        )

    df_metrics["augmentation_suggestions"] = "; ".join(suggestions)

    logger.info("Distribución de clases:\n%s", class_counts.to_string())
    logger.info("Ratio de desbalance: %.2f", imbalance_ratio)
    logger.info("Entropía normalizada: %.4f", normalized_entropy)
    for s in suggestions:
        logger.info("  💡 %s", s)

    return df_metrics


# ═════════════════════════════════════════════════════════════════════════════
#  NODO 3: ANÁLISIS DE ESPECIFICACIONES DE IMAGEN
# ═════════════════════════════════════════════════════════════════════════════

def analyze_image_specs(clean_data: pd.DataFrame) -> pd.DataFrame:
    """Inspecciona resoluciones, canales y variabilidad de las imágenes.

    Calcula:
    - Estadísticas descriptivas de resolución (min, max, media, mediana, std,
      percentiles 25/75, moda).
    - Distribución de aspect ratios y megapíxeles.
    - Verificación de uniformidad de canales (RGB).
    - Desglose de resolución por cámara.
    - Recomendación de resolución de resize basada en tamaños estándar
      de detección de objetos.

    Args:
        clean_data: DataFrame limpio.

    Returns:
        DataFrame con una fila por estadística y la recomendación de resize.
    """
    logger.info("=" * 60)
    logger.info("NODO 3: Análisis de Especificaciones de Imagen")
    logger.info("=" * 60)

    # Imágenes únicas (evitar contar anotaciones múltiples)
    img_cols = ["image_id", "image_width", "image_height", "channels", "camera_name"]
    df_images = clean_data[img_cols].drop_duplicates(subset=["image_id"]).copy()

    df_images["aspect_ratio"] = df_images["image_width"] / df_images["image_height"]
    df_images["megapixels"] = (
        df_images["image_width"] * df_images["image_height"]
    ) / 1e6

    # ── Estadísticas descriptivas ──
    stat_functions = [
        ("count", lambda s: s.count()),
        ("min", lambda s: s.min()),
        ("max", lambda s: s.max()),
        ("mean", lambda s: s.mean()),
        ("median", lambda s: s.median()),
        ("std", lambda s: s.std()),
        ("p25", lambda s: s.quantile(0.25)),
        ("p75", lambda s: s.quantile(0.75)),
        ("mode", lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan),
    ]

    stat_rows: list[dict] = []
    for stat_name, func in stat_functions:
        row: dict[str, Any] = {"metric": stat_name}
        for col, alias in [
            ("image_width", "width"),
            ("image_height", "height"),
            ("aspect_ratio", "aspect_ratio"),
            ("megapixels", "megapixels"),
        ]:
            val = func(df_images[col])
            row[alias] = (
                int(val) if stat_name in ("count", "mode") and alias in ("width", "height")
                else round(float(val), 4)
            )
        stat_rows.append(row)

    df_stats = pd.DataFrame(stat_rows)

    # ── Verificación de canales ──
    unique_channels = df_images["channels"].unique()
    all_rgb = bool(all(c == 3 for c in unique_channels))

    # ── Resolución por cámara ──
    camera_res = (
        df_images.groupby("camera_name")
        .agg(
            width_mode=("image_width", lambda s: int(s.mode().iloc[0])),
            height_mode=("image_height", lambda s: int(s.mode().iloc[0])),
            count=("image_id", "count"),
        )
        .reset_index()
    )

    # ── Recomendación de resize ──
    median_w = float(df_images["image_width"].median())
    median_h = float(df_images["image_height"].median())
    standard_sizes = [320, 416, 512, 640, 768, 896, 1024, 1280]
    recommended_size = min(
        standard_sizes, key=lambda s: abs(s - min(median_w, median_h) * 0.5)
    )

    rec_row = pd.DataFrame([{
        "metric": "recommended_resize",
        "width": recommended_size,
        "height": recommended_size,
        "aspect_ratio": 1.0,
        "megapixels": round((recommended_size ** 2) / 1e6, 4),
    }])
    df_stats = pd.concat([df_stats, rec_row], ignore_index=True)

    # Metadatos adicionales como columnas
    df_stats["all_rgb"] = all_rgb
    df_stats["unique_channels"] = str(unique_channels.tolist())

    cam_summary = "; ".join(
        f"{row['camera_name']}: {row['width_mode']}x{row['height_mode']} "
        f"({row['count']} imgs)"
        for _, row in camera_res.iterrows()
    )
    df_stats["camera_breakdown"] = cam_summary

    logger.info("Imágenes únicas analizadas: %d", len(df_images))
    logger.info("Resolución mediana: %dx%d", int(median_w), int(median_h))
    logger.info("Todos RGB: %s", all_rgb)
    logger.info("Resize recomendado: %dx%d", recommended_size, recommended_size)
    logger.info("Desglose por cámara:\n%s", camera_res.to_string(index=False))

    return df_stats


# ═════════════════════════════════════════════════════════════════════════════
#  NODO 4: ANÁLISIS DE BOUNDING BOXES
# ═════════════════════════════════════════════════════════════════════════════

def analyze_bounding_boxes(clean_data: pd.DataFrame) -> pd.DataFrame:
    """Análisis detallado de las bounding boxes del dataset.

    Calcula por cada clase:
    - Área en píxeles y relativa al área de imagen (%).
    - Aspect ratio ``w/h`` (media, mediana, desviación estándar).
    - Segmentación por tamaño según el estándar COCO:
        * Pequeño: área < 32² = 1024 px²
        * Mediano: 1024 ≤ área < 96² = 9216 px²
        * Grande: área ≥ 9216 px²
    - Porcentaje de objetos lejanos (< 0.1 % del área de imagen).
    - Porcentaje de objetos diminutos (< 16×16 px equivalente).

    Args:
        clean_data: DataFrame limpio.

    Returns:
        DataFrame con una fila por clase y métricas de bounding box.
    """
    logger.info("=" * 60)
    logger.info("NODO 4: Análisis de Bounding Boxes")
    logger.info("=" * 60)

    df = clean_data.copy()

    # Dimensiones en píxeles
    df["bbox_w_px"] = df["bbox_w_norm"] * df["image_width"]
    df["bbox_h_px"] = df["bbox_h_norm"] * df["image_height"]
    df["bbox_area_px"] = df["bbox_w_px"] * df["bbox_h_px"]
    df["image_area_px"] = df["image_width"] * df["image_height"]
    df["bbox_area_relative"] = (df["bbox_area_px"] / df["image_area_px"]) * 100
    df["bbox_aspect_ratio"] = df["bbox_w_px"] / df["bbox_h_px"].replace(0, np.nan)

    # Segmentación por tamaño (COCO)
    conditions = [
        df["bbox_area_px"] < 1024,
        (df["bbox_area_px"] >= 1024) & (df["bbox_area_px"] < 9216),
        df["bbox_area_px"] >= 9216,
    ]
    size_labels = ["Small (<32²)", "Medium (32²-96²)", "Large (>96²)"]
    df["size_category"] = np.select(conditions, size_labels, default="Unknown")

    # Objetos lejanos y diminutos
    df["is_distant"] = df["bbox_area_relative"] < 0.1
    df["is_tiny"] = df["bbox_area_px"] < 256  # < 16×16

    # ── Resumen por clase ──
    summary_records: list[dict] = []
    for cls_name in sorted(df["class_label"].unique()):
        cls_df = df[df["class_label"] == cls_name]
        size_dist = cls_df["size_category"].value_counts(normalize=True)

        summary_records.append({
            "class_label": cls_name,
            "total_boxes": len(cls_df),
            "area_px_mean": round(float(cls_df["bbox_area_px"].mean()), 2),
            "area_px_median": round(float(cls_df["bbox_area_px"].median()), 2),
            "area_px_std": round(float(cls_df["bbox_area_px"].std()), 2),
            "area_relative_mean_pct": round(
                float(cls_df["bbox_area_relative"].mean()), 4
            ),
            "aspect_ratio_mean": round(
                float(cls_df["bbox_aspect_ratio"].mean()), 4
            ),
            "aspect_ratio_median": round(
                float(cls_df["bbox_aspect_ratio"].median()), 4
            ),
            "aspect_ratio_std": round(
                float(cls_df["bbox_aspect_ratio"].std()), 4
            ),
            "pct_small": round(
                float(size_dist.get("Small (<32²)", 0)) * 100, 2
            ),
            "pct_medium": round(
                float(size_dist.get("Medium (32²-96²)", 0)) * 100, 2
            ),
            "pct_large": round(
                float(size_dist.get("Large (>96²)", 0)) * 100, 2
            ),
            "pct_distant": round(float(cls_df["is_distant"].mean()) * 100, 2),
            "pct_tiny": round(float(cls_df["is_tiny"].mean()) * 100, 2),
            "width_px_mean": round(float(cls_df["bbox_w_px"].mean()), 2),
            "height_px_mean": round(float(cls_df["bbox_h_px"].mean()), 2),
        })

    df_summary = pd.DataFrame(summary_records)

    for _, row in df_summary.iterrows():
        logger.info(
            "  %s: %d boxes | Área media: %.0f px² | AR: %.2f | "
            "S/M/L: %.0f%%/%.0f%%/%.0f%%",
            row["class_label"], int(row["total_boxes"]),
            row["area_px_mean"], row["aspect_ratio_mean"],
            row["pct_small"], row["pct_medium"], row["pct_large"],
        )

    total_distant = int(df["is_distant"].sum())
    logger.info(
        "Total objetos lejanos (<0.1%% área): %d (%.1f%%)",
        total_distant, total_distant / len(df) * 100,
    )

    return df_summary


# ═════════════════════════════════════════════════════════════════════════════
#  NODO 5: ANÁLISIS DE CONDICIONES AMBIENTALES
# ═════════════════════════════════════════════════════════════════════════════

def analyze_environmental_conditions(
    clean_data: pd.DataFrame,
) -> pd.DataFrame:
    """Clasifica y agrupa imágenes según condiciones del entorno.

    Analiza:
    - Distribución por hora del día (Day / Night / Dawn/Dusk).
    - Distribución climática (Sunny / Rainy / Foggy).
    - Niveles de oclusión de objetos.
    - Análisis de solapamiento (overlap) entre bounding boxes del mismo frame
      mediante cálculo de IoU pairwise en una muestra.

    Args:
        clean_data: DataFrame limpio.

    Returns:
        DataFrame consolidado con ``condition_type``, ``condition_value``
        y métricas asociadas.
    """
    logger.info("=" * 60)
    logger.info("NODO 5: Análisis de Condiciones Ambientales")
    logger.info("=" * 60)

    df = clean_data.copy()

    # ── Distribución por hora del día ──
    time_dist = (
        df.groupby("time_of_day")
        .agg(
            n_annotations=("image_id", "count"),
            n_images=("image_id", "nunique"),
            n_frames=("frame_id", "nunique"),
        )
        .reset_index()
    )
    time_dist["pct_annotations"] = round(
        time_dist["n_annotations"] / time_dist["n_annotations"].sum() * 100, 2
    )

    # ── Distribución por clima ──
    weather_dist = (
        df.groupby("weather")
        .agg(
            n_annotations=("image_id", "count"),
            n_images=("image_id", "nunique"),
            n_frames=("frame_id", "nunique"),
        )
        .reset_index()
    )
    weather_dist["pct_annotations"] = round(
        weather_dist["n_annotations"] / weather_dist["n_annotations"].sum() * 100, 2
    )

    # ── Distribución de oclusión ──
    occlusion_dist = (
        df.groupby("occlusion_level")
        .agg(n_annotations=("image_id", "count"))
        .reset_index()
    )
    occlusion_dist["pct"] = round(
        occlusion_dist["n_annotations"]
        / occlusion_dist["n_annotations"].sum()
        * 100,
        2,
    )

    # ── Análisis de solapamiento (IoU) ──
    overlap_stats = _compute_overlap_stats(df)

    # ── Construir DataFrame consolidado ──
    records: list[dict] = []

    for _, row in time_dist.iterrows():
        records.append({
            "condition_type": "time_of_day",
            "condition_value": row["time_of_day"],
            "n_annotations": int(row["n_annotations"]),
            "n_unique_images": int(row["n_images"]),
            "n_unique_frames": int(row["n_frames"]),
            "pct_of_total": float(row["pct_annotations"]),
        })

    for _, row in weather_dist.iterrows():
        records.append({
            "condition_type": "weather",
            "condition_value": row["weather"],
            "n_annotations": int(row["n_annotations"]),
            "n_unique_images": int(row["n_images"]),
            "n_unique_frames": int(row["n_frames"]),
            "pct_of_total": float(row["pct_annotations"]),
        })

    for _, row in occlusion_dist.iterrows():
        records.append({
            "condition_type": "occlusion",
            "condition_value": row["occlusion_level"],
            "n_annotations": int(row["n_annotations"]),
            "n_unique_images": 0,
            "n_unique_frames": 0,
            "pct_of_total": float(row["pct"]),
        })

    records.append({
        "condition_type": "overlap",
        "condition_value": "mean_objects_per_image",
        "n_annotations": int(overlap_stats["mean_objects_per_image"]),
        "n_unique_images": int(overlap_stats["total_images"]),
        "n_unique_frames": int(overlap_stats["total_frames"]),
        "pct_of_total": round(overlap_stats["pct_images_with_overlap"], 2),
    })

    df_env = pd.DataFrame(records)

    logger.info("Distribución por hora del día:\n%s", time_dist.to_string(index=False))
    logger.info("Distribución por clima:\n%s", weather_dist.to_string(index=False))
    logger.info("Distribución de oclusión:\n%s", occlusion_dist.to_string(index=False))
    logger.info(
        "Imágenes con solapamiento: %.1f%%",
        overlap_stats["pct_images_with_overlap"],
    )

    return df_env


def _compute_overlap_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula estadísticas de solapamiento entre bboxes de una misma imagen.

    Para evitar O(n²) sobre todo el dataset, se muestrea un máximo de 500
    imágenes y se calcula intersección pairwise entre sus bounding boxes.
    """
    rng = np.random.default_rng(42)

    total_images = int(df["image_id"].nunique())
    total_frames = int(df["frame_id"].nunique())
    objects_per_image = df.groupby("image_id").size()
    mean_obj = float(objects_per_image.mean())

    sample_images = df["image_id"].unique()
    max_sample = 500
    if len(sample_images) > max_sample:
        sample_images = rng.choice(sample_images, max_sample, replace=False)

    overlap_count = 0

    for img_id in sample_images:
        img_df = df.loc[df["image_id"] == img_id]
        if len(img_df) < 2:
            continue

        boxes = img_df[
            ["bbox_cx_norm", "bbox_cy_norm", "bbox_w_norm", "bbox_h_norm"]
        ].values

        # Centro → esquinas
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        has_overlap = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                inter_w = max(0.0, min(x2[i], x2[j]) - max(x1[i], x1[j]))
                inter_h = max(0.0, min(y2[i], y2[j]) - max(y1[i], y1[j]))
                if inter_w * inter_h > 0:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if has_overlap:
            overlap_count += 1

    pct = (overlap_count / len(sample_images) * 100) if len(sample_images) > 0 else 0.0

    return {
        "total_images": total_images,
        "total_frames": total_frames,
        "mean_objects_per_image": mean_obj,
        "pct_images_with_overlap": pct,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  NODO 6: GENERACIÓN DEL REPORTE EDA
# ═════════════════════════════════════════════════════════════════════════════

def generate_eda_report(
    class_balance: pd.DataFrame,
    image_specs: pd.DataFrame,
    bbox_analysis: pd.DataFrame,
    environmental_analysis: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Consolida métricas y genera el reporte visual del EDA.

    Produce:
    - 6 figuras PNG de alta resolución en ``data/08_reporting/figures/``.
    - 1 reporte HTML interactivo auto-contenido con imágenes embebidas
      en base64 (sin dependencias externas).
    - 1 DataFrame resumen consolidado exportado a Parquet.

    Gráficos generados:
        01. Distribución de clases (bar + pie)
        02. Resolución por cámara + resize recomendado
        03. Distribución de área de bounding boxes
        04. Aspect ratio de bounding boxes por clase
        05. Condiciones ambientales (hora, clima, oclusión)
        06. Objetos difíciles de detectar (lejanos + diminutos)

    Args:
        class_balance: Salida del nodo ``analyze_class_balance``.
        image_specs: Salida del nodo ``analyze_image_specs``.
        bbox_analysis: Salida del nodo ``analyze_bounding_boxes``.
        environmental_analysis: Salida del nodo
            ``analyze_environmental_conditions``.
        parameters: Diccionario con ``report_output_dir``.

    Returns:
        DataFrame con resumen consolidado de todas las métricas del EDA.
    """
    logger.info("=" * 60)
    logger.info("NODO 6: Generación del Reporte EDA")
    logger.info("=" * 60)

    report_dir = Path(parameters.get("report_output_dir", "data/08_reporting"))
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Configurar estilo de gráficos ──
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
    })

    figure_paths: Dict[str, str] = {}

    # ── 01: Distribución de Clases ──
    figure_paths["class_distribution"] = _plot_class_distribution(
        class_balance, figures_dir,
    )

    # ── 02: Resolución de Imágenes ──
    figure_paths["image_resolution"] = _plot_image_resolution(
        image_specs, figures_dir,
    )

    # ── 03: Área de Bounding Boxes ──
    figure_paths["bbox_area"] = _plot_bbox_area(bbox_analysis, figures_dir)

    # ── 04: Aspect Ratio ──
    figure_paths["bbox_aspect_ratio"] = _plot_bbox_aspect_ratio(
        bbox_analysis, figures_dir,
    )

    # ── 05: Condiciones Ambientales ──
    figure_paths["environmental"] = _plot_environmental(
        environmental_analysis, figures_dir,
    )

    # ── 06: Objetos Difíciles ──
    figure_paths["difficult_objects"] = _plot_difficult_objects(
        bbox_analysis, figures_dir,
    )

    # ── Generar HTML ──
    html_path = report_dir / "eda_report.html"
    _generate_html_report(
        html_path=html_path,
        figure_paths=figure_paths,
        class_balance=class_balance,
        image_specs=image_specs,
        bbox_analysis=bbox_analysis,
        environmental_analysis=environmental_analysis,
    )
    logger.info("  ✓ Reporte HTML generado: %s", html_path)

    # ── DataFrame resumen consolidado ──
    df_summary = _build_summary_dataframe(
        class_balance, image_specs, bbox_analysis,
        environmental_analysis, figure_paths, html_path,
    )

    logger.info("=" * 60)
    logger.info("REPORTE EDA COMPLETO ✓")
    logger.info("  Figuras generadas: %d", len(figure_paths))
    logger.info("  Reporte HTML: %s", html_path)
    logger.info("  Métricas consolidadas: %d", len(df_summary))
    logger.info("=" * 60)

    return df_summary


# ── Funciones de plotting ─────────────────────────────────────────────────────

def _plot_class_distribution(
    class_balance: pd.DataFrame, figures_dir: Path,
) -> str:
    """01: Bar chart + pie chart de distribución de clases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_classes = len(class_balance)
    colors = sns.color_palette("husl", n_classes)

    # Bar chart
    bars = axes[0].bar(
        class_balance["class_label"], class_balance["count"],
        color=colors, edgecolor="white", linewidth=1.2,
    )
    axes[0].set_title(
        "Distribución de Clases — Frecuencia Absoluta", fontweight="bold",
    )
    axes[0].set_xlabel("Clase")
    axes[0].set_ylabel("Cantidad de Anotaciones")
    for bar, val in zip(bars, class_balance["count"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_height() * 0.02,
            f"{int(val):,}", ha="center", va="bottom",
            fontweight="bold", fontsize=10,
        )

    # Pie chart
    axes[1].pie(
        class_balance["count"],
        labels=class_balance["class_label"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=[0.05] * n_classes,
        shadow=True,
    )
    axes[1].set_title("Distribución de Clases — Proporción", fontweight="bold")

    plt.tight_layout()
    path = figures_dir / "01_class_distribution.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  ✓ Generado: %s", path.name)
    return str(path)


def _plot_image_resolution(
    image_specs: pd.DataFrame, figures_dir: Path,
) -> str:
    """02: Resolución por cámara + resize recomendado."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Camera breakdown
    cam_info = (
        image_specs["camera_breakdown"].iloc[0]
        if "camera_breakdown" in image_specs.columns else ""
    )
    cam_parts = [c.strip() for c in cam_info.split(";") if c.strip()]

    if cam_parts:
        cam_names, cam_widths, cam_heights = [], [], []
        for part in cam_parts:
            name = part.split(":")[0].strip()
            dims_str = part.split(":")[1].strip().split("(")[0].strip()
            w, h = dims_str.split("x")
            cam_names.append(name)
            cam_widths.append(int(w))
            cam_heights.append(int(h))

        x = np.arange(len(cam_names))
        width = 0.35
        axes[0].bar(
            x - width / 2, cam_widths, width,
            label="Ancho", color="#3498db", edgecolor="white",
        )
        axes[0].bar(
            x + width / 2, cam_heights, width,
            label="Alto", color="#e74c3c", edgecolor="white",
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cam_names, rotation=30, ha="right")
        axes[0].set_title("Resolución por Cámara", fontweight="bold")
        axes[0].set_ylabel("Píxeles")
        axes[0].legend()

    # Resize recomendado
    rec_row = image_specs[image_specs["metric"] == "recommended_resize"]
    if len(rec_row) > 0:
        rec_size = int(rec_row["width"].iloc[0])
        standard_sizes = [320, 416, 512, 640, 768, 896, 1024, 1280]
        bar_colors = [
            "#2ecc71" if s == rec_size else "#bdc3c7" for s in standard_sizes
        ]
        axes[1].bar(
            [str(s) for s in standard_sizes], standard_sizes,
            color=bar_colors, edgecolor="white",
        )
        axes[1].set_title(
            f"Resize Recomendado: {rec_size}×{rec_size}", fontweight="bold",
        )
        axes[1].set_xlabel("Tamaños Estándar (px)")
        axes[1].set_ylabel("Dimensión")
        axes[1].axhline(
            y=rec_size, color="#e74c3c", linestyle="--", alpha=0.7,
            label=f"Recomendado: {rec_size}",
        )
        axes[1].legend()

    plt.tight_layout()
    path = figures_dir / "02_image_resolution.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  ✓ Generado: %s", path.name)
    return str(path)


def _plot_bbox_area(
    bbox_analysis: pd.DataFrame, figures_dir: Path,
) -> str:
    """03: Distribución de área y segmentación de tamaño de bboxes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    classes = bbox_analysis["class_label"].values
    n_cls = len(classes)
    x = np.arange(n_cls)
    colors = sns.color_palette("husl", n_cls)

    # Area media por clase
    axes[0].bar(x, bbox_analysis["area_px_mean"], color=colors, edgecolor="white")
    axes[0].errorbar(
        x, bbox_analysis["area_px_mean"],
        yerr=bbox_analysis["area_px_std"],
        fmt="none", ecolor="gray", capsize=5,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_title("Área Media de BBox por Clase", fontweight="bold")
    axes[0].set_ylabel("Área (px²)")
    axes[0].set_xlabel("Clase")

    # Size categories stacked bar
    size_data = (
        bbox_analysis[["class_label", "pct_small", "pct_medium", "pct_large"]]
        .set_index("class_label")
    )
    size_data.plot(
        kind="bar", stacked=True, ax=axes[1],
        color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="white",
    )
    axes[1].set_title(
        "Distribución de Tamaños (Estándar COCO)", fontweight="bold",
    )
    axes[1].set_ylabel("Porcentaje (%)")
    axes[1].set_xlabel("Clase")
    axes[1].legend(
        ["Pequeño (<32²)", "Mediano (32²-96²)", "Grande (>96²)"],
        loc="upper right", fontsize=9,
    )
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    path = figures_dir / "03_bbox_area_distribution.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  ✓ Generado: %s", path.name)
    return str(path)


def _plot_bbox_aspect_ratio(
    bbox_analysis: pd.DataFrame, figures_dir: Path,
) -> str:
    """04: Aspect ratio (w/h) de bounding boxes por clase."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ar = bbox_analysis[
        ["class_label", "aspect_ratio_mean", "aspect_ratio_median", "aspect_ratio_std"]
    ]
    x = np.arange(len(ar))
    w = 0.35

    ax.bar(
        x - w / 2, ar["aspect_ratio_mean"], w,
        label="Media", color="#3498db", edgecolor="white",
    )
    ax.bar(
        x + w / 2, ar["aspect_ratio_median"], w,
        label="Mediana", color="#e67e22", edgecolor="white",
    )
    ax.errorbar(
        x - w / 2, ar["aspect_ratio_mean"],
        yerr=ar["aspect_ratio_std"],
        fmt="none", ecolor="gray", capsize=5,
    )
    ax.axhline(
        y=1.0, color="#e74c3c", linestyle="--", alpha=0.6, label="Cuadrado (1:1)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ar["class_label"])
    ax.set_title("Aspect Ratio (w/h) de BBox por Clase", fontweight="bold")
    ax.set_ylabel("Aspect Ratio")
    ax.set_xlabel("Clase")
    ax.legend()

    plt.tight_layout()
    path = figures_dir / "04_bbox_aspect_ratio.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  ✓ Generado: %s", path.name)
    return str(path)


def _plot_environmental(
    env_analysis: pd.DataFrame, figures_dir: Path,
) -> str:
    """05: Distribución de condiciones ambientales."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hora del día
    time_data = env_analysis[env_analysis["condition_type"] == "time_of_day"]
    if len(time_data) > 0:
        time_colors = {
            "Day": "#f1c40f", "Night": "#2c3e50", "Dawn/Dusk": "#e67e22",
        }
        colors_list = [
            time_colors.get(v, "#95a5a6") for v in time_data["condition_value"]
        ]
        axes[0].pie(
            time_data["n_annotations"],
            labels=time_data["condition_value"],
            autopct="%1.1f%%", colors=colors_list,
            startangle=90, explode=[0.03] * len(time_data),
        )
        axes[0].set_title("Distribución — Hora del Día", fontweight="bold")

    # Clima
    weather_data = env_analysis[env_analysis["condition_type"] == "weather"]
    if len(weather_data) > 0:
        weather_colors = {
            "Sunny": "#f39c12", "Rainy": "#3498db", "Foggy": "#95a5a6",
        }
        colors_list = [
            weather_colors.get(v, "#bdc3c7")
            for v in weather_data["condition_value"]
        ]
        axes[1].pie(
            weather_data["n_annotations"],
            labels=weather_data["condition_value"],
            autopct="%1.1f%%", colors=colors_list,
            startangle=90, explode=[0.03] * len(weather_data),
        )
        axes[1].set_title("Distribución — Clima", fontweight="bold")

    # Oclusión
    occ_data = env_analysis[env_analysis["condition_type"] == "occlusion"]
    if len(occ_data) > 0:
        occ_colors = {
            "NOT_OCCLUDED": "#2ecc71",
            "PARTIALLY_OCCLUDED": "#f39c12",
            "MOSTLY_OCCLUDED": "#e74c3c",
        }
        colors_list = [
            occ_colors.get(v, "#bdc3c7") for v in occ_data["condition_value"]
        ]
        bars = axes[2].barh(
            occ_data["condition_value"], occ_data["pct_of_total"],
            color=colors_list, edgecolor="white",
        )
        axes[2].set_title("Nivel de Oclusión", fontweight="bold")
        axes[2].set_xlabel("Porcentaje (%)")
        for bar, val in zip(bars, occ_data["pct_of_total"]):
            axes[2].text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold",
            )

    plt.tight_layout()
    path = figures_dir / "05_environmental_conditions.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  ✓ Generado: %s", path.name)
    return str(path)


def _plot_difficult_objects(
    bbox_analysis: pd.DataFrame, figures_dir: Path,
) -> str:
    """06: Objetos difíciles de detectar (lejanos + diminutos)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_cls = len(bbox_analysis)

    # Lejanos
    axes[0].bar(
        bbox_analysis["class_label"], bbox_analysis["pct_distant"],
        color=sns.color_palette("Reds_r", n_cls), edgecolor="white",
    )
    axes[0].set_title(
        "Objetos Lejanos por Clase (< 0.1% área imagen)", fontweight="bold",
    )
    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_xlabel("Clase")

    # Diminutos
    axes[1].bar(
        bbox_analysis["class_label"], bbox_analysis["pct_tiny"],
        color=sns.color_palette("Oranges_r", n_cls), edgecolor="white",
    )
    axes[1].set_title(
        "Objetos Diminutos por Clase (< 16×16 px equiv.)", fontweight="bold",
    )
    axes[1].set_ylabel("Porcentaje (%)")
    axes[1].set_xlabel("Clase")

    plt.tight_layout()
    path = figures_dir / "06_difficult_objects.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  ✓ Generado: %s", path.name)
    return str(path)


# ── Generación del reporte HTML ───────────────────────────────────────────────

def _generate_html_report(
    html_path: Path,
    figure_paths: Dict[str, str],
    class_balance: pd.DataFrame,
    image_specs: pd.DataFrame,
    bbox_analysis: pd.DataFrame,
    environmental_analysis: pd.DataFrame,
) -> None:
    """Genera un reporte HTML auto-contenido con imágenes base64 embebidas."""

    # Embeber imágenes como base64
    embedded: Dict[str, str] = {}
    for name, fpath in figure_paths.items():
        with open(fpath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            embedded[name] = f"data:image/png;base64,{encoded}"

    # Construir tarjetas de métricas de objetos difíciles
    difficult_cards = ""
    for _, row in bbox_analysis.iterrows():
        difficult_cards += (
            f'<div class="metric-card">'
            f'<div class="value">{row["pct_distant"]:.1f}%</div>'
            f'<div class="label">{row["class_label"]} — Lejanos</div>'
            f'</div>\n'
        )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waymo EDA Report — Análisis Exploratorio de Datos</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            line-height: 1.6;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            text-align: center;
            padding: 3rem 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        header p {{ color: #a0a0a0; font-size: 1.1rem; }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(5px);
        }}
        .section h2 {{
            font-size: 1.5rem;
            color: #00d2ff;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(0,210,255,0.3);
        }}
        .section img {{
            width: 100%;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,210,255,0.15);
            color: #00d2ff;
            font-weight: 600;
        }}
        tr:hover {{ background: rgba(255,255,255,0.03); }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.08);
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-3px); }}
        .metric-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #00d2ff;
        }}
        .metric-card .label {{
            font-size: 0.85rem;
            color: #a0a0a0;
            margin-top: 0.3rem;
        }}
        .suggestion {{
            background: rgba(46,204,113,0.15);
            border-left: 4px solid #2ecc71;
            padding: 1rem 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }}
        .suggestion strong {{ color: #2ecc71; }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: #555;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>&#x1F697; Waymo EDA Report</h1>
            <p>Análisis Exploratorio de Datos — Dataset de Percepción</p>
            <p style="margin-top:0.5rem;font-size:0.9rem;color:#666;">
                Generado automáticamente por el pipeline de Kedro
            </p>
        </header>

        <!-- 1. Distribución de Clases -->
        <div class="section">
            <h2>&#x1F4CA; 1. Distribución de Clases</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="value">{len(class_balance)}</div>
                    <div class="label">Clases Detectadas</div>
                </div>
                <div class="metric-card">
                    <div class="value">{class_balance['imbalance_ratio_global'].iloc[0]}</div>
                    <div class="label">Ratio de Desbalance</div>
                </div>
                <div class="metric-card">
                    <div class="value">{class_balance['normalized_entropy'].iloc[0]}</div>
                    <div class="label">Entropía Normalizada</div>
                </div>
                <div class="metric-card">
                    <div class="value">{int(class_balance['count'].sum()):,}</div>
                    <div class="label">Total Anotaciones</div>
                </div>
            </div>
            <img src="{embedded.get('class_distribution', '')}"
                 alt="Distribución de Clases">
            {_df_to_html_table(class_balance[['class_label','count','percentage','suggested_weight']])}
            <div class="suggestion">
                <strong>&#x1F4A1; Sugerencia:</strong>
                {class_balance['augmentation_suggestions'].iloc[0]}
            </div>
        </div>

        <!-- 2. Especificaciones de Imagen -->
        <div class="section">
            <h2>&#x1F5BC; 2. Especificaciones de Imagen</h2>
            <img src="{embedded.get('image_resolution', '')}"
                 alt="Resolución de Imágenes">
            {_df_to_html_table(image_specs[['metric','width','height','aspect_ratio','megapixels']])}
        </div>

        <!-- 3. Bounding Boxes -->
        <div class="section">
            <h2>&#x1F4E6; 3. Análisis de Bounding Boxes</h2>
            <img src="{embedded.get('bbox_area', '')}"
                 alt="Distribución de Área de BBox">
            <img src="{embedded.get('bbox_aspect_ratio', '')}"
                 alt="Aspect Ratio de BBox">
            {_df_to_html_table(bbox_analysis)}
        </div>

        <!-- 4. Objetos Difíciles -->
        <div class="section">
            <h2>&#x1F50D; 4. Objetos Difíciles de Detectar</h2>
            <img src="{embedded.get('difficult_objects', '')}"
                 alt="Objetos Difíciles">
            <div class="metric-grid">
                {difficult_cards}
            </div>
        </div>

        <!-- 5. Condiciones Ambientales -->
        <div class="section">
            <h2>&#x1F324; 5. Condiciones Ambientales</h2>
            <img src="{embedded.get('environmental', '')}"
                 alt="Condiciones Ambientales">
            {_df_to_html_table(environmental_analysis)}
        </div>

        <footer>
            <p>Waymo EDA Pipeline &mdash; Kedro &ge; 0.19.x</p>
            <p>Reporte generado automáticamente</p>
        </footer>
    </div>
</body>
</html>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a una tabla HTML con estilos del reporte."""
    lines = ['<table>', '<thead><tr>']
    for col in df.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")
    for _, row in df.iterrows():
        lines.append("<tr>")
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                formatted = f"{val:.4f}" if abs(val) < 1 else f"{val:,.2f}"
            else:
                formatted = str(val)
            lines.append(f"<td>{formatted}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


# ── Construcción del DataFrame resumen ────────────────────────────────────────

def _build_summary_dataframe(
    class_balance: pd.DataFrame,
    image_specs: pd.DataFrame,
    bbox_analysis: pd.DataFrame,
    environmental_analysis: pd.DataFrame,
    figure_paths: Dict[str, str],
    html_path: Path,
) -> pd.DataFrame:
    """Construye el DataFrame de resumen consolidado del EDA."""
    records: list[dict] = []

    # Métricas de overview
    records.extend([
        {
            "section": "overview",
            "metric": "total_classes",
            "value": str(len(class_balance)),
        },
        {
            "section": "overview",
            "metric": "imbalance_ratio",
            "value": str(class_balance["imbalance_ratio_global"].iloc[0]),
        },
        {
            "section": "overview",
            "metric": "entropy_normalized",
            "value": str(class_balance["normalized_entropy"].iloc[0]),
        },
        {
            "section": "overview",
            "metric": "augmentation_suggestion",
            "value": class_balance["augmentation_suggestions"].iloc[0],
        },
    ])

    # Especificaciones de imagen
    rec_row = image_specs[image_specs["metric"] == "recommended_resize"]
    if len(rec_row) > 0:
        records.append({
            "section": "image_specs",
            "metric": "recommended_resize",
            "value": f"{int(rec_row['width'].iloc[0])}x{int(rec_row['height'].iloc[0])}",
        })

    all_rgb = image_specs["all_rgb"].iloc[0] if "all_rgb" in image_specs.columns else "N/A"
    records.append({
        "section": "image_specs",
        "metric": "all_rgb",
        "value": str(all_rgb),
    })

    # Bounding boxes
    for _, row in bbox_analysis.iterrows():
        records.append({
            "section": "bbox_analysis",
            "metric": f"{row['class_label']}_total_boxes",
            "value": str(int(row["total_boxes"])),
        })
        records.append({
            "section": "bbox_analysis",
            "metric": f"{row['class_label']}_pct_small",
            "value": f"{row['pct_small']}%",
        })

    # Condiciones ambientales
    for _, row in environmental_analysis.iterrows():
        records.append({
            "section": "environmental",
            "metric": f"{row['condition_type']}_{row['condition_value']}",
            "value": f"{row['pct_of_total']}%",
        })

    # Rutas de archivos generados
    for name, fpath in figure_paths.items():
        records.append({
            "section": "report_files",
            "metric": f"figure_{name}",
            "value": fpath,
        })
    records.append({
        "section": "report_files",
        "metric": "html_report",
        "value": str(html_path),
    })

    return pd.DataFrame(records)
