"""
src/dataset.py
==============
Pipeline de datos desde archivos TFRecord de Waymo Open Dataset para
clasificación de objetos en navegación segura.

Formato confirmado del TFRecord real:
  - Cada record es un proto binario serializado de Waymo (~2.8 MB).
  - Contiene 8 JPEGs por record (5 cámaras frontales + 3 cámaras laterales):
      JPEG 0-4: resolución 1079×972×3  (cámaras frontales)
      JPEG 5-7: resolución  587×972×3  (cámaras laterales)
  - Los JPEGs están embebidos en el proto y se extraen por marcador binario.
  - El format NO usa tf.train.Example plano: es un proto binario Waymo.

Estrategia de parseo:
  1. Parser primario: extrae JPEGs directamente por marcador FF D8 FF / FF D9.
  2. Fallback: si el parser primario falla, activa el modo de inspección que
     imprime el esquema completo del Example para diagnóstico.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constantes ───────────────────────────────────────────────────────────────
TARGET_SIZE: Tuple[int, int] = (224, 224)
NUM_CHANNELS: int = 3

# Índice de cámara a usar por defecto (0 = frontal principal, 1079×972)
# 0–4: cámaras frontales (1079×972) | 5–7: laterales (587×972)
DEFAULT_CAMERA_IDX: int = 0

# Número de JPEGs por record confirmado
JPEGS_PER_RECORD: int = 8

# Marcadores JPEG
JPEG_SOI = b"\xff\xd8\xff"   # Start of Image
JPEG_EOI = b"\xff\xd9"       # End of Image


# ─── Feature spec estándar (fallback tf.train.Example) ────────────────────────
# Usado solo si el TFRecord tiene formato Example plano.
FEATURE_SPEC_STANDARD = {
    "image/encoded":          tf.io.FixedLenFeature([], tf.string,  default_value=""),
    "image/height":           tf.io.FixedLenFeature([], tf.int64,   default_value=0),
    "image/width":            tf.io.FixedLenFeature([], tf.int64,   default_value=0),
    "image/class/label":      tf.io.VarLenFeature(tf.int64),
}

ALTERNATIVE_IMAGE_KEYS = [
    "image/encoded", "image/image", "image_jpeg",
    "jpeg/encoded", "raw_image", "encoded_image",
]


# ═════════════════════════════════════════════════════════════════════════════
#  PARSER PRIMARIO — Proto binario Waymo (formato confirmado)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_jpegs_from_proto(raw_bytes: bytes) -> List[bytes]:
    """Extrae todos los JPEGs embebidos en un proto binario de Waymo.

    El proto contiene múltiples imágenes JPEG contiguas. Se identifican
    por sus marcadores SOI (FF D8 FF) y EOI (FF D9).

    Args:
        raw_bytes: Bytes crudos del record TFRecord.

    Returns:
        Lista de bytes de cada JPEG encontrado.
    """
    jpegs, pos = [], 0
    while True:
        start = raw_bytes.find(JPEG_SOI, pos)
        if start < 0:
            break
        end = raw_bytes.find(JPEG_EOI, start)
        if end < 0:
            break
        jpegs.append(raw_bytes[start: end + 2])
        pos = end + 2
    return jpegs


def _parse_waymo_proto_record(
    raw_record: tf.Tensor,
    camera_idx: int = DEFAULT_CAMERA_IDX,
    label: int = 0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parsea un record de Waymo proto y devuelve (imagen, etiqueta).

    Extrae el JPEG de la cámara indicada por ``camera_idx``, lo decodifica,
    redimensiona a 224×224 y normaliza en [0, 1].

    Args:
        raw_record: Tensor bytes del record serializado.
        camera_idx: Índice de cámara a extraer (0-7).
        label: Etiqueta de clase (asignada externamente si no está en el proto).

    Returns:
        Tupla ``(imagen [224, 224, 3] float32, etiqueta int32)``.
    """
    # Usar tf.numpy_function para la extracción de bytes en Python
    def extract_jpeg(raw_bytes_np):
        jpegs = _extract_jpegs_from_proto(bytes(raw_bytes_np.numpy()))
        if not jpegs or camera_idx >= len(jpegs):
            # Fallback: imagen negra si la cámara no está disponible
            import numpy as np
            return np.zeros((224, 224, 3), dtype=np.uint8).tobytes()
        return jpegs[camera_idx]

    jpeg_bytes = tf.py_function(
        func=lambda r: tf.constant(
            _extract_jpegs_from_proto(r.numpy())[camera_idx]
            if len(_extract_jpegs_from_proto(r.numpy())) > camera_idx
            else b"\xff\xd8\xff\xe0" + b"\x00" * 100
        ),
        inp=[raw_record],
        Tout=tf.string,
    )

    image = tf.io.decode_jpeg(jpeg_bytes, channels=NUM_CHANNELS)
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [*TARGET_SIZE, NUM_CHANNELS])

    label_tensor = tf.constant(label, dtype=tf.int32)
    return image, label_tensor


# ═════════════════════════════════════════════════════════════════════════════
#  PARSER SECUNDARIO — tf.train.Example plano (formato estándar TF)
# ═════════════════════════════════════════════════════════════════════════════

def _inspect_tfrecord_keys(tfrecord_path: str, n_records: int = 2) -> None:
    """Imprime todas las llaves y tipos de los primeros N records.

    Útil para diagnosticar el esquema cuando el parseo estándar falla.
    """
    logger.warning("=" * 60)
    logger.warning("MODO INSPECCIÓN — Esquema real del TFRecord")
    logger.warning("=" * 60)

    raw_ds = tf.data.TFRecordDataset(tfrecord_path)

    for i, raw_record in enumerate(raw_ds.take(n_records)):
        data = raw_record.numpy()
        logger.info("Record %d: %d bytes", i + 1, len(data))

        # Contar JPEGs directamente en los bytes
        jpegs = _extract_jpegs_from_proto(data)
        logger.info("  JPEGs embebidos en el proto: %d", len(jpegs))
        for j, jpeg in enumerate(jpegs):
            try:
                img = tf.io.decode_jpeg(jpeg)
                logger.info("    JPEG %d: %s bytes → shape %s", j, len(jpeg), img.shape)
            except Exception:
                logger.info("    JPEG %d: %s bytes → decode ERROR", j, len(jpeg))

        # Intentar también como tf.train.Example
        try:
            example = tf.train.Example()
            example.ParseFromString(data)
            for key in sorted(example.features.feature.keys()):
                feat = example.features.feature[key]
                kind = feat.WhichOneof("kind")
                if kind == "bytes_list":
                    n = len(feat.bytes_list.value)
                    sz = len(feat.bytes_list.value[0]) if n else 0
                    logger.info("  [bytes ] %-50s n=%d, sz=%d", key, n, sz)
                elif kind == "int64_list":
                    v = list(feat.int64_list.value)[:5]
                    logger.info("  [int64 ] %-50s %s", key, v)
                elif kind == "float_list":
                    v = [round(x, 4) for x in list(feat.float_list.value)[:5]]
                    logger.info("  [float ] %-50s %s", key, v)
        except Exception as e:
            logger.info("  tf.train.Example parse: %s", e)

    logger.warning("=" * 60)


def _detect_format(tfrecord_path: str) -> str:
    """Detecta si el TFRecord es formato Waymo proto o tf.train.Example.

    Returns:
        ``"waymo_proto"`` o ``"tf_example"``.
    """
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    raw_record = next(iter(raw_ds)).numpy()

    jpegs = _extract_jpegs_from_proto(raw_record)
    if len(jpegs) >= 1:
        try:
            tf.io.decode_jpeg(jpegs[0])
            logger.info(
                "Formato detectado: Waymo proto binario (%d JPEGs por record)",
                len(jpegs),
            )
            return "waymo_proto"
        except Exception:
            pass

    # Intentar tf.train.Example con llave de imagen
    try:
        example = tf.train.Example()
        example.ParseFromString(raw_record)
        for key in ALTERNATIVE_IMAGE_KEYS:
            feat = example.features.feature.get(key)
            if feat and feat.HasField("bytes_list") and feat.bytes_list.value:
                if len(feat.bytes_list.value[0]) > 1000:
                    logger.info("Formato detectado: tf.train.Example (llave='%s')", key)
                    return "tf_example"
    except Exception:
        pass

    logger.warning("Formato desconocido — activando inspección")
    _inspect_tfrecord_keys(tfrecord_path)
    return "waymo_proto"  # Asumir Waymo como default


def _parse_tf_example(
    raw_record: tf.Tensor,
    image_key: str = "image/encoded",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parsea un record como tf.train.Example plano."""
    spec = {
        image_key: tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image/class/label": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(raw_record, spec)

    image = tf.io.decode_jpeg(parsed[image_key], channels=NUM_CHANNELS)
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [*TARGET_SIZE, NUM_CHANNELS])

    labels = tf.sparse.to_dense(parsed["image/class/label"])
    label = tf.cond(
        tf.size(labels) > 0,
        true_fn=lambda: tf.cast(labels[0], tf.int32),
        false_fn=lambda: tf.constant(0, dtype=tf.int32),
    )
    return image, label


# ═════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(
    tfrecord_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    shuffle_buffer: int = 500,
    repeat: bool = True,
    camera_idx: int = DEFAULT_CAMERA_IDX,
    label: int = 0,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Carga el TFRecord de Waymo y devuelve un ``tf.data.Dataset``.

    Detecta automáticamente si el TFRecord es:
    - **Waymo proto binario** (formato de este proyecto): extrae JPEGs
      por marcador binario.
    - **tf.train.Example plano** (formato TF estándar): parsea con
      ``tf.io.parse_single_example``.

    Pipeline aplicado:
    ``TFRecordDataset → [shuffle] → map(parse+resize+normalize) → batch → prefetch``

    Args:
        tfrecord_path: Ruta al archivo ``.tfrecord``.
        batch_size: Tamaño del batch.
        shuffle: Si True, mezcla los records.
        shuffle_buffer: Tamaño del buffer de shuffle.
        repeat: Si True, repite infinitamente (necesario para ``model.fit``
            con ``steps_per_epoch``).
        camera_idx: Índice de cámara a extraer (0=frontal principal).
            0–4: frontales (1079×972) | 5–7: laterales (587×972).
        label: Etiqueta de clase asignada a todos los records (int).
            Para datos multi-clase, extiende este parser según tu esquema.
        prefetch: Batches a pre-cargar (``tf.data.AUTOTUNE`` recomendado).

    Returns:
        ``tf.data.Dataset`` con elementos ``(imagen [B,224,224,3], label [B])``.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(tfrecord_path)
    if not path.exists():
        raise FileNotFoundError(f"TFRecord no encontrado: {tfrecord_path}")

    logger.info("Cargando: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    fmt = _detect_format(tfrecord_path)

    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type="",
        num_parallel_reads=tf.data.AUTOTUNE,
    )

    if shuffle:
        raw_ds = raw_ds.shuffle(buffer_size=shuffle_buffer, seed=42)

    if fmt == "waymo_proto":
        _cam = camera_idx
        _lbl = label

        def parse_fn(raw_record):
            jpeg_bytes = tf.py_function(
                func=lambda r: tf.constant(
                    _safe_extract_jpeg(r.numpy(), _cam)
                ),
                inp=[raw_record],
                Tout=tf.string,
            )
            image = tf.io.decode_jpeg(jpeg_bytes, channels=NUM_CHANNELS)
            image = tf.image.resize(image, TARGET_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.ensure_shape(image, [*TARGET_SIZE, NUM_CHANNELS])
            return image, tf.constant(_lbl, dtype=tf.int32)

    else:
        def parse_fn(raw_record):
            return _parse_tf_example(raw_record)

    dataset = raw_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(prefetch)

    logger.info(
        "Dataset listo — formato=%s | cámara=%d | batch=%d | shuffle=%s",
        fmt, camera_idx, batch_size, shuffle,
    )
    return dataset


def _safe_extract_jpeg(raw_bytes: bytes, camera_idx: int) -> bytes:
    """Extrae el JPEG de la cámara indicada de forma segura."""
    jpegs = _extract_jpegs_from_proto(raw_bytes)
    if not jpegs:
        # Imagen JPEG negra mínima de 224×224 como fallback
        import numpy as np
        import io
        from PIL import Image  # noqa: F401 — solo si PIL está disponible
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        buf = io.BytesIO()
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(arr).save(buf, format="JPEG")
            return buf.getvalue()
        except ImportError:
            # Retornar JPEG mínimo codificado manualmente (1×1 negro)
            return (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
                b"\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06"
                b"\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b"
                b"\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
                b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\xc0"
                b"\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
                b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06"
                b"\x07\x08\x09\x0a\x0b\xff\xda\x00\x08\x01\x01\x00\x00"
                b"?\x00\xfb\xd2\x8a\x00\xff\xd9"
            )
    idx = min(camera_idx, len(jpegs) - 1)
    return jpegs[idx]


def count_records(tfrecord_path: str) -> int:
    """Cuenta el número total de records en el TFRecord.

    Cada record de Waymo contiene 8 imágenes (una por cámara).
    ``steps_per_epoch = count_records(path) // batch_size``

    Args:
        tfrecord_path: Ruta al archivo ``.tfrecord``.

    Returns:
        Número total de records (frames).
    """
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path):
        count += 1
    logger.info("Total records (frames): %d | Total imágenes: %d",
                count, count * JPEGS_PER_RECORD)
    return count
