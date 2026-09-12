"""
src/model.py
============
Modelo de clasificación de imágenes para navegación segura.

Ofrece dos variantes configurables via ``build_model()``:

  - ``"mobilenetv2"`` (recomendado): Transfer Learning con MobileNetV2
    preentrenado en ImageNet. Congela la base y entrena solo el clasificador.
    Ideal cuando los datos son limitados (~1 K–50 K imágenes).

  - ``"cnn"`` : CNN personalizada ligera de 4 bloques convolucionales.
    Útil para experimentar sin depender de pesos preentrenados o cuando
    el dominio difiere mucho de ImageNet.

Ambas variantes:
  - Aceptan entrada [224, 224, 3] normalizada en [0, 1].
  - Usan Batch Normalization + Dropout para regularización.
  - Tienen una cabeza de clasificación completamente configurable.
"""

import logging
from typing import List, Optional, Literal

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

logger = logging.getLogger(__name__)

# ─── Tipo de modelo ───────────────────────────────────────────────────────────
ModelVariant = Literal["mobilenetv2", "cnn"]

# ─── Constantes por defecto ───────────────────────────────────────────────────
INPUT_SHAPE = (224, 224, 3)
L2_REG = 1e-4


def build_mobilenetv2(
    num_classes: int,
    input_shape: tuple = INPUT_SHAPE,
    dropout_rate: float = 0.3,
    fine_tune_at: Optional[int] = None,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Construye un clasificador con Transfer Learning sobre MobileNetV2.

    Arquitectura:
    ::

        Input (224×224×3)
          └─ MobileNetV2 [frozen] (1280-dim feature map)
               └─ GlobalAveragePooling2D
                    └─ BatchNormalization
                         └─ Dropout(dropout_rate)
                              └─ Dense(256, relu, L2)
                                   └─ BatchNormalization
                                        └─ Dropout(dropout_rate/2)
                                             └─ Dense(num_classes, softmax)

    Args:
        num_classes: Número de clases de salida.
        input_shape: Forma del tensor de entrada ``(H, W, C)``.
        dropout_rate: Tasa de dropout en la cabeza de clasificación.
        fine_tune_at: Si se especifica, descongela las capas desde este
            índice en adelante en la base MobileNetV2 para fine-tuning.
            Úsalo en una segunda fase de entrenamiento con lr muy baja.
        learning_rate: Tasa de aprendizaje inicial del optimizador Adam.

    Returns:
        Modelo Keras compilado.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ── Base preentrenada ──
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Congelar todo en fase 1

    # Fine-tuning parcial: descongelar capas desde fine_tune_at
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        logger.info(
            "Fine-tuning activado desde capa %d/%d",
            fine_tune_at, len(base_model.layers),
        )

    # ── Preprocesamiento esperado por MobileNetV2 ──
    # La entrada llega normalizada [0,1]; MobileNetV2 espera [-1,1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    # ── Extracción de features ──
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Cabeza de clasificación ──
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_1",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop_2")(x)

    # Salida: softmax para clasificación multiclase
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_Classifier")

    # ── Compilación ──
    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    logger.info(
        "MobileNetV2 construido — clases=%d | dropout=%.2f | lr=%.0e | "
        "parámetros entrenables=%s",
        num_classes, dropout_rate, learning_rate,
        f"{model.count_params():,}",
    )
    return model


def build_custom_cnn(
    num_classes: int,
    input_shape: tuple = INPUT_SHAPE,
    filters: List[int] = None,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Construye una CNN personalizada de 4 bloques convolucionales.

    Arquitectura (con ``filters=[32, 64, 128, 256]`` por defecto):
    ::

        Input (224×224×3)
          └─ Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
               └─ Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)
                    └─ Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)
                         └─ Conv2D(256, 3×3) → BN → ReLU → GlobalAvgPool
                              └─ Dense(256, relu, L2) → Dropout
                                   └─ Dense(num_classes, softmax)

    Args:
        num_classes: Número de clases de salida.
        input_shape: Forma del tensor de entrada ``(H, W, C)``.
        filters: Lista de filtros por bloque convolucional.
        dropout_rate: Tasa de dropout antes de la capa de salida.
        learning_rate: Tasa de aprendizaje del optimizador Adam.

    Returns:
        Modelo Keras compilado.
    """
    if filters is None:
        filters = [32, 64, 128, 256]

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    # ── Bloques convolucionales ──
    for i, n_filters in enumerate(filters):
        x = layers.Conv2D(
            n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f"conv_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)

        # MaxPooling en todos los bloques excepto el último (usamos GlobalAvgPool)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{i+1}")(x)

    # ── Cabeza de clasificación ──
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = keras.Model(inputs, outputs, name="CustomCNN_Classifier")

    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    logger.info(
        "CNN personalizada construida — clases=%d | bloques=%s | dropout=%.2f | "
        "parámetros=%s",
        num_classes, filters, dropout_rate,
        f"{model.count_params():,}",
    )
    return model


def build_model(
    num_classes: int,
    variant: ModelVariant = "mobilenetv2",
    **kwargs,
) -> keras.Model:
    """Factoría unificada: construye el modelo según la variante elegida.

    Args:
        num_classes: Número de clases de salida del clasificador.
        variant: ``"mobilenetv2"`` (Transfer Learning) o ``"cnn"``
            (CNN personalizada desde cero).
        **kwargs: Argumentos adicionales pasados al constructor del modelo
            (p. ej. ``dropout_rate``, ``learning_rate``, ``fine_tune_at``).

    Returns:
        Modelo Keras compilado listo para ``model.fit()``.

    Raises:
        ValueError: Si ``variant`` no es una opción válida.

    Example::

        model = build_model(num_classes=4, variant="mobilenetv2",
                            dropout_rate=0.3, learning_rate=1e-3)
        model.summary()
    """
    if variant == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, **kwargs)
    elif variant == "cnn":
        return build_custom_cnn(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Variante '{variant}' no válida. Usa 'mobilenetv2' o 'cnn'."
        )
