"""
train.py
========
Script principal de entrenamiento para el clasificador de objetos en
navegación segura con TensorFlow/Keras.

Uso:
    python train.py

    # Con opciones:
    python train.py --tfrecord data/train.tfrecord \\
                    --num_classes 4 \\
                    --model mobilenetv2 \\
                    --epochs 20 \\
                    --batch_size 32

Flujo:
    1. Parsea argumentos de configuración.
    2. Cuenta los records para calcular steps_per_epoch.
    3. Carga el dataset desde TFRecord (src/dataset.py).
    4. Construye el modelo (src/model.py).
    5. Configura callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
       TensorBoard.
    6. Ejecuta model.fit() — Fase 1 (base congelada).
    7. (Opcional) Fine-tuning — Fase 2 (base parcialmente descongelada).
    8. Guarda el modelo final en SavedModel format.
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Asegurar que src/ esté en el path de Python ──
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
from src.dataset import load_dataset, count_records
from src.model import build_model

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN POR DEFECTO
# ═════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    # ── Datos ──
    "tfrecord":     "data/train.tfrecord",   # Ruta al TFRecord de entrenamiento
    "val_tfrecord": None,                    # Ruta al TFRecord de validación (opcional)
    "image_key":    None,                    # None = detección automática

    # ── Modelo ──
    "model":        "mobilenetv2",           # "mobilenetv2" | "cnn"
    "num_classes":  4,                       # Nº de clases (ajusta a tu dataset)
    "dropout":      0.3,
    "learning_rate": 1e-3,

    # ── Entrenamiento ──
    "batch_size":   32,
    "epochs":       20,                      # Fase 1: base congelada
    "fine_tune":    False,                   # Activar Fase 2 fine-tuning
    "fine_tune_epochs": 10,                  # Épocas adicionales de fine-tuning
    "fine_tune_at": 100,                     # Capa desde la que descongelar
    "fine_tune_lr": 1e-5,                    # LR muy baja para fine-tuning

    # ── Salidas ──
    "output_dir":   "outputs",              # Dir para checkpoints y logs
    "model_name":   "safe_nav_classifier",
}


# ═════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento — Clasificador de Objetos para Navegación Segura",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Datos
    data_grp = parser.add_argument_group("Datos")
    data_grp.add_argument("--tfrecord",     default=DEFAULTS["tfrecord"],
                          help="Ruta al TFRecord de entrenamiento")
    data_grp.add_argument("--val_tfrecord", default=DEFAULTS["val_tfrecord"],
                          help="Ruta al TFRecord de validación (opcional)")
    data_grp.add_argument("--image_key",    default=DEFAULTS["image_key"],
                          help="Llave de la feature de imagen (None=auto)")

    # Modelo
    model_grp = parser.add_argument_group("Modelo")
    model_grp.add_argument("--model",        default=DEFAULTS["model"],
                           choices=["mobilenetv2", "cnn"],
                           help="Arquitectura del modelo")
    model_grp.add_argument("--num_classes",  default=DEFAULTS["num_classes"],
                           type=int, help="Número de clases de salida")
    model_grp.add_argument("--dropout",      default=DEFAULTS["dropout"],
                           type=float, help="Tasa de dropout")
    model_grp.add_argument("--learning_rate", default=DEFAULTS["learning_rate"],
                           type=float, help="Learning rate inicial")

    # Entrenamiento
    train_grp = parser.add_argument_group("Entrenamiento")
    train_grp.add_argument("--batch_size",   default=DEFAULTS["batch_size"],
                           type=int, help="Tamaño de batch")
    train_grp.add_argument("--epochs",       default=DEFAULTS["epochs"],
                           type=int, help="Épocas de entrenamiento (Fase 1)")
    train_grp.add_argument("--fine_tune",    action="store_true",
                           help="Activar fine-tuning (Fase 2)")
    train_grp.add_argument("--fine_tune_epochs", default=DEFAULTS["fine_tune_epochs"],
                           type=int, help="Épocas adicionales de fine-tuning")
    train_grp.add_argument("--fine_tune_at", default=DEFAULTS["fine_tune_at"],
                           type=int, help="Índice de capa desde la que descongelar")
    train_grp.add_argument("--fine_tune_lr", default=DEFAULTS["fine_tune_lr"],
                           type=float, help="Learning rate para fine-tuning")

    # Salidas
    out_grp = parser.add_argument_group("Salidas")
    out_grp.add_argument("--output_dir",  default=DEFAULTS["output_dir"],
                         help="Directorio de salida para checkpoints y logs")
    out_grp.add_argument("--model_name",  default=DEFAULTS["model_name"],
                         help="Nombre base del modelo guardado")

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

def build_callbacks(output_dir: Path, model_name: str) -> list:
    """Configura los callbacks de Keras para el entrenamiento.

    Callbacks incluidos:
    - **ModelCheckpoint**: Guarda el mejor modelo según val_loss.
    - **EarlyStopping**: Detiene si val_loss no mejora en 5 épocas.
    - **ReduceLROnPlateau**: Reduce LR x0.3 si val_loss estanca 3 épocas.
    - **TensorBoard**: Logs en ``output_dir/logs/`` para visualización.

    Args:
        output_dir: Directorio raíz de salidas.
        model_name: Nombre base del archivo de checkpoint.

    Returns:
        Lista de callbacks de Keras.
    """
    ckpt_dir = output_dir / "checkpoints"
    log_dir  = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{model_name}_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss" if True else "loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if True else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if True else "loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
    ]

    logger.info("Callbacks configurados:")
    logger.info("  ✓ ModelCheckpoint → %s", ckpt_path)
    logger.info("  ✓ EarlyStopping  (patience=5)")
    logger.info("  ✓ ReduceLROnPlateau (factor=0.3, patience=3)")
    logger.info("  ✓ TensorBoard → %s", log_dir)

    return callbacks


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Función principal de entrenamiento."""
    args = parse_args()

    # ── Resumen de configuración ──
    logger.info("=" * 60)
    logger.info("CLASIFICADOR DE OBJETOS — NAVEGACIÓN SEGURA")
    logger.info("TensorFlow %s | GPU disponibles: %s",
                tf.__version__,
                tf.config.list_physical_devices("GPU") or "ninguna (CPU)")
    logger.info("=" * 60)
    logger.info("Configuración:")
    for k, v in vars(args).items():
        logger.info("  %-20s = %s", k, v)
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Contar records para steps_per_epoch ──
    logger.info("Contando records en el TFRecord de entrenamiento…")
    n_train = count_records(args.tfrecord)
    steps_per_epoch = max(1, n_train // args.batch_size)

    n_val = 0
    val_dataset = None
    validation_steps = None

    if args.val_tfrecord and Path(args.val_tfrecord).exists():
        n_val = count_records(args.val_tfrecord)
        validation_steps = max(1, n_val // args.batch_size)
        logger.info("Records: train=%d | val=%d", n_train, n_val)
    else:
        logger.warning(
            "No se especificó dataset de validación o no existe. "
            "El entrenamiento correrá sin validación. "
            "Los callbacks de monitor usarán 'loss' en lugar de 'val_loss'."
        )
        logger.info("Records de entrenamiento: %d", n_train)

    # ── 2. Cargar datasets ──
    train_dataset = load_dataset(
        tfrecord_path=args.tfrecord,
        batch_size=args.batch_size,
        shuffle=True,
        repeat=True,
        camera_idx=0,    # Cámara frontal principal (1079×972)
        label=0,         # Etiqueta base; extiende para multi-clase
    )

    if args.val_tfrecord and Path(args.val_tfrecord).exists():
        val_dataset = load_dataset(
            tfrecord_path=args.val_tfrecord,
            batch_size=args.batch_size,
            shuffle=False,
            repeat=True,
            camera_idx=0,
            label=0,
        )

    # ── 3. Construir modelo ──
    logger.info("Construyendo modelo '%s' con %d clases…", args.model, args.num_classes)
    model = build_model(
        num_classes=args.num_classes,
        variant=args.model,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
    )
    model.summary(print_fn=logger.info)

    # ── 4. Configurar callbacks ──
    callbacks = build_callbacks(output_dir, args.model_name)

    # Ajustar monitor si no hay validación
    if val_dataset is None:
        for cb in callbacks:
            if hasattr(cb, "monitor"):
                cb.monitor = cb.monitor.replace("val_loss", "loss")
                cb.monitor = cb.monitor.replace("val_accuracy", "accuracy")

    # ── 5. Entrenamiento — Fase 1: Base congelada ──
    logger.info("")
    logger.info("━" * 60)
    logger.info("FASE 1 — Entrenamiento con base congelada")
    logger.info("  Épocas: %d | steps_per_epoch: %d", args.epochs, steps_per_epoch)
    logger.info("━" * 60)

    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Fase 1 completada.")
    logger.info(
        "  Mejor loss: %.4f | Mejor accuracy: %.4f",
        min(history.history.get("val_loss", history.history["loss"])),
        max(history.history.get("val_accuracy", history.history["accuracy"])),
    )

    # ── 6. Fine-tuning — Fase 2 (solo MobileNetV2) ──
    if args.fine_tune and args.model == "mobilenetv2":
        logger.info("")
        logger.info("━" * 60)
        logger.info("FASE 2 — Fine-tuning (capas ≥ %d descongeladas)", args.fine_tune_at)
        logger.info("  LR: %.0e | Épocas adicionales: %d",
                    args.fine_tune_lr, args.fine_tune_epochs)
        logger.info("━" * 60)

        # Reconstruir con fine-tuning activado
        model = build_model(
            num_classes=args.num_classes,
            variant="mobilenetv2",
            dropout_rate=args.dropout,
            learning_rate=args.fine_tune_lr,    # LR muy reducida
            fine_tune_at=args.fine_tune_at,
        )

        # Cargar los mejores pesos de Fase 1
        best_ckpt = output_dir / "checkpoints" / f"{args.model_name}_best.keras"
        if best_ckpt.exists():
            model.load_weights(str(best_ckpt))
            logger.info("Pesos cargados desde: %s", best_ckpt)

        callbacks_ft = build_callbacks(
            output_dir / "fine_tune", f"{args.model_name}_finetune"
        )
        if val_dataset is None:
            for cb in callbacks_ft:
                if hasattr(cb, "monitor"):
                    cb.monitor = cb.monitor.replace("val_loss", "loss")

        history_ft = model.fit(
            train_dataset,
            epochs=args.fine_tune_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks_ft,
            verbose=1,
        )

        logger.info("Fine-tuning completado.")
    elif args.fine_tune and args.model == "cnn":
        logger.warning(
            "Fine-tuning solo disponible para MobileNetV2. "
            "Omitiendo Fase 2 para el modelo CNN personalizado."
        )

    # ── 7. Guardar modelo final ──
    # Keras 3 requiere extensión .keras para el formato nativo.
    # Usa model.export() si necesitas formato SavedModel (TFLite/TFServing).
    saved_model_path = output_dir / f"{args.model_name}.keras"
    model.save(str(saved_model_path))
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Modelo guardado en: %s", saved_model_path)
    logger.info("  Para cargarlo:   tf.keras.models.load_model('%s')", saved_model_path)
    logger.info("  Para TFServing:  model.export('outputs/saved_model/')")
    logger.info("  Para TensorBoard: tensorboard --logdir %s", output_dir / "logs")
    logger.info("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
