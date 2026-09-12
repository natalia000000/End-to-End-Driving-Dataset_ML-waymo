"""
Definición del pipeline de EDA para el dataset Waymo.

Este pipeline implementa 6 nodos secuenciales/paralelos que procesan
datos de imágenes y anotaciones para generar un reporte integral:

    raw_data → ingest_and_clean → clean_data
                                      │
                    ┌─────────────────┼─────────────────┐─────────────────┐
                    ▼                 ▼                 ▼                 ▼
            class_balance     image_specs       bbox_analysis     environmental
                    │                 │                 │                 │
                    └─────────────────┴─────────────────┴─────────────────┘
                                              │
                                      generate_report
                                              │
                                      eda_summary + plots
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    analyze_bounding_boxes,
    analyze_class_balance,
    analyze_environmental_conditions,
    analyze_image_specs,
    generate_eda_report,
    ingest_and_clean_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de EDA con los 6 nodos del análisis.

    Los nodos de análisis (2-5) se ejecutan en paralelo al no tener
    dependencias entre sí. Kedro resuelve el DAG automáticamente.

    Returns:
        Pipeline de Kedro con la secuencia completa del EDA.
    """
    return pipeline(
        [
            # ── Nodo 1: Ingesta y Limpieza ──
            node(
                func=ingest_and_clean_data,
                inputs="params:ingestion",
                outputs="clean_data",
                name="ingest_and_clean_data_node",
                tags=["ingestion", "cleaning"],
            ),
            # ── Nodo 2: Balance de Clases ──
            node(
                func=analyze_class_balance,
                inputs="clean_data",
                outputs="class_balance_metrics",
                name="analyze_class_balance_node",
                tags=["analysis", "class_balance"],
            ),
            # ── Nodo 3: Especificaciones de Imagen ──
            node(
                func=analyze_image_specs,
                inputs="clean_data",
                outputs="image_specs_metrics",
                name="analyze_image_specs_node",
                tags=["analysis", "image_specs"],
            ),
            # ── Nodo 4: Bounding Boxes ──
            node(
                func=analyze_bounding_boxes,
                inputs="clean_data",
                outputs="bbox_analysis",
                name="analyze_bounding_boxes_node",
                tags=["analysis", "bounding_boxes"],
            ),
            # ── Nodo 5: Condiciones Ambientales ──
            node(
                func=analyze_environmental_conditions,
                inputs="clean_data",
                outputs="environmental_analysis",
                name="analyze_environmental_conditions_node",
                tags=["analysis", "environmental"],
            ),
            # ── Nodo 6: Generación del Reporte ──
            node(
                func=generate_eda_report,
                inputs=[
                    "class_balance_metrics",
                    "image_specs_metrics",
                    "bbox_analysis",
                    "environmental_analysis",
                    "params:report",
                ],
                outputs="eda_summary_report",
                name="generate_eda_report_node",
                tags=["reporting"],
            ),
        ],
        tags=["data_eda"],
    )
