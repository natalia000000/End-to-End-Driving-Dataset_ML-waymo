"""Registro de pipelines del proyecto — API estándar Kedro ≥ 0.19."""

from kedro.pipeline import Pipeline

from waymo_eda.pipelines.data_eda.pipeline import create_pipeline as create_data_eda_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Registra todos los pipelines disponibles en el proyecto.

    Returns:
        Diccionario con pipelines nombrados. ``__default__`` se ejecuta
        cuando se invoca ``kedro run`` sin especificar un pipeline.
    """
    data_eda = create_data_eda_pipeline()

    return {
        "__default__": data_eda,
        "data_eda": data_eda,
    }
