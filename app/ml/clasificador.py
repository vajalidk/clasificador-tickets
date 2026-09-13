"""
Carga los modelos entrenados (categoria y urgencia) y expone una unica
funcion publica, `clasificar(texto)`, usada por el endpoint /clasificar.

Los modelos se cargan una sola vez (patron singleton a nivel de modulo) y se
reutilizan en cada request: cargar un .joblib desde disco en cada
clasificacion seria un desperdicio de I/O y CPU innecesario bajo el free
tier de Cloud Run.

La version activa de los modelos se resuelve leyendo `models/latest.json`
(ver entrenar_modelo.py), nunca un nombre de archivo fijo, para que un
rollback de modelo (editar ese JSON) no requiera cambios de codigo.
"""

import json
import logging
import threading
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

# app/ml/clasificador.py -> app/ml -> app -> raiz del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
LATEST_PATH = MODELS_DIR / "latest.json"

_lock = threading.Lock()
_modelo_categoria = None
_modelo_urgencia = None
_version_cargada = None


class ModelosNoDisponiblesError(RuntimeError):
    """Se lanza cuando no se pueden cargar los modelos entrenados."""


def _cargar_modelos() -> None:
    """Carga (o recarga, si `latest.json` apunta a otra version) los dos
    modelos en memoria. Es thread-safe porque Flask + gunicorn pueden
    atender requests concurrentes con varios workers/threads."""
    global _modelo_categoria, _modelo_urgencia, _version_cargada

    if not LATEST_PATH.exists():
        raise ModelosNoDisponiblesError(
            f"No se encontro {LATEST_PATH}. Ejecuta primero: "
            "python generar_dataset.py && python entrenar_modelo.py"
        )

    with LATEST_PATH.open(encoding="utf-8") as f:
        info = json.load(f)

    version = info["version_activa"]
    if _modelo_categoria is not None and _version_cargada == version:
        return  # ya esta cargada la version activa, no hay nada que hacer

    with _lock:
        # doble chequeo dentro del lock por si otro hilo ya recargo
        if _modelo_categoria is not None and _version_cargada == version:
            return

        archivo_categoria = info["modelos"]["categoria"]["archivo"]
        archivo_urgencia = info["modelos"]["urgencia"]["archivo"]

        ruta_categoria = MODELS_DIR / archivo_categoria
        ruta_urgencia = MODELS_DIR / archivo_urgencia
        if not ruta_categoria.exists() or not ruta_urgencia.exists():
            raise ModelosNoDisponiblesError(
                f"El archivo de modelo referenciado en latest.json no existe "
                f"({ruta_categoria} / {ruta_urgencia})."
            )

        _modelo_categoria = joblib.load(ruta_categoria)
        _modelo_urgencia = joblib.load(ruta_urgencia)
        _version_cargada = version

        logger.info(
            "Modelos cargados (version=%s): categoria=%s urgencia=%s",
            version, archivo_categoria, archivo_urgencia,
        )


def clasificar(texto: str) -> dict:
    """Clasifica un ticket de soporte.

    Devuelve un dict con:
      - categoria: str, la clase predicha por el modelo de categoria.
      - urgencia: str, la clase predicha por el modelo de urgencia.
      - confianza: float (0-1, redondeado a 2 decimales), la probabilidad
        que el modelo de categoria asigna a la clase predicha. Se usa la
        confianza de categoria (no de urgencia) porque es el campo que la
        API expone como "nivel de confianza" de la clasificacion principal.
    """
    _cargar_modelos()

    proba_categoria = _modelo_categoria.predict_proba([texto])[0]
    idx = int(proba_categoria.argmax())
    categoria = str(_modelo_categoria.classes_[idx])
    confianza = round(float(proba_categoria[idx]), 2)

    urgencia = str(_modelo_urgencia.predict([texto])[0])

    return {"categoria": categoria, "urgencia": urgencia, "confianza": confianza}
