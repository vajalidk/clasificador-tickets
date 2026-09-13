"""
Endpoints de la API:

    POST /clasificar         -> clasifica un ticket de soporte
    GET  /health              -> healthcheck (incluye verificacion de DB)
    GET  /dashboard            -> panel HTML con graficas (Chart.js)
    GET  /api/estadisticas      -> datos JSON que consume el dashboard
    GET  /apidocs               -> lo expone flasgger automaticamente

Las funciones de `app.db.supabase_client` se importan como modulo (no se
hace `from ... import funcion`) para que los tests puedan hacer monkeypatch
de `db.insertar_clasificacion`, `db.check_conexion`, etc. sin depender de una
conexion real a Postgres.
"""

import logging

from flask import Blueprint, abort, jsonify, render_template, request

from app import limiter
from app.db import supabase_client as db
from app.ml.clasificador import clasificar

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)

MAX_TEXTO_LENGTH = 2000
UMBRAL_CONFIANZA_DUDOSA = 0.6


@main_bp.route("/clasificar", methods=["POST"])
@limiter.limit("20 per minute")
def endpoint_clasificar():
    """Clasifica un ticket de soporte al cliente.
    ---
    tags:
      - Clasificacion
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - texto
          properties:
            texto:
              type: string
              example: "No puedo acceder a mi cuenta desde ayer, es urgente"
    responses:
      200:
        description: Clasificacion exitosa
        schema:
          type: object
          properties:
            categoria:
              type: string
              example: "soporte técnico"
            urgencia:
              type: string
              example: "alta"
            confianza:
              type: number
              format: float
              example: 0.87
      400:
        description: Texto invalido (ausente, vacio o demasiado largo)
      429:
        description: Se supero el limite de 20 solicitudes por minuto
    """
    payload = request.get_json(silent=True)
    texto = _validar_texto(payload)

    resultado = clasificar(texto)
    _persistir_clasificacion(texto, resultado)

    return jsonify(resultado), 200


def _validar_texto(payload) -> str:
    """Valida el body de /clasificar y devuelve el texto ya saneado, o
    aborta la request con 400 si la validacion falla."""
    if not isinstance(payload, dict) or "texto" not in payload:
        _abortar_400("El campo 'texto' es obligatorio.")

    texto = payload["texto"]
    if not isinstance(texto, str):
        _abortar_400("El campo 'texto' debe ser una cadena de texto.")

    texto = texto.strip()
    if not texto:
        _abortar_400("El campo 'texto' no puede estar vacio.")

    if len(texto) > MAX_TEXTO_LENGTH:
        _abortar_400(
            f"El campo 'texto' excede la longitud maxima de " f"{MAX_TEXTO_LENGTH} caracteres."
        )

    return texto


def _abortar_400(mensaje: str):
    abort(400, description=mensaje)


def _persistir_clasificacion(texto: str, resultado: dict) -> None:
    """Guarda la clasificacion en `clasificaciones` y, si la confianza es
    baja, tambien en `predicciones_dudosas` para revision humana posterior
    (ver plan de mantenimiento del modelo en el README).

    Un fallo de base de datos se registra en logs pero NO hace fallar la
    request: el ticket ya fue clasificado correctamente y no tiene sentido
    devolver un error 500 al usuario por un problema de persistencia que no
    afecta el resultado que esta pidiendo.
    """
    try:
        db.insertar_clasificacion(
            texto, resultado["categoria"], resultado["urgencia"], resultado["confianza"]
        )
        if resultado["confianza"] < UMBRAL_CONFIANZA_DUDOSA:
            db.insertar_prediccion_dudosa(
                texto,
                resultado["categoria"],
                resultado["urgencia"],
                resultado["confianza"],
            )
            logger.info(
                "Clasificacion con confianza baja (%.2f) registrada para revision manual",
                resultado["confianza"],
            )
    except Exception:
        logger.exception("No se pudo persistir la clasificacion en la base de datos")


@main_bp.route("/health", methods=["GET"])
def endpoint_health():
    """Healthcheck del servicio, incluyendo la conexion a la base de datos.
    ---
    tags:
      - Sistema
    responses:
      200:
        description: El servicio y la base de datos responden correctamente
        schema:
          type: object
          properties:
            status:
              type: string
              example: "ok"
      503:
        description: La base de datos no responde
        schema:
          type: object
          properties:
            status:
              type: string
              example: "error"
            detalle:
              type: string
    """
    try:
        db.check_conexion()
    except Exception as e:
        logger.error("Healthcheck fallo: %s", e)
        return jsonify({"status": "error", "detalle": str(e)}), 503

    return jsonify({"status": "ok"}), 200


@main_bp.route("/dashboard", methods=["GET"])
def endpoint_dashboard():
    """Panel de estadisticas (HTML + Chart.js).
    ---
    tags:
      - Dashboard
    produces:
      - text/html
    responses:
      200:
        description: Pagina HTML del dashboard
    """
    return render_template("dashboard.html")


@main_bp.route("/api/estadisticas", methods=["GET"])
def endpoint_estadisticas():
    """Datos agregados que consume el dashboard.
    ---
    tags:
      - Dashboard
    responses:
      200:
        description: Estadisticas agregadas de clasificaciones
        schema:
          type: object
          properties:
            por_categoria:
              type: array
              items:
                type: object
            por_urgencia:
              type: array
              items:
                type: object
            por_dia:
              type: array
              items:
                type: object
            total_clasificaciones:
              type: integer
            total_predicciones_dudosas:
              type: integer
      500:
        description: Error al consultar la base de datos
    """
    try:
        estadisticas = db.obtener_estadisticas()
    except Exception:
        logger.exception("Error obteniendo estadisticas")
        abort(500, description="No se pudieron obtener las estadisticas.")

    return jsonify(estadisticas), 200
