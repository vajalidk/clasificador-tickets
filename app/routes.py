"""
Endpoints de la API:

    POST  /clasificar             -> clasifica un ticket de soporte
    GET   /health                  -> healthcheck (incluye verificacion de DB)
    GET   /nuevo-ticket             -> formulario publico "Mesa de Ayuda"
    GET   /dashboard                 -> panel HTML con graficas (Chart.js)
    GET   /tickets                    -> listado completo con filtros
    GET   /api/estadisticas            -> datos JSON que consume el dashboard
    GET   /api/tickets                  -> listado JSON paginado/filtrable
    PATCH /api/tickets/<id>/estado       -> marcar resuelto/pendiente (admin)
    GET   /apidocs                        -> lo expone flasgger automaticamente

Las funciones de `app.db.supabase_client` se importan como modulo (no se
hace `from ... import funcion`) para que los tests puedan hacer monkeypatch
de `db.insertar_clasificacion`, `db.check_conexion`, etc. sin depender de una
conexion real a Postgres.
"""

import logging
import os

from flask import Blueprint, abort, jsonify, render_template, request

from app import limiter
from app.db import supabase_client as db
from app.ml.clasificador import clasificar

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)

MAX_TEXTO_LENGTH = 2000
MAX_NOMBRE_LENGTH = 120
MAX_CORREO_LENGTH = 150
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
            nombre:
              type: string
              example: "Ana Garcia"
            correo:
              type: string
              example: "ana@ejemplo.com"
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
    nombre = _validar_opcional(payload, "nombre", MAX_NOMBRE_LENGTH)
    correo = _validar_opcional(payload, "correo", MAX_CORREO_LENGTH)

    resultado = clasificar(texto)
    _persistir_clasificacion(texto, resultado, nombre, correo)

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


def _validar_opcional(payload, campo: str, largo_maximo: int) -> str | None:
    """Nombre/correo son opcionales: si vienen, deben ser texto y respetar
    un largo maximo; si no vienen o vienen vacios, se guarda NULL."""
    if not isinstance(payload, dict):
        return None
    valor = payload.get(campo)
    if valor is None:
        return None
    if not isinstance(valor, str):
        _abortar_400(f"El campo '{campo}' debe ser una cadena de texto.")
    valor = valor.strip()
    if not valor:
        return None
    if len(valor) > largo_maximo:
        _abortar_400(f"El campo '{campo}' excede la longitud maxima de {largo_maximo} caracteres.")
    return valor


def _abortar_400(mensaje: str):
    abort(400, description=mensaje)


def _persistir_clasificacion(
    texto: str, resultado: dict, nombre: str | None = None, correo: str | None = None
) -> None:
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
            texto,
            resultado["categoria"],
            resultado["urgencia"],
            resultado["confianza"],
            nombre=nombre,
            correo=correo,
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


@main_bp.route("/nuevo-ticket", methods=["GET"])
def endpoint_nuevo_ticket():
    """Formulario publico para enviar un ticket de soporte (pensado para
    usuarios finales, no para desarrolladores) - envia internamente al
    mismo POST /clasificar via fetch().
    ---
    tags:
      - Dashboard
    produces:
      - text/html
    responses:
      200:
        description: Pagina HTML del formulario
    """
    return render_template("nuevo_ticket.html")


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
            confianza_promedio:
              type: number
              format: float
            recientes:
              type: array
              items:
                type: object
      500:
        description: Error al consultar la base de datos
    """
    try:
        estadisticas = db.obtener_estadisticas()
    except Exception:
        logger.exception("Error obteniendo estadisticas")
        abort(500, description="No se pudieron obtener las estadisticas.")

    return jsonify(estadisticas), 200


ESTADOS_VALIDOS = {"pendiente", "resuelto"}
ORDENES_VALIDOS = {"reciente", "antiguo", "urgencia", "confianza"}
LIMITE_MAXIMO_TICKETS = 100


@main_bp.route("/tickets", methods=["GET"])
def endpoint_tickets_html():
    """Listado completo de tickets, con filtros (HTML).
    ---
    tags:
      - Dashboard
    produces:
      - text/html
    responses:
      200:
        description: Pagina HTML del listado de tickets
    """
    return render_template("tickets.html")


@main_bp.route("/api/tickets", methods=["GET"])
def endpoint_api_tickets():
    """Listado paginado y filtrable de tickets.
    ---
    tags:
      - Dashboard
    parameters:
      - in: query
        name: categoria
        type: string
        required: false
      - in: query
        name: urgencia
        type: string
        required: false
        enum: [alta, media, baja]
      - in: query
        name: estado
        type: string
        required: false
        enum: [pendiente, resuelto]
      - in: query
        name: orden
        type: string
        required: false
        enum: [reciente, antiguo, urgencia, confianza]
      - in: query
        name: page
        type: integer
        required: false
      - in: query
        name: limite
        type: integer
        required: false
    responses:
      200:
        description: Lista de tickets con metadatos de paginacion
        schema:
          type: object
          properties:
            tickets:
              type: array
              items:
                type: object
            total:
              type: integer
            limite:
              type: integer
            offset:
              type: integer
      400:
        description: Parametro de filtro/orden invalido
      500:
        description: Error al consultar la base de datos
    """
    categoria = request.args.get("categoria") or None
    urgencia = request.args.get("urgencia") or None
    estado = request.args.get("estado") or None
    orden = request.args.get("orden", "reciente")

    if urgencia and urgencia not in {"alta", "media", "baja"}:
        _abortar_400("El parametro 'urgencia' debe ser alta, media o baja.")
    if estado and estado not in ESTADOS_VALIDOS:
        _abortar_400("El parametro 'estado' debe ser pendiente o resuelto.")
    if orden not in ORDENES_VALIDOS:
        _abortar_400(f"El parametro 'orden' debe ser uno de: {', '.join(sorted(ORDENES_VALIDOS))}.")

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        _abortar_400("El parametro 'page' debe ser un numero entero.")
    try:
        limite = int(request.args.get("limite", 20))
    except ValueError:
        _abortar_400("El parametro 'limite' debe ser un numero entero.")
    limite = max(1, min(limite, LIMITE_MAXIMO_TICKETS))

    try:
        resultado = db.listar_clasificaciones(
            categoria=categoria,
            urgencia=urgencia,
            estado=estado,
            orden=orden,
            limite=limite,
            offset=(page - 1) * limite,
        )
    except Exception:
        logger.exception("Error listando tickets")
        abort(500, description="No se pudieron obtener los tickets.")

    return jsonify(resultado), 200


def _verificar_admin() -> None:
    """Gate minimo de administrador para acciones de escritura sobre
    tickets (marcar resuelto/pendiente). No es un sistema de cuentas de
    usuario completo (fuera del alcance de este proyecto de portafolio):
    es un token compartido, configurado como variable de entorno
    ADMIN_TOKEN, que el panel de administracion (/tickets) le pide al
    usuario antes de mostrar los botones de accion. Si ADMIN_TOKEN no esta
    configurado en el servidor, la funcion de administrador queda
    deshabilitada por completo (fail-safe: nunca se permite por accidente
    con un token vacio en ambos lados)."""
    token_esperado = os.environ.get("ADMIN_TOKEN")
    if not token_esperado:
        abort(503, description="La funcion de administrador no esta configurada en el servidor.")

    token_recibido = request.headers.get("X-Admin-Token", "")
    if token_recibido != token_esperado:
        abort(401, description="Token de administrador invalido.")


@main_bp.route("/api/tickets/<int:ticket_id>/estado", methods=["PATCH"])
def endpoint_actualizar_estado(ticket_id: int):
    """Marca un ticket como resuelto o pendiente. Requiere el header
    'X-Admin-Token' con el valor configurado en la variable de entorno
    ADMIN_TOKEN del servidor.
    ---
    tags:
      - Clasificacion
    consumes:
      - application/json
    parameters:
      - in: path
        name: ticket_id
        type: integer
        required: true
      - in: header
        name: X-Admin-Token
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - estado
          properties:
            estado:
              type: string
              enum: [pendiente, resuelto]
    responses:
      200:
        description: Estado actualizado
      400:
        description: Estado invalido
      401:
        description: Token de administrador invalido
      404:
        description: El ticket no existe
      503:
        description: Funcion de administrador no configurada en el servidor
    """
    _verificar_admin()

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("estado") not in ESTADOS_VALIDOS:
        _abortar_400("El campo 'estado' debe ser 'pendiente' o 'resuelto'.")

    try:
        actualizado = db.actualizar_estado_ticket(ticket_id, payload["estado"])
    except Exception:
        logger.exception("Error actualizando el estado del ticket %s", ticket_id)
        abort(500, description="No se pudo actualizar el ticket.")

    if actualizado is None:
        abort(404, description=f"No existe un ticket con id {ticket_id}.")

    return jsonify(actualizado), 200
