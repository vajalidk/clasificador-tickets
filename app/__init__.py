"""
Application factory de la API. Se usa el patron "application factory"
(en vez de crear el objeto `Flask` a nivel de modulo) para poder construir
multiples instancias de la app con configuraciones distintas en los tests
(por ejemplo, con rate limiting deshabilitado) sin efectos secundarios entre
ellas.
"""

import logging
import os

from flasgger import Swagger
from flask import Flask, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

# `limiter` se define a nivel de modulo (sin `app` todavia) para que
# app/routes.py pueda importarlo con `from app import limiter` y decorar
# sus endpoints con @limiter.limit(...). Se enlaza a la instancia real de
# Flask despues, dentro de create_app(), con limiter.init_app(app).
limiter = Limiter(key_func=get_remote_address, default_limits=[])


def _configurar_logging() -> None:
    """Logging estructurado a stdout: nivel configurable por entorno
    (INFO en produccion) y timestamp en cada linea. Nunca se usa `print`
    para eventos de la aplicacion, tanto para poder filtrar por nivel como
    para que Cloud Run capture correctamente los logs con severidad."""
    nivel = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def create_app(config_overrides: dict | None = None) -> Flask:
    _configurar_logging()
    logger = logging.getLogger(__name__)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024  # 100 KB, cota generosa para el body JSON

    if config_overrides:
        app.config.update(config_overrides)

    limiter.init_app(app)

    app.config.setdefault("SWAGGER", {
        "title": "Clasificador de Tickets de Soporte - API",
        "uiversion": 3,
        "specs_route": "/apidocs/",
    })
    Swagger(app)

    # Import diferido para evitar import circular: app/routes.py hace
    # `from app import limiter`, lo que requiere que este modulo ya se haya
    # ejecutado hasta la linea de `limiter = Limiter(...)` de arriba.
    from app.routes import main_bp

    app.register_blueprint(main_bp)

    _registrar_manejadores_error(app)

    logger.info("Aplicacion Flask inicializada (entorno=%s)", os.environ.get("FLASK_ENV", "production"))
    return app


def _registrar_manejadores_error(app: Flask) -> None:
    """Manejador de errores centralizado: cualquier error (validacion,
    rate limit, 404, o una excepcion no controlada) devuelve SIEMPRE JSON
    consistente `{"error": "..."}`, nunca una pagina HTML de error."""

    @app.errorhandler(HTTPException)
    def manejar_http_exception(error: HTTPException):
        return jsonify({"error": error.description or error.name}), error.code

    @app.errorhandler(Exception)
    def manejar_excepcion_generica(error: Exception):
        logging.getLogger(__name__).exception("Error no controlado")
        return jsonify({"error": "Error interno del servidor"}), 500
