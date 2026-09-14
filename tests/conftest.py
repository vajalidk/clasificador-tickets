"""
Fixtures compartidas para los tests de la API.

Decision clave: los tests NUNCA se conectan a un Postgres/Supabase real.
`app.db.supabase_client` (importado en app/routes.py como `db`) se
reemplaza por un `MagicMock`. Esto hace que la suite:
  - corra en CI sin necesitar credenciales ni una red externa,
  - sea rapida (sin latencia de red a una base de datos),
  - sea deterministica (no depende del estado previo de ninguna tabla).

Los tests de este proyecto verifican el CONTRATO HTTP de la API
(status codes, forma del JSON, validaciones, rate limiting), no el driver
de Postgres en si — eso es responsabilidad de pruebas de integracion
separadas (fuera del alcance de este proyecto de portafolio).
"""

from unittest.mock import MagicMock

import pytest

from app import create_app, limiter
from app import routes as routes_module


@pytest.fixture
def mock_db(monkeypatch):
    """Sustituye el modulo de base de datos usado por app/routes.py."""
    mock = MagicMock()
    mock.check_conexion.return_value = True
    mock.obtener_estadisticas.return_value = {
        "por_categoria": [],
        "por_urgencia": [],
        "por_dia": [],
        "total_clasificaciones": 0,
        "total_predicciones_dudosas": 0,
        "confianza_promedio": None,
        "recientes": [],
    }
    mock.listar_clasificaciones.return_value = {
        "tickets": [],
        "total": 0,
        "limite": 20,
        "offset": 0,
    }
    mock.actualizar_estado_ticket.return_value = {"id": 1, "estado": "resuelto"}
    monkeypatch.setattr(routes_module, "db", mock)
    return mock


@pytest.fixture
def app(mock_db):
    """Crea una instancia aislada de la app para cada test (application
    factory pattern), con el mock de base de datos ya inyectado."""
    flask_app = create_app({"TESTING": True})
    yield flask_app


@pytest.fixture
def client(app):
    """Cliente HTTP de pruebas. Resetea el estado del rate limiter antes de
    cada test: Flask-Limiter usa almacenamiento en memoria a nivel de
    proceso, compartido entre todas las apps creadas en el mismo test run,
    asi que sin este reset un test podria heredar el contador de llamadas
    de un test anterior y fallar de forma intermitente."""
    limiter.reset()
    return app.test_client()
