"""
Cliente de base de datos (Postgres/Supabase) para las tablas `clasificaciones`
y `predicciones_dudosas`.

Todas las funciones publicas de este modulo son las que usa `app/routes.py`.
En los tests (ver tests/conftest.py) estas funciones se reemplazan por mocks,
por lo que el resto de la aplicacion nunca depende de una conexion real a
Postgres para poder ejecutarse en CI.

La cadena de conexion se lee siempre de la variable de entorno
`DATABASE_URL` (nunca hardcodeada). Se usa un pool de conexiones pequeno
(1-5) porque en Cloud Run cada instancia del contenedor puede atender varias
requests concurrentes con pocos workers/threads de gunicorn.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import psycopg2.pool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.SimpleConnectionPool | None = None


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "La variable de entorno DATABASE_URL no esta configurada. "
            "Revisa tu .env (ver .env.example)."
        )
    return url


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=5, dsn=_database_url()
        )
        logger.info("Pool de conexiones a Postgres inicializado")
    return _pool


# Reintentos con backoff exponencial: Supabase (como cualquier Postgres
# gestionado en un tier gratuito) puede tener caidas breves de red o
# "cold starts" tras periodos de inactividad. 3 intentos con espera creciente
# (1s, 2s, 4s) cubre esos casos sin bloquear la respuesta HTTP por demasiado
# tiempo.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(psycopg2.OperationalError),
    reraise=True,
)
def _obtener_conexion():
    return _get_pool().getconn()


def _liberar_conexion(conn) -> None:
    if _pool is not None and conn is not None:
        _pool.putconn(conn)


@contextmanager
def _cursor(commit: bool = False):
    conn = _obtener_conexion()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        _liberar_conexion(conn)


def check_conexion() -> bool:
    """Usado por GET /health. Lanza una excepcion si la conexion falla."""
    with _cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
    return True


def insertar_clasificacion(
    texto: str, categoria: str, urgencia: str, confianza: float
) -> dict:
    fecha_hora = datetime.now(timezone.utc)
    with _cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO clasificaciones (texto, categoria, urgencia, confianza, fecha_hora)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, fecha_hora
            """,
            (texto, categoria, urgencia, confianza, fecha_hora),
        )
        return dict(cur.fetchone())


def insertar_prediccion_dudosa(
    texto: str, categoria: str, urgencia: str, confianza: float
) -> dict:
    fecha_hora = datetime.now(timezone.utc)
    with _cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO predicciones_dudosas (texto, categoria, urgencia, confianza, fecha_hora)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, fecha_hora
            """,
            (texto, categoria, urgencia, confianza, fecha_hora),
        )
        return dict(cur.fetchone())


def obtener_estadisticas() -> dict:
    """Agregaciones para el dashboard: tickets por categoria, por urgencia,
    y por dia en los ultimos 7 dias. Todas las agregaciones se calculan en
    SQL (no en Python) para aprovechar los indices en `categoria` y
    `fecha_hora` y no traer filas de mas a la aplicacion."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT categoria, COUNT(*) AS total
            FROM clasificaciones
            GROUP BY categoria
            ORDER BY total DESC
            """
        )
        por_categoria = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT urgencia, COUNT(*) AS total
            FROM clasificaciones
            GROUP BY urgencia
            ORDER BY total DESC
            """
        )
        por_urgencia = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT DATE(fecha_hora) AS dia, COUNT(*) AS total
            FROM clasificaciones
            WHERE fecha_hora >= (NOW() - INTERVAL '7 days')
            GROUP BY dia
            ORDER BY dia ASC
            """
        )
        por_dia = [
            {"dia": r["dia"].isoformat(), "total": r["total"]} for r in cur.fetchall()
        ]

        cur.execute("SELECT COUNT(*) AS total FROM clasificaciones")
        total_clasificaciones = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS total FROM predicciones_dudosas")
        total_dudosas = cur.fetchone()["total"]

    return {
        "por_categoria": por_categoria,
        "por_urgencia": por_urgencia,
        "por_dia": por_dia,
        "total_clasificaciones": total_clasificaciones,
        "total_predicciones_dudosas": total_dudosas,
    }
