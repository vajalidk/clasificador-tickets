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
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
        _pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, dsn=_database_url())
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
    texto: str,
    categoria: str,
    urgencia: str,
    confianza: float,
    fecha_hora: datetime | None = None,
    nombre: str | None = None,
    correo: str | None = None,
) -> dict:
    """`fecha_hora` es opcional y normalmente se omite (se usa el momento
    actual). Solo se pasa explicitamente desde `seed_demo_data.py`, para
    poder distribuir tickets de demostracion en los ultimos N dias y que
    la grafica de "tickets por dia" del dashboard se vea realista.

    `nombre`/`correo` son opcionales: solo los manda el formulario
    /nuevo-ticket (seccion 2.3.1); un cliente que use la API directamente
    (Swagger, curl, integraciones) puede omitirlos sin problema."""
    fecha_hora = fecha_hora or datetime.now(timezone.utc)
    with _cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO clasificaciones
                (texto, categoria, urgencia, confianza, fecha_hora, nombre, correo)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, fecha_hora
            """,
            (texto, categoria, urgencia, confianza, fecha_hora, nombre, correo),
        )
        return dict(cur.fetchone())


def insertar_prediccion_dudosa(
    texto: str,
    categoria: str,
    urgencia: str,
    confianza: float,
    fecha_hora: datetime | None = None,
) -> dict:
    fecha_hora = fecha_hora or datetime.now(timezone.utc)
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


def obtener_estadisticas(urgencias: list[str] | None = None, dia: str | None = None) -> dict:
    """Agregaciones para el dashboard: tickets por categoria, por urgencia,
    por dia en los ultimos 7 dias, confianza promedio y las clasificaciones
    mas recientes. Todas las agregaciones se calculan en SQL (no en
    Python) para aprovechar los indices en `categoria` y `fecha_hora` y no
    traer filas de mas a la aplicacion.

    `urgencias` y `dia` habilitan el filtrado cruzado del dashboard (click
    en la dona de urgencia / en un punto de la grafica de tendencia): TODAS
    las agregaciones (categoria, urgencia, totales, confianza, recientes)
    se recalculan con esos filtros aplicados, excepto `por_dia` — esa
    grafica es el mecanismo de seleccion en si mismo, asi que siempre
    muestra los ultimos 7 dias completos (solo respeta el filtro de
    urgencia, no el de `dia`, para poder seguir eligiendo otro dia)."""
    condiciones = []
    parametros: list = []
    if urgencias:
        condiciones.append("urgencia = ANY(%s)")
        parametros.append(urgencias)
    if dia:
        condiciones.append("DATE(fecha_hora) = %s")
        parametros.append(dia)
    where = f"WHERE {' AND '.join(condiciones)}" if condiciones else ""

    condiciones_dia_chart = list(condiciones)
    parametros_dia_chart = list(parametros)
    if dia:
        # por_dia no se filtra por el dia seleccionado (ver docstring).
        condiciones_dia_chart = condiciones_dia_chart[:-1]
        parametros_dia_chart = parametros_dia_chart[:-1]

    with _cursor() as cur:
        cur.execute(
            f"""
            SELECT categoria, COUNT(*) AS total
            FROM clasificaciones
            {where}
            GROUP BY categoria
            ORDER BY total DESC
            """,
            parametros,
        )
        por_categoria = [dict(r) for r in cur.fetchall()]

        # Orden fijo (alta/media/baja) en vez de ORDER BY total DESC: el
        # dashboard le asigna un color semantico a cada urgencia (rojo,
        # naranja, verde) y necesita un orden estable para que la leyenda
        # y los colores no cambien de posicion segun los datos.
        cur.execute(
            f"""
            SELECT urgencia, COUNT(*) AS total
            FROM clasificaciones
            {where}
            GROUP BY urgencia
            ORDER BY CASE urgencia
                WHEN 'alta' THEN 1
                WHEN 'media' THEN 2
                WHEN 'baja' THEN 3
                ELSE 4
            END
            """,
            parametros,
        )
        por_urgencia = [dict(r) for r in cur.fetchall()]

        condiciones_ventana = ["fecha_hora >= (NOW() - INTERVAL '7 days')"] + condiciones_dia_chart
        cur.execute(
            f"""
            SELECT DATE(fecha_hora) AS dia, COUNT(*) AS total
            FROM clasificaciones
            WHERE {' AND '.join(condiciones_ventana)}
            GROUP BY dia
            ORDER BY dia ASC
            """,
            parametros_dia_chart,
        )
        por_dia = [{"dia": r["dia"].isoformat(), "total": r["total"]} for r in cur.fetchall()]

        cur.execute(f"SELECT COUNT(*) AS total FROM clasificaciones {where}", parametros)
        total_clasificaciones = cur.fetchone()["total"]

        cur.execute(f"SELECT COUNT(*) AS total FROM predicciones_dudosas {where}", parametros)
        total_dudosas = cur.fetchone()["total"]

        cur.execute(
            f"SELECT ROUND(AVG(confianza)::numeric, 2) AS promedio FROM clasificaciones {where}",
            parametros,
        )
        fila_promedio = cur.fetchone()["promedio"]
        confianza_promedio = float(fila_promedio) if fila_promedio is not None else None

        cur.execute(
            f"""
            SELECT id, texto, categoria, urgencia, confianza, fecha_hora, nombre, correo, estado
            FROM clasificaciones
            {where}
            ORDER BY fecha_hora DESC
            LIMIT 10
            """,
            parametros,
        )
        recientes = [_fila_ticket(r) for r in cur.fetchall()]

    return {
        "por_categoria": por_categoria,
        "por_urgencia": por_urgencia,
        "por_dia": por_dia,
        "total_clasificaciones": total_clasificaciones,
        "total_predicciones_dudosas": total_dudosas,
        "confianza_promedio": confianza_promedio,
        "recientes": recientes,
    }


def _fila_ticket(r: dict) -> dict:
    """Serializa una fila de `clasificaciones` a un dict JSON-friendly,
    compartido entre `obtener_estadisticas` (recientes) y
    `listar_clasificaciones` (listado completo con filtros)."""
    return {
        "id": r["id"],
        "texto": r["texto"],
        "categoria": r["categoria"],
        "urgencia": r["urgencia"],
        "confianza": r["confianza"],
        "fecha_hora": r["fecha_hora"].isoformat(),
        "nombre": r["nombre"],
        "correo": r["correo"],
        "estado": r["estado"],
    }


ORDENES_VALIDOS = {
    "reciente": "fecha_hora DESC",
    "antiguo": "fecha_hora ASC",
    "urgencia": (
        "CASE urgencia WHEN 'alta' THEN 1 WHEN 'media' THEN 2 WHEN 'baja' THEN 3 ELSE 4 END, "
        "fecha_hora DESC"
    ),
    "confianza": "confianza ASC",
}


def listar_clasificaciones(
    categoria: str | None = None,
    urgencia: str | None = None,
    estado: str | None = None,
    orden: str = "reciente",
    limite: int = 20,
    offset: int = 0,
) -> dict:
    """Listado paginado y filtrable de tickets para GET /tickets y
    GET /api/tickets. `orden` se valida contra ORDENES_VALIDOS (whitelist)
    antes de interpolarse en el SQL - nunca se concatena un valor que
    venga directo del usuario sin pasar por esa whitelist, para evitar
    inyeccion SQL en la clausula ORDER BY (que no admite placeholders
    %s de psycopg2)."""
    clausula_orden = ORDENES_VALIDOS.get(orden, ORDENES_VALIDOS["reciente"])

    condiciones = []
    parametros: list = []
    if categoria:
        condiciones.append("categoria = %s")
        parametros.append(categoria)
    if urgencia:
        condiciones.append("urgencia = %s")
        parametros.append(urgencia)
    if estado:
        condiciones.append("estado = %s")
        parametros.append(estado)
    where = f"WHERE {' AND '.join(condiciones)}" if condiciones else ""

    with _cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS total FROM clasificaciones {where}", parametros)
        total = cur.fetchone()["total"]

        cur.execute(
            f"""
            SELECT id, texto, categoria, urgencia, confianza, fecha_hora, nombre, correo, estado
            FROM clasificaciones
            {where}
            ORDER BY {clausula_orden}
            LIMIT %s OFFSET %s
            """,
            parametros + [limite, offset],
        )
        tickets = [_fila_ticket(r) for r in cur.fetchall()]

    return {"tickets": tickets, "total": total, "limite": limite, "offset": offset}


def actualizar_estado_ticket(ticket_id: int, estado: str) -> dict | None:
    """Marca un ticket como 'pendiente' o 'resuelto'. Devuelve None si el
    id no existe (para que el endpoint pueda responder 404)."""
    with _cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE clasificaciones
            SET estado = %s
            WHERE id = %s
            RETURNING id, estado
            """,
            (estado, ticket_id),
        )
        fila = cur.fetchone()
        return dict(fila) if fila else None
