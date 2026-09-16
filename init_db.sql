-- ---------------------------------------------------------------------
-- Esquema de base de datos para el Clasificador de Tickets de Soporte.
-- Ejecutar una sola vez sobre el proyecto de Supabase (o cualquier
-- Postgres), por ejemplo desde el SQL Editor de supabase.com.
--
-- Tablas:
--   clasificaciones      -> historial de TODAS las clasificaciones hechas
--   predicciones_dudosas -> copia de las clasificaciones con confianza < 0.6,
--                           pensada como cola de revision humana para el
--                           reentrenamiento periodico (ver README, plan de
--                           mantenimiento del modelo).
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS clasificaciones (
    id          BIGSERIAL PRIMARY KEY,
    texto       TEXT NOT NULL,
    categoria   TEXT NOT NULL,
    urgencia    TEXT NOT NULL,
    confianza   REAL NOT NULL CHECK (confianza >= 0 AND confianza <= 1),
    fecha_hora  TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
    -- Datos opcionales de quien envia el ticket (formulario /nuevo-ticket).
    nombre      TEXT,
    correo      TEXT,
    -- Titulo corto opcional (campo "Asunto" del formulario). Si viene vacio,
    -- la app genera uno automaticamente a partir de la primera oracion de
    -- `texto` (ver _generar_titulo en app/db/supabase_client.py).
    asunto      TEXT,
    -- Ciclo de vida del ticket: 'pendiente' -> 'resuelto'.
    estado      TEXT NOT NULL DEFAULT 'pendiente' CHECK (estado IN ('pendiente', 'resuelto')),
    -- Se llena al marcar 'resuelto' y se limpia si se revierte a
    -- 'pendiente' (ver actualizar_estado_ticket).
    fecha_resuelto TIMESTAMPTZ
);

-- Los filtros mas comunes del dashboard/listado son "agregar por
-- categoria", "agregar por rango de fechas" y "filtrar por estado".
CREATE INDEX IF NOT EXISTS idx_clasificaciones_fecha_hora
    ON clasificaciones (fecha_hora);

CREATE INDEX IF NOT EXISTS idx_clasificaciones_categoria
    ON clasificaciones (categoria);

CREATE INDEX IF NOT EXISTS idx_clasificaciones_estado
    ON clasificaciones (estado);


CREATE TABLE IF NOT EXISTS predicciones_dudosas (
    id          BIGSERIAL PRIMARY KEY,
    texto       TEXT NOT NULL,
    categoria   TEXT NOT NULL,
    urgencia    TEXT NOT NULL,
    confianza   REAL NOT NULL CHECK (confianza >= 0 AND confianza <= 1),
    fecha_hora  TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
    -- Campos para el flujo de revision manual: se completan cuando un
    -- humano revisa el caso (ver plan de mantenimiento del modelo).
    revisado         BOOLEAN NOT NULL DEFAULT FALSE,
    categoria_correcta TEXT,
    urgencia_correcta  TEXT
);

CREATE INDEX IF NOT EXISTS idx_predicciones_dudosas_fecha_hora
    ON predicciones_dudosas (fecha_hora);

CREATE INDEX IF NOT EXISTS idx_predicciones_dudosas_revisado
    ON predicciones_dudosas (revisado);
