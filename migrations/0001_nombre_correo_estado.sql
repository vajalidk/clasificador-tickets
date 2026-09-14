-- Migracion para bases de datos que ya tenian la tabla `clasificaciones`
-- creada antes de agregar nombre/correo/estado (ver init_db.sql para el
-- esquema completo en un proyecto nuevo). Idempotente: se puede correr
-- mas de una vez sin error.

ALTER TABLE clasificaciones ADD COLUMN IF NOT EXISTS nombre TEXT;
ALTER TABLE clasificaciones ADD COLUMN IF NOT EXISTS correo TEXT;
ALTER TABLE clasificaciones ADD COLUMN IF NOT EXISTS estado TEXT NOT NULL DEFAULT 'pendiente';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'clasificaciones_estado_check'
    ) THEN
        ALTER TABLE clasificaciones
            ADD CONSTRAINT clasificaciones_estado_check
            CHECK (estado IN ('pendiente', 'resuelto'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_clasificaciones_estado ON clasificaciones (estado);
