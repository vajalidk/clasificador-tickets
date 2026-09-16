-- Migracion para bases de datos que ya tenian la tabla `clasificaciones`
-- creada antes de agregar `fecha_resuelto` (se llena al marcar un ticket
-- como resuelto, se limpia si se revierte a pendiente). Idempotente.

ALTER TABLE clasificaciones ADD COLUMN IF NOT EXISTS fecha_resuelto TIMESTAMPTZ;
