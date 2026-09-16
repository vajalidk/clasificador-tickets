-- Migracion para bases de datos que ya tenian la tabla `clasificaciones`
-- creada antes de agregar el campo "asunto" (titulo corto opcional del
-- ticket). Idempotente: se puede correr mas de una vez sin error.

ALTER TABLE clasificaciones ADD COLUMN IF NOT EXISTS asunto TEXT;
