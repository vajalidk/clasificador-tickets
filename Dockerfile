# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------
# Etapa 1: build
# Instala las dependencias en un virtualenv aislado. Se hace en una etapa
# separada para que las herramientas de compilacion (si pip necesitara
# alguna) no terminen en la imagen final: solo se copia el resultado
# (/opt/venv), nunca el cache de pip ni artefactos intermedios.
# ---------------------------------------------------------------------
FROM python:3.12-slim AS build

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------
# Etapa 2: runtime
# Imagen final, minima: solo el virtualenv ya resuelto, el codigo de la
# app y los modelos entrenados. Corre como usuario no-root (buena
# practica de seguridad: si un dia se descubre una vulnerabilidad remota
# en el proceso, no corre con privilegios de root dentro del contenedor).
# ---------------------------------------------------------------------
FROM python:3.12-slim

RUN groupadd --system appuser && \
    useradd --system --gid appuser --home /app --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Se copian los modelos ya entrenados (generar_dataset.py + entrenar_modelo.py
# se corrieron antes de construir la imagen y sus .joblib estan versionados
# en el repo, ver .gitignore). Esto hace que el build de Docker sea rapido y
# 100% determinista: no depende de reentrenar un modelo con aleatoriedad
# dentro del propio build.
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser run.py ./run.py
COPY --chown=appuser:appuser models ./models

USER appuser

# Cloud Run inyecta $PORT en tiempo de ejecucion (normalmente 8080); el
# default de abajo solo se usa para `docker run` local sin -e PORT=....
ENV PORT=8080
EXPOSE 8080

# Forma shell (no exec) a proposito: es la unica forma de que $PORT se
# expanda en tiempo de arranque del contenedor.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 run:app
