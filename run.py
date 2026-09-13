"""
Punto de entrada de la aplicacion.

- Desarrollo local: `python run.py` (usa el servidor de desarrollo de Flask).
- Produccion (Docker/Cloud Run): `gunicorn --bind 0.0.0.0:$PORT run:app`
  (ver Dockerfile). gunicorn importa el objeto `app` definido aqui, por lo
  que el bloque `if __name__ == "__main__"` de abajo nunca se ejecuta en
  produccion.
"""

import os

from dotenv import load_dotenv

# Carga variables de entorno desde .env en desarrollo local. En Cloud Run
# las variables se inyectan directamente en el entorno del contenedor, por
# lo que este archivo simplemente no existira y load_dotenv() no hace nada.
load_dotenv()

from app import create_app  # noqa: E402 (import despues de load_dotenv a proposito)

app = create_app()

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=puerto, debug=debug)
