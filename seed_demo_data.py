"""
Genera tickets de ejemplo variados, los clasifica con el modelo real
(`app.ml.clasificador`) y los inserta directamente en Supabase
(`clasificaciones` / `predicciones_dudosas`), con fechas distribuidas en
los ultimos N dias.

Por que un script separado en vez de pegarle a POST /clasificar muchas
veces: ese endpoint tiene un rate limit de 20 solicitudes/minuto (seccion
2 del README) para proteger el free tier - generar, por ejemplo, 60
tickets de demo tardaria varios minutos y competiria con el limite real
de la app. Este script se salta la API por completo: reutiliza
`clasificar()` directamente (el mismo modelo, el mismo calculo de
confianza) y escribe en la base de datos con las mismas funciones que usa
`app/routes.py`, asi que el resultado es indistinguible de tickets reales
para el dashboard - solo que se generan en segundos, no en minutos.

Uso:
    python seed_demo_data.py                  # 60 tickets, ultimos 7 dias
    python seed_demo_data.py --cantidad 150 --dias 14
    python seed_demo_data.py --seed 42        # reproducible

Requiere que DATABASE_URL este configurada (.env o variable de entorno) -
apunta a la MISMA base de datos que usa la app en produccion, asi que los
tickets generados apareceran en el dashboard real.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from app.db import supabase_client as db  # noqa: E402
from app.ml.clasificador import clasificar  # noqa: E402
from generar_dataset import (  # noqa: E402
    CATEGORIAS,
    PREFIJOS_URGENCIA,
    elegir_urgencia,
    rellenar_plantilla,
)

UMBRAL_CONFIANZA_DUDOSA = 0.6


def generar_texto_variado(rng: random.Random) -> str:
    """Reutiliza las plantillas de generar_dataset.py, pero con una
    semilla independiente de la del dataset de entrenamiento (seed=42):
    el objetivo aqui es variedad para la demo, no reproducir el dataset
    original."""
    categoria_key = rng.choice(list(CATEGORIAS.keys()))
    plantilla = rng.choice(CATEGORIAS[categoria_key])
    texto_base = rellenar_plantilla(rng, plantilla)

    urgencia_deseada = elegir_urgencia(rng, categoria_key)
    prefijo = rng.choice(PREFIJOS_URGENCIA[urgencia_deseada])
    texto = (prefijo + texto_base).strip()
    return texto[0].upper() + texto[1:]


def fecha_aleatoria_en_ventana(rng: random.Random, dias: int) -> datetime:
    segundos_atras = rng.uniform(0, dias * 24 * 3600)
    return datetime.now(timezone.utc) - timedelta(seconds=segundos_atras)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera y clasifica tickets de ejemplo, insertandolos directamente "
            "en Supabase (sin pasar por la API/rate limit) para poblar el "
            "dashboard con datos de demostracion."
        )
    )
    parser.add_argument(
        "--cantidad", type=int, default=60, help="Cuantos tickets generar (default: 60)"
    )
    parser.add_argument(
        "--dias",
        type=int,
        default=7,
        help="Distribuir las fechas en los ultimos N dias (default: 7, igual "
        "que la ventana que muestra el dashboard)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla para reproducibilidad (opcional)"
    )
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        print(
            "ERROR: configura DATABASE_URL (copia .env.example a .env y "
            "completa tu connection string de Supabase) antes de correr este script.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(args.seed)

    print(f"Generando y clasificando {args.cantidad} tickets de demo...")
    dudosos = 0

    for i in range(args.cantidad):
        texto = generar_texto_variado(rng)
        resultado = clasificar(texto)
        fecha_hora = fecha_aleatoria_en_ventana(rng, args.dias)

        db.insertar_clasificacion(
            texto,
            resultado["categoria"],
            resultado["urgencia"],
            resultado["confianza"],
            fecha_hora=fecha_hora,
        )

        if resultado["confianza"] < UMBRAL_CONFIANZA_DUDOSA:
            db.insertar_prediccion_dudosa(
                texto,
                resultado["categoria"],
                resultado["urgencia"],
                resultado["confianza"],
                fecha_hora=fecha_hora,
            )
            dudosos += 1

        if (i + 1) % 10 == 0 or (i + 1) == args.cantidad:
            print(f"  {i + 1}/{args.cantidad} insertados...")

    print(
        f"\nListo: {args.cantidad} tickets insertados en 'clasificaciones' "
        f"({dudosos} de ellos tambien en 'predicciones_dudosas' por baja confianza), "
        f"distribuidos en los ultimos {args.dias} dias."
    )
    print("Recarga /dashboard para verlos reflejados en las graficas.")


if __name__ == "__main__":
    main()
