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
import unicodedata
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


def generar_texto_variado(rng: random.Random) -> tuple[str, str]:
    """Reutiliza las plantillas de generar_dataset.py, pero con una
    semilla independiente de la del dataset de entrenamiento (seed=42):
    el objetivo aqui es variedad para la demo, no reproducir el dataset
    original. Devuelve tambien la categoria de la plantilla usada (no
    necesariamente la que el modelo termine prediciendo), para poder
    generar un "asunto" acorde al tema del ticket."""
    categoria_key = rng.choice(list(CATEGORIAS.keys()))
    plantilla = rng.choice(CATEGORIAS[categoria_key])
    texto_base = rellenar_plantilla(rng, plantilla)

    urgencia_deseada = elegir_urgencia(rng, categoria_key)
    prefijo = rng.choice(PREFIJOS_URGENCIA[urgencia_deseada])
    texto = (prefijo + texto_base).strip()
    return texto[0].upper() + texto[1:], categoria_key


def fecha_aleatoria_en_ventana(rng: random.Random, dias: int) -> datetime:
    segundos_atras = rng.uniform(0, dias * 24 * 3600)
    return datetime.now(timezone.utc) - timedelta(seconds=segundos_atras)


# ---------------------------------------------------------------------------
# Datos de contacto (nombre/correo) y "asunto" simulados: le dan a la demo
# la misma pinta que tickets reales enviados desde /nuevo-ticket, en vez de
# aparecer todos como "Sin nombre" en /tickets.
# ---------------------------------------------------------------------------

NOMBRES = [
    "Ana",
    "Luis",
    "Maria",
    "Jose",
    "Laura",
    "Carlos",
    "Sofia",
    "Miguel",
    "Fernanda",
    "Javier",
    "Paola",
    "Ricardo",
    "Alejandra",
    "Diego",
    "Gabriela",
    "Roberto",
    "Daniela",
    "Francisco",
    "Valeria",
    "Eduardo",
    "Monica",
    "Sergio",
    "Patricia",
    "Alberto",
    "Claudia",
    "Raul",
    "Adriana",
    "Hector",
    "Veronica",
    "Jorge",
]

APELLIDOS = [
    "Garcia",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Perez",
    "Sanchez",
    "Ramirez",
    "Torres",
    "Flores",
    "Rivera",
    "Gomez",
    "Diaz",
    "Reyes",
    "Morales",
    "Cruz",
    "Ortiz",
    "Gutierrez",
    "Chavez",
    "Ramos",
    "Mendoza",
    "Vazquez",
    "Castillo",
    "Jimenez",
    "Moreno",
    "Romero",
    "Alvarez",
    "Aguilar",
    "Medina",
    "Herrera",
]

DOMINIOS_CORREO = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com.mx", "live.com.mx"]

ASUNTOS_POR_CATEGORIA = {
    "facturación": [
        "Problema con mi factura",
        "Cobro duplicado en mi cuenta",
        "Solicitud de reembolso",
        "Duda sobre mi plan de pago",
        "Error en el monto facturado",
        "Cancelacion de suscripcion",
    ],
    "soporte técnico": [
        "No puedo iniciar sesion",
        "La aplicacion se cierra sola",
        "Error al sincronizar mis datos",
        "Problema de conexion con el servidor",
        "La pagina no carga correctamente",
        "Fallo al subir archivos",
    ],
    "queja": [
        "Mal servicio de atencion al cliente",
        "Tiempo de espera excesivo",
        "Producto llego danado",
        "Inconformidad con la resolucion anterior",
        "Trato inadecuado de un representante",
    ],
    "información general": [
        "Consulta sobre horarios de atencion",
        "Pregunta sobre metodos de pago",
        "Duda sobre el proceso de registro",
        "Informacion sobre planes disponibles",
        "Consulta general del servicio",
    ],
}


def _sin_acentos(cadena: str) -> str:
    normalizado = unicodedata.normalize("NFKD", cadena)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def generar_nombre(rng: random.Random) -> str:
    return f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)}"


def generar_correo(rng: random.Random, nombre: str) -> str:
    partes = [_sin_acentos(p).lower() for p in nombre.split()]
    sufijo = str(rng.randint(1, 99)) if rng.random() < 0.2 else ""
    return f"{'.'.join(partes)}{sufijo}@{rng.choice(DOMINIOS_CORREO)}"


def generar_asunto(rng: random.Random, categoria_key: str) -> str:
    return rng.choice(ASUNTOS_POR_CATEGORIA[categoria_key])


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
    parser.add_argument(
        "--prob-correo",
        type=float,
        default=0.75,
        help="Probabilidad (0-1) de que un ticket incluya correo, ademas del "
        "nombre (default: 0.75, un cuarto de los tickets queda sin correo)",
    )
    parser.add_argument(
        "--sin-contacto",
        action="store_true",
        help="No generar nombre/correo/asunto: tickets anonimos, como el "
        "comportamiento original de este script",
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
    con_correo = 0

    for i in range(args.cantidad):
        texto, categoria_key = generar_texto_variado(rng)
        resultado = clasificar(texto)
        fecha_hora = fecha_aleatoria_en_ventana(rng, args.dias)

        nombre = asunto = correo = None
        if not args.sin_contacto:
            nombre = generar_nombre(rng)
            asunto = generar_asunto(rng, categoria_key)
            if rng.random() < args.prob_correo:
                correo = generar_correo(rng, nombre)
                con_correo += 1

        db.insertar_clasificacion(
            texto,
            resultado["categoria"],
            resultado["urgencia"],
            resultado["confianza"],
            fecha_hora=fecha_hora,
            nombre=nombre,
            correo=correo,
            asunto=asunto,
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
    if not args.sin_contacto:
        print(f"Con nombre y asunto: {args.cantidad}. Con correo: {con_correo}.")
    print("Recarga /dashboard para verlos reflejados en las graficas.")


if __name__ == "__main__":
    main()
