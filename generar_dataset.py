"""
Genera un dataset sintetico de tickets de soporte al cliente en espanol.

El dataset combina plantillas de texto por categoria con frases que indican
un nivel de urgencia (alta / media / baja), mezcladas de forma aleatoria pero
reproducible (semilla fija) para simular la variabilidad de tickets reales
sin depender de ningun servicio externo ni de datos con informacion personal.

Salida: data/tickets_dataset.csv con columnas: texto, categoria, urgencia
"""

import csv
import random
from pathlib import Path

SEED = 42
N_TICKETS = 700
OUTPUT_PATH = Path(__file__).parent / "data" / "tickets_dataset.csv"

# ---------------------------------------------------------------------------
# Plantillas base por categoria. Cada plantilla puede contener placeholders
# ({producto}, {numero}, {monto}, {dias}) que se rellenan al azar para
# aumentar la variabilidad lexica del dataset.
# ---------------------------------------------------------------------------

PRODUCTOS = [
    "la app movil",
    "el plan premium",
    "mi suscripcion",
    "el servicio web",
    "la plataforma",
    "mi cuenta",
    "el sistema",
    "el panel de control",
]

BASE_FACTURACION = [
    "Me cobraron {monto} de mas en la factura de este mes y no entiendo por que.",
    "No reconozco un cargo de {monto} en mi tarjeta relacionado con {producto}.",
    "Quiero solicitar el reembolso del pago duplicado del pedido {numero}.",
    "Mi factura del pedido {numero} no coincide con el plan que contrate.",
    "Necesito una copia de la factura del mes pasado para mi contabilidad.",
    "Se me renovo la suscripcion automaticamente y no queria renovarla.",
    "El metodo de pago fue rechazado pero igual me generaron el cobro.",
    "Quiero cambiar el plan de facturacion de mensual a anual.",
    "Por que el precio de {producto} subio sin previo aviso.",
    "Solicito la cancelacion de mi suscripcion y el reembolso proporcional.",
    "El descuento que me ofrecieron no se aplico en la factura {numero}.",
    "Necesito la factura fiscal con los datos de mi empresa para el pedido {numero}.",
    "Me siguen cobrando {producto} aunque cancele el servicio hace tiempo.",
    "Quiero saber por que aparecen dos cargos identicos de {monto} este mes.",
    "Necesito actualizar la tarjeta de credito con la que pago {producto}.",
    "No me llega el recibo de pago del pedido {numero} a mi correo.",
    "Quisiera cambiar mi ciclo de facturacion, me gustaria pagar cada semana.",
    "El cupon de descuento que aplique no se ve reflejado en el total a pagar.",
    "Me gustaria saber si puedo pagar {producto} en varias mensualidades.",
    "Se me duplico el cobro anual de {producto}, necesito que me regresen uno.",
    "Cual es la diferencia de precio entre el plan actual y el plan superior.",
    "Necesito la factura a nombre de mi empresa, no a mi nombre personal.",
]

BASE_SOPORTE_TECNICO = [
    "No puedo iniciar sesion en {producto} desde ayer, me marca error de servidor.",
    "{producto} se cierra solo cada vez que intento subir un archivo.",
    "La aplicacion no carga las imagenes y se queda en blanco.",
    "Recibo el error 500 al intentar guardar los cambios en {producto}.",
    "El boton de exportar datos no responde en ningun navegador.",
    "No me llega el correo de verificacion para activar mi cuenta.",
    "La sincronizacion entre el celular y la web de {producto} no funciona.",
    "Perdi acceso a mi cuenta despues de la ultima actualizacion.",
    "El sistema muestra datos desactualizados desde hace {dias} dias.",
    "No logro restablecer mi contrasena, el enlace del correo esta roto.",
    "La integracion con la API deja de responder despues de unos minutos.",
    "{producto} consume demasiada memoria y se congela el navegador.",
    "La pantalla se queda cargando indefinidamente al abrir {producto}.",
    "Me aparece un mensaje de error que dice 'token invalido' al entrar.",
    "No puedo adjuntar archivos mayores a 5 MB en {producto}.",
    "El buscador interno de {producto} no encuentra resultados que si existen.",
    "Mi cuenta se cerro sola en medio de una sesion de trabajo, varias veces.",
    "Los reportes de {producto} tardan mas de 10 minutos en generarse.",
    "No recibo las notificaciones push en mi celular desde la actualizacion.",
    "{producto} muestra un error de certificado de seguridad al conectarse.",
    "El calendario de {producto} no sincroniza bien con Google Calendar.",
    "No puedo cambiar mi contrasena, el formulario no guarda los cambios.",
]

BASE_QUEJA = [
    "Llevo {dias} dias esperando respuesta y nadie me contesta, es inaceptable.",
    "El servicio al cliente me trato de forma grosera en el chat de soporte.",
    "Esta es la tercera vez que reporto el mismo problema con {producto} y nadie lo resuelve.",
    "Estoy muy decepcionado con la calidad de {producto}, no cumple lo prometido.",
    "Cancele mi cuenta hace {dias} dias y todavia me siguen cobrando.",
    "Pedi hablar con un supervisor y me colgaron la llamada.",
    "El soporte tecnico cerro mi ticket sin resolver el problema.",
    "Ya escribi tres veces por este mismo tema y siempre me responden lo mismo sin ayudar.",
    "Quiero presentar una queja formal por el trato recibido en la sucursal.",
    "Es la peor experiencia que he tenido con un servicio en linea.",
    "Me prometieron una solucion en 24 horas y ya pasaron {dias} dias sin novedades.",
    "Nadie en el equipo de soporte parece saber como resolver mi caso.",
    "Estoy a punto de cancelar mi cuenta por lo mal que me han atendido.",
    "El representante me dejo en espera mas de una hora y luego colgo la llamada.",
    "Voy a reportar esta empresa porque no respetan lo que ofrecen en su publicidad.",
    "No puedo creer que despues de tantos correos nadie me haya dado una solucion.",
    "El nivel de servicio ha bajado muchisimo comparado con hace un año.",
    "Me siento estafado, pague por un servicio premium que nunca funciono bien.",
    "Cada vez que llamo me transfieren con alguien distinto y repito todo de nuevo.",
]

BASE_INFORMACION_GENERAL = [
    "Cuales son los horarios de atencion al cliente.",
    "Como puedo cambiar el correo asociado a mi cuenta.",
    "Que metodos de pago aceptan para {producto}.",
    "Quisiera saber si {producto} tiene version para empresas.",
    "Donde puedo consultar los terminos y condiciones del servicio.",
    "Tienen alguna guia para empezar a usar {producto}.",
    "Como contacto al equipo de ventas para una demo.",
    "Cuanto tiempo tarda en procesarse una solicitud de baja.",
    "Necesito informacion sobre las tarifas para mas de 10 usuarios.",
    "Existe alguna app movil disponible para {producto}.",
    "Podrian confirmarme si el servicio esta disponible en mi pais.",
    "Como puedo actualizar mis datos de facturacion.",
    "Tienen algun plan especial para estudiantes o para organizaciones sin fines de lucro.",
    "Quisiera saber si {producto} es compatible con Mac y con Windows.",
    "Me podrian compartir el enlace a la documentacion tecnica de la API.",
    "Cuantos usuarios puedo agregar en el plan basico de {producto}.",
    "Ofrecen algun periodo de prueba gratuito antes de contratar {producto}.",
    "Donde puedo ver el historial completo de mis pedidos anteriores.",
    "Que diferencia hay entre el soporte estandar y el soporte prioritario.",
    "Como puedo invitar a un compañero de trabajo a mi cuenta de {producto}.",
    "En que idiomas esta disponible la interfaz de {producto}.",
]

# Frases que se anteponen para marcar el nivel de urgencia percibido.
PREFIJOS_URGENCIA = {
    "alta": [
        "Urgente: ",
        "Es critico, necesito una respuesta hoy mismo. ",
        "Necesito ayuda inmediata, esto esta afectando mi trabajo. ",
        "Por favor atiendan esto lo antes posible, no puedo esperar mas. ",
        "Esto es una emergencia para mi negocio. ",
        "Necesito una solucion ya, esto no puede esperar. ",
        "Por favor prioricen mi caso, es sumamente urgente. ",
    ],
    "media": [
        "",
        "Agradeceria una respuesta en los proximos dias. ",
        "No es una emergencia, pero me gustaria resolverlo pronto. ",
        "Cuando tengan oportunidad, ",
        "Me gustaria que lo revisaran esta semana si es posible. ",
    ],
    "baja": [
        "Sin ninguna prisa, ",
        "Cuando puedan, sin urgencia: ",
        "Solo quiero comentar lo siguiente, no es urgente. ",
        "En algun momento les agradeceria revisar esto: ",
        "",
        "No hay prisa alguna, solo queria dejarlo comentado: ",
    ],
}

CATEGORIAS = {
    "facturación": BASE_FACTURACION,
    "soporte técnico": BASE_SOPORTE_TECNICO,
    "queja": BASE_QUEJA,
    "información general": BASE_INFORMACION_GENERAL,
}

# Distribucion realista de urgencia por categoria: las quejas y fallas
# tecnicas tienden a ser mas urgentes, la informacion general casi nunca.
DISTRIBUCION_URGENCIA = {
    "facturación": {"alta": 0.35, "media": 0.45, "baja": 0.20},
    "soporte técnico": {"alta": 0.45, "media": 0.40, "baja": 0.15},
    "queja": {"alta": 0.55, "media": 0.35, "baja": 0.10},
    "información general": {"alta": 0.05, "media": 0.35, "baja": 0.60},
}


def elegir_urgencia(rng: random.Random, categoria: str) -> str:
    pesos = DISTRIBUCION_URGENCIA[categoria]
    niveles = list(pesos.keys())
    probabilidades = list(pesos.values())
    return rng.choices(niveles, weights=probabilidades, k=1)[0]


def rellenar_plantilla(rng: random.Random, plantilla: str) -> str:
    return plantilla.format(
        producto=rng.choice(PRODUCTOS),
        numero=rng.randint(1000, 9999),
        monto=f"${rng.randint(5, 500)}",
        dias=rng.randint(2, 15),
    )


def generar_tickets(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    categorias_ciclo = list(CATEGORIAS.keys())
    tickets = []

    # Distribuye los N tickets de forma pareja entre las 4 categorias.
    for i in range(n):
        categoria = categorias_ciclo[i % len(categorias_ciclo)]
        plantillas = CATEGORIAS[categoria]
        plantilla = rng.choice(plantillas)
        texto_base = rellenar_plantilla(rng, plantilla)

        urgencia = elegir_urgencia(rng, categoria)
        prefijo = rng.choice(PREFIJOS_URGENCIA[urgencia])
        texto = (prefijo + texto_base).strip()
        # Normaliza mayuscula inicial tras concatenar prefijo + texto base.
        texto = texto[0].upper() + texto[1:]

        tickets.append({"texto": texto, "categoria": categoria, "urgencia": urgencia})

    rng.shuffle(tickets)
    return tickets


def main() -> None:
    tickets = generar_tickets(N_TICKETS, SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["texto", "categoria", "urgencia"])
        writer.writeheader()
        writer.writerows(tickets)

    print(f"Dataset generado con {len(tickets)} tickets en: {OUTPUT_PATH}")
    conteo_categorias = {}
    conteo_urgencia = {}
    for t in tickets:
        conteo_categorias[t["categoria"]] = conteo_categorias.get(t["categoria"], 0) + 1
        conteo_urgencia[t["urgencia"]] = conteo_urgencia.get(t["urgencia"], 0) + 1

    print("Distribucion por categoria:", conteo_categorias)
    print("Distribucion por urgencia:", conteo_urgencia)


if __name__ == "__main__":
    main()
