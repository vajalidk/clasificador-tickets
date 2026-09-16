"""
Tests de las funciones puras de app/db/supabase_client.py (sin conexion a
Postgres): generacion del titulo mostrado en las tablas y el serializador
de filas `_fila_ticket`. El resto del modulo (todo lo que ejecuta SQL) se
prueba indirectamente via tests/test_api.py con la base de datos mockeada.
"""

from datetime import datetime, timezone

from app.db.supabase_client import _fila_ticket, _generar_titulo_y_detalle


def test_generar_titulo_usa_asunto_si_existe_y_detalle_es_el_texto_completo():
    titulo, detalle = _generar_titulo_y_detalle("cualquier texto de detalle", "Mi asunto")
    assert titulo == "Mi asunto"
    assert detalle == "cualquier texto de detalle"


def test_generar_titulo_usa_primera_oracion_y_detalle_es_el_resto():
    texto = "No puedo entrar a mi cuenta. Ya intente resetear la contrasena."
    titulo, detalle = _generar_titulo_y_detalle(texto, None)
    assert titulo == "No puedo entrar a mi cuenta."
    assert detalle == "Ya intente resetear la contrasena."


def test_generar_titulo_de_una_sola_oracion_no_deja_detalle_duplicado():
    texto = "Se me olvido mi contrasena y no puedo entrar."
    titulo, detalle = _generar_titulo_y_detalle(texto, None)
    assert titulo == texto
    assert detalle == ""


def test_generar_titulo_trunca_oracion_larga():
    texto = "a" * 200
    titulo, _detalle = _generar_titulo_y_detalle(texto, None)
    assert titulo.endswith("…")
    assert len(titulo) == 81  # 80 caracteres + elipsis


def _fila_base(**overrides) -> dict:
    fila = {
        "id": 1,
        "texto": "Algo paso con mi factura.",
        "asunto": None,
        "categoria": "facturación",
        "urgencia": "media",
        "confianza": 0.85,
        "fecha_hora": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "nombre": None,
        "correo": None,
        "estado": "pendiente",
        "fecha_resuelto": None,
    }
    fila.update(overrides)
    return fila


def test_fila_ticket_genera_titulo_cuando_no_hay_asunto():
    resultado = _fila_ticket(_fila_base())
    assert resultado["titulo"] == "Algo paso con mi factura."
    assert resultado["detalle"] == ""
    assert resultado["texto"] == "Algo paso con mi factura."


def test_fila_ticket_serializa_fecha_resuelto_como_iso():
    fila = _fila_base(estado="resuelto", fecha_resuelto=datetime(2024, 1, 2, tzinfo=timezone.utc))
    resultado = _fila_ticket(fila)
    assert resultado["fecha_resuelto"] == "2024-01-02T00:00:00+00:00"


def test_fila_ticket_fecha_resuelto_nula_si_pendiente():
    resultado = _fila_ticket(_fila_base())
    assert resultado["fecha_resuelto"] is None
