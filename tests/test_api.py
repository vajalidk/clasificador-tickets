"""
Tests de contrato HTTP para la API del clasificador de tickets.

Cubren exactamente lo exigido para este proyecto:
  - /health responde 200.
  - /clasificar con texto valido responde 200 con las 3 llaves y tipos
    correctos.
  - /clasificar sin texto, con texto vacio y con texto demasiado largo
    responden 400.
  - /clasificar respeta el rate limit (20/min) y responde 429 al excederlo.

La base de datos esta mockeada (ver conftest.py): estos tests validan el
comportamiento de la API, no la conexion real a Postgres.
"""

MAX_TEXTO_LENGTH = 2000


def test_health_responde_200(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_nuevo_ticket_responde_200_con_html(client):
    resp = client.get("/nuevo-ticket")

    assert resp.status_code == 200
    assert b"Mesa de Ayuda" in resp.data


def test_clasificar_texto_valido_devuelve_200_con_forma_correcta(client):
    resp = client.post(
        "/clasificar",
        json={"texto": "No puedo acceder a mi cuenta desde ayer, es urgente"},
    )

    assert resp.status_code == 200
    data = resp.get_json()

    assert set(data.keys()) == {"categoria", "urgencia", "confianza"}
    assert isinstance(data["categoria"], str) and data["categoria"]
    assert isinstance(data["urgencia"], str) and data["urgencia"]
    assert isinstance(data["confianza"], float)
    assert 0.0 <= data["confianza"] <= 1.0


def test_clasificar_sin_campo_texto_devuelve_400(client):
    resp = client.post("/clasificar", json={})

    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_clasificar_texto_vacio_devuelve_400(client):
    resp = client.post("/clasificar", json={"texto": "   "})

    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_clasificar_texto_demasiado_largo_devuelve_400(client):
    resp = client.post("/clasificar", json={"texto": "a" * (MAX_TEXTO_LENGTH + 1)})

    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_clasificar_respeta_rate_limit_de_20_por_minuto(client):
    for numero_intento in range(20):
        resp = client.post("/clasificar", json={"texto": "prueba de carga"})
        assert resp.status_code == 200, f"Intento {numero_intento + 1} deberia pasar"

    resp_excedida = client.post("/clasificar", json={"texto": "una peticion de mas"})

    assert resp_excedida.status_code == 429
    assert "error" in resp_excedida.get_json()
