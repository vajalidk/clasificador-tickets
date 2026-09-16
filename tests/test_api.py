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


def test_clasificar_guarda_nombre_y_correo_opcionales(client, mock_db):
    resp = client.post(
        "/clasificar",
        json={"texto": "No puedo entrar a mi cuenta", "nombre": "Ana", "correo": "ana@ejemplo.com"},
    )

    assert resp.status_code == 200
    _args, kwargs = mock_db.insertar_clasificacion.call_args
    assert kwargs["nombre"] == "Ana"
    assert kwargs["correo"] == "ana@ejemplo.com"


def test_clasificar_guarda_asunto_opcional(client, mock_db):
    resp = client.post(
        "/clasificar",
        json={"texto": "No puedo entrar a mi cuenta", "asunto": "Problema de acceso"},
    )

    assert resp.status_code == 200
    _args, kwargs = mock_db.insertar_clasificacion.call_args
    assert kwargs["asunto"] == "Problema de acceso"
    # El texto guardado es el mensaje limpio, sin el asunto concatenado.
    args, _kwargs = mock_db.insertar_clasificacion.call_args
    assert args[0] == "No puedo entrar a mi cuenta"


def test_tickets_html_responde_200(client):
    resp = client.get("/tickets")

    assert resp.status_code == 200
    assert b"Todos los tickets" in resp.data


def test_api_tickets_responde_200_con_forma_correcta(client):
    resp = client.get("/api/tickets")

    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data.keys()) == {"tickets", "total", "limite", "offset"}


def test_api_tickets_urgencia_invalida_devuelve_400(client):
    resp = client.get("/api/tickets?urgencia=critica")

    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_patch_estado_sin_admin_token_configurado_devuelve_503(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    resp = client.patch("/api/tickets/1/estado", json={"estado": "resuelto"})

    assert resp.status_code == 503


def test_patch_estado_con_token_invalido_devuelve_401(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "token-correcto")

    resp = client.patch(
        "/api/tickets/1/estado",
        json={"estado": "resuelto"},
        headers={"X-Admin-Token": "token-incorrecto"},
    )

    assert resp.status_code == 401


def test_patch_estado_con_token_valido_actualiza_ticket(client, monkeypatch, mock_db):
    monkeypatch.setenv("ADMIN_TOKEN", "token-correcto")

    resp = client.patch(
        "/api/tickets/1/estado",
        json={"estado": "resuelto"},
        headers={"X-Admin-Token": "token-correcto"},
    )

    assert resp.status_code == 200
    assert resp.get_json() == {"id": 1, "estado": "resuelto"}


def test_patch_estado_ticket_inexistente_devuelve_404(client, monkeypatch, mock_db):
    monkeypatch.setenv("ADMIN_TOKEN", "token-correcto")
    mock_db.actualizar_estado_ticket.return_value = None

    resp = client.patch(
        "/api/tickets/999/estado",
        json={"estado": "resuelto"},
        headers={"X-Admin-Token": "token-correcto"},
    )

    assert resp.status_code == 404
