# Clasificador de Tickets de Soporte con Pipeline Automatizado

API que recibe texto de soporte al cliente en espanol y devuelve
**categoria**, **urgencia** y **nivel de confianza**, con historial
persistente en Postgres, dashboard de estadisticas y despliegue automatico
100% gratuito (Render + Supabase + GitHub Actions).

> Este README se construye seccion por seccion junto con el proyecto. Cada
> apartado se completa a medida que se implementa esa parte del sistema.

---

## 1. Dataset y modelo de IA

### 1.1 Dataset sintetico

[`generar_dataset.py`](generar_dataset.py) genera 250 tickets de soporte en
espanol y los guarda en [`data/tickets_dataset.csv`](data/tickets_dataset.csv)
con columnas `texto,categoria,urgencia`.

En vez de escribir 250 frases a mano, el script combina:

- **Plantillas base por categoria** (`facturación`, `soporte técnico`,
  `queja`, `información general`), con placeholders (`{producto}`,
  `{numero}`, `{monto}`, `{dias}`) que se rellenan al azar para variar el
  vocabulario.
- **Prefijos de urgencia** (`alta` / `media` / `baja`) que se anteponen al
  texto base, con una distribucion de probabilidad distinta por categoria
  (por ejemplo, una `queja` tiene mas probabilidad de ser `alta` que una
  pregunta de `información general`), para que el dataset sea realista y no
  perfectamente balanceado de forma artificial.
- Una **semilla fija** (`random.seed(42)` a traves de `random.Random(42)`)
  para que el dataset generado sea siempre identico y el pipeline completo
  sea reproducible bit a bit.

### 1.2 Entrenamiento

[`entrenar_modelo.py`](entrenar_modelo.py) entrena **dos modelos
independientes** sobre el mismo dataset:

| Modelo | Predice | Pipeline |
|---|---|---|
| `categoria_v{fecha}.joblib` | categoria (4 clases) | `TfidfVectorizer` + `LinearSVC` + `CalibratedClassifierCV` |
| `urgencia_v{fecha}.joblib` | urgencia (3 clases) | `TfidfVectorizer` + `LinearSVC` + `CalibratedClassifierCV` |

**Por que `LinearSVC` + `CalibratedClassifierCV` en vez de
`LogisticRegression` directamente:** con vectores TF-IDF (alta
dimensionalidad, pocos ejemplos por clase), los SVM lineales suelen dar
fronteras de decision mas robustas que la regresion logistica simple. El
problema es que un SVM no produce probabilidades de forma nativa (solo una
distancia al hiperplano). `CalibratedClassifierCV` resuelve esto entrenando
el SVM con validacion cruzada interna y ajustando una funcion sigmoide sobre
sus scores, obteniendo un `predict_proba` con probabilidades reales y bien
calibradas — indispensable porque el campo `confianza` de la API depende de
que esa probabilidad sea confiable, no un numero arbitrario.

El script, al ejecutarse:

1. Carga el CSV y separa 80/20 train/test de forma estratificada.
2. Entrena y evalua sobre el 20% de test, imprimiendo **accuracy, un
   `classification_report` (precision/recall/F1 por clase) y la matriz de
   confusion** para cada uno de los dos modelos.
3. Reentrena el modelo final con el 100% de los datos (el dataset sintetico
   es pequeno; una vez medido el desempeno de forma honesta con el split de
   test, no tiene sentido dejar ese 20% fuera del modelo que se despliega).
4. Guarda cada modelo versionado por fecha en `models/`, por ejemplo:
   `models/categoria_v20260913.joblib`.
5. Actualiza `models/latest.json`, que apunta a la version activa de cada
   modelo junto con sus metricas. **La API en produccion solo lee este
   archivo**, nunca un nombre de archivo fijo — esto permite hacer
   **rollback instantaneo** a una version anterior (editando el JSON para
   apuntar a un `.joblib` mas viejo) sin reentrenar ni redeployar codigo.

Metricas obtenidas en la corrida de referencia (250 tickets, semilla 42):

- **Categoria**: accuracy 0.98 en test (50 ejemplos).
- **Urgencia**: accuracy 0.90 en test (50 ejemplos).

(El detalle completo por clase se imprime en consola al ejecutar el script y
queda tambien guardado dentro de `models/latest.json`.)

### 1.3 Por que un modelo clasico (TF-IDF + SVM) y no un LLM via API

Esta es la decision de arquitectura mas importante del proyecto de IA, y se
tomo deliberadamente en contra de usar un LLM (OpenAI, Anthropic, etc.) via
API:

1. **Cero costo por request.** El objetivo explicito del proyecto es "no
   usar servicios de pago en ningun punto del pipeline". Un LLM via API
   cobra por token; un modelo TF-IDF + SVM entrenado localmente no tiene
   costo marginal alguno por clasificacion.
2. **Cero dependencia de conexion externa.** Si la API de un proveedor de
   LLM tiene una caida, un rate-limit, o cambia su contrato, `/clasificar`
   dejaria de funcionar. Con un modelo `.joblib` cargado en memoria, la
   clasificacion es una operacion local, determinista y sin red.
3. **Corre dentro del limite de memoria/CPU de un hosting gratuito (Render).**
   Los modelos de este proyecto pesan unos pocos MB y clasifican en
   milisegundos con CPU minima. Cargar un modelo de lenguaje (aunque fuera
   uno pequeno, local) para un dominio de 4 categorias seria
   sobre-ingenieria y competiria por la memoria limitada del contenedor
   gratuito.
4. **El dominio es acotado y estable.** Clasificar entre 4 categorias y 3
   niveles de urgencia es un problema de clasificacion de texto clasico y
   bien resuelto por TF-IDF + modelos lineales desde hace mas de una
   decada. Un LLM aporta capacidad de generalizacion a dominios abiertos
   que aqui no se necesita, a cambio de latencia, costo y una dependencia
   externa innecesaria.
5. **Interpretabilidad y control de version.** Un modelo lineal sobre
   TF-IDF es auditable (se pueden inspeccionar los pesos por termino) y
   versionado explicitamente (`models/latest.json`), algo mucho mas dificil
   de garantizar con un modelo de terceros que puede cambiar sin previo
   aviso ("model drift" del propio proveedor).

### 1.4 Plan de mantenimiento del modelo (deteccion de drift y reentrenamiento)

Ningun modelo de clasificacion de texto se mantiene preciso para siempre: el
vocabulario de los usuarios cambia, aparecen productos nuevos, y la
distribucion real de tickets se distancia poco a poco del dataset sintetico
inicial ("data drift"). El plan de mantenimiento de este proyecto es:

1. **Captura automatica de casos dudosos.** Cada vez que `/clasificar`
   devuelve una `confianza < 0.6` para la categoria predicha, la
   clasificacion se guarda tanto en `clasificaciones` como en una tabla
   adicional, `predicciones_dudosas` (ver seccion 2, `init_db.sql`). Esta
   tabla es, en efecto, una cola de "casos para revision humana".
2. **Revision manual periodica.** Un humano (el propio equipo de soporte o
   quien mantenga el proyecto) revisa `predicciones_dudosas` periodicamente
   (por ejemplo, semanalmente) y corrige la etiqueta real de esos tickets.
3. **Reentrenamiento incremental.** Los tickets corregidos se incorporan a
   `data/tickets_dataset.csv` (o a un dataset ampliado) y se vuelve a correr
   `entrenar_modelo.py`. Como el script ya es reproducible y versiona cada
   modelo por fecha, este reentrenamiento no sobreescribe nada: genera una
   nueva version y el pointer `models/latest.json` se actualiza solo si las
   nuevas metricas son iguales o mejores que las anteriores.
4. **Rollback seguro.** Si una nueva version entrenada resulta peor (por
   ejemplo, por un lote de correcciones manuales con errores), basta con
   editar `models/latest.json` para volver a apuntar a la version anterior
   — no hace falta revertir codigo ni volver a desplegar la imagen Docker
   completa (los `.joblib` de todas las versiones quedan conservados en
   `models/`).
5. **Metrica de vigilancia continua.** El `GET /api/estadisticas` (seccion
   2) expone la proporcion de clasificaciones con baja confianza en el
   tiempo; un aumento sostenido de esa proporcion es la senal de alerta de
   que el modelo esta "quedandose viejo" frente a los tickets reales y de
   que toca revisar `predicciones_dudosas` y reentrenar.

---

## 2. Backend (Flask)

### 2.1 Arquitectura

```mermaid
flowchart LR
    Cliente[Cliente / navegador] -->|POST /clasificar| API
    Cliente -->|GET /health| API
    Cliente -->|GET /dashboard| API
    Cliente -->|GET /apidocs| API

    subgraph API[Flask app - application factory]
        Routes[app/routes.py] --> ML[app/ml/clasificador.py]
        Routes --> DB[app/db/supabase_client.py]
        Routes --> Limiter[Flask-Limiter 20 req/min]
        Routes --> Swagger[flasgger /apidocs]
    end

    ML -->|joblib.load| Modelos[(models/*.joblib\n+ latest.json)]
    DB -->|psycopg2 pool + tenacity retry| Postgres[(Supabase Postgres)]
```

- **`app/__init__.py`** implementa el patron *application factory*
  (`create_app()`), en vez de crear un `Flask()` global. Esto permite crear
  instancias de la app aisladas en los tests (seccion 3), cada una con su
  propia configuracion, sin que un test contamine el estado de otro.
- **`app/routes.py`** contiene los 5 endpoints y llama a `app/ml` y
  `app/db` a traves de sus funciones publicas — nunca accede a joblib o a
  psycopg2 directamente. Esto mantiene la logica de negocio (validacion,
  persistencia condicional) separada de los detalles de implementacion del
  modelo y de la base de datos.
- **`app/ml/clasificador.py`** carga los `.joblib` una sola vez (singleton
  con lock para uso thread-safe) y resuelve la version activa leyendo
  `models/latest.json` en cada llamada (asi un rollback de modelo se ve
  reflejado sin reiniciar el proceso, apenas cambie ese archivo).
- **`app/db/supabase_client.py`** encapsula toda la logica de Postgres:
  pool de conexiones (`psycopg2.pool.SimpleConnectionPool`, 1-5 conexiones)
  y reintentos con backoff exponencial via `tenacity` ante
  `psycopg2.OperationalError` (cubre caidas de red breves o el "cold start"
  tipico de un Postgres gestionado en tier gratuito tras inactividad).

### 2.2 Decisiones tecnicas no funcionales

- **Logging estructurado, nunca `print`.** `_configurar_logging()` en
  `app/__init__.py` configura `logging.basicConfig` con timestamp y nivel
  configurable por `LOG_LEVEL` (INFO por defecto). Todos los modulos usan
  `logging.getLogger(__name__)`. Render captura stdout automaticamente,
  y un log estructurado con nivel permite filtrar y alertar en produccion;
  un `print` no se puede filtrar ni tiene severidad.
- **Rate limiting con `Flask-Limiter` (20/min por IP en `/clasificar`).**
  Protege el free tier de Supabase/Render de un abuso accidental o
  deliberado. *Limitacion conocida y documentada*: el backend de
  almacenamiento por defecto de Flask-Limiter es en memoria del proceso; en
  un despliegue con **mas de una instancia** corriendo en paralelo, cada
  instancia llevaria su propio contador (el limite real efectivo seria
  `20 x num_instancias`). Para este proyecto de portafolio,
  que corre en el free tier con bajo trafico, se acepta esta limitacion en
  vez de anadir una dependencia de pago (Redis gestionado) solo para tener
  un contador distribuido. Si el trafico creciera, la migracion natural
  seria un backend `redis://` para Flask-Limiter.
- **Documentacion OpenAPI con `flasgger`.** Se eligio sobre
  `flask-smorest` porque permite documentar cada endpoint con un docstring
  YAML dentro de la misma funcion (menor ceremonia que definir schemas
  Marshmallow separados para un proyecto de este tamano), exponiendo
  `/apidocs` de forma automatica.
- **Manejo de errores centralizado.** `_registrar_manejadores_error()`
  registra un `@app.errorhandler(HTTPException)` (cubre 400, 404, 429, etc.)
  y un `@app.errorhandler(Exception)` como red de seguridad final para
  cualquier excepcion no prevista. Ambos devuelven siempre
  `{"error": "..."}` en JSON con el status code correspondiente — nunca la
  pagina HTML de error por defecto de Flask/Werkzeug.
- **Fallo de persistencia no rompe la respuesta al usuario.** Si
  `/clasificar` no puede escribir en Postgres (por ejemplo, Supabase esta
  caido), el error se registra con `logger.exception(...)` pero la API
  igual devuelve 200 con la clasificacion ya calculada: el calculo del
  modelo es correcto y no depende de la base de datos, por lo que fallar la
  request completa por un problema de persistencia seria degradar
  innecesariamente la experiencia del usuario. (Este es un tradeoff
  explicito: se prioriza disponibilidad de la funcion principal sobre
  consistencia estricta del historial.)

### 2.3 Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| `POST` | `/clasificar` | Clasifica un ticket: `{"texto": "..."}` -> `{"categoria", "urgencia", "confianza"}` |
| `GET` | `/health` | 200 si la app y la DB responden; 503 con detalle si la DB falla |
| `GET` | `/nuevo-ticket` | Formulario publico (HTML) para que un usuario final envie un ticket, sin tocar la API directamente |
| `GET` | `/dashboard` | HTML con Chart.js, consume `/api/estadisticas` |
| `GET` | `/api/estadisticas` | JSON con agregaciones para el dashboard |
| `GET` | `/apidocs` | Swagger UI autogenerado (flasgger) |

### 2.3.1 `/nuevo-ticket`: formulario para usuarios finales

Swagger (`/apidocs`) es documentacion para desarrolladores, no una interfaz
pensada para que alguien sin conocimientos tecnicos reporte un problema.
[`app/templates/nuevo_ticket.html`](app/templates/nuevo_ticket.html) es una
pagina publica con la estetica de una "Mesa de Ayuda" corporativa (campos
de Nombre/Correo opcionales, Asunto y Mensaje) que:

1. Combina Asunto + Mensaje en un solo texto y lo envia via `fetch()` a
   `POST /clasificar` (el mismo endpoint que ya existia, sin duplicar
   logica de clasificacion ni de persistencia).
2. Muestra el resultado como una confirmacion de "ticket recibido",
   revelando de forma transparente la categoria/urgencia/confianza que el
   modelo detecto — una decision deliberada de este proyecto de
   portafolio: el objetivo es exhibir el sistema de IA funcionando, a
   diferencia de un ticket real de soporte donde ese detalle interno
   normalmente se le ocultaria al usuario final.
3. Maneja los mismos errores que ya devuelve la API (400 por texto vacio,
   429 por rate limit) mostrando un mensaje legible en vez de un JSON
   crudo.

No se agrego ningun endpoint ni tabla nueva — es una capa de presentacion
sobre `/clasificar` que ya existia desde la seccion 2.

### 2.4 Base de datos (Supabase / Postgres)

El esquema completo esta en [`init_db.sql`](init_db.sql): tablas
`clasificaciones` y `predicciones_dudosas`, cada una con indices en
`fecha_hora` y (en `clasificaciones`) en `categoria`, que son los dos
patrones de consulta que usa `GET /api/estadisticas`.

**Como crear el proyecto gratuito en Supabase:**

1. Crear una cuenta en [supabase.com](https://supabase.com) (tier gratuito,
   sin tarjeta requerida para el plan free).
2. "New Project" -> elegir un nombre, una contrasena para la base de datos
   (guardala, la necesitas para el connection string) y una region cercana.
3. Esperar ~2 minutos a que se aprovisione el proyecto.
4. Ir al boton **"Connect"** (arriba del proyecto) -> pestaña
   **"Connection string"** -> metodo **"Session pooler"** (no "Direct
   connection" - ver el porque en la seccion 7.2, tiene que ver con que la
   mayoria de hosting gratuito solo tiene salida IPv4) -> tipo **URI**, y
   copiar la cadena (tiene la forma
   `postgresql://postgres.<project-ref>:[password]@aws-0-<region>.pooler.supabase.com:5432/postgres`).
5. Pegar esa URL en tu `.env` local como `DATABASE_URL` (ver
   `.env.example`), y mas adelante como variable de entorno en el
   dashboard de Render (seccion 7.2) para el despliegue.
6. Abrir el **SQL Editor** de Supabase, pegar el contenido de
   `init_db.sql` y ejecutarlo una sola vez para crear las tablas.

### 2.4.1 Poblar el dashboard con datos de demostracion (`seed_demo_data.py`)

Con un dashboard recien creado, las graficas de la seccion 2.3 se ven
vacias hasta que existan varias clasificaciones. En vez de mandar tickets
uno por uno (chocando ademas contra el rate limit de 20/min de
`/clasificar`), [`seed_demo_data.py`](seed_demo_data.py) genera texto
variado reutilizando las plantillas de `generar_dataset.py`, lo clasifica
con el modelo real (`app.ml.clasificador.clasificar`) y lo inserta
**directamente en Supabase** (mismas funciones de `app/db/supabase_client.py`
que usa la API), con fechas distribuidas aleatoriamente en los ultimos N
dias para que la grafica de "tickets por dia" se vea realista:

```bash
# Con tu .env configurado (DATABASE_URL apuntando a tu Supabase):
python seed_demo_data.py                    # 60 tickets, ultimos 7 dias
python seed_demo_data.py --cantidad 150 --dias 14
python seed_demo_data.py --seed 42          # reproducible
```

El resultado es indistinguible de tickets reales para `/dashboard` (usa
el mismo modelo y las mismas tablas) — solo se genera en segundos en vez
de minutos, y sin gastar el rate limit real de la API.

### 2.5 Ejemplos curl

```bash
# Clasificar un ticket
curl -X POST http://localhost:5000/clasificar \
  -H "Content-Type: application/json" \
  -d '{"texto": "No puedo acceder a mi cuenta desde ayer, es urgente"}'
# -> {"categoria": "soporte técnico", "urgencia": "alta", "confianza": 0.87}

# Healthcheck
curl http://localhost:5000/health
# -> {"status": "ok"}

# Estadisticas (usadas por el dashboard)
curl http://localhost:5000/api/estadisticas

# Dashboard (abrir en el navegador)
open http://localhost:5000/dashboard   # macOS
# o simplemente visitar la URL en cualquier navegador

# Documentacion interactiva
open http://localhost:5000/apidocs
```

---

## Como ejecutar localmente (backend completo)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash / Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edita .env con tu DATABASE_URL real de Supabase (ver seccion 2.4)

python generar_dataset.py
python entrenar_modelo.py

python run.py
# -> servidor de desarrollo en http://localhost:5000
```

---

## 3. Pruebas automatizadas y calidad de codigo

### 3.1 Tests (`tests/`)

[`tests/conftest.py`](tests/conftest.py) reemplaza `app.db.supabase_client`
por un `MagicMock` (via `monkeypatch`) antes de crear cada app de prueba, y
resetea el estado del rate limiter (`limiter.reset()`) antes de cada test.
Esto se documenta explicitamente porque son las dos decisiones que hacen que
la suite sea rapida y determinista en CI:

- **Sin DB real**: no se usa sqlite en memoria como sustituto de Postgres
  porque el SQL de `supabase_client.py` usa sintaxis especifica de Postgres
  (`RETURNING`, `NOW() AT TIME ZONE 'utc'`, `INTERVAL '7 days'`) que sqlite
  no entiende. Mockear el modulo completo prueba el contrato HTTP de la API
  (que es lo que pide la especificacion) sin acoplar los tests a un motor
  de base de datos concreto.
- **Reset del rate limiter**: Flask-Limiter guarda sus contadores en un
  almacenamiento en memoria a nivel de proceso. Sin resetearlo entre tests,
  el test de rate limit podria "heredar" requests de otro test que corrio
  antes en el mismo proceso de pytest y fallar de forma intermitente.

[`tests/test_api.py`](tests/test_api.py) cubre exactamente lo pedido:
`/health` -> 200; `/clasificar` valido -> 200 con `categoria` (str),
`urgencia` (str) y `confianza` (float 0-1); `/clasificar` sin texto, con
texto vacio y con texto > 2000 caracteres -> 400; y 20 requests validas
seguidas de una 21a que debe responder 429.

```bash
pip install -r requirements-dev.txt
pytest -v
```

### 3.2 Calidad de codigo (`black`, `isort`, `flake8`)

- **`pyproject.toml`**: configura `black` e `isort` (perfil `black`) con
  `line-length = 100`. Se eligio 100 en vez del default de black (88)
  porque varias funciones tienen docstrings YAML de flasgger y consultas
  SQL multilinea que se ven forzadas y menos legibles con 88 columnas.
- **`.flake8`**: mismo `max-line-length = 100` para no pelear con black, e
  ignora `E203`/`W503` (ambas reglas contradicen el estilo que aplica black
  de forma automatica: espacios en slices y saltos de linea antes de un
  operador binario).
- **`.pre-commit-config.yaml`**: corre `black`, `isort` y `flake8` en cada
  `git commit` local. Instalacion:

  ```bash
  pip install -r requirements-dev.txt
  pre-commit install          # una sola vez por clon del repo
  ```

  A partir de ahi, cada commit local pasa por los 3 checks automaticamente.
  El mismo trio se vuelve a correr en GitHub Actions (seccion 6) como red de
  seguridad, por si alguien commitea con `--no-verify` o sin `pre-commit`
  instalado.

### 3.3 `requirements.txt` vs `requirements-dev.txt`

Se separaron a proposito: `requirements.txt` son las dependencias que
**corren en produccion** (las que instala el Dockerfile), mientras que
`requirements-dev.txt` (que incluye a `requirements.txt` con `-r`) agrega
`pytest`, `black`, `isort`, `flake8` y `pre-commit` — herramientas que solo
se usan en desarrollo local y en CI. Esto mantiene la imagen Docker final
mas pequena y evita instalar herramientas de testing en el contenedor que
se despliega en produccion.

### 3.4 Dependabot (`.github/dependabot.yml`)

Configurado para abrir PRs automaticos semanales ante actualizaciones de:
dependencias de Python (`pip`), la imagen base del `Dockerfile` (`docker`) y
las GitHub Actions usadas en los workflows (`github-actions`). Esto cubre
las tres superficies de dependencias del proyecto sin necesidad de
revisarlas manualmente.

---

---

## 4. Docker

> **Nota sobre la version de Python**: la especificacion original de este
> proyecto pedia `python:3.11-slim`. Al validar el pipeline de CI contra
> PyPI real se detecto que `numpy==2.5.3` y `scipy==1.18.1` (resueltos por
> `pip freeze` al fijar versiones en la seccion 1) **ya no publican wheels
> para Python 3.11** — solo para 3.12 en adelante. Como el objetivo
> explicito es que el pipeline de CI/CD quede en verde de verdad (no solo
> en teoria), se ajusto tanto el `Dockerfile` como `ci-cd.yml` a
> **Python 3.12**, que si esta soportado por todas las dependencias
> fijadas. Se documenta aqui como una desviacion deliberada de la
> especificacion original, motivada por disponibilidad real de paquetes.

### 4.1 Por que multi-stage

[`Dockerfile`](Dockerfile) tiene dos etapas:

1. **`build`**: crea un virtualenv en `/opt/venv` e instala
   `requirements.txt` ahi dentro.
2. **runtime** (imagen final, sin nombre): parte de `python:3.12-slim` otra
   vez, limpio, y copia **solo** el virtualenv ya resuelto
   (`COPY --from=build /opt/venv /opt/venv`), el codigo de `app/`, `run.py`
   y `models/`.

La ventaja de separar build de runtime: si en el futuro alguna dependencia
de `requirements.txt` necesitara compilarse desde codigo fuente (por
ejemplo, un wheel sin build precompilado para alguna arquitectura), las
herramientas de compilacion (`gcc`, headers de desarrollo, etc.) solo
existirian en la etapa `build` y nunca inflarian el tamano de la imagen que
realmente se despliega en produccion.

### 4.2 Modelos: se copian, no se reentrenan en el build

El Dockerfile hace `COPY models ./models` en vez de correr
`entrenar_modelo.py` durante el build. Se eligio esto deliberadamente:

- **Reproducibilidad**: el build de Docker no depende de que scikit-learn
  entrene exactamente igual en la maquina de CI que en local (aunque el
  script es determinista con semilla fija, cualquier entrenamiento durante
  el build seria trabajo redundante).
- **Velocidad de build**: copiar 2 archivos `.joblib` de unos pocos KB es
  instantaneo; entrenar (aunque sea rapido, segundos) sigue siendo mas
  lento y anade una dependencia innecesaria a pandas/scikit-learn/el
  dataset CSV dentro de la imagen final.
- Es coherente con el flujo documentado en la seccion 1: los modelos se
  versionan explicitamente en `models/` y el pointer `models/latest.json`
  decide cual esta activo — el Dockerfile simplemente respeta ese
  contrato.

### 4.3 Seguridad: usuario no-root

Se crea un usuario de sistema `appuser` (`groupadd --system` /
`useradd --system`) y todos los `COPY` de la etapa final usan
`--chown=appuser:appuser`, seguido de `USER appuser` antes del `CMD`. Si
alguna vez se descubriera una vulnerabilidad de ejecucion remota de codigo
en el proceso de gunicorn/Flask, el proceso no correria con privilegios de
root dentro del contenedor — una practica de seguridad estandar para
contenedores en produccion.

### 4.4 Puerto dinamico (`$PORT`) y arranque

```dockerfile
ENV PORT=8080
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 run:app
```

La mayoria de plataformas de hosting de contenedores (Render, Cloud Run,
etc.) inyectan la variable `PORT` en tiempo de ejecucion y esperan que el
contenedor escuche ahi; el `ENV PORT=8080` de arriba es solo el valor por
defecto para pruebas locales con `docker run` sin `-e PORT=...`. El `CMD`
se escribe en **forma shell**
(no en forma exec/array) a proposito: es la unica forma en que `$PORT` se
expande al arrancar el contenedor en vez de pasarse como texto literal.

### 4.5 `.dockerignore`

Excluye del contexto de build: `.git`, `.github`, `tests/`, herramientas de
desarrollo (`requirements-dev.txt`, `.pre-commit-config.yaml`, `.flake8`,
`pytest.ini`), `.venv`, cache de Python, `.env`/`.env.example` y `data/`
(el CSV crudo no hace falta en producción, solo los `.joblib` ya
entrenados). Esto acelera el build (menos contexto que enviar al daemon de
Docker) y evita que un secreto local (`.env`) termine copiado a una imagen.

### 4.6 Como construir y correr la imagen localmente

```bash
docker build -t clasificador-tickets .

docker run -p 8080:8080 \
  -e DATABASE_URL="postgresql://usuario:password@host:5432/postgres" \
  -e PORT=8080 \
  clasificador-tickets

curl http://localhost:8080/health
```

> **Nota de este entorno de desarrollo**: la maquina donde se escribio este
> Dockerfile no tiene Docker Desktop instalado, por lo que el build no se
> pudo ejecutar end-to-end aqui mismo — se valido manualmente linea por
> linea (rutas de `COPY` consistentes con `WORKDIR /app` y con como
> `app/ml/clasificador.py` resuelve `models/` de forma relativa al proyecto,
> permisos de `appuser` sobre `/opt/venv` y `/app`, forma shell del `CMD`
> para expandir `$PORT`). El workflow de CI/CD (seccion 6) construye esta
> misma imagen en un runner de GitHub Actions (Linux) en cada push a
> `main`, lo cual sirve como la validacion real end-to-end antes de
> desplegar (seccion 7, Render).

---

## 5. Repositorio GitHub

### 5.1 Estrategia de ramas

- **`main`**: siempre desplegable. Cada push a `main` dispara el deploy a
  Render (seccion 6). Se protege en GitHub (Settings -> Branches -> Add
  branch ruleset para `main`): exigir que los checks `lint`, `seguridad` y
  `tests` pasen antes de poder mergear, y prohibir el push directo sin PR
  (esto se configura manualmente en la UI de GitHub por el dueño del repo;
  no es algo que un script pueda activar de forma remota via git).
- **`develop`**: rama de trabajo diario. Los `feature/*` se mergean aqui
  primero; sirve como zona de integracion antes de promover a `main`.
- **`feature/*`**: una rama por cambio puntual (ej. `feature/ci-cd-pipeline`,
  la rama en la que se desarrollo la seccion 6 de este mismo README). Se
  abre un Pull Request de `feature/*` hacia `develop` (o hacia `main` para
  un hotfix urgente), donde corren los checks de CI antes de mergear.

```
feature/*  ──PR──►  develop  ──PR──►  main  ──push──► deploy automatico
```

### 5.2 Conventional Commits

Todos los commits de este repo siguen [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/):

| Prefijo | Uso | Ejemplo real de este repo |
|---|---|---|
| `feat:` | funcionalidad nueva | `feat: agrega endpoint de estadisticas` |
| `fix:` | correccion de bug | `fix: corrige timezone en fecha_hora` |
| `docs:` | solo documentacion | `docs: documenta flujo de ramas y Conventional Commits` |
| `test:` | tests o config de tests | `test: agrega suite de pytest, calidad de codigo y dependabot` |
| `ci:` | pipelines/workflows | `ci: agrega pipeline de CI/CD con lint, seguridad, tests y deploy` |
| `chore:` | mantenimiento (deps, config) | `chore: actualiza flask-limiter` (lo que abrira Dependabot) |
| `build:` | build system / Docker | `build: agrega Dockerfile multi-stage para produccion` |

### 5.3 Por que no se "reescribio" el historial retroactivamente

Las secciones 1-3 de este proyecto se commitearon directamente en `main`
(antes de que existiera `develop`) porque en ese punto no habia todavia
nada desplegable ni un pipeline de CI que proteger. A partir de la seccion
6 (este mismo cambio), el trabajo nuevo sigue el flujo `feature/* -> develop
-> main` descrito arriba.

---

## 6. CI/CD con GitHub Actions

### 6.1 Los 5 jobs de `.github/workflows/ci-cd.yml`

```mermaid
flowchart LR
    A[push / pull_request] --> B[lint]
    A --> C[seguridad: gitleaks]
    A --> D[tests: pytest]
    A --> G[docker-build: build + smoke test]
    B --> E{needs}
    C --> E
    D --> E
    G --> E
    E -->|solo si push a main Y todos OK| F[deploy: dispara Render]
```

1. **`lint`**: `black --check`, `isort --check`, `flake8` sobre todo el
   codigo. Falla si algo no esta formateado o tiene un error de estilo.
2. **`seguridad`**: corre `gitleaks` (accion oficial `gitleaks-action`)
   sobre el historial completo del repo (`fetch-depth: 0`), buscando
   credenciales filtradas por accidente. Se eligio gitleaks sobre
   trufflehog por ser mas rapido y no requerir cuenta/API key para uso
   basico en CI. `.gitleaks.toml` extiende el ruleset por defecto y solo
   agrega una excepcion para `.env.example` (que a proposito contiene un
   connection string con forma real pero con password de placeholder).
3. **`tests`**: instala `requirements-dev.txt` y corre `pytest -v` (los 6
   tests de la seccion 3, con la base de datos mockeada).
4. **`docker-build`**: construye la imagen del `Dockerfile` (seccion 4) en
   el runner de GitHub Actions (Linux, con Docker preinstalado) y corre un
   smoke test: levanta el contenedor y verifica que `/health` responda
   *algun* codigo HTTP (503 es el esperado sin una DB real — lo que se
   valida es que gunicorn/Flask arrancan dentro del contenedor, no la
   conectividad a Postgres). Esta imagen **no se publica a ningun
   registry** aqui, solo se valida que compila y arranca — Render
   construye su propia copia de la imagen a partir del mismo `Dockerfile`
   al desplegar (seccion 7). Este job existe porque la maquina de
   desarrollo no tenia Docker instalado (seccion 4) — este es el punto
   donde el Dockerfile se valida de verdad, end-to-end.
5. **`deploy`**: **solo corre si los 4 jobs anteriores pasaron** (`needs:
   [lint, seguridad, tests, docker-build]`) **y** el evento es un push
   directo a `main` (nunca en un Pull Request ni en otra rama). Si falta
   cualquiera de esos checks, GitHub Actions simplemente no ejecuta este
   job — el pipeline completo queda en rojo y no se despliega nada,
   cumpliendo el requisito de "si algo falla, no se despliega".

> **Nota sobre la plataforma de despliegue**: la especificacion original de
> este proyecto proponia Google Cloud Run. Durante la configuracion real
> con una cuenta de Google Cloud en Mexico, el flujo de verificacion de
> facturacion (formulario de RFC/regimen fiscal) fallo repetidamente con
> errores del lado de Google (`OR_BACR2_59`) sin una solucion clara y
> disponible en el momento. Se decidio migrar el despliegue a **Render**
> (render.com), que ofrece el mismo tipo de despliegue gratuito de
> contenedores Docker con una verificacion de cuenta mucho mas simple (sin
> campos fiscales especificos de un pais). El `Dockerfile` de la seccion 4
> no cambio en absoluto — Render lo usa directamente, igual que hubiera
> hecho Cloud Run.

### 6.2 Deploy sin bloquear el pipeline si aun no hay credenciales de Render

El job `deploy` primero verifica (`steps.check`) si existe el secret
`RENDER_DEPLOY_HOOK_URL`. Si no existe (por ejemplo, recien clonaste este
repo y todavia no configuraste Render), el paso de disparo del deploy se
salta limpiamente (`if: steps.check.outputs.listo == 'true'`) en vez de
fallar con un error críptico. Esto es una decision deliberada no exigida
explicitamente por la especificacion: preferimos que el pipeline muestre
"omitido" en vez de "fallido" cuando el motivo es simplemente que el
usuario aun no completo la seccion 7 (configuracion de Render), para no
confundir un problema de configuracion pendiente con un bug real del
codigo.

### 6.3 Secrets de GitHub necesarios (Settings -> Secrets and variables -> Actions)

| Secret | Contenido | Usado por |
|---|---|---|
| `RENDER_DEPLOY_HOOK_URL` | URL secreta del "Deploy Hook" del servicio en Render (seccion 7.3) | dispara el deploy con un simple `curl -X POST` |

`DATABASE_URL` y `LOG_LEVEL` **ya no son secrets de GitHub** — se
configuran directamente como variables de entorno en el dashboard de
Render (seccion 7.2), porque es Render (no GitHub Actions) quien corre el
contenedor en produccion.

`GITHUB_TOKEN` (usado por gitleaks para poder comentar en PRs) lo provee
GitHub automaticamente, no hace falta crearlo.

### 6.4 Job separado para Pull Requests

El trigger `pull_request: branches: [main]` hace que `lint`, `seguridad` y
`tests` corran en cada PR hacia `main` — exactamente el "lint + tests sin
deploy" que pide la especificacion — porque el job `deploy` tiene la
condicion `github.event_name == 'push'`, que es falsa para un evento de
tipo `pull_request`.

---

## 7. Despliegue gratuito (Render)

Guia paso a paso para dejar la API corriendo en una URL publica sin gastar
un centavo.

### 7.1 Cuenta y servicio en Render

1. Crear una cuenta en [render.com](https://render.com) — el boton
   **"Sign up with GitHub"** es el mas directo, porque de una vez deja
   autorizado el acceso a tus repos.
2. **New -> Web Service** -> selecciona el repositorio
   `clasificador-tickets`.
3. Render detecta el `Dockerfile` automaticamente y lo usa como metodo de
   build (no elijas un runtime de Python manual).
4. En **Instance Type**, selecciona **Free** ($0/month) — no el plan de
   $7/mes que Render sugiere por defecto para "mas potencia".
5. Antes de desplegar, configura las dos secciones siguientes
   (Environment Variables y Auto-Deploy).

### 7.2 Variables de entorno en Render

En la seccion **"Environment Variables"** del formulario de creacion (o
despues, en **Settings -> Environment** del servicio ya creado):

| Key | Value |
|---|---|
| `DATABASE_URL` | El connection string de Supabase (seccion 2.4) — **usa el modo "Session pooler", no "Direct connection"** (ver nota de IPv4 abajo) |
| `LOG_LEVEL` | `INFO` |

> **Por que "Session pooler" y no "Direct connection"**: Supabase ofrece
> conexion directa por IPv6 de forma gratuita; IPv4 es un add-on de pago.
> La mayoria de plataformas de hosting gratuito (incluyendo Render y Cloud
> Run) solo tienen salida a internet por IPv4. Usar "Direct connection"
> haría que el despliegue nunca pudiera conectarse a la base de datos. El
> "Session pooler" de Supabase si soporta IPv4 sin costo extra, y se
> comporta como una conexion de sesion normal (compatible con el pool de
> conexiones de `psycopg2` que usa `app/db/supabase_client.py`).

### 7.3 Desactivar Auto-Deploy y crear el Deploy Hook

Por defecto, Render redespliega automaticamente en cada push a `main`,
**sin pasar por nuestro pipeline de CI**. Para que el deploy solo ocurra
si `lint`/`seguridad`/`tests`/`docker-build` pasan (igual que se penso
originalmente para Cloud Run):

1. En el formulario de creacion (o en **Settings** del servicio), busca
   **"Auto-Deploy"** y cambialo de **"On Commit"** a **"Off"**.
2. Click en **"Deploy web service"** para crear el servicio (este primer
   deploy es manual, para que el servicio exista).
3. Una vez creado, ve a **Settings** del servicio -> busca la seccion
   **"Deploy Hook"** -> copia esa URL (empieza con
   `https://api.render.com/deploy/srv-...`).
4. En GitHub: **Settings -> Secrets and variables -> Actions -> New
   repository secret** -> crea `RENDER_DEPLOY_HOOK_URL` con esa URL.

A partir de aqui, cada push a `main` que pase los 4 checks de calidad hace
que GitHub Actions llame a esa URL, y Render reconstruye y despliega la
ultima version — exactamente el mismo gate que se diseño para Cloud Run,
solo con una plataforma distinta por debajo.

### 7.4 URL publica

Render asigna una URL con la forma:

```
https://clasificador-tickets.onrender.com
```

Una vez desplegado, esa URL expone:

| Endpoint | URL |
|---|---|
| Clasificar | `POST https://clasificador-tickets.onrender.com/clasificar` |
| Health | `GET https://clasificador-tickets.onrender.com/health` |
| Dashboard | `GET https://clasificador-tickets.onrender.com/dashboard` |
| Swagger | `GET https://clasificador-tickets.onrender.com/apidocs` |

> Esta seccion se actualizara con la URL exacta (el nombre real del
> servicio puede variar) en cuanto el primer deploy corra exitosamente.

### 7.5 UptimeRobot: monitoreo gratuito + reducir cold starts

Render "duerme" las instancias gratuitas tras un periodo sin trafico para
no cobrar por tiempo inactivo; el efecto secundario es un "cold start"
(unos 10-30 segundos) en la primera request tras la inactividad.

1. Crear una cuenta gratuita en [uptimerobot.com](https://uptimerobot.com).
2. **Add New Monitor**: tipo `HTTP(s)`, URL =
   `https://clasificador-tickets.onrender.com/health`, intervalo = 5
   minutos (el minimo del plan gratuito).
3. Esto cumple dos funciones a la vez: alerta por correo si `/health`
   empieza a devolver algo distinto de 200, y mantiene la instancia
   "tibia" (recibiendo trafico cada 5 minutos) para que los usuarios
   reales casi nunca sufran el cold start.

---

## 8. Variables de entorno y seguridad

Resumen de las medidas de seguridad aplicadas en todo el proyecto (cada una
ya se implemento en secciones anteriores; aqui se consolidan):

| Medida | Donde | Seccion |
|---|---|---|
| `.env.example` con todas las llaves, sin valores reales | [`.env.example`](.env.example) | 2 |
| `.env` real ignorado por git | [`.gitignore`](.gitignore) | 2 |
| Ninguna credencial hardcodeada (siempre `os.environ`) | `app/db/supabase_client.py`, `run.py` | 2 |
| `DATABASE_URL` vive solo en el dashboard de Render y en `.env` local, nunca en GitHub ni en el repo | Render (seccion 7.2) | 7 |
| El unico secret de GitHub (`RENDER_DEPLOY_HOOK_URL`) no expone credenciales de la app, solo dispara un deploy | `.github/workflows/ci-cd.yml` | 6-7 |
| Escaneo de secretos filtrados en cada push/PR | `gitleaks` + `.gitleaks.toml` | 6 |
| Usuario no-root en el contenedor | `Dockerfile` | 4 |
| Rate limiting contra abuso del free tier | `Flask-Limiter` en `/clasificar` | 2 |

**Que hacer si un secret se filtra por accidente**: rotarlo inmediatamente
(regenerar el password de Supabase, o regenerar el Deploy Hook de Render
desde su dashboard) y revocar el anterior — cambiar el valor en el secret
de GitHub no invalida una credencial que ya fue expuesta en el historial
de git; hay que invalidarla en el proveedor (Supabase/Render) directamente.
`gitleaks` en CI esta ahi como red de seguridad para detectar esto lo antes
posible, no como sustituto de revisar el `git diff` antes de cada commit.

---

## Estructura completa del repositorio

```
.
├── .github/
│   ├── workflows/
│   │   └── ci-cd.yml              # lint + seguridad + tests + deploy
│   └── dependabot.yml              # actualizaciones automaticas de dependencias
├── app/
│   ├── __init__.py                 # application factory (create_app)
│   ├── routes.py                    # los 5 endpoints
│   ├── ml/
│   │   └── clasificador.py           # carga modelos .joblib y expone clasificar(texto)
│   ├── db/
│   │   └── supabase_client.py        # conexion a Postgres con pool + reintentos
│   ├── templates/
│   │   ├── dashboard.html             # panel de estadisticas (Chart.js)
│   │   └── nuevo_ticket.html           # formulario publico "Mesa de Ayuda"
│   └── static/
├── data/
│   └── tickets_dataset.csv          # dataset sintetico (generado)
├── models/
│   ├── categoria_v{fecha}.joblib    # modelo de categoria (generado)
│   ├── urgencia_v{fecha}.joblib     # modelo de urgencia (generado)
│   └── latest.json                  # puntero a la version activa
├── tests/
│   ├── conftest.py                   # fixtures (app, client, mock de DB)
│   └── test_api.py                    # tests del contrato HTTP de la API
├── generar_dataset.py
├── entrenar_modelo.py
├── seed_demo_data.py                # pobla el dashboard con datos de demo
├── run.py                           # entry point (dev server / gunicorn)
├── init_db.sql                      # CREATE TABLE de clasificaciones y predicciones_dudosas
├── Dockerfile                        # build multi-stage, usuario no-root
├── .dockerignore
├── .gitleaks.toml                    # excepcion para .env.example
├── requirements.txt
├── requirements-dev.txt             # + pytest/black/isort/flake8/pre-commit
├── pytest.ini
├── pyproject.toml                    # config de black/isort
├── .flake8
├── .pre-commit-config.yaml
├── .env.example
├── .gitignore
└── README.md
```
