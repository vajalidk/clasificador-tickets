"""
Entrena los dos modelos de clasificacion de tickets de soporte:

1. categoria: facturacion / soporte tecnico / queja / informacion general
2. urgencia:  alta / media / baja

Ambos modelos usan el mismo tipo de pipeline:
    TfidfVectorizer -> LinearSVC -> CalibratedClassifierCV

Se eligio LinearSVC + CalibratedClassifierCV en vez de LogisticRegression
porque, en datasets de texto pequenos y de alta dimensionalidad (como este,
con TF-IDF de n-gramas), los SVM lineales suelen separar mejor las clases;
CalibratedClassifierCV envuelve el SVM (que por si solo no da probabilidades)
con calibracion sigmoide/isotonica via validacion cruzada, para obtener un
`predict_proba` con probabilidades reales y bien calibradas, en vez de un
simple softmax de la distancia al hiperplano.

El script es reproducible (semilla fija), imprime metricas de evaluacion y
guarda cada modelo con un nombre versionado por fecha, actualizando
`models/latest.json` para permitir rollback instantaneo (apuntar de nuevo a
una version anterior sin reentrenar).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "tickets_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
LATEST_PATH = MODELS_DIR / "latest.json"

# Stopwords en espanol reducidas a las mas frecuentes y menos informativas
# para este dominio. Se evita depender de nltk/spacy para no anadir pesos ni
# descargas extra al contenedor de produccion.
STOPWORDS_ES = [
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "durante",
    "e",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "erais",
    "eran",
    "eras",
    "eres",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estas",
    "este",
    "esto",
    "estos",
    "fue",
    "fueron",
    "ha",
    "han",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "me",
    "mi",
    "mis",
    "mucho",
    "muchos",
    "muy",
    "nada",
    "ni",
    "no",
    "nos",
    "nosotros",
    "o",
    "os",
    "otra",
    "otras",
    "otro",
    "otros",
    "para",
    "pero",
    "poco",
    "por",
    "porque",
    "que",
    "quien",
    "se",
    "segun",
    "ser",
    "si",
    "sin",
    "sobre",
    "su",
    "sus",
    "te",
    "tener",
    "ti",
    "tiene",
    "todo",
    "todos",
    "tu",
    "tus",
    "un",
    "una",
    "uno",
    "unos",
    "y",
    "ya",
    "yo",
]


def construir_pipeline() -> Pipeline:
    """Crea el pipeline TF-IDF + LinearSVC calibrado, compartido por ambos
    modelos (categoria y urgencia). Cada llamada crea una instancia nueva
    para no compartir estado entre los dos entrenamientos."""
    vectorizador = TfidfVectorizer(
        stop_words=STOPWORDS_ES,
        ngram_range=(1, 2),
        max_features=3000,
        min_df=1,
        sublinear_tf=True,
    )
    svm_base = LinearSVC(random_state=RANDOM_STATE, class_weight="balanced")
    # cv=3: con ~200 ejemplos de entrenamiento y hasta 4 clases, un cv mayor
    # dejaria muy pocos ejemplos por clase en cada fold de calibracion.
    clasificador = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
    return Pipeline(
        [
            ("tfidf", vectorizador),
            ("clf", clasificador),
        ]
    )


def entrenar_y_evaluar(
    nombre_tarea: str,
    X: pd.Series,
    y: pd.Series,
) -> tuple[Pipeline, dict]:
    """Entrena un pipeline para una tarea (categoria o urgencia), imprime
    metricas de evaluacion en el set de prueba y devuelve el pipeline
    entrenado con TODOS los datos (para maximizar el uso del dataset
    pequeno) junto con las metricas calculadas sobre el split de prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, zero_division=0)
    reporte_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    matriz = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    print(f"\n{'=' * 70}")
    print(f"Modelo: {nombre_tarea}")
    print(f"{'=' * 70}")
    print(f"Accuracy en test ({len(y_test)} ejemplos): {acc:.4f}\n")
    print("Reporte de clasificacion (precision / recall / F1 por clase):")
    print(reporte)
    print(f"Matriz de confusion (orden de clases: {sorted(y.unique())}):")
    print(matriz)

    # Reentrena con el 100% de los datos para el modelo que se despliega:
    # con un dataset sintetico tan pequeno, no tiene sentido "desperdiciar"
    # el 20% de test en el modelo final una vez que ya medimos su
    # desempeno de forma honesta arriba.
    pipeline_final = construir_pipeline()
    pipeline_final.fit(X, y)

    metricas = {
        "accuracy_test": round(float(acc), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": reporte_dict,
    }
    return pipeline_final, metricas


def guardar_modelo(pipeline: Pipeline, nombre: str, version: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = MODELS_DIR / f"{nombre}_v{version}.joblib"
    joblib.dump(pipeline, ruta)
    return ruta


def actualizar_latest(
    version: str,
    ruta_categoria: Path,
    ruta_urgencia: Path,
    metricas_categoria: dict,
    metricas_urgencia: dict,
) -> None:
    """Actualiza el puntero a la version activa de los modelos. La app en
    produccion siempre lee este archivo para saber que .joblib cargar, lo
    que permite hacer rollback a una version anterior con un solo cambio
    (editar este JSON) sin necesidad de reentrenar ni redeplegar codigo."""
    contenido = {
        "version_activa": version,
        "actualizado_en": datetime.now(timezone.utc).isoformat(),
        "modelos": {
            "categoria": {
                "archivo": ruta_categoria.name,
                "metricas": metricas_categoria,
            },
            "urgencia": {
                "archivo": ruta_urgencia.name,
                "metricas": metricas_urgencia,
            },
        },
    }
    with LATEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)
    print(f"\nPuntero de version activa actualizado: {LATEST_PATH}")


def main() -> None:
    if not DATA_PATH.exists():
        print(
            f"ERROR: no se encontro el dataset en {DATA_PATH}. "
            "Ejecuta primero: python generar_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["texto", "categoria", "urgencia"])
    print(f"Dataset cargado: {len(df)} tickets desde {DATA_PATH}")

    pipeline_categoria, metricas_categoria = entrenar_y_evaluar(
        "categoria", df["texto"], df["categoria"]
    )
    pipeline_urgencia, metricas_urgencia = entrenar_y_evaluar(
        "urgencia", df["texto"], df["urgencia"]
    )

    version = datetime.now().strftime("%Y%m%d")
    ruta_categoria = guardar_modelo(pipeline_categoria, "categoria", version)
    ruta_urgencia = guardar_modelo(pipeline_urgencia, "urgencia", version)

    print(f"\nModelo de categoria guardado en: {ruta_categoria}")
    print(f"Modelo de urgencia guardado en:  {ruta_urgencia}")

    actualizar_latest(version, ruta_categoria, ruta_urgencia, metricas_categoria, metricas_urgencia)

    print("\nEntrenamiento completado con exito.")


if __name__ == "__main__":
    main()
