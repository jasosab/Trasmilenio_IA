"""
ACTIVIDAD 3: Aprendizaje Supervisado — TransMilenio Bogotá
==========================================================
Se usa la misma base de conocimiento de la Actividad 2 para
generar un dataset etiquetado y entrenar un clasificador MLP.

PROBLEMA DE CLASIFICACIÓN:
  Dado el par (estación_origen, estación_destino),
  predecir si la ruta óptima requiere transbordo.
    Clase 0 → Ruta directa (sin transbordo)
    Clase 1 → Ruta con al menos un transbordo

FEATURES por cada par:
  1. distancia_manhattan   : |x_o - x_d| + |y_o - y_d|
  2. diff_x                : |x_o - x_d|
  3. diff_y                : |y_o - y_d|
  4. misma_troncal         : 1 si misma troncal, 0 si no
  5. troncal_origen_enc    : troncal del origen (Label Encoded)
  6. troncal_destino_enc   : troncal del destino (Label Encoded)

PIPELINE:
  Generación dataset → Split 80/20 → StandardScaler
  → MLPClassifier → Accuracy + Reporte completo

Curso  : Inteligencia Artificial / Análisis de Datos
Univ.  : Universidad Ibero — Facultad de Ingeniería
"""

# ──────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────
import sys
import os

# Permite importar los módulos de la actividad 2 aunque se ejecute
# desde otro directorio (útil en Google Colab: sube los archivos y
# ejecuta este script en la misma carpeta).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from itertools import permutations

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Módulos propios de la Actividad 2
from knowledge_base import ESTACIONES, construir_grafo
from search_engine import astar

# ──────────────────────────────────────────────────────────────────
# 2. GENERACIÓN DEL DATASET
# ──────────────────────────────────────────────────────────────────

def generar_dataset():
    """
    Genera un dataset de todos los pares ordenados (origen, destino).

    Para cada par calcula features e invoca A* para determinar si
    la ruta óptima requiere transbordo (etiqueta).

    Retorna:
        X  : array (n_muestras, n_features)
        y  : array (n_muestras,) con valores 0 o 1
        feature_names : lista de nombres de features
    """
    grafo = construir_grafo()
    estaciones = list(ESTACIONES.keys())

    # Codificar troncales como enteros
    troncales = sorted(set(info["troncal"] for info in ESTACIONES.values()))
    troncal_a_id = {t: i for i, t in enumerate(troncales)}

    X_rows = []
    y_rows = []

    total = len(estaciones) * (len(estaciones) - 1)
    procesados = 0

    print(f"\n[Generación dataset] Pares a evaluar: {total}")
    print("  (Ejecutando A* para cada par — puede tardar unos segundos...)\n")

    for origen, destino in permutations(estaciones, 2):
        resultado = astar(grafo, origen, destino)

        if resultado is None:
            procesados += 1
            continue  # Par inconexo — lo omitimos

        _, tiempo_total, n_transbordos = resultado

        # ── Features ──────────────────────────────────────────────
        xo = ESTACIONES[origen]["x"]
        yo = ESTACIONES[origen]["y"]
        xd = ESTACIONES[destino]["x"]
        yd = ESTACIONES[destino]["y"]

        dist_manhattan = abs(xo - xd) + abs(yo - yd)
        diff_x         = abs(xo - xd)
        diff_y         = abs(yo - yd)
        misma_troncal  = int(ESTACIONES[origen]["troncal"] ==
                             ESTACIONES[destino]["troncal"])
        tronc_o        = troncal_a_id[ESTACIONES[origen]["troncal"]]
        tronc_d        = troncal_a_id[ESTACIONES[destino]["troncal"]]

        X_rows.append([dist_manhattan, diff_x, diff_y,
                        misma_troncal, tronc_o, tronc_d])

        # ── Etiqueta ──────────────────────────────────────────────
        # 0 = ruta directa, 1 = requiere transbordo
        y_rows.append(1 if n_transbordos > 0 else 0)

        procesados += 1

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    print(f"  Muestras generadas : {len(X)}")
    print(f"  Sin transbordo (0) : {sum(y == 0)} ({100*sum(y==0)/len(y):.1f}%)")
    print(f"  Con transbordo (1) : {sum(y == 1)} ({100*sum(y==1)/len(y):.1f}%)")

    feature_names = [
        "distancia_manhattan",
        "diff_x",
        "diff_y",
        "misma_troncal",
        "troncal_origen_enc",
        "troncal_destino_enc",
    ]

    return X, y, feature_names


# ──────────────────────────────────────────────────────────────────
# 3. ENTRENAMIENTO Y EVALUACIÓN
# ──────────────────────────────────────────────────────────────────

def entrenar_y_evaluar(X, y, feature_names):
    """
    Divide los datos 80/20, normaliza, entrena MLP y evalúa.
    """
    print("\n" + "═" * 58)
    print("  PASO 1: SPLIT DE DATOS  (80% entrenamiento / 20% prueba)")
    print("═" * 58)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"  Muestras entrenamiento : {len(X_train)}")
    print(f"  Muestras prueba        : {len(X_test)}")

    # ── Normalización ─────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  PASO 2: NORMALIZACIÓN  (StandardScaler)")
    print("═" * 58)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # aprende media/std en train
    X_test_sc  = scaler.transform(X_test)        # aplica a test (sin filtración)

    print("  Media de features (train):", np.round(scaler.mean_, 2))
    print("  Desv. std (train)        :", np.round(scaler.scale_, 2))

    # ── Modelo: MLPClassifier ─────────────────────────────────────
    print("\n" + "═" * 58)
    print("  PASO 3: MODELO  —  MLPClassifier (Red Neuronal MLP)")
    print("═" * 58)

    modelo = MLPClassifier(
        hidden_layer_sizes=(64, 32),   # 2 capas ocultas
        activation="relu",             # función de activación ReLU
        solver="adam",                 # optimizador Adam (variante de SGD)
        max_iter=500,                  # máximo de épocas
        random_state=42,
        early_stopping=True,           # detiene si no mejora en val.
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    print(f"  Arquitectura : entrada({X_train.shape[1]}) → 64 → 32 → salida(2)")
    print(f"  Activación   : ReLU")
    print(f"  Optimizador  : Adam")
    print(f"  Max épocas   : 500")

    print("\n  Entrenando...")
    modelo.fit(X_train_sc, y_train)
    print(f"  Convergió en {modelo.n_iter_} iteraciones (épocas).")

    # ── Evaluación ────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  PASO 4: EVALUACIÓN  —  Datos de prueba (20%)")
    print("═" * 58)

    y_pred = modelo.predict(X_test_sc)

    acc_train = accuracy_score(y_train, modelo.predict(X_train_sc))
    acc_test  = accuracy_score(y_test,  y_pred)

    print(f"\n  Accuracy entrenamiento : {acc_train * 100:.2f}%")
    print(f"  Accuracy prueba        : {acc_test  * 100:.2f}%")

    print("\n  Reporte de clasificación (prueba):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Sin transbordo", "Con transbordo"]
    ))

    print("  Matriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    etiquetas = ["Sin transb.", "Con transb."]
    ancho = 14
    print(f"  {'':>{ancho}} | {etiquetas[0]:>{ancho}} | {etiquetas[1]:>{ancho}}")
    print(f"  {'-'*(ancho*3 + 6)}")
    for i, fila in enumerate(cm):
        pred_label = f"Real: {etiquetas[i]}"
        print(f"  {pred_label:>{ancho}} | {fila[0]:>{ancho}} | {fila[1]:>{ancho}}")

    return modelo, scaler


# ──────────────────────────────────────────────────────────────────
# 4. DEMO: PREDICCIONES SOBRE NUEVOS PARES
# ──────────────────────────────────────────────────────────────────

def demo_predicciones(modelo, scaler):
    """
    Muestra predicciones del modelo sobre pares de ejemplo.
    También ejecuta A* para comparar la predicción con la realidad.
    """
    grafo = construir_grafo()

    pares_demo = [
        ("Portal Norte",   "Portal Sur"),       # misma troncal Caracas
        ("Portal Suba",    "Portal Sur"),        # troncales distintas, transbordo esperado
        ("Portal 80",      "Portal Américas"),   # troncales distintas
        ("Mártires",       "Ricaurte"),          # misma troncal, vecinos directos
        ("NQS Calle 30",   "Av. Jiménez"),       # distintas, transbordo esperado
        ("Calle 100",      "Calle 72"),          # misma troncal, cercanas
    ]

    print("\n" + "═" * 58)
    print("  PASO 5: DEMO — Predicciones sobre pares nuevos")
    print("═" * 58)
    print(f"\n  {'Origen':<22} {'Destino':<22} {'Pred':<18} {'Real'}")
    print(f"  {'-'*80}")

    troncales = sorted(set(info["troncal"] for info in ESTACIONES.values()))
    troncal_a_id = {t: i for i, t in enumerate(troncales)}

    for origen, destino in pares_demo:
        if origen not in ESTACIONES or destino not in ESTACIONES:
            continue

        xo = ESTACIONES[origen]["x"]
        yo = ESTACIONES[origen]["y"]
        xd = ESTACIONES[destino]["x"]
        yd = ESTACIONES[destino]["y"]

        features = np.array([[
            abs(xo - xd) + abs(yo - yd),
            abs(xo - xd),
            abs(yo - yd),
            int(ESTACIONES[origen]["troncal"] == ESTACIONES[destino]["troncal"]),
            troncal_a_id[ESTACIONES[origen]["troncal"]],
            troncal_a_id[ESTACIONES[destino]["troncal"]],
        ]])

        features_sc = scaler.transform(features)
        pred = modelo.predict(features_sc)[0]
        pred_str = "Con transbordo" if pred == 1 else "Sin transbordo"

        resultado = astar(grafo, origen, destino)
        real_transb = resultado[2] if resultado else "?"
        real_str = "Con transbordo" if real_transb > 0 else "Sin transbordo"
        correcto = "✓" if pred_str == real_str else "✗"

        print(f"  {origen:<22} {destino:<22} {pred_str:<18} {real_str} {correcto}")

    print()


# ──────────────────────────────────────────────────────────────────
# 5. PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
==============================================================
  ACTIVIDAD 3 - Aprendizaje Supervisado  (TransMilenio)
      Universidad Ibero - IA / Analisis de Datos
==============================================================
Problema : Clasificar si una ruta requiere transbordo
Modelo   : MLPClassifier (Red Neuronal Multicapa)
Libreria : scikit-learn
""")

    # 1. Generar datos etiquetados
    X, y, feature_names = generar_dataset()

    # 2. Entrenar y evaluar
    modelo, scaler = entrenar_y_evaluar(X, y, feature_names)

    # 3. Demo interactiva de predicciones
    demo_predicciones(modelo, scaler)
