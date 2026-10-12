# Documentación técnica del código — Actividad 3

**Archivo:** `actividad3_ml.py`  
**Curso:** Inteligencia Artificial / Análisis de Datos  
**Universidad:** Ibero — Facultad de Ingeniería

---

## Tabla de contenidos

1. [Visión general del script](#1-visión-general-del-script)
2. [Imports y dependencias](#2-imports-y-dependencias)
3. [Función `generar_dataset()`](#3-función-generar_dataset)
4. [Función `entrenar_y_evaluar()`](#4-función-entrenar_y_evaluar)
5. [Función `demo_predicciones()`](#5-función-demo_predicciones)
6. [Punto de entrada `__main__`](#6-punto-de-entrada-__main__)
7. [Flujo completo de datos](#7-flujo-completo-de-datos)
8. [Resultados obtenidos](#8-resultados-obtenidos)

---

## 1. Visión general del script

El script resuelve un **problema de clasificación binaria**:

> Dado un par de estaciones (origen, destino), predecir si la ruta óptima requiere al menos un transbordo.

| Clase | Valor | Significado |
|-------|-------|-------------|
| 0 | Sin transbordo | La ruta se hace en una sola línea |
| 1 | Con transbordo | La ruta requiere cambiar de línea al menos una vez |

El pipeline completo es:

```
knowledge_base.py + search_engine.py
        |
        v
generar_dataset()          <- crea X (features) e y (etiquetas)
        |
        v
train_test_split()         <- divide 80% entrenamiento / 20% prueba
        |
        v
StandardScaler             <- normaliza las features
        |
        v
MLPClassifier.fit()        <- entrena la red neuronal
        |
        v
accuracy_score + report    <- evalúa sobre el 20% de prueba
        |
        v
demo_predicciones()        <- prueba el modelo con pares nuevos
```

---

## 2. Imports y dependencias

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

Estas tres líneas aseguran que Python pueda encontrar los módulos `knowledge_base` y `search_engine` aunque el script se ejecute desde un directorio diferente (por ejemplo, cuando se sube a Google Colab).

- `os.path.abspath(__file__)` obtiene la ruta absoluta del script actual.
- `os.path.dirname(...)` extrae el directorio que lo contiene.
- `sys.path.insert(0, ...)` agrega ese directorio al inicio del path de búsqueda de módulos.

```python
import numpy as np
from itertools import permutations
```

- `numpy` se usa para construir los arrays `X` e `y` y para operaciones numéricas.
- `permutations` genera todos los pares ordenados de estaciones: (A→B) y (B→A) son pares distintos porque la dirección puede importar en una red de transporte.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Todas estas importaciones pertenecen a `scikit-learn`:

| Clase / función | Rol en el script |
|-----------------|-----------------|
| `MLPClassifier` | El modelo de red neuronal que se entrena |
| `train_test_split` | Divide el dataset en entrenamiento y prueba |
| `StandardScaler` | Normaliza las features a media 0 y varianza 1 |
| `accuracy_score` | Calcula el porcentaje de predicciones correctas |
| `classification_report` | Muestra precision, recall y F1 por clase |
| `confusion_matrix` | Tabla de verdaderos/falsos positivos y negativos |

```python
from knowledge_base import ESTACIONES, construir_grafo
from search_engine import astar
```

Se reutilizan directamente los módulos de la Actividad 2:
- `ESTACIONES`: diccionario con la información de cada estación (troncal, coordenadas x/y).
- `construir_grafo()`: construye el grafo bidireccional a partir de las reglas de conexión.
- `astar()`: implementación del algoritmo A* que devuelve `(ruta, tiempo, n_transbordos)`.

---

## 3. Función `generar_dataset()`

### Propósito

Crear el dataset etiquetado que servirá de entrada al modelo. No existe un archivo CSV externo; los datos se generan programáticamente usando el grafo de TransMilenio.

### Paso 1 — Codificación de troncales

```python
troncales = sorted(set(info["troncal"] for info in ESTACIONES.values()))
troncal_a_id = {t: i for i, t in enumerate(troncales)}
```

Las troncales son cadenas de texto (`"Caracas"`, `"NQS"`, etc.). Los modelos de ML solo trabajan con números, por eso se crea un diccionario que mapea cada troncal a un entero único:

```
Americas → 0
Caracas  → 1
Calle80  → 2
NQS      → 3
Suba     → 4
```

### Paso 2 — Iteración sobre todos los pares

```python
for origen, destino in permutations(estaciones, 2):
    resultado = astar(grafo, origen, destino)
    if resultado is None:
        continue
    _, tiempo_total, n_transbordos = resultado
```

`permutations(estaciones, 2)` genera todos los pares ordenados de estaciones sin repetición del mismo elemento. Con 35 estaciones hay hasta 35 × 34 = 1 190 pares. Si A* no encuentra ruta (estaciones desconectadas), el par se descarta.

El resultado de A* es una tupla `(ruta, tiempo_total, n_transbordos)`. Solo se usa `n_transbordos` para construir la etiqueta; el tiempo y la ruta completa se descartan con `_`.

### Paso 3 — Cálculo de features

```python
xo = ESTACIONES[origen]["x"]
yo = ESTACIONES[origen]["y"]
xd = ESTACIONES[destino]["x"]
yd = ESTACIONES[destino]["y"]

dist_manhattan = abs(xo - xd) + abs(yo - yd)   # feature 1
diff_x         = abs(xo - xd)                   # feature 2
diff_y         = abs(yo - yd)                   # feature 3
misma_troncal  = int(...)                        # feature 4
tronc_o        = troncal_a_id[...]              # feature 5
tronc_d        = troncal_a_id[...]              # feature 6
```

Cada par de estaciones se convierte en un vector de 6 números. Las coordenadas `x` e `y` en `ESTACIONES` son valores enteros que representan la posición aproximada de cada estación en una cuadrícula abstracta de la ciudad:

| Feature | Descripción | Rango típico |
|---------|-------------|--------------|
| `distancia_manhattan` | Distancia total en la cuadrícula | 0 – 9 |
| `diff_x` | Diferencia horizontal | 0 – 4 |
| `diff_y` | Diferencia vertical | 0 – 9 |
| `misma_troncal` | 1 si comparten línea, 0 si no | 0 o 1 |
| `troncal_origen_enc` | Línea del origen codificada | 0 – 4 |
| `troncal_destino_enc` | Línea del destino codificada | 0 – 4 |

### Paso 4 — Etiqueta

```python
y_rows.append(1 if n_transbordos > 0 else 0)
```

La etiqueta es simplemente `1` si A* tuvo que cruzar al menos una conexión marcada como `transbordo=True` en el grafo, y `0` si la ruta fue completamente directa.

### Paso 5 — Construcción de arrays NumPy

```python
X = np.array(X_rows, dtype=float)
y = np.array(y_rows, dtype=int)
```

Se convierten las listas Python a arrays NumPy:
- `X` tiene forma `(n_muestras, 6)`.
- `y` tiene forma `(n_muestras,)`.

---

## 4. Función `entrenar_y_evaluar()`

### 4.1 Split 80/20

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `test_size=0.20` | 20% | Reserva el 20% de muestras para prueba |
| `random_state=42` | 42 | Semilla fija → resultados reproducibles |
| `stratify=y` | `y` | Mantiene la proporción de clases en ambas particiones |

`stratify=y` es importante porque el dataset está desbalanceado (≈18% sin transbordo, ≈82% con transbordo). Sin estratificación, el 20% de prueba podría quedar con una distribución muy distinta.

### 4.2 Normalización

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

`StandardScaler` transforma cada feature restando su media y dividiendo entre su desviación estándar:

```
x_normalizado = (x - media) / std
```

**Por qué dos llamadas distintas:**  
- `fit_transform(X_train)`: calcula media y std **solo con los datos de entrenamiento**, luego aplica la transformación.  
- `transform(X_test)`: aplica la **misma** media y std calculadas en entrenamiento. Si se usara `fit_transform` en el test, habría *data leakage* (el modelo estaría usando información del test durante el entrenamiento).

### 4.3 Definición del modelo

```python
modelo = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
```

Descripción de cada parámetro:

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `hidden_layer_sizes=(64, 32)` | Tupla | Dos capas ocultas: la primera con 64 neuronas, la segunda con 32 |
| `activation="relu"` | ReLU | Función de activación en las capas ocultas. Permite aprender relaciones no lineales sin el problema de gradientes que desaparecen (vanishing gradient) |
| `solver="adam"` | Adam | Optimizador adaptativo. Ajusta la tasa de aprendizaje automáticamente durante el entrenamiento |
| `max_iter=500` | 500 | Número máximo de épocas (pasadas completas sobre el dataset) |
| `random_state=42` | 42 | Semilla para reproducibilidad |
| `early_stopping=True` | — | Detiene el entrenamiento si la pérdida de validación no mejora durante 20 épocas seguidas |
| `validation_fraction=0.1` | 10% | Fracción del conjunto de entrenamiento usada para monitorear early stopping |
| `n_iter_no_change=20` | 20 | Número de épocas sin mejora antes de detener |

La arquitectura completa de la red es:

```
Capa de entrada  : 6 neuronas  (una por feature)
Capa oculta 1    : 64 neuronas (activación ReLU)
Capa oculta 2    : 32 neuronas (activación ReLU)
Capa de salida   : 2 neuronas  (softmax -> probabilidad de clase 0 y clase 1)
```

### 4.4 Entrenamiento

```python
modelo.fit(X_train_sc, y_train)
```

Esta línea ejecuta todo el proceso de aprendizaje: propagación hacia adelante, cálculo de la pérdida (cross-entropy), retropropagación del gradiente y actualización de los pesos con Adam. Al finalizar, `modelo.n_iter_` indica en cuántas épocas convergió.

### 4.5 Evaluación

```python
y_pred = modelo.predict(X_test_sc)
acc_train = accuracy_score(y_train, modelo.predict(X_train_sc))
acc_test  = accuracy_score(y_test, y_pred)
```

Se obtienen predicciones sobre el conjunto de prueba (que el modelo nunca vio durante el entrenamiento) y se calcula la exactitud.

```python
print(classification_report(y_test, y_pred, target_names=["Sin transbordo", "Con transbordo"]))
```

El `classification_report` muestra para cada clase:
- **Precision**: de los que predije como clase X, ¿cuántos realmente lo eran?
- **Recall**: de todos los que realmente son clase X, ¿cuántos detecté?
- **F1-score**: media armónica entre precision y recall.

```python
cm = confusion_matrix(y_test, y_pred)
```

La matriz de confusión tiene la siguiente estructura:

```
                  Pred: Sin transb.   Pred: Con transb.
Real: Sin transb.       VP (33)            FN (8)
Real: Con transb.       FP (2)             VN (182)
```

- **VP (Verdaderos Positivos)**: predijo "sin transbordo" y era correcto.
- **VN (Verdaderos Negativos)**: predijo "con transbordo" y era correcto.
- **FP (Falso Positivo)**: predijo "sin transbordo" pero era con transbordo.
- **FN (Falso Negativo)**: predijo "con transbordo" pero era sin transbordo.

---

## 5. Función `demo_predicciones()`

```python
features = np.array([[...]])
features_sc = scaler.transform(features)
pred = modelo.predict(features_sc)[0]
```

Para cada par de ejemplo se construye manualmente el vector de features (exactamente igual que en `generar_dataset()`), se normaliza con el **mismo** `scaler` ya entrenado, y se obtiene la predicción.

Luego se ejecuta A* para obtener el valor real y comparar:

```python
resultado = astar(grafo, origen, destino)
real_transb = resultado[2] if resultado else "?"
correcto = "✓" if pred_str == real_str else "✗"
```

Esto permite verificar visualmente si el modelo generaliza correctamente a pares específicos.

---

## 6. Punto de entrada `__main__`

```python
if __name__ == "__main__":
    X, y, feature_names = generar_dataset()
    modelo, scaler = entrenar_y_evaluar(X, y, feature_names)
    demo_predicciones(modelo, scaler)
```

El bloque `if __name__ == "__main__"` garantiza que el script solo ejecute la lógica principal cuando se corre directamente (`python actividad3_ml.py`), no cuando es importado como módulo por otro script.

Las tres funciones se llaman en secuencia y se pasan los objetos necesarios entre ellas: `X` e `y` van a `entrenar_y_evaluar()`, y el `modelo` y `scaler` ya entrenados van a `demo_predicciones()`.

---

## 7. Flujo completo de datos

```
ESTACIONES (dict)
    |
    +---> permutations() ------> 1190 pares (origen, destino)
    |                                  |
    |                             astar() para cada par
    |                                  |
    +---> coordenadas x, y -----> 6 features por par
    |
    v
X: (1122, 6)   y: (1122,)
    |
    v
train_test_split(stratify=y)
    |
    +---> X_train (897, 6)    y_train (897,)
    |         |
    |    StandardScaler.fit_transform()
    |         |
    |    X_train_sc (897, 6)
    |         |
    |    MLPClassifier.fit()
    |         |
    |    modelo entrenado
    |
    +---> X_test (225, 6)     y_test (225,)
              |
         StandardScaler.transform()   <- mismos parámetros que en train
              |
         X_test_sc (225, 6)
              |
         modelo.predict()
              |
         y_pred (225,)
              |
         accuracy_score / classification_report / confusion_matrix
```

---

## 8. Resultados obtenidos

| Métrica | Valor |
|---------|-------|
| Muestras totales | 1 122 |
| Muestras entrenamiento | 897 |
| Muestras prueba | 225 |
| Épocas hasta convergencia | 24 |
| Accuracy entrenamiento | 95.76% |
| Accuracy prueba | **95.56%** |
| F1 "Sin transbordo" | 0.87 |
| F1 "Con transbordo" | **0.97** |

La diferencia mínima entre accuracy de entrenamiento (95.76%) y de prueba (95.56%) indica que el modelo **no está sobreajustado**: generaliza bien a datos nuevos.
