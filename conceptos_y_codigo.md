# Conceptos de Aprendizaje Supervisado y su implementación en el código

**Proyecto:** Sistema Inteligente de Rutas — TransMilenio  
**Curso:** Inteligencia Artificial / Análisis de Datos  
**Universidad:** Ibero — Facultad de Ingeniería  
**Referencia:** Hurbans, R. (2020). *Grokking Artificial Intelligence Algorithms*. Manning.

---

## Tabla de contenidos

1. [¿Qué es el aprendizaje supervisado?](#1-qué-es-el-aprendizaje-supervisado)
2. [Definición del problema de clasificación](#2-definición-del-problema-de-clasificación)
3. [Dataset etiquetado](#3-dataset-etiquetado)
4. [Features (variables de entrada)](#4-features-variables-de-entrada)
5. [Split de datos: entrenamiento y prueba](#5-split-de-datos-entrenamiento-y-prueba)
6. [Normalización de datos](#6-normalización-de-datos)
7. [El modelo: Perceptrón Multicapa (MLP)](#7-el-modelo-perceptrón-multicapa-mlp)
8. [Entrenamiento: el método `.fit()`](#8-entrenamiento-el-método-fit)
9. [Evaluación: accuracy y métricas adicionales](#9-evaluación-accuracy-y-métricas-adicionales)
10. [Early stopping](#10-early-stopping)
11. [Conexión con la Actividad 2](#11-conexión-con-la-actividad-2)

---

## 1. ¿Qué es el aprendizaje supervisado?

El **aprendizaje supervisado** es la rama del Machine Learning donde el algoritmo aprende a partir de ejemplos que ya tienen una respuesta conocida (etiqueta). El objetivo es que el modelo encuentre la función que relaciona las entradas con las salidas, de modo que pueda predecir correctamente en datos que nunca ha visto.

```
Aprendizaje supervisado:
    Entrada (X) + Etiqueta conocida (y) ---> Modelo aprende f(X) ≈ y
    Datos nuevos (X_nuevo)              ---> Modelo predice y_nuevo
```

**En este proyecto:** se le entrega al modelo información sobre dos estaciones (sus posiciones, sus líneas) y una etiqueta que indica si la ruta entre ellas requiere transbordo. Tras el entrenamiento, el modelo puede predecir esa respuesta para cualquier par nuevo.

---

## 2. Definición del problema de clasificación

Dentro del aprendizaje supervisado existen dos tipos principales de problemas:

- **Regresión:** la salida es un número continuo (ej. predecir el tiempo de viaje en minutos).
- **Clasificación:** la salida es una categoría discreta (ej. ¿requiere transbordo? sí/no).

Este proyecto resuelve un problema de **clasificación binaria** porque solo hay dos clases posibles.

| Clase | Etiqueta numérica | Significado real |
|-------|-------------------|-----------------|
| Sin transbordo | `0` | La ruta óptima va por una sola línea |
| Con transbordo | `1` | La ruta óptima cruza al menos una vez de línea |

**Líneas del código donde se define la etiqueta:**

```python
# actividad3_ml.py — línea 116
y_rows.append(1 if n_transbordos > 0 else 0)
```

El valor `n_transbordos` proviene directamente del algoritmo A* de la Actividad 2 (`search_engine.py`). Si A* encontró al menos un paso marcado con `transbordo=True` en el grafo, la etiqueta es `1`; de lo contrario, `0`.

---

## 3. Dataset etiquetado

En el aprendizaje supervisado el dataset es la colección de pares `(entrada, etiqueta)`. Aquí no existe un archivo CSV externo: los datos se **generan programáticamente** aprovechando el grafo de conocimiento ya construido en la Actividad 2.

**Concepto:** un dataset etiquetado es el "libro de texto" del modelo. Cuantos más ejemplos representativos tenga, mejor aprenderá.

**En el código** (líneas 88–116):

```python
for origen, destino in permutations(estaciones, 2):   # todos los pares posibles
    resultado = astar(grafo, origen, destino)           # ejecuta A* real
    if resultado is None:
        continue                                        # descarta pares sin ruta
    _, tiempo_total, n_transbordos = resultado

    X_rows.append([dist_manhattan, diff_x, diff_y,
                   misma_troncal, tronc_o, tronc_d])   # features del par
    y_rows.append(1 if n_transbordos > 0 else 0)       # etiqueta real
```

`permutations(estaciones, 2)` genera todos los pares ordenados (A→B y B→A son distintos porque la red puede no ser perfectamente simétrica). Se usan A→B y B→A como muestras independientes, lo que maximiza el tamaño del dataset con los datos disponibles.

**Resultado:** 1 122 muestras etiquetadas a partir de 35 estaciones.

---

## 4. Features (variables de entrada)

Las **features** son las variables numéricas que el modelo usa para aprender. El arte del Machine Learning está en elegir features que contengan información útil para predecir la etiqueta.

### Razonamiento detrás de cada feature

| # | Feature | Por qué es útil |
|---|---------|-----------------|
| 1 | `distancia_manhattan` | A mayor distancia geográfica, más probable un transbordo |
| 2 | `diff_x` | Diferencia horizontal: cruzar la ciudad de oeste a este casi siempre implica cambio de línea |
| 3 | `diff_y` | Diferencia vertical: moverse a lo largo de una misma troncal no requiere transbordo |
| 4 | `misma_troncal` | Si origen y destino están en la misma línea, la ruta suele ser directa |
| 5 | `troncal_origen_enc` | La línea de origen influye: algunas están más conectadas que otras |
| 6 | `troncal_destino_enc` | Idem para el destino |

**Código de cálculo de features** (líneas 98–112):

```python
xo = ESTACIONES[origen]["x"]
yo = ESTACIONES[origen]["y"]
xd = ESTACIONES[destino]["x"]
yd = ESTACIONES[destino]["y"]

dist_manhattan = abs(xo - xd) + abs(yo - yd)      # feature 1
diff_x         = abs(xo - xd)                      # feature 2
diff_y         = abs(yo - yd)                      # feature 3
misma_troncal  = int(ESTACIONES[origen]["troncal"] # feature 4
                  == ESTACIONES[destino]["troncal"])
tronc_o        = troncal_a_id[ESTACIONES[origen]["troncal"]]  # feature 5
tronc_d        = troncal_a_id[ESTACIONES[destino]["troncal"]] # feature 6
```

Las features 5 y 6 son el resultado de la **codificación de etiquetas** (*Label Encoding*): transformar texto (`"Caracas"`, `"NQS"`) en números (`1`, `3`) porque los modelos ML no procesan cadenas de texto directamente.

```python
# líneas 76–77
troncales = sorted(set(info["troncal"] for info in ESTACIONES.values()))
troncal_a_id = {t: i for i, t in enumerate(troncales)}
```

---

## 5. Split de datos: entrenamiento y prueba

**Concepto:** Si entrenamos y evaluamos el modelo con los mismos datos, no sabremos si realmente aprendió o simplemente memorizó las respuestas. Por eso se divide el dataset:

- **Conjunto de entrenamiento (80%):** el modelo aprende de estos datos.
- **Conjunto de prueba (20%):** se usa exclusivamente para medir qué tan bien generaliza el modelo a datos nuevos.

Esta división replica el escenario real: el modelo aprende del pasado y se evalúa con situaciones que no ha visto.

**Código** (líneas 151–153):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

El parámetro `stratify=y` es fundamental cuando las clases están desbalanceadas (el 82% de los ejemplos tienen transbordo). Sin él, el 20% de prueba podría quedar con una proporción diferente, haciendo la evaluación poco representativa.

Con `random_state=42` la división es siempre la misma, lo que hace los resultados **reproducibles**: cualquier persona que ejecute el script obtendrá exactamente las mismas particiones.

---

## 6. Normalización de datos

**Concepto:** Las redes neuronales son sensibles a la escala de los datos. Si una feature varía entre 0 y 9 y otra entre 0 y 1, la primera puede dominar el aprendizaje simplemente por tener valores más grandes, aunque no sea más informativa. La normalización pone todas las features en la misma escala.

El `StandardScaler` aplica la transformación **Z-score**:

```
x_norm = (x - media) / desviacion_estandar
```

Después de la normalización, cada feature tiene media ≈ 0 y desviación estándar ≈ 1.

**Código** (líneas 163–165):

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # aprende estadísticas del train
X_test_sc  = scaler.transform(X_test)        # aplica las mismas al test
```

**Regla de oro — no contaminar el test:**  
Se llama `fit_transform` solo sobre `X_train`. El `scaler` aprende la media y la desviación estándar de los datos de entrenamiento, y luego aplica esos mismos valores al conjunto de prueba con `transform`. Si se volviera a llamar `fit` sobre el test, el modelo indirectamente "vería" información del test durante el entrenamiento (*data leakage*), inflando artificialmente el accuracy.

---

## 7. El modelo: Perceptrón Multicapa (MLP)

### El Perceptrón (base conceptual)

Un Perceptrón simple es la unidad básica: recibe entradas numéricas, las multiplica por pesos, suma los resultados más un sesgo (*bias*) y aplica una función de activación para producir una salida.

```
salida = f( w1·x1 + w2·x2 + ... + wn·xn + b )
```

El Perceptrón simple solo puede resolver problemas **linealmente separables**. Para problemas más complejos se usa el **Perceptrón Multicapa (MLP)**, que apila varias capas de neuronas.

### Arquitectura del MLP en este proyecto

```
Entrada (6 features)
    │
    ▼
[Capa oculta 1: 64 neuronas — activación ReLU]
    │
    ▼
[Capa oculta 2: 32 neuronas — activación ReLU]
    │
    ▼
[Capa de salida: 2 neuronas — softmax]
    │
    ▼
Clase 0 (sin transbordo) o Clase 1 (con transbordo)
```

**Código de definición del modelo** (líneas 175–184):

```python
modelo = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # 2 capas ocultas
    activation="relu",             # ReLU en capas ocultas
    solver="adam",                 # optimizador Adam
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
```

### Función de activación ReLU

**ReLU** (*Rectified Linear Unit*) es la función de activación más usada en redes profundas:

```
ReLU(z) = max(0, z)
```

Si la entrada es negativa, la salida es 0. Si es positiva, pasa tal cual. Esto introduce **no linealidad** en la red (necesaria para aprender patrones complejos) sin el problema de gradientes que desaparecen que tiene la función sigmoide.

En el código se especifica con `activation="relu"`.

### Optimizador Adam

El entrenamiento ajusta los pesos minimizando la función de pérdida (diferencia entre la predicción y la etiqueta real). **Adam** es un algoritmo de optimización que adapta automáticamente la tasa de aprendizaje para cada parámetro. Es una variante mejorada del descenso por gradiente estocástico (SGD).

En el código se especifica con `solver="adam"`.

---

## 8. Entrenamiento: el método `.fit()`

**Concepto:** El entrenamiento es el proceso donde el modelo ajusta sus pesos internos para minimizar el error en los datos de entrenamiento. En cada época:

1. Se presentan los datos de entrenamiento en lotes (*batches*).
2. Se calcula la salida de la red (propagación hacia adelante).
3. Se mide el error entre la salida y la etiqueta real (función de pérdida).
4. Se calcula el gradiente del error respecto a cada peso (retropropagación).
5. Adam actualiza los pesos en la dirección que reduce el error.

**Código** (línea 192):

```python
modelo.fit(X_train_sc, y_train)
```

Una sola línea en scikit-learn ejecuta todo ese proceso. Al finalizar, `modelo.n_iter_` indica en cuántas épocas convergió la red (en este caso, 24 épocas, mucho menos que el máximo de 500).

---

## 9. Evaluación: accuracy y métricas adicionales

### Accuracy (exactitud)

La métrica más intuitiva: porcentaje de predicciones correctas.

```
Accuracy = (predicciones correctas) / (total de predicciones)
```

**Código** (líneas 202–203):

```python
acc_train = accuracy_score(y_train, modelo.predict(X_train_sc))
acc_test  = accuracy_score(y_test,  y_pred)
```

Se mide en **ambos conjuntos** para detectar sobreajuste (*overfitting*): si el accuracy de entrenamiento es muy alto pero el de prueba es bajo, el modelo memorizó los datos pero no aprendió a generalizar. En este proyecto la diferencia es mínima (95.76% vs 95.56%), lo que indica buen balance.

### Precision, Recall y F1

Para problemas con clases desbalanceadas, el accuracy solo puede ser engañoso. Si el 82% de los pares tienen transbordo, un modelo que siempre diga "con transbordo" tendría 82% de accuracy sin haber aprendido nada.

**Código** (líneas 209–212):

```python
print(classification_report(
    y_test, y_pred,
    target_names=["Sin transbordo", "Con transbordo"]
))
```

- **Precision** de "Sin transbordo": de todos los que predijo como directos, ¿cuántos lo eran realmente? (94%)
- **Recall** de "Sin transbordo": de todos los que realmente eran directos, ¿cuántos detectó? (80%)
- **F1**: media armónica entre precision y recall (0.87)

El recall de 80% para "Sin transbordo" indica que el modelo falla en 8 de cada 41 rutas directas, clasificándolas como rutas con transbordo. Esto es aceptable dado que esa clase es minoritaria.

### Matriz de confusión

**Código** (líneas 215–222):

```python
cm = confusion_matrix(y_test, y_pred)
```

Resultado obtenido:

```
                  Pred: Sin transb.   Pred: Con transb.
Real: Sin transb.        33                  8
Real: Con transb.         2                182
```

- 33 rutas directas identificadas correctamente.
- 8 rutas directas clasificadas erróneamente como con transbordo (falso negativo).
- 2 rutas con transbordo clasificadas erróneamente como directas (falso positivo).
- 182 rutas con transbordo identificadas correctamente.

---

## 10. Early stopping

**Concepto:** El sobreajuste ocurre cuando la red memoriza los datos de entrenamiento en lugar de aprender patrones generales. Una forma de prevenirlo es detener el entrenamiento cuando la pérdida sobre un pequeño subconjunto de validación deja de mejorar.

**Código** (líneas 181–183):

```python
early_stopping=True,
validation_fraction=0.1,
n_iter_no_change=20,
```

- `validation_fraction=0.1`: se reserva el 10% del conjunto de entrenamiento como "validación interna" (distinto del conjunto de prueba del 20%).
- `n_iter_no_change=20`: si la pérdida no mejora en 20 épocas consecutivas, el entrenamiento se detiene automáticamente.

En la práctica, el modelo convergió en solo 24 épocas, lo que indica que aprendió rápidamente sin necesitar las 500 épocas máximas.

---

## 11. Conexión con la Actividad 2

La Actividad 2 implementó un **sistema experto basado en reglas** (BFS y A*) para encontrar rutas. La Actividad 3 usa esos mismos datos para construir un **modelo de aprendizaje automático**.

| Aspecto | Actividad 2 (Búsqueda) | Actividad 3 (Supervisado) |
|---------|----------------------|--------------------------|
| Enfoque | Reglas explícitas + grafo | Aprendizaje desde ejemplos |
| Algoritmo | BFS, A* | MLPClassifier |
| Salida | Ruta completa paso a paso | Predicción binaria (¿transbordo?) |
| Generalización | Exacta para el grafo definido | Aproximada para cualquier par |
| Módulos reutilizados | `knowledge_base.py`, `search_engine.py` | Los mismos, como fuente de datos |

**Punto de conexión clave:** el algoritmo A* de la Actividad 2 actúa como **oráculo** para generar las etiquetas del dataset de entrenamiento:

```python
# search_engine.py — astar() devuelve cuántos transbordos tiene la ruta óptima
resultado = astar(grafo, origen, destino)
_, _, n_transbordos = resultado

# actividad3_ml.py — eso se convierte en la etiqueta del ejemplo
y_rows.append(1 if n_transbordos > 0 else 0)
```

El flujo es: A* calcula la verdad → la verdad se convierte en etiqueta → el MLP aprende a aproximar esa verdad solo con features geométricas, sin necesitar ejecutar A*. Una vez entrenado, el modelo puede responder en microsegundos a la pregunta "¿esta ruta tendrá transbordo?" sin recorrer el grafo.
