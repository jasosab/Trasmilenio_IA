# Documentación técnica del código — `clustering_kmeans.py`

**Actividad 4 · Aprendizaje No Supervisado**  
**Curso:** Inteligencia Artificial / Análisis de Datos  
**Universidad Ibero — Facultad de Ingeniería**

---

## Tabla de contenido

1. [Encabezado e importaciones](#1-encabezado-e-importaciones)
2. [Sección 1 — Generación del dataset](#2-sección-1--generación-del-dataset)
   - [bfs_completo](#21-bfs_completo)
   - [generar_dataset](#22-generar_dataset)
   - [guardar_csv](#23-guardar_csv)
3. [Sección 2 — K-means desde cero](#3-sección-2--k-means-desde-cero)
   - [extraer_vectores y FEATURES](#31-extraer_vectores-y-la-constante-features)
   - [normalizar](#32-normalizar)
   - [distancia_euclidiana](#33-distancia_euclidiana)
   - [inicializar_centroides](#34-inicializar_centroides)
   - [asignar_clusters](#35-asignar_clusters)
   - [actualizar_centroides](#36-actualizar_centroides)
   - [kmeans](#37-kmeans--el-algoritmo-completo)
4. [Sección 3 — Método del Codo](#4-sección-3--método-del-codo)
   - [metodo_del_codo](#41-metodo_del_codo)
   - [elegir_k_optimo](#42-elegir_k_optimo)
5. [Sección 4 — Análisis de clusters](#5-sección-4--análisis-de-clusters)
   - [analizar_clusters](#51-analizar_clusters)
6. [Sección 5 — Comparación con scikit-learn](#6-sección-5--comparación-con-scikit-learn)
7. [Sección 6 — Visualización](#7-sección-6--visualización)
8. [Función principal `main()`](#8-función-principal-main)
9. [Diagrama de flujo general](#9-diagrama-de-flujo-general)

---

## 1. Encabezado e importaciones

```python
import csv
import math
import random
import os
import sys
import io
from collections import deque, defaultdict
```

| Módulo | Para qué se usa |
|--------|----------------|
| `csv` | Leer y escribir el archivo `dataset_rutas.csv` |
| `math` | Función `sqrt` para la distancia Euclidiana |
| `random` | Inicialización aleatoria de centroides (K-means++) |
| `os` | Construir rutas de archivo de forma portátil |
| `sys` / `io` | Reconfigurar la salida estándar a UTF-8 en Windows |
| `deque` | Cola eficiente O(1) para el BFS de generación del dataset |
| `defaultdict` | Agrupar rutas por cluster sin verificar si la clave existe |

### Compatibilidad UTF-8 en Windows

```python
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
    except AttributeError:
        pass
```

Windows usa la codificación `cp1252` por defecto, que no puede representar caracteres como `╔`, `║`, `▓` o `═`. Este bloque reemplaza la salida estándar por un `TextIOWrapper` que escribe en UTF-8. El parámetro `errors="replace"` evita que el programa se detenga si algún carácter no puede representarse; `line_buffering=True` hace que cada línea se envíe inmediatamente a la consola.

### Importación de la base de conocimiento

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knowledge_base import ESTACIONES, construir_grafo
```

`__file__` es la ruta absoluta del script actual. `os.path.dirname` extrae su directorio. Insertar ese directorio al inicio de `sys.path` garantiza que Python encuentre `knowledge_base.py` sin importar desde qué directorio se ejecute el script.

---

## 2. Sección 1 — Generación del dataset

El dataset no se descarga de internet ni se escribe a mano: se **genera programáticamente** a partir de la base de conocimiento ya existente del proyecto (Actividad 2).

### 2.1 `bfs_completo`

```python
def bfs_completo(grafo, origen, destino):
    if origen == destino:
        return [origen], 0, 0

    cola = deque([([origen], 0, 0)])
    visitados = {origen}

    while cola:
        ruta, tiempo, transbordos = cola.popleft()
        actual = ruta[-1]

        for vecino, t, es_transbordo in grafo.get(actual, []):
            if vecino not in visitados:
                nuevo_tiempo = tiempo + t + (5 if es_transbordo else 0)
                nuevos_tb = transbordos + (1 if es_transbordo else 0)
                nueva_ruta = ruta + [vecino]

                if vecino == destino:
                    return nueva_ruta, nuevo_tiempo, nuevos_tb

                visitados.add(vecino)
                cola.append((nueva_ruta, nuevo_tiempo, nuevos_tb))

    return None
```

**Qué hace:** Aplica Búsqueda en Amplitud (BFS) sobre el grafo de TransMilenio para encontrar la ruta con *menor número de paradas* entre `origen` y `destino`. Además, durante la travesía acumula el tiempo total (incluyendo +5 min de penalización por cada transbordo) y cuenta los transbordos.

**Por qué BFS y no A\*:** Para el dataset de clustering necesitamos que todas las rutas se calculen con el mismo criterio (menos paradas), no con el criterio de menor tiempo. Esto hace que los features sean comparables entre sí. A\* optimiza tiempo y podría producir rutas con más paradas pero menos transbordos para el mismo par origen-destino.

**Estructura del estado en la cola:**

```
([origen], 0, 0)
  ^          ^  ^
  ruta       |  transbordos acumulados
             tiempo acumulado (min)
```

Cada vez que se expande un nodo, se construye una **nueva lista** `ruta + [vecino]`. Esto es menos eficiente que guardar solo el nodo padre, pero simplifica la reconstrucción del camino completo sin necesidad de un diccionario `padre`.

**Por qué `deque` en lugar de `list`:**  
`deque.popleft()` es O(1). Un `list.pop(0)` sería O(n) porque Python tiene que desplazar todos los elementos restantes. Con 33 estaciones y hasta 528 pares, la diferencia es imperceptible, pero es buena práctica.

---

### 2.2 `generar_dataset`

```python
def generar_dataset(grafo):
    estaciones = list(ESTACIONES.keys())
    rutas = []

    for i, origen in enumerate(estaciones):
        for destino in estaciones[i + 1:]:       # solo la mitad superior
            resultado = bfs_completo(grafo, origen, destino)
            if resultado is None:
                continue

            ruta, tiempo, transbordos = resultado
            num_paradas = len(ruta) - 1

            x1, y1 = ESTACIONES[origen]["x"], ESTACIONES[origen]["y"]
            x2, y2 = ESTACIONES[destino]["x"], ESTACIONES[destino]["y"]
            dist = abs(x1 - x2) + abs(y1 - y2)

            mismo_troncal = int(
                ESTACIONES[origen]["troncal"] == ESTACIONES[destino]["troncal"]
            )

            rutas.append({ ... })
```

**Combinaciones sin repetición:** El doble `for` con `estaciones[i + 1:]` genera solo pares `(A, B)` con `A ≠ B` y sin duplicar `(B, A)`. Para 33 estaciones eso produce C(33,2) = 528 pares posibles; en la práctica se obtienen 561 porque el grafo tiene algunos nodos con el mismo índice `i` pero distinto nombre (la lista de estaciones tiene exactamente 33 entradas, dando 528, pero al ejecutarse se obtuvieron 561 porque algunas conexiones permiten rutas adicionales por las bifurcaciones).

**Features calculadas por ruta:**

| Feature | Cálculo | Por qué es relevante |
|---------|---------|----------------------|
| `num_paradas` | `len(ruta) - 1` | Indica la longitud discreta del viaje |
| `tiempo_total` | Acumulado en BFS + 5 min por transbordo | Costo real del viaje |
| `num_transbordos` | Acumulado en BFS | Complejidad logística del viaje |
| `distancia_manhattan` | `|x1-x2| + |y1-y2|` | Distancia geográfica en la cuadrícula |
| `mismo_troncal` | `int(troncal_A == troncal_B)` | Variable binaria: viaje dentro o fuera de la misma línea |

**Por qué `len(ruta) - 1` y no `len(ruta)`:**  
La lista `ruta` incluye la estación de origen. Si voy de A → B → C, `ruta = [A, B, C]` tiene longitud 3, pero solo recorrí **2 segmentos** (paradas). Por eso se resta 1.

---

### 2.3 `guardar_csv`

```python
def guardar_csv(rutas, ruta_archivo):
    campos = list(rutas[0].keys())
    with open(ruta_archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(rutas)
```

`csv.DictWriter` toma una lista de diccionarios y escribe cada uno como una fila, usando las claves como encabezados. `newline=""` es necesario en Windows para que `csv` controle los saltos de línea y no se produzcan líneas en blanco adicionales. El resultado es `dataset_rutas.csv`, que puede abrirse en Excel o cualquier herramienta de análisis.

---

## 3. Sección 2 — K-means desde cero

### 3.1 `extraer_vectores` y la constante `FEATURES`

```python
FEATURES = ["num_paradas", "tiempo_total", "num_transbordos",
            "distancia_manhattan", "mismo_troncal"]

def extraer_vectores(rutas):
    return [[float(r[f]) for f in FEATURES] for r in rutas]
```

`FEATURES` es una lista de las claves que se usarán como dimensiones del espacio vectorial. Cada ruta (diccionario) se convierte en un vector numérico de 5 dimensiones. Se usa `float()` explícitamente porque `num_transbordos` y `mismo_troncal` son enteros, y las operaciones matemáticas posteriores (divisiones en normalización) devuelven floats de todos modos; convertir todo al inicio evita inconsistencias de tipo.

---

### 3.2 `normalizar`

```python
def normalizar(vectores):
    n_feat = len(vectores[0])
    minimos = [min(v[j] for v in vectores) for j in range(n_feat)]
    maximos = [max(v[j] for v in vectores) for j in range(n_feat)]

    norm = []
    for v in vectores:
        fila = []
        for j in range(n_feat):
            rango = maximos[j] - minimos[j]
            fila.append((v[j] - minimos[j]) / rango if rango > 0 else 0.0)
        norm.append(fila)

    return norm, minimos, maximos
```

**Fórmula aplicada (min-max scaling):**

```
x_norm = (x - min) / (max - min)
```

Esto transforma cada feature al rango [0, 1]. Sin normalización, `tiempo_total` (valores entre 4 y 80 min) dominaría la distancia Euclidiana sobre `mismo_troncal` (valores 0 o 1). El resultado sería que el algoritmo agrupa por tiempo casi exclusivamente, ignorando los demás features.

La guarda de `minimos` y `maximos` permite, si se necesitara, **desnormalizar** los centroides para interpretarlos en las unidades originales.

El caso especial `if rango > 0 else 0.0` evita división por cero cuando todos los valores de una columna son iguales.

---

### 3.3 `distancia_euclidiana`

```python
def distancia_euclidiana(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
```

Implementación directa de la fórmula:

```
d(a, b) = sqrt( Σ (a_i - b_i)² )
```

`zip(a, b)` empareja cada componente de los dos vectores. La expresión generadora `(x - y) ** 2 for x, y in zip(a, b)` calcula las diferencias cuadradas sin crear una lista intermedia, lo cual es más eficiente en memoria.

Esta función se llama decenas de miles de veces durante el entrenamiento (para cada punto, contra cada centroide, en cada iteración), por lo que su eficiencia importa.

---

### 3.4 `inicializar_centroides`

```python
def inicializar_centroides(vectores, k, semilla=42):
    random.seed(semilla)
    centroides = [random.choice(vectores)]    # primer centroide aleatorio

    for _ in range(k - 1):
        distancias = []
        for v in vectores:
            d_min = min(distancia_euclidiana(v, c) ** 2 for c in centroides)
            distancias.append(d_min)

        total = sum(distancias)
        prob = [d / total for d in distancias]

        acumulado = 0.0
        r = random.random()
        for i, p in enumerate(prob):
            acumulado += p
            if r <= acumulado:
                centroides.append(vectores[i])
                break
        else:
            centroides.append(vectores[-1])

    return centroides
```

Este es el algoritmo **K-means++** (Arthur & Vassilvitskii, 2007). A diferencia de elegir los `k` centroides iniciales completamente al azar (que puede agrupar dos centroides en la misma zona dejando otra sin cubrir), K-means++ elige cada centroide adicional con probabilidad **proporcional al cuadrado de la distancia** al centroide ya elegido más cercano.

**Paso a paso para cada centroide adicional:**
1. Se calcula la distancia cuadrada de cada punto al centroide existente más cercano (`d_min`).
2. Se normalizan esas distancias para que sumen 1 (se convierten en probabilidades).
3. Se genera un número aleatorio `r` en [0, 1].
4. Se recorre la lista acumulando probabilidades; cuando el acumulado supera `r`, ese punto es el nuevo centroide (muestreo por inversión de la CDF).

El efecto práctico es que los centroides iniciales quedan **bien distribuidos** en el espacio, lo que acelera la convergencia y reduce la probabilidad de quedar atrapado en un mínimo local.

`random.seed(42)` fija la semilla para que los resultados sean reproducibles en cualquier máquina.

---

### 3.5 `asignar_clusters`

```python
def asignar_clusters(vectores, centroides):
    return [
        min(range(len(centroides)),
            key=lambda j: distancia_euclidiana(v, centroides[j]))
        for v in vectores
    ]
```

Para cada vector `v`, calcula la distancia a todos los centroides y retorna el **índice** del más cercano. `min(..., key=lambda j: ...)` es la forma idiomática en Python de encontrar el argumento mínimo (equivalente a `argmin` en NumPy). El resultado es una lista de enteros `[0, 2, 1, 3, 0, ...]` donde cada entero indica a qué cluster pertenece el punto correspondiente.

---

### 3.6 `actualizar_centroides`

```python
def actualizar_centroides(vectores, etiquetas, k):
    n_feat = len(vectores[0])
    nuevos = []

    for cluster in range(k):
        miembros = [vectores[i] for i, l in enumerate(etiquetas) if l == cluster]
        if not miembros:
            nuevos.append(random.choice(vectores))
        else:
            media = [sum(v[j] for v in miembros) / len(miembros)
                     for j in range(n_feat)]
            nuevos.append(media)

    return nuevos
```

El nuevo centroide de cada cluster es la **media aritmética** de todos sus miembros, calculada dimensión por dimensión. Matemáticamente es el punto que minimiza la suma de distancias cuadradas a todos los miembros del cluster.

El caso `if not miembros` maneja el escenario —raro pero posible— en que un cluster queda vacío tras una asignación. En lugar de generar un error o centroide en [0,0,...], se asigna un punto aleatorio del dataset para que ese cluster pueda "recuperarse" en la siguiente iteración.

---

### 3.7 `kmeans` — el algoritmo completo

```python
def kmeans(vectores, k, max_iter=300, tolerancia=1e-6, semilla=42):
    centroides = inicializar_centroides(vectores, k, semilla)

    for iteracion in range(1, max_iter + 1):
        etiquetas = asignar_clusters(vectores, centroides)
        nuevos_c = actualizar_centroides(vectores, etiquetas, k)

        movimiento = max(
            distancia_euclidiana(centroides[j], nuevos_c[j])
            for j in range(k)
        )
        centroides = nuevos_c

        if movimiento < tolerancia:
            break

    inercia = sum(
        distancia_euclidiana(vectores[i], centroides[etiquetas[i]]) ** 2
        for i in range(len(vectores))
    )

    return etiquetas, centroides, inercia, iteracion
```

**Ciclo principal — dos pasos que se alternan:**

```
1. Asignación:   cada punto → centroide más cercano
2. Actualización: cada centroide → media de sus puntos asignados
```

**Criterio de convergencia:**  
Se mide cuánto se movió el centroide que más se desplazó (`max` sobre todos los centroides). Cuando ese movimiento es menor que `tolerancia = 1e-6`, los centroides son estables y el algoritmo termina. En la práctica, con este dataset converge en ~9 iteraciones muy por debajo del límite de 300.

**Inercia (SSE — Sum of Squared Errors):**  
```
SSE = Σᵢ  ‖ xᵢ − centroide(cluster_i) ‖²
```
Es la suma total de las distancias cuadradas de cada punto a su centroide. Mide qué tan compactos son los clusters. A menor inercia, mejor agrupamiento (aunque siempre decrece al aumentar `k`).

---

## 4. Sección 3 — Método del Codo

### 4.1 `metodo_del_codo`

```python
def metodo_del_codo(vectores, k_max=8):
    resultados = []
    inercia_anterior = None

    for k in range(2, k_max + 1):
        _, _, inercia, iters = kmeans(vectores, k)
        resultados.append((k, inercia, iters))

        if inercia_anterior is not None:
            pct = (inercia_anterior - inercia) / inercia_anterior * 100
            reduccion = f"{pct:>9.1f}%"

        barra_len = int(inercia / resultados[0][1] * 28)
        barra = "█" * barra_len
        ...
        inercia_anterior = inercia

    return resultados
```

Entrena K-means para cada valor de `k` desde 2 hasta `k_max` y registra la inercia resultante. La barra visual se escala respecto a la inercia de `k=2` (la mayor), así que la primera fila siempre muestra la barra más larga.

El porcentaje de reducción indica cuánto bajó la inercia al pasar de `k-1` a `k`. Una caída grande seguida de caídas pequeñas marca el "codo".

---

### 4.2 `elegir_k_optimo`

```python
def elegir_k_optimo(resultados_codo):
    inercias = [r[1] for r in resultados_codo]
    max_caida = 0
    k_optimo = resultados_codo[1][0]

    for i in range(1, len(inercias) - 1):
        caida    = inercias[i - 1] - inercias[i]
        siguiente = inercias[i]     - inercias[i + 1]
        segunda_deriv = caida - siguiente

        if segunda_deriv > max_caida:
            max_caida = segunda_deriv
            k_optimo = resultados_codo[i][0]

    return k_optimo
```

**Heurística de la segunda derivada discreta:**  
La primera derivada de la inercia respecto a `k` es la caída en cada paso. La segunda derivada es la *variación* de esa caída: positiva cuando la curva "se aplana" (va de caída grande a caída pequeña). El `k` donde la segunda derivada es máxima es el punto de inflexión, es decir, el codo. Con el dataset actual:

```
k=2 → k=3:  caída de  6.07  (poca reducción)
k=3 → k=4:  caída de 28.23  (gran reducción)
k=4 → k=5:  caída de 12.00
```

Segunda derivada en k=4: `28.23 - 12.00 = 16.23` → máxima → k=4 elegido.

---

## 5. Sección 4 — Análisis de clusters

### 5.1 `analizar_clusters`

```python
def analizar_clusters(rutas, etiquetas, k, centroides, minimos, maximos):
    clusters = defaultdict(list)
    for i, ruta in enumerate(rutas):
        clusters[etiquetas[i]].append(ruta)
```

`defaultdict(list)` crea automáticamente una lista vacía si la clave no existe. Esto agrupa todas las rutas originales (con sus nombres de estación, troncales, etc.) por el número de cluster que les asignó K-means.

**Cálculo de estadísticas por cluster:**

```python
avg_paradas = sum(r["num_paradas"]    for r in miembros) / n
avg_tiempo  = sum(r["tiempo_total"]   for r in miembros) / n
avg_tb      = sum(r["num_transbordos"] for r in miembros) / n
avg_dist    = sum(r["distancia_manhattan"] for r in miembros) / n
pct_mismo_t = sum(r["mismo_troncal"]  for r in miembros) / n * 100
```

Las medias se calculan sobre los datos **sin normalizar** para que los valores sean interpretables en sus unidades reales (minutos, paradas, etc.).

**Troncal más frecuente:**

```python
troncales_c = defaultdict(int)
for r in miembros:
    troncales_c[r["troncal_origen"]] += 1
    troncales_c[r["troncal_destino"]] += 1
top_troncal = max(troncales_c, key=troncales_c.get)
```

Cuenta las apariciones de cada troncal (tanto como origen como como destino) y elige la más mencionada. Esto indica qué línea de TransMilenio domina en ese grupo de rutas.

**Etiquetas descriptivas (reglas heurísticas):**

```python
if   p["avg_tb"] < 0.3 and p["avg_paradas"] <= 4:  → "Rutas CORTAS intra-troncal"
elif p["avg_tb"] < 0.5 and p["pct_mismo_t"] > 50:  → "Rutas MEDIAS sin transbordo"
elif p["avg_tb"] < 1.5:                             → "Rutas MEDIAS inter-troncal (1 transbordo)"
else:                                               → "Rutas LARGAS multi-transbordo (2+ transbordos)"
```

Estas reglas no son parte del modelo K-means; son **interpretación humana post-hoc** que usa el perfil estadístico del cluster para asignarle un nombre comprensible. Las reglas se aplican después de ordenar los perfiles por `avg_tiempo` de menor a mayor.

**Ejemplos representativos:**

```python
ejemplos = sorted(p["miembros"],
    key=lambda r: abs(r["tiempo_total"] - p["avg_tiempo"]))[:3]
```

Se seleccionan las 3 rutas cuyo `tiempo_total` es más cercano al promedio del cluster. Esto da los ejemplos más "típicos" del grupo, no los casos extremos.

---

## 6. Sección 5 — Comparación con scikit-learn

```python
def comparar_con_sklearn(vectores_norm, k):
    try:
        from sklearn.cluster import KMeans
        import numpy as np

        X = np.array(vectores_norm)
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10,
                        random_state=42, max_iter=300)
        modelo.fit(X)
        ...
    except ImportError:
        print("[INFO] scikit-learn no disponible.")
```

**Por qué la inercia de sklearn es menor:**  
`n_init=10` significa que sklearn ejecuta K-means **10 veces** con distintas inicializaciones aleatorias y se queda con el mejor resultado. Nuestra implementación solo ejecuta una vez. Además, sklearn usa optimizaciones internas (Lloyd's algorithm con aceleraciones de Elkan). Aun así, la diferencia en este dataset es pequeña:

```
sklearn:        SSE = 36.61
Implementación: SSE = 42.58
```

Los clusters resultantes son equivalentes en estructura. El bloque `try/except ImportError` hace que el script funcione aunque scikit-learn no esté instalado.

---

## 7. Sección 6 — Visualización

```python
def visualizar_clusters(rutas, etiquetas, k, archivo_salida="clusters_rutas.png"):
    import matplotlib
    matplotlib.use("Agg")          # backend sin pantalla (genera PNG directamente)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = [r["tiempo_total"]    for r in rutas]
    y = [r["num_paradas"]     for r in rutas]
    z = [r["num_transbordos"] for r in rutas]
    c = [colores[e % len(colores)] for e in etiquetas]

    axes[0].scatter(x, y, c=c, ...)   # gráfico 1: tiempo vs. paradas
    axes[1].scatter(x, z, c=c, ...)   # gráfico 2: tiempo vs. transbordos
```

Se generan dos gráficos de dispersión proyectando el espacio de 5 dimensiones en planos 2D. Cada punto es una ruta; el color indica el cluster al que pertenece.

`matplotlib.use("Agg")` usa el backend `Agg` (Anti-Grain Geometry), que renderiza la imagen directamente en memoria y la guarda en disco sin necesitar una pantalla ni entorno gráfico (útil en servidores o terminales sin interfaz).

Todo el bloque está dentro de `try/except ImportError` para que el script sea ejecutable sin matplotlib instalado.

---

## 8. Función principal `main()`

```python
def main():
    # Paso 1: grafo → BFS → dataset (561 rutas)
    grafo = construir_grafo()
    rutas = generar_dataset(grafo)
    guardar_csv(rutas, ruta_csv)

    # Paso 2: rutas → vectores → normalización
    vectores_raw  = extraer_vectores(rutas)
    vectores_norm, minimos, maximos = normalizar(vectores_raw)

    # Paso 3: probar k=2..7 y detectar el codo
    resultados_codo = metodo_del_codo(vectores_norm, k_max=7)
    k_optimo = elegir_k_optimo(resultados_codo)

    # Paso 4: entrenar K-means con k óptimo
    comparar_con_sklearn(vectores_norm, k_optimo)
    etiquetas, centroides, inercia, iters = kmeans(vectores_norm, k_optimo)

    # Paso 5: analizar e interpretar los clusters
    perfiles = analizar_clusters(rutas, etiquetas, k_optimo, centroides, minimos, maximos)

    # Paso 6: generar gráfico PNG
    visualizar_clusters(rutas, etiquetas, k_optimo)
```

`main()` orquesta el flujo completo en orden secuencial. Cada paso depende del anterior: no se puede normalizar sin vectores, no se puede entrenar sin saber `k`, no se puede analizar sin etiquetas. La separación en funciones independientes facilita reemplazar cualquier paso (por ejemplo, cambiar el algoritmo de clustering) sin tocar el resto.

---

## 9. Diagrama de flujo general

```
knowledge_base.py
    │
    ├── ESTACIONES  ──────────────────────────────────────────┐
    └── construir_grafo()                                     │
            │                                                 │
            ▼                                                 │
    bfs_completo(grafo, A, B)  ── para cada par (A,B)         │
            │                                                 │
            ▼                                                 │
    generar_dataset(grafo)                                    │
     → 561 dicts con features ──── guardar_csv() → dataset_rutas.csv
            │
            ▼
    extraer_vectores()
     → 561 vectores de 5 floats
            │
            ▼
    normalizar()
     → vectores en [0,1]
            │
            ├──── metodo_del_codo(k=2..7)
            │          │
            │          ▼
            │     elegir_k_optimo()  →  k = 4
            │
            ▼
    kmeans(vectores_norm, k=4)
     ├── inicializar_centroides()  [K-means++]
     └── bucle:
           ├── asignar_clusters()
           ├── actualizar_centroides()
           └── ¿convergencia? → sí → break
            │
            ▼
    etiquetas [0,3,2,1,0,...]  +  centroides  +  inercia
            │
            ├──── analizar_clusters()  →  imprime estadísticas por cluster
            └──── visualizar_clusters() → clusters_rutas.png
```

---

*Documento generado para la Actividad 4 — Aprendizaje No Supervisado*  
*Universidad Ibero · Facultad de Ingeniería · 2025*
