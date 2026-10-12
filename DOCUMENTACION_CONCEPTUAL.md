# Documentación conceptual — Aprendizaje No Supervisado con K-means

**Actividad 4 · Aprendizaje No Supervisado**  
**Curso:** Inteligencia Artificial / Análisis de Datos  
**Universidad Ibero — Facultad de Ingeniería**

---

## Tabla de contenido

1. [Aprendizaje supervisado vs. no supervisado](#1-aprendizaje-supervisado-vs-no-supervisado)
2. [Clustering: encontrar estructura sin etiquetas](#2-clustering-encontrar-estructura-sin-etiquetas)
3. [El algoritmo K-means](#3-el-algoritmo-k-means)
4. [K-means++: inicialización inteligente](#4-k-means-inicialización-inteligente)
5. [Normalización de features](#5-normalización-de-features)
6. [Distancia Euclidiana como medida de similitud](#6-distancia-euclidiana-como-medida-de-similitud)
7. [Inercia (SSE) como métrica de calidad](#7-inercia-sse-como-métrica-de-calidad)
8. [El Método del Codo para elegir k](#8-el-método-del-codo-para-elegir-k)
9. [Convergencia del algoritmo](#9-convergencia-del-algoritmo)
10. [Cómo se aplica al contexto de TransMilenio](#10-cómo-se-aplica-al-contexto-de-transmilenio)
11. [Interpretación de los resultados](#11-interpretación-de-los-resultados)
12. [Limitaciones del enfoque](#12-limitaciones-del-enfoque)

---

## 1. Aprendizaje supervisado vs. no supervisado

En inteligencia artificial, la palabra *supervisado* indica si el algoritmo aprende a partir de datos que ya tienen la respuesta correcta.

| Característica | Supervisado | No supervisado |
|----------------|-------------|----------------|
| ¿Hay etiquetas? | Sí (`spam` / `no spam`, `ruta rápida` / `ruta lenta`) | No |
| ¿Qué aprende? | A predecir la etiqueta de nuevos datos | A encontrar estructura interna en los datos |
| Ejemplos de algoritmos | Regresión lineal, árbol de decisión, SVM | K-means, DBSCAN, PCA |
| ¿Cómo se mide el éxito? | Comparando la predicción con la etiqueta real | Con métricas internas como inercia o silueta |

**En este proyecto:** no hay ningún campo `tipo_de_ruta = "larga"` en el dataset. Las rutas son solo números. El algoritmo tiene que descubrir por sí solo si existen grupos naturales y cuáles son, sin que nadie le diga cuántas categorías buscar (en principio) ni cuáles son sus nombres.

> **Líneas clave:** El dataset se genera en `generar_dataset()` (línea 80). En ningún momento se escribe una etiqueta como `"corta"` o `"larga"`. Las etiquetas descriptivas solo aparecen *después* del entrenamiento, en `analizar_clusters()` (línea 352), como interpretación humana de lo que el modelo encontró.

---

## 2. Clustering: encontrar estructura sin etiquetas

*Clustering* (o agrupamiento) es la tarea de dividir un conjunto de observaciones en grupos —llamados *clusters*— de modo que los objetos dentro del mismo grupo sean más similares entre sí que con los objetos de otros grupos.

**Intuición geométrica:** si representamos cada ruta como un punto en un espacio de 5 dimensiones (una por feature), el clustering busca "nubes de puntos" densas. Los puntos dentro de la misma nube se parecen; los de nubes distintas son diferentes.

```
Espacio de features (simplificado a 2D para visualizar):

tiempo_total
     │
  80 │        ●●●●●   ← cluster de rutas largas
     │      ●●●
  40 │   ◆◆◆◆◆        ← cluster de rutas medias
     │ ◆◆◆
  10 │▲▲▲▲             ← cluster de rutas cortas
     └──────────────────── num_paradas
        2   4   6   8
```

> **Líneas clave:** La representación geométrica se construye en `extraer_vectores()` (línea 150) y `normalizar()` (línea 158). Cada llamada a `distancia_euclidiana()` (línea 179) mide la distancia entre dos puntos de ese espacio.

---

## 3. El algoritmo K-means

K-means es el algoritmo de clustering más utilizado. Su nombre viene de que agrupa los datos en **k grupos**, donde el centro de cada grupo es la **media** (*mean*) de sus miembros.

### Idea central

El algoritmo busca minimizar la suma de distancias cuadradas de cada punto a su centroide:

```
Objetivo: minimizar SSE = Σᵢ  ‖ xᵢ − centroide(cluster_i) ‖²
```

donde `xᵢ` es un punto de datos y `centroide(cluster_i)` es el centro del cluster al que pertenece.

### Los dos pasos que se repiten

**Paso E (Expectation) — Asignación:**  
Cada punto se asigna al centroide más cercano. El cluster de cada punto cambia potencialmente en cada iteración.

```python
# clustering_kmeans.py — línea 216
def asignar_clusters(vectores, centroides):
    return [
        min(range(len(centroides)),
            key=lambda j: distancia_euclidiana(v, centroides[j]))
        for v in vectores
    ]
```

*Concepto → código:* `min(..., key=lambda j: distancia_euclidiana(...))` es exactamente la regla "asignar al centroide más cercano". Para cada punto `v`, se prueban todos los centroides `j` y se elige el que minimiza la distancia.

---

**Paso M (Maximization) — Actualización:**  
Los centroides se recalculan como la media de todos los puntos asignados a ese cluster.

```python
# clustering_kmeans.py — línea 225
def actualizar_centroides(vectores, etiquetas, k):
    for cluster in range(k):
        miembros = [vectores[i] for i, l in enumerate(etiquetas) if l == cluster]
        media = [sum(v[j] for v in miembros) / len(miembros)
                 for j in range(n_feat)]
        nuevos.append(media)
```

*Concepto → código:* `sum(v[j] for v in miembros) / len(miembros)` es la media aritmética de la j-ésima coordenada de todos los miembros del cluster. Esto se hace para cada una de las 5 dimensiones.

---

**Bucle principal:**

```python
# clustering_kmeans.py — línea 262
for iteracion in range(1, max_iter + 1):
    etiquetas = asignar_clusters(vectores, centroides)   # Paso E
    nuevos_c  = actualizar_centroides(vectores, etiquetas, k)  # Paso M

    movimiento = max(distancia_euclidiana(centroides[j], nuevos_c[j])
                     for j in range(k))
    centroides = nuevos_c

    if movimiento < tolerancia:
        break
```

*Concepto → código:* Los dos pasos se alternan en cada iteración del `for`. La variable `movimiento` mide cuánto se desplazó el centroide que más se movió. Cuando ese movimiento es despreciable (`< 1e-6`), los clusters son estables.

### Garantía del algoritmo

K-means **siempre converge** porque en cada iteración el SSE puede mantenerse igual o disminuir, pero nunca aumentar. Como el número de posibles asignaciones es finito (cada punto puede estar en uno de `k` clusters), el algoritmo no puede ciclar indefinidamente.

La convergencia **no garantiza** el mínimo global del SSE — puede quedar atrapado en un mínimo local. Por eso se usan buenas inicializaciones (K-means++) y, en producción, múltiples ejecuciones.

---

## 4. K-means++: inicialización inteligente

La versión estándar de K-means elige los centroides iniciales al azar. Esto puede resultar en dos centroides en la misma zona del espacio, ignorando otra zona completamente — un mal punto de partida que produce clusters de mala calidad.

**K-means++** soluciona esto: cada nuevo centroide se elige con probabilidad *proporcional al cuadrado de la distancia* al centroide ya elegido más cercano. Los puntos lejos de los centroides existentes tienen más probabilidad de ser elegidos como próximos centroides.

```
Probabilidad de elegir el punto xᵢ como siguiente centroide:

         d(xᵢ, centroide_más_cercano)²
P(xᵢ) = ─────────────────────────────────────
         Σⱼ  d(xⱼ, centroide_más_cercano)²
```

**Implementación en el código:**

```python
# clustering_kmeans.py — líneas 194–211
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
```

*Concepto → código:*
- `d_min = min(... ** 2 for c in centroides)` — distancia cuadrada al centroide más cercano de los ya elegidos.
- `prob = [d / total for d in distancias]` — normalización para convertir en distribución de probabilidad.
- El bucle `for i, p in enumerate(prob): acumulado += p; if r <= acumulado` es el método de la **transformada inversa** para muestrear una distribución discreta arbitraria: equivale a elegir aleatoriamente un punto con probabilidad `prob[i]`.

**Efecto en este dataset:** los 4 centroides iniciales quedan distribuidos en zonas distintas del espacio (una ruta corta, una media, una larga intra-troncal, una larga inter-troncal), y el algoritmo converge en solo 9 iteraciones.

---

## 5. Normalización de features

Las 5 features tienen escalas muy diferentes:

| Feature | Rango típico | Unidad |
|---------|-------------|--------|
| `num_paradas` | 1 – 12 | paradas |
| `tiempo_total` | 4 – 80 | minutos |
| `num_transbordos` | 0 – 4 | transbordos |
| `distancia_manhattan` | 1 – 9 | unidades de cuadrícula |
| `mismo_troncal` | 0 o 1 | booleano |

Sin normalización, la distancia Euclidiana entre dos puntos estaría dominada casi exclusivamente por `tiempo_total` (valores hasta 80) mientras que `mismo_troncal` (valores 0 o 1) apenas influiría.

**Ejemplo concreto:**  
Ruta A: `[2, 10, 0, 2, 1]`  
Ruta B: `[8, 60, 2, 7, 0]`

Sin normalizar: `d = sqrt((2-8)² + (10-60)² + ...) ≈ sqrt(36 + 2500 + ...) ≈ 50.4`  
El término `(10-60)² = 2500` domina completamente.

Con normalización min-max todo queda en [0,1] y cada feature contribuye equitativamente.

**Implementación:**

```python
# clustering_kmeans.py — líneas 158–176
def normalizar(vectores):
    minimos = [min(v[j] for v in vectores) for j in range(n_feat)]
    maximos = [max(v[j] for v in vectores) for j in range(n_feat)]

    norm = []
    for v in vectores:
        fila = [(v[j] - minimos[j]) / (maximos[j] - minimos[j])
                if (maximos[j] - minimos[j]) > 0 else 0.0
                for j in range(n_feat)]
        norm.append(fila)
    return norm, minimos, maximos
```

*Concepto → código:* La fórmula `(v[j] - minimos[j]) / (maximos[j] - minimos[j])` es directamente la normalización min-max. El mínimo de cada columna pasa a 0, el máximo pasa a 1, y el resto se escala proporcionalmente.

> **Nota importante:** la normalización se aplica sobre los `vectores_raw` (línea 559 de `main()`), *antes* de pasarlos al Método del Codo y a K-means. Sin embargo, al interpretar los clusters en `analizar_clusters()`, se usan las rutas originales sin normalizar para que los promedios sean legibles en minutos y paradas reales.

---

## 6. Distancia Euclidiana como medida de similitud

K-means necesita una función que diga "qué tan parecidos son dos puntos". La elección estándar es la **distancia Euclidiana**, que es la extensión n-dimensional del teorema de Pitágoras:

```
d(a, b) = √( (a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)² )
```

Dos rutas son "similares" si tienen valores parecidos en todas las features simultáneamente. Una ruta de 2 paradas / 8 min / 0 transbordos es similar a otra de 3 paradas / 10 min / 0 transbordos, pero muy diferente de una de 9 paradas / 50 min / 2 transbordos.

```python
# clustering_kmeans.py — línea 179
def distancia_euclidiana(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
```

*Concepto → código:* `(x - y) ** 2 for x, y in zip(a, b)` genera las diferencias cuadradas componente a componente. `sum(...)` las suma. `math.sqrt(...)` toma la raíz cuadrada. Es la fórmula literal.

**Por qué distancia y no similitud:** K-means minimiza distancia (menor es mejor). Una similitud sería lo inverso (mayor es mejor), y habría que reformular el objetivo. Ambos enfoques son válidos; la distancia es la convención estándar.

---

## 7. Inercia (SSE) como métrica de calidad

La **inercia** (también llamada SSE, *Sum of Squared Errors*, o WCSS, *Within-Cluster Sum of Squares*) es la métrica principal que K-means optimiza:

```
SSE = Σᵢ₌₁ⁿ  ‖ xᵢ − μ(cluster_i) ‖²
```

donde `μ(cluster_i)` es el centroide del cluster al que pertenece `xᵢ`.

**Propiedades importantes:**
- Siempre decrece cuando aumenta `k` (más clusters → cada cluster es más compacto).
- Es 0 cuando `k = n` (cada punto es su propio cluster — inútil).
- Es máxima cuando `k = 1` (todos en el mismo grupo).

```python
# clustering_kmeans.py — líneas 276–280
inercia = sum(
    distancia_euclidiana(vectores[i], centroides[etiquetas[i]]) ** 2
    for i in range(len(vectores))
)
```

*Concepto → código:* Para cada punto `vectores[i]`, se calcula su distancia al centroide de su cluster (`centroides[etiquetas[i]]`), se eleva al cuadrado, y se suman todos. Esto es exactamente la fórmula del SSE.

**Resultado en este proyecto:**

```
k=4: SSE = 42.58  (implementación propia, un solo intento)
k=4: SSE = 36.61  (scikit-learn con 10 intentos — mejor mínimo local)
```

La diferencia se debe a que sklearn prueba 10 inicializaciones diferentes y se queda con la de menor SSE. Aun así, los clusters resultantes describen la misma estructura.

---

## 8. El Método del Codo para elegir k

K-means requiere que el usuario especifique `k` de antemano. ¿Cómo saber cuál es el mejor valor?

El **Método del Codo** (*Elbow Method*) se basa en trazar la inercia como función de `k` y buscar el punto donde la curva "dobla como un codo": agregar más clusters a partir de ahí reduce la inercia muy poco y no justifica la complejidad adicional.

```
SSE
│
76 │ ●
   │  \
70 │   ●
   │    \
42 │     ●  ← CODO aquí (k=4)
   │       \
30 │        ●
   │         \
25 │          ●
   └────────────── k
       2  3  4  5  6
```

**Implementación del codo automático — segunda derivada discreta:**

```python
# clustering_kmeans.py — líneas 332–338
for i in range(1, len(inercias) - 1):
    caida      = inercias[i - 1] - inercias[i]      # pendiente actual
    siguiente  = inercias[i]     - inercias[i + 1]  # pendiente siguiente
    segunda_deriv = caida - siguiente                # variación de la pendiente

    if segunda_deriv > max_caida:
        max_caida = segunda_deriv
        k_optimo  = resultados_codo[i][0]
```

*Concepto → código:*  
- `caida` es la primera derivada discreta (cuánto bajó la inercia en este paso).
- `segunda_deriv = caida - siguiente` es la segunda derivada discreta (si la curva se aplana, `siguiente < caida`, por lo que `segunda_deriv > 0`).
- El codo es el punto donde la curva pasa de "caer rápido" a "caer despacio". Matemáticamente, es donde la segunda derivada es **máxima**.

Con los datos del proyecto:

| k | Inercia | Caída | Siguiente caída | Segunda derivada |
|---|---------|-------|-----------------|------------------|
| 3 | 70.82 | 6.07 | 28.24 | -22.17 |
| 4 | 42.58 | 28.24 | 12.03 | **+16.21** ← máxima |
| 5 | 30.59 | 12.03 | 5.50 | +6.53 |

El máximo de la segunda derivada está en k=4 → k óptimo = 4.

---

## 9. Convergencia del algoritmo

**¿Por qué K-means siempre converge?**

En cada iteración se realizan dos pasos que juntos no pueden aumentar el SSE:

1. **Asignación:** reasignar un punto a un centroide más cercano solo puede mantener o disminuir su contribución al SSE.
2. **Actualización:** reemplazar el centroide por la media de sus miembros es la operación que minimiza el SSE dentro del cluster (la media minimiza la suma de cuadrados).

Como el SSE nunca sube y el número de configuraciones posibles es finito, el algoritmo debe estabilizarse en algún momento.

**Criterio implementado:**

```python
# clustering_kmeans.py — líneas 266–274
movimiento = max(
    distancia_euclidiana(centroides[j], nuevos_c[j])
    for j in range(k)
)
centroides = nuevos_c

if movimiento < tolerancia:   # tolerancia = 1e-6
    break
```

*Concepto → código:* En lugar de comparar el SSE antes y después (que requeriría recalcularlo dos veces), se mide el **desplazamiento máximo** de los centroides. Si ningún centroide se movió más de `1e-6` unidades, los clusters son estables y se detiene. En este dataset se alcanza la convergencia en 9 iteraciones.

---

## 10. Cómo se aplica al contexto de TransMilenio

### El dataset como espacio vectorial

Cada ruta de TransMilenio se describe con 5 números:

```
Ruta: Portal Norte → Portal Sur
  num_paradas        = 9
  tiempo_total       = 38 min
  num_transbordos    = 0
  distancia_manhattan = 9
  mismo_troncal      = 1
```

Este vector `[9, 38, 0, 9, 1]` ubica la ruta como un punto en el espacio R⁵. Rutas parecidas quedarán cerca en ese espacio.

### Generación del dataset desde la base de conocimiento

```python
# clustering_kmeans.py — líneas 92–122
for i, origen in enumerate(estaciones):
    for destino in estaciones[i + 1:]:
        resultado = bfs_completo(grafo, origen, destino)
        ...
        rutas.append({
            "num_paradas":         num_paradas,
            "tiempo_total":        tiempo,
            "num_transbordos":     transbordos,
            "distancia_manhattan": dist,
            "mismo_troncal":       mismo_troncal,
        })
```

*Concepto → código:* BFS explora el grafo para encontrar la ruta de menor número de paradas entre cada par de estaciones. Esa ruta se convierte en un punto del dataset. No hay etiquetas — solo los 5 números que describen las características del viaje.

### Por qué estas 5 features y no otras

| Feature | Qué captura | Relevancia para clustering |
|---------|-------------|---------------------------|
| `num_paradas` | Longitud de la ruta | Separa viajes cortos de largos |
| `tiempo_total` | Costo temporal real | La métrica más práctica para el usuario |
| `num_transbordos` | Complejidad del viaje | Diferencia rutas directas de rutas con cambios |
| `distancia_manhattan` | Separación geográfica | Independiente del grafo; refleja dónde están las estaciones |
| `mismo_troncal` | Tipo de conexión | Distingue viajes intra-línea de viajes inter-línea |

`num_paradas` y `tiempo_total` están correlacionadas (más paradas → más tiempo), pero no son idénticas (los tiempos entre estaciones varían y los transbordos añaden 5 minutos). Tenerlas ambas refuerza la señal de "longitud del viaje".

---

## 11. Interpretación de los resultados

### Clusters encontrados (k = 4)

El modelo no sabe qué significa "Caracas" ni "transbordo". Solo ve números. Al converger, encuentra 4 grupos naturales que, cuando los examinamos, corresponden exactamente a categorías que un experto en transporte reconocería:

| Cluster | Perfil numérico | Interpretación humana |
|---------|----------------|----------------------|
| C2 | ~2 paradas, ~9 min, 0.10 transbordos, 100% misma troncal | Viajes cortos dentro de una línea |
| C1 | ~5.5 paradas, ~24 min, 0.23 transbordos, 100% misma troncal | Viajes moderados en la misma línea |
| C0 | ~4 paradas, ~26 min, 1.32 transbordos, 0% misma troncal | Viajes que cruzan una línea con un transbordo |
| C3 | ~7.4 paradas, ~45 min, 2.11 transbordos, 0% misma troncal | Viajes largos que cruzan varias líneas |

**Esto es lo esencial del aprendizaje no supervisado:** el modelo encontró esas 4 categorías sin que nadie le dijera que existen, basándose únicamente en la geometría de los datos.

### Código de interpretación post-hoc

```python
# clustering_kmeans.py — líneas 401–410
for p in perfiles:
    if p["avg_tb"] < 0.3 and p["avg_paradas"] <= 4:
        label = "Rutas CORTAS intra-troncal"
    elif p["avg_tb"] < 0.5 and p["pct_mismo_t"] > 50:
        label = "Rutas MEDIAS sin transbordo"
    elif p["avg_tb"] < 1.5:
        label = "Rutas MEDIAS con 1 transbordo inter-troncal"
    else:
        label = "Rutas LARGAS multi-transbordo (2+ transbordos)"
```

Estas reglas `if/elif` no son parte del modelo. Son **interpretación humana post-hoc**: después de que K-means formó los grupos, un humano examina el perfil estadístico y asigna un nombre descriptivo. El modelo solo produjo los números 0, 1, 2, 3 como etiquetas — los nombres vienen del análisis.

---

## 12. Limitaciones del enfoque

### K-means asume clusters esféricos

K-means usa distancia Euclidiana y centroides como medias, lo que implícitamente asume que los clusters tienen forma aproximadamente esférica en el espacio de features. Si los grupos reales tuvieran formas arbitrarias (curvas, anillos), K-means los partiría incorrectamente. En este dataset no es un problema porque los grupos sí tienden a ser compactos y esféricos.

### k debe especificarse de antemano

El Método del Codo da una guía, pero no una respuesta única. La heurística de la segunda derivada eligió k=4 automáticamente, pero un analista podría argumentar k=3 o k=5.

```python
# clustering_kmeans.py — línea 568
k_optimo = elegir_k_optimo(resultados_codo)
```

Si se quisiera probar otro valor de k, basta con pasar `k_optimo = 3` directamente a `kmeans()`.

### La inercia siempre mejora con más k

Como se mencionó, SSE siempre baja al aumentar k. El Método del Codo no da una respuesta matemáticamente exacta — es una heurística visual. En datasets más complejos puede ser ambiguo.

### Sensibilidad a la inicialización

Aunque K-means++ reduce el problema, distintas semillas pueden producir clusters ligeramente diferentes. La semilla `random.seed(42)` garantiza reproducibilidad, pero si se cambia la semilla, el resultado puede variar.

```python
# clustering_kmeans.py — línea 189
random.seed(semilla)   # semilla=42 por defecto en kmeans()
```

Para mitigar esto, en producción se ejecutaría K-means múltiples veces (como hace sklearn con `n_init=10`) y se conservaría el resultado con menor SSE.

---

*Documento generado para la Actividad 4 — Aprendizaje No Supervisado*  
*Universidad Ibero · Facultad de Ingeniería · 2025*
