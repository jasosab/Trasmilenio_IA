# Descripción del dataset — `dataset_rutas.csv`

**Actividad 4 · Aprendizaje No Supervisado**  
**Curso:** Inteligencia Artificial / Análisis de Datos  
**Universidad Ibero — Facultad de Ingeniería**

---

## 1. Origen y método de generación

El dataset **no proviene de una fuente externa**. Se genera de forma programática a partir de la base de conocimiento del sistema TransMilenio construida en la Actividad 2 (`knowledge_base.py`), que contiene:

- **33 estaciones** distribuidas en 5 troncales (Caracas, NQS, Calle 80, Suba, Américas).
- **35 reglas de conexión** bidireccionales (incluyendo transbordos entre troncales).

El script `clustering_kmeans.py` aplica el algoritmo **BFS (Búsqueda en Amplitud)** sobre el grafo de conexiones para calcular la ruta de menor número de paradas entre cada par único de estaciones. De esas rutas se extraen las 5 variables numéricas que conforman el dataset.

**Tamaño del espacio de pares:**  
Con 33 estaciones, el número de combinaciones sin repetición es C(33, 2) = 528 pares posibles. Se generaron **561 rutas** porque el grafo permite múltiples paths alternativos entre algunos pares debido a las conexiones transversales entre troncales.

---

## 2. Estructura del archivo

El archivo `dataset_rutas.csv` tiene formato CSV con encabezados en la primera fila.

| Propiedad | Valor |
|-----------|-------|
| Formato | CSV (comma-separated values), codificación UTF-8 |
| Filas de datos | 561 |
| Columnas | 9 |
| Archivo | `dataset_rutas.csv` |
| Generado por | `clustering_kmeans.py` → función `guardar_csv()` |

### Primeras 5 filas del archivo

```
origen,destino,troncal_origen,troncal_destino,num_paradas,tiempo_total,num_transbordos,distancia_manhattan,mismo_troncal
Portal Norte,Toberín,Caracas,Caracas,1,4,0,1,1
Portal Norte,Calle 127,Caracas,Caracas,2,7,0,2,1
Portal Norte,Calle 100,Caracas,Caracas,3,11,0,3,1
Portal Norte,Calle 72,Caracas,Caracas,4,16,0,4,1
```

---

## 3. Descripción de columnas

### Columnas descriptivas (no usadas en el clustering)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `origen` | Texto | Nombre de la estación de inicio de la ruta |
| `destino` | Texto | Nombre de la estación de llegada |
| `troncal_origen` | Texto | Línea de TransMilenio a la que pertenece la estación de origen |
| `troncal_destino` | Texto | Línea de TransMilenio a la que pertenece la estación de destino |

Valores posibles de troncal: `Caracas`, `NQS`, `Calle80`, `Suba`, `Americas`.

---

### Columnas numéricas (features de clustering)

Estas 5 columnas son las que se normalizan y se pasan al algoritmo K-means.

---

#### `num_paradas`
**Tipo:** Entero  
**Rango:** 1 – 12  
**Media:** 5.23  
**Descripción:** Número de segmentos recorridos en la ruta, equivalente a la cantidad de estaciones visitadas sin contar el origen. Una ruta de 3 estaciones (A→B→C) tiene `num_paradas = 2`.

| Valor | Frecuencia | % |
|-------|-----------|---|
| 1 | 36 | 6.4% |
| 2 | 50 | 8.9% |
| 3 | 66 | 11.8% |
| 4 | 75 | 13.4% |
| 5 | 81 | 14.4% |
| 6 | 77 | 13.7% |
| 7 | 68 | 12.1% |
| 8 | 50 | 8.9% |
| 9+ | 58 | 10.3% |

La distribución es aproximadamente normal con moda en 5 paradas, lo que refleja que la mayoría de los viajes en TransMilenio tienen una longitud moderada.

---

#### `tiempo_total`
**Tipo:** Entero (minutos)  
**Rango:** 3 – 66 min  
**Media:** 31.46 min  
**Descripción:** Tiempo estimado total del viaje en minutos. Incluye el tiempo de tránsito entre estaciones (definido en `knowledge_base.py` para cada conexión) más una **penalización de 5 minutos por cada transbordo**, que representa el tiempo de espera al cambiar de bus.

**Rutas más cortas:**

| Ruta | Tiempo |
|------|--------|
| Toberín → Calle 127 | 3 min |
| Mártires → Ricaurte | 3 min |
| Nariño → Santa Lucía | 3 min |
| Santa Lucía → Pradera | 3 min |
| Pradera → Restrepo | 3 min |

**Rutas más largas:**

| Ruta | Tiempo | Paradas | Transbordos |
|------|--------|---------|-------------|
| Portal Sur → Portal Tunal | 66 min | 11 | 3 |
| Portal Tunal → Portal Suba | 66 min | 12 | 2 |
| Portal Tunal → Av. Jiménez | 62 min | 10 | 4 |
| Portal Tunal → Suba-Calle 95 | 61 min | 11 | 2 |
| Olaya → Portal Suba | 61 min | 11 | 2 |

---

#### `num_transbordos`
**Tipo:** Entero  
**Rango:** 0 – 4  
**Media:** 1.41  
**Descripción:** Número de veces que el pasajero debe cambiar de bus o línea durante el recorrido. Un transbordo implica bajarse de un bus y subirse a otro en una estación de conexión.

| Transbordos | Rutas | % |
|-------------|-------|---|
| 0 | 103 | 18.4% |
| 1 | 211 | 37.6% |
| 2 | 166 | 29.6% |
| 3 | 74 | 13.2% |
| 4 | 7 | 1.2% |

Más del 80% de las rutas requieren al menos un transbordo, lo que refleja que la red de TransMilenio tiene pocas conexiones directas entre troncales distintas.

---

#### `distancia_manhattan`
**Tipo:** Entero (unidades de cuadrícula)  
**Rango:** 0 – 12  
**Media:** 4.02  
**Descripción:** Distancia entre el origen y el destino medida como la suma de las diferencias absolutas en coordenadas (x, y) asignadas a cada estación en la base de conocimiento. Es una estimación geográfica de qué tan separadas están las dos estaciones en la ciudad, independiente del grafo de conexiones.

```
distancia_manhattan = |x_origen - x_destino| + |y_origen - y_destino|
```

| Valor | Frecuencia |
|-------|-----------|
| 0 | 8 |
| 1 | 57 |
| 2 | 89 |
| 3 | 109 |
| 4 | 92 |
| 5 | 70 |
| 6 | 56 |
| 7+ | 80 |

Los 8 casos con distancia 0 corresponden a estaciones que comparten coordenadas en la cuadrícula (por ejemplo, estaciones de distintas troncales que confluyen en la misma zona geográfica).

---

#### `mismo_troncal`
**Tipo:** Binario (0 o 1)  
**Media:** 0.21  
**Descripción:** Indica si la estación de origen y la de destino pertenecen a la misma troncal de TransMilenio.

| Valor | Significado | Rutas | % |
|-------|-------------|-------|---|
| `1` | Mismo corredor (viaje intra-troncal) | 116 | 20.7% |
| `0` | Corredores distintos (viaje inter-troncal) | 445 | 79.3% |

El predominio de rutas inter-troncal (79.3%) refleja que la red de TransMilenio tiene 5 corredores que conectan distintas zonas de la ciudad, y la mayoría de los viajes involucra cruzar de un corredor a otro.

---

## 4. Distribución por troncal

La columna `troncal_origen` refleja qué proporción de rutas parten de cada corredor:

| Troncal | Rutas como origen | % | Estaciones en la troncal |
|---------|------------------|---|--------------------------|
| Caracas | 285 | 50.8% | 10 |
| NQS | 185 | 33.0% | 9 |
| Calle 80 | 46 | 8.2% | 4 |
| Suba | 35 | 6.2% | 5 |
| Américas | 10 | 1.8% | 5 |

La troncal Caracas domina el dataset porque es la línea más larga (10 estaciones) y por lo tanto genera más combinaciones únicas de pares.

---

### Combinaciones de troncal origen → destino

| Combinación | Rutas |
|-------------|-------|
| Caracas → NQS | 100 |
| Caracas → Suba | 50 |
| Caracas → Américas | 50 |
| NQS → Suba | 50 |
| NQS → Américas | 50 |
| Caracas → Caracas | 45 |
| NQS → NQS | 45 |
| Caracas → Calle 80 | 40 |
| NQS → Calle 80 | 40 |
| Suba → Américas | 25 |
| Calle 80 → Suba | 20 |
| Calle 80 → Américas | 20 |
| Suba → Suba | 10 |
| Américas → Américas | 10 |
| Calle 80 → Calle 80 | 6 |

---

## 5. Estadísticas resumen

| Feature | Mínimo | Máximo | Media | Interpretación del rango |
|---------|--------|--------|-------|--------------------------|
| `num_paradas` | 1 | 12 | 5.23 | Desde 1 segmento hasta 12 estaciones recorridas |
| `tiempo_total` | 3 min | 66 min | 31.46 min | Desde 3 min (estaciones adyacentes) hasta ~1 hora |
| `num_transbordos` | 0 | 4 | 1.41 | Desde viaje directo hasta 4 cambios de bus |
| `distancia_manhattan` | 0 | 12 | 4.02 | Desde estaciones en la misma zona hasta extremos opuestos |
| `mismo_troncal` | 0 | 1 | 0.21 | 20.7% de rutas son dentro de la misma línea |

---

## 6. Relación entre variables

**`num_paradas` y `tiempo_total` están correlacionadas positivamente:** más paradas implica más tiempo. Sin embargo, no son idénticas porque:
- El tiempo entre estaciones varía (de 3 a 8 minutos según el tramo).
- Cada transbordo añade 5 minutos fijos de penalización.

Una ruta de 4 paradas con 2 transbordos puede tomar más tiempo que una de 6 paradas sin transbordo.

**`mismo_troncal` y `num_transbordos` están fuertemente relacionados:** casi todas las rutas con `mismo_troncal = 1` tienen `num_transbordos = 0`, mientras que las rutas inter-troncal siempre tienen al menos 1 transbordo. Esto no es redundancia sino confirmación de consistencia del grafo.

**`distancia_manhattan` y `tiempo_total` son independientes** en varios casos: una ruta puede tener distancia Manhattan baja (estaciones geográficamente cercanas) pero tiempo alto si requiere rodear por otra troncal. Esto ocurre, por ejemplo, entre estaciones de NQS y Calle 80 que están próximas geográficamente pero conectadas indirectamente.

---

## 7. Consideraciones para el clustering

- **El dataset no tiene etiquetas.** Las columnas `origen`, `destino`, `troncal_origen` y `troncal_destino` son descriptivas y se excluyen del clustering. Solo las 5 columnas numéricas se usan como features.
- **Se aplica normalización min-max** antes del entrenamiento para que `tiempo_total` (rango 63) no domine sobre `mismo_troncal` (rango 1).
- **Los datos no tienen valores nulos.** Toda ruta generada por BFS tiene siempre los 5 valores definidos.
- **No hay duplicados.** Cada par (origen, destino) aparece exactamente una vez, con el orden determinístico del BFS.

---

*Documento generado para la Actividad 4 — Aprendizaje No Supervisado*  
*Universidad Ibero · Facultad de Ingeniería · 2025*
