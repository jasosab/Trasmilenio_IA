# Descripción de los datos — Actividad 3

**Proyecto:** Sistema Inteligente de Rutas — TransMilenio Bogotá  
**Curso:** Inteligencia Artificial / Análisis de Datos  
**Universidad:** Ibero — Facultad de Ingeniería

---

## 1. Origen de los datos

Los datos **no provienen de un archivo externo**. Se generan programáticamente a partir de la base de conocimiento construida en la Actividad 2 (`knowledge_base.py`), que representa la red de TransMilenio como un conjunto de hechos y reglas lógicas.

El proceso de generación tiene dos etapas:

1. Se toman todos los pares ordenados de estaciones posibles usando `permutations(estaciones, 2)`.
2. Para cada par se ejecuta el algoritmo A* (`search_engine.py`) y se registra si la ruta óptima requiere transbordo.

Esto convierte el **conocimiento experto del grafo** en un **dataset tabular etiquetado** apto para Machine Learning.

---

## 2. Estaciones de la red (fuente primaria)

La red cuenta con **35 estaciones** distribuidas en 5 troncales. Cada estación tiene los siguientes atributos en `knowledge_base.py`:

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `nombre` | string | Nombre oficial de la estación |
| `troncal` | string | Línea a la que pertenece |
| `x` | entero | Posición horizontal en la cuadrícula (1–5) |
| `y` | entero | Posición vertical en la cuadrícula (1–10) |

### Distribución por troncal

| Troncal | Estaciones | Descripción geográfica |
|---------|-----------|------------------------|
| Caracas | 10 | Eje Norte-Sur principal |
| NQS | 9 | Norte-Quito-Sur, paralela a Caracas |
| Suba | 5 | Corredor noroccidental |
| Américas | 5 | Corredor suroccidental |
| Calle 80 | 4 | Corredor occidental |

### Coordenadas de las estaciones

Las coordenadas `x` e `y` son valores enteros que ubican cada estación en una **cuadrícula abstracta** de la ciudad. No representan coordenadas geográficas reales (latitud/longitud), sino posiciones relativas que permiten calcular distancias aproximadas entre estaciones.

| Troncal | Rango x | Rango y |
|---------|---------|---------|
| Caracas | x = 4 | y = 1 a 10 |
| NQS | x = 2–3 | y = 1 a 7 |
| Suba | x = 1–3 | y = 8 a 10 |
| Américas | x = 1–4 | y = 3 a 4 |
| Calle 80 | x = 1–3 | y = 6 |

---

## 3. Conexiones de la red (reglas)

La red tiene **25 conexiones** definidas explícitamente en `CONEXIONES` dentro de `knowledge_base.py`. Cada conexión es una regla de la forma:

```
conecta(origen, destino, tiempo_minutos, es_transbordo)
```

Todas las conexiones son **bidireccionales**: si existe A → B, también existe B → A con el mismo tiempo.

### Atributos de cada conexión

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `origen` | string | Nombre de la estación de partida |
| `destino` | string | Nombre de la estación de llegada |
| `tiempo_minutos` | entero | Tiempo de viaje entre las dos estaciones |
| `transbordo` | booleano | `True` si implica cambiar de línea |

### Distribución de conexiones

| Tipo | Cantidad | Porcentaje |
|------|----------|------------|
| Directas (misma línea) | 19 | 76% |
| Con transbordo (cambio de línea) | 6 | 24% |

### Rango de tiempos por conexión

| Estadístico | Valor |
|-------------|-------|
| Mínimo | 3 min |
| Máximo | 8 min |
| Promedio | ~4.3 min |

---

## 4. Dataset generado para el modelo

### Proceso de generación

A partir de las 35 estaciones se generan todos los pares ordenados posibles:

```
Pares candidatos = 35 × 34 = 1 190
Pares con ruta válida (A* encuentra camino) = 1 122
Pares descartados (sin ruta) = 68
```

Los 68 pares descartados corresponden a combinaciones entre estaciones de troncales que no tienen ninguna conexión transversal entre sí en el grafo definido.

### Dimensiones del dataset

| Dimensión | Valor |
|-----------|-------|
| Número de muestras | 1 122 |
| Número de features | 6 |
| Variable objetivo (etiqueta) | 1 (binaria) |

---

## 5. Variables de entrada (features)

Cada muestra del dataset es un vector de 6 variables numéricas calculadas a partir de los atributos de las dos estaciones involucradas:

| # | Nombre | Tipo | Fórmula | Rango |
|---|--------|------|---------|-------|
| 1 | `distancia_manhattan` | Numérico continuo | `|x_o − x_d| + |y_o − y_d|` | 1 – 9 |
| 2 | `diff_x` | Numérico continuo | `|x_o − x_d|` | 0 – 4 |
| 3 | `diff_y` | Numérico continuo | `|y_o − y_d|` | 0 – 9 |
| 4 | `misma_troncal` | Binario | `1` si misma línea, `0` si no | 0 o 1 |
| 5 | `troncal_origen_enc` | Categórico codificado | Label Encoding de la troncal del origen | 0 – 4 |
| 6 | `troncal_destino_enc` | Categórico codificado | Label Encoding de la troncal del destino | 0 – 4 |

### Codificación de troncales (Label Encoding)

| Troncal | Código numérico |
|---------|----------------|
| Americas | 0 |
| Caracas | 1 |
| Calle80 | 2 |
| NQS | 3 |
| Suba | 4 |

### Estadísticas descriptivas de las features (sobre las 1 122 muestras)

| Feature | Media | Desv. std | Mínimo | Máximo |
|---------|-------|-----------|--------|--------|
| distancia_manhattan | ≈ 4.01 | ≈ 2.20 | 1 | 9 |
| diff_x | ≈ 1.15 | ≈ 0.91 | 0 | 4 |
| diff_y | ≈ 2.86 | ≈ 2.02 | 0 | 9 |
| misma_troncal | ≈ 0.20 | ≈ 0.40 | 0 | 1 |
| troncal_origen_enc | ≈ 2.19 | ≈ 1.24 | 0 | 4 |
| troncal_destino_enc | ≈ 2.17 | ≈ 1.25 | 0 | 4 |

> Los valores de media y desviación estándar provienen del `StandardScaler` ajustado sobre el conjunto de entrenamiento (80% de las muestras).

---

## 6. Variable objetivo (etiqueta)

La etiqueta `y` indica si la ruta óptima calculada por A* requiere al menos un transbordo:

| Valor | Clase | Significado |
|-------|-------|-------------|
| `0` | Sin transbordo | La ruta se completa dentro de una sola línea |
| `1` | Con transbordo | La ruta requiere cambiar de línea al menos una vez |

### Distribución de clases

| Clase | Muestras | Porcentaje |
|-------|----------|------------|
| 0 — Sin transbordo | 206 | 18.4% |
| 1 — Con transbordo | 916 | 81.6% |

El dataset está **desbalanceado**: los pares que requieren transbordo son aproximadamente 4.5 veces más frecuentes que los directos. Esto refleja la realidad de la red: la mayoría de los viajes entre estaciones de distintas troncales necesitan al menos un cambio de línea.

Para mitigar el efecto del desbalance se usó `stratify=y` en el split, garantizando que ambas particiones (entrenamiento y prueba) mantengan la misma proporción 18%/82%.

---

## 7. División del dataset

| Partición | Muestras | Proporción |
|-----------|----------|------------|
| Entrenamiento | 897 | 80% |
| Prueba | 225 | 20% |

La división se realizó con `random_state=42` para garantizar la reproducibilidad de los resultados.

---

## 8. Preprocesamiento aplicado

| Paso | Técnica | Herramienta |
|------|---------|-------------|
| Generación de pares | `permutations(estaciones, 2)` | `itertools` |
| Etiquetado | Ejecución de A* sobre cada par | `search_engine.astar()` |
| Codificación de texto | Label Encoding de troncales | Diccionario manual |
| División | Split estratificado 80/20 | `sklearn.model_selection.train_test_split` |
| Normalización | Z-score por feature | `sklearn.preprocessing.StandardScaler` |

No se aplicaron técnicas de balanceo de clases (como SMOTE o class weighting) porque el modelo alcanzó un F1 de 0.87 en la clase minoritaria sin necesitarlas.

---

## 9. Trazabilidad de los datos

Toda la cadena desde los datos crudos hasta el dataset final es reproducible con el mismo código:

```
knowledge_base.py
    ESTACIONES (35 entradas)  ──────────────────────────────┐
    CONEXIONES (25 reglas)    ──> construir_grafo() ──> grafo│
                                                             │
actividad3_ml.py                                            │
    generar_dataset()                                        │
        permutations(estaciones, 2) ──> 1 190 pares          │
        astar(grafo, origen, destino) <──────────────────────┘
        calcular 6 features por par
        asignar etiqueta (0 o 1)
        ──> X (1122, 6)  y (1122,)
```

Ejecutar `python actividad3_ml.py` regenera el dataset completo desde cero en cada corrida.
