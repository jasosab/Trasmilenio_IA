"""
CLUSTERING DE RUTAS — TransMilenio Bogotá (K-means)
====================================================
Actividad 4: Aprendizaje No Supervisado

Genera un dataset de rutas entre pares de estaciones y aplica
el algoritmo K-means para descubrir agrupaciones naturales
sin utilizar etiquetas previas.

Características (features) de cada ruta:
  - num_paradas       : cantidad de estaciones recorridas
  - tiempo_total      : minutos estimados del viaje
  - num_transbordos   : número de cambios de bus/línea
  - distancia_manhattan: distancia en la cuadrícula de coordenadas
  - mismo_troncal     : 1 si origen y destino están en la misma troncal

Algoritmo: K-means implementado desde cero (Python puro).
           Opcionalmente usa scikit-learn si está disponible.

Curso: Inteligencia Artificial / Análisis de Datos
Universidad Ibero — Facultad de Ingeniería
"""

import csv
import math
import random
import os
import sys
import io
from collections import deque, defaultdict

# Forzar UTF-8 en Windows para que los caracteres especiales se muestren bien
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
    except AttributeError:
        pass

# Asegurarse de importar desde el mismo directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knowledge_base import ESTACIONES, construir_grafo

# ─────────────────────────────────────────────────────────────
# SECCIÓN 1 — GENERACIÓN DEL DATASET
# ─────────────────────────────────────────────────────────────

def bfs_completo(grafo, origen, destino):
    """
    BFS que retorna (ruta, tiempo_total, num_transbordos).
    Cuenta el tiempo acumulado y los transbordos en el camino.
    Retorna None si no existe ruta.
    """
    if origen == destino:
        return [origen], 0, 0

    # Estado: (ruta, tiempo, transbordos)
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


def generar_dataset(grafo):
    """
    Crea un dataset con TODAS las rutas posibles entre pares
    de estaciones (combinaciones sin repetición).

    Retorna: lista de diccionarios con las features de cada ruta.
    """
    estaciones = list(ESTACIONES.keys())
    rutas = []

    print("  Generando dataset de rutas...", end="", flush=True)

    for i, origen in enumerate(estaciones):
        for destino in estaciones[i + 1:]:
            resultado = bfs_completo(grafo, origen, destino)
            if resultado is None:
                continue

            ruta, tiempo, transbordos = resultado
            num_paradas = len(ruta) - 1  # segmentos recorridos

            # Distancia Manhattan entre coordenadas del grafo
            x1 = ESTACIONES[origen]["x"]
            y1 = ESTACIONES[origen]["y"]
            x2 = ESTACIONES[destino]["x"]
            y2 = ESTACIONES[destino]["y"]
            dist = abs(x1 - x2) + abs(y1 - y2)

            mismo_troncal = int(
                ESTACIONES[origen]["troncal"] == ESTACIONES[destino]["troncal"]
            )

            rutas.append({
                "origen":             origen,
                "destino":            destino,
                "troncal_origen":     ESTACIONES[origen]["troncal"],
                "troncal_destino":    ESTACIONES[destino]["troncal"],
                "num_paradas":        num_paradas,
                "tiempo_total":       tiempo,
                "num_transbordos":    transbordos,
                "distancia_manhattan": dist,
                "mismo_troncal":      mismo_troncal,
            })

    print(f" {len(rutas)} rutas generadas.")
    return rutas


def guardar_csv(rutas, ruta_archivo):
    """Guarda el dataset en un archivo CSV."""
    if not rutas:
        return

    campos = list(rutas[0].keys())
    with open(ruta_archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(rutas)

    print(f"  Dataset guardado en: {ruta_archivo}")


# ─────────────────────────────────────────────────────────────
# SECCIÓN 2 — K-MEANS DESDE CERO (Python puro)
# ─────────────────────────────────────────────────────────────

FEATURES = ["num_paradas", "tiempo_total", "num_transbordos",
            "distancia_manhattan", "mismo_troncal"]


def extraer_vectores(rutas):
    """
    Extrae los vectores de features numéricos de cada ruta.
    Retorna una lista de listas de floats.
    """
    return [[float(r[f]) for f in FEATURES] for r in rutas]


def normalizar(vectores):
    """
    Normalización min-max por columna → valores en [0, 1].
    Evita que features con escalas grandes dominen la distancia.
    Retorna: (vectores_norm, minimos, maximos)
    """
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


def distancia_euclidiana(a, b):
    """Distancia Euclidiana entre dos vectores."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def inicializar_centroides(vectores, k, semilla=42):
    """
    Inicialización K-means++ para centroides más dispersos.
    Reduce el riesgo de convergencia a óptimos locales.
    """
    random.seed(semilla)
    centroides = [random.choice(vectores)]

    for _ in range(k - 1):
        # Distancia cuadrada de cada punto al centroide más cercano
        distancias = []
        for v in vectores:
            d_min = min(distancia_euclidiana(v, c) ** 2 for c in centroides)
            distancias.append(d_min)

        # Selección proporcional a la distancia cuadrada (K-means++)
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


def asignar_clusters(vectores, centroides):
    """Asigna cada vector al centroide más cercano."""
    return [
        min(range(len(centroides)),
            key=lambda j: distancia_euclidiana(v, centroides[j]))
        for v in vectores
    ]


def actualizar_centroides(vectores, etiquetas, k):
    """Recalcula el centroide de cada cluster como la media de sus puntos."""
    n_feat = len(vectores[0])
    nuevos = []

    for cluster in range(k):
        miembros = [vectores[i] for i, l in enumerate(etiquetas) if l == cluster]
        if not miembros:
            # Si un cluster queda vacío, reinicializar con punto aleatorio
            nuevos.append(random.choice(vectores))
        else:
            media = [sum(v[j] for v in miembros) / len(miembros)
                     for j in range(n_feat)]
            nuevos.append(media)

    return nuevos


def kmeans(vectores, k, max_iter=300, tolerancia=1e-6, semilla=42):
    """
    Algoritmo K-means completo.

    Parámetros:
        vectores  : lista de puntos (listas de floats)
        k         : número de clusters
        max_iter  : iteraciones máximas
        tolerancia: umbral de convergencia (movimiento de centroides)
        semilla   : semilla aleatoria para reproducibilidad

    Retorna:
        etiquetas : lista de int (cluster asignado a cada punto)
        centroides: lista de centroides finales
        inercia   : suma de distancias cuadradas al centroide (SSE)
        iteraciones: número de iteraciones hasta convergencia
    """
    centroides = inicializar_centroides(vectores, k, semilla)

    for iteracion in range(1, max_iter + 1):
        etiquetas = asignar_clusters(vectores, centroides)
        nuevos_c = actualizar_centroides(vectores, etiquetas, k)

        # Verificar convergencia
        movimiento = max(
            distancia_euclidiana(centroides[j], nuevos_c[j])
            for j in range(k)
        )
        centroides = nuevos_c

        if movimiento < tolerancia:
            break

    # Calcular inercia (SSE — Sum of Squared Errors)
    inercia = sum(
        distancia_euclidiana(vectores[i], centroides[etiquetas[i]]) ** 2
        for i in range(len(vectores))
    )

    return etiquetas, centroides, inercia, iteracion


# ─────────────────────────────────────────────────────────────
# SECCIÓN 3 — MÉTODO DEL CODO (Elbow Method)
# ─────────────────────────────────────────────────────────────

def metodo_del_codo(vectores, k_max=8):
    """
    Prueba K-means para k = 2..k_max y retorna la inercia de cada k.
    Permite identificar visualmente el "codo" donde agregar más clusters
    ya no reduce significativamente la inercia.
    """
    print("\n  ── Método del Codo (Elbow Method) ──")
    print(f"  {'k':>3}  {'Inercia':>12}  {'Reducción':>10}  Barra")
    print(f"  {'─'*3}  {'─'*12}  {'─'*10}  {'─'*30}")

    resultados = []
    inercia_anterior = None

    for k in range(2, k_max + 1):
        _, _, inercia, iters = kmeans(vectores, k)
        resultados.append((k, inercia, iters))

        reduccion = ""
        if inercia_anterior is not None:
            pct = (inercia_anterior - inercia) / inercia_anterior * 100
            reduccion = f"{pct:>9.1f}%"

        barra_len = int(inercia / resultados[0][1] * 28)
        barra = "█" * barra_len

        print(f"  {k:>3}  {inercia:>12.4f}  {reduccion:>10}  {barra}")
        inercia_anterior = inercia

    return resultados


def elegir_k_optimo(resultados_codo):
    """
    Heurística de la segunda derivada para detectar el 'codo'.
    El k óptimo es donde la reducción de inercia cae más bruscamente.
    """
    if len(resultados_codo) < 3:
        return resultados_codo[0][0]

    inercias = [r[1] for r in resultados_codo]
    max_caida = 0
    k_optimo = resultados_codo[1][0]

    for i in range(1, len(inercias) - 1):
        caida = inercias[i - 1] - inercias[i]
        siguiente = inercias[i] - inercias[i + 1]
        segunda_deriv = caida - siguiente
        if segunda_deriv > max_caida:
            max_caida = segunda_deriv
            k_optimo = resultados_codo[i][0]

    return k_optimo


# ─────────────────────────────────────────────────────────────
# SECCIÓN 4 — ANÁLISIS E INTERPRETACIÓN DE CLUSTERS
# ─────────────────────────────────────────────────────────────

ETIQUETAS_CLUSTER = {
    # Se asignan dinámicamente según el perfil de cada cluster
}


def analizar_clusters(rutas, etiquetas, k, centroides, minimos, maximos):
    """
    Analiza cada cluster: calcula estadísticas promedio y
    asigna una etiqueta descriptiva según el perfil del cluster.
    """
    clusters = defaultdict(list)
    for i, ruta in enumerate(rutas):
        clusters[etiquetas[i]].append(ruta)

    print(f"\n{'═'*60}")
    print(f"  ANÁLISIS DE CLUSTERS (k = {k})")
    print(f"{'═'*60}")

    perfiles = []
    for c in range(k):
        miembros = clusters[c]
        n = len(miembros)
        if n == 0:
            continue

        avg_paradas   = sum(r["num_paradas"]        for r in miembros) / n
        avg_tiempo    = sum(r["tiempo_total"]        for r in miembros) / n
        avg_tb        = sum(r["num_transbordos"]     for r in miembros) / n
        avg_dist      = sum(r["distancia_manhattan"] for r in miembros) / n
        pct_mismo_t   = sum(r["mismo_troncal"]       for r in miembros) / n * 100

        # Contar troncales más frecuentes en el cluster
        troncales_c = defaultdict(int)
        for r in miembros:
            troncales_c[r["troncal_origen"]] += 1
            troncales_c[r["troncal_destino"]] += 1
        top_troncal = max(troncales_c, key=troncales_c.get)

        perfiles.append({
            "cluster":      c,
            "n":            n,
            "avg_paradas":  avg_paradas,
            "avg_tiempo":   avg_tiempo,
            "avg_tb":       avg_tb,
            "avg_dist":     avg_dist,
            "pct_mismo_t":  pct_mismo_t,
            "top_troncal":  top_troncal,
            "miembros":     miembros,
        })

    # Ordenar clusters de "más corto" a "más largo"
    perfiles.sort(key=lambda p: p["avg_tiempo"])

    etiquetas_descriptivas = []
    for p in perfiles:
        if p["avg_tb"] < 0.3 and p["avg_paradas"] <= 4:
            label = "Rutas CORTAS intra-troncal"
        elif p["avg_tb"] < 0.5 and p["pct_mismo_t"] > 50:
            label = "Rutas MEDIAS sin transbordo"
        elif p["avg_tb"] < 1.5:
            label = "Rutas MEDIAS con 1 transbordo inter-troncal"
        else:
            label = "Rutas LARGAS multi-transbordo (2+ transbordos)"
        etiquetas_descriptivas.append(label)

    for i, p in enumerate(perfiles):
        label = etiquetas_descriptivas[i]
        print(f"\n  Cluster {p['cluster']} — {label}")
        print(f"  {'─'*56}")
        print(f"    Rutas en el cluster  : {p['n']:>4}")
        print(f"    Paradas promedio     : {p['avg_paradas']:>6.1f}")
        print(f"    Tiempo promedio      : {p['avg_tiempo']:>6.1f} min")
        print(f"    Transbordos prom.    : {p['avg_tb']:>6.2f}")
        print(f"    Distancia prom.      : {p['avg_dist']:>6.2f} ud.")
        print(f"    % misma troncal      : {p['pct_mismo_t']:>6.1f}%")
        print(f"    Troncal más frecuente: {p['top_troncal']}")

        # Mostrar 3 ejemplos representativos (los más cercanos al centroide)
        print(f"\n    Ejemplos representativos:")
        # Ordenar por tiempo total para elegir ejemplos variados
        ejemplos = sorted(p["miembros"], key=lambda r: abs(r["tiempo_total"] - p["avg_tiempo"]))[:3]
        for r in ejemplos:
            tb_str = f"  {r['num_transbordos']} transbordo(s)" if r["num_transbordos"] > 0 else "  sin transbordo"
            print(f"      • {r['origen']:<22} → {r['destino']:<22}")
            print(f"        {r['num_paradas']} paradas, {r['tiempo_total']} min,{tb_str}")

    return perfiles


# ─────────────────────────────────────────────────────────────
# SECCIÓN 5 — COMPARACIÓN CON SCIKIT-LEARN (opcional)
# ─────────────────────────────────────────────────────────────

def comparar_con_sklearn(vectores_norm, k):
    """
    Ejecuta K-means de scikit-learn con los mismos parámetros
    para validar que nuestra implementación produce resultados similares.
    """
    try:
        from sklearn.cluster import KMeans
        import numpy as np

        X = np.array(vectores_norm)
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10,
                        random_state=42, max_iter=300)
        modelo.fit(X)

        inercia_sk = modelo.inertia_
        print(f"\n  Scikit-learn KMeans:")
        print(f"    Inercia (SSE)  : {inercia_sk:.4f}")
        print(f"    Iteraciones    : {modelo.n_iter_}")
        print(f"  Implementación propia:")

    except ImportError:
        print("\n  [INFO] scikit-learn no disponible. Se usa solo la implementación propia.")
    except Exception as e:
        print(f"\n  [AVISO] Error al usar scikit-learn: {e}")


# ─────────────────────────────────────────────────────────────
# SECCIÓN 6 — VISUALIZACIÓN (opcional, requiere matplotlib)
# ─────────────────────────────────────────────────────────────

def visualizar_clusters(rutas, etiquetas, k, archivo_salida="clusters_rutas.png"):
    """
    Genera un gráfico de dispersión 2D (tiempo vs. paradas)
    coloreado por cluster.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # no requiere pantalla
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colores = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
                   "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("K-means — Clustering de Rutas TransMilenio\n"
                     "Universidad Ibero · IA / Análisis de Datos",
                     fontsize=13, fontweight="bold")

        x = [r["tiempo_total"]        for r in rutas]
        y = [r["num_paradas"]         for r in rutas]
        z = [r["num_transbordos"]     for r in rutas]
        c = [colores[e % len(colores)] for e in etiquetas]

        # Gráfico 1: Tiempo vs Paradas
        axes[0].scatter(x, y, c=c, alpha=0.6, s=40, edgecolors="white", linewidths=0.4)
        axes[0].set_xlabel("Tiempo total (min)", fontsize=11)
        axes[0].set_ylabel("Número de paradas",  fontsize=11)
        axes[0].set_title("Tiempo vs. Paradas", fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Gráfico 2: Tiempo vs Transbordos
        axes[1].scatter(x, z, c=c, alpha=0.6, s=40, edgecolors="white", linewidths=0.4)
        axes[1].set_xlabel("Tiempo total (min)",   fontsize=11)
        axes[1].set_ylabel("Número de transbordos", fontsize=11)
        axes[1].set_title("Tiempo vs. Transbordos", fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Leyenda
        leyenda = [mpatches.Patch(color=colores[i % len(colores)],
                                  label=f"Cluster {i}")
                   for i in range(k)]
        fig.legend(handles=leyenda, loc="lower center", ncol=k,
                   bbox_to_anchor=(0.5, -0.04), fontsize=10)

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        dir_salida = os.path.dirname(os.path.abspath(__file__))
        ruta_img = os.path.join(dir_salida, archivo_salida)
        plt.savefig(ruta_img, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Gráfico guardado en: {ruta_img}")

    except ImportError:
        print("  [INFO] matplotlib no disponible. Se omite la visualización.")
    except Exception as e:
        print(f"  [AVISO] No se pudo generar el gráfico: {e}")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    BANNER = """
╔══════════════════════════════════════════════════════════╗
║   K-MEANS CLUSTERING — Rutas TransMilenio Bogotá         ║
║   Actividad 4: Aprendizaje No Supervisado                ║
║   Universidad Ibero · IA / Análisis de Datos             ║
╚══════════════════════════════════════════════════════════╝
"""
    print(BANNER)

    # ── Paso 1: Generar dataset ──────────────────────────────
    print("PASO 1 — Generación del dataset")
    print("─" * 50)
    grafo = construir_grafo()
    rutas = generar_dataset(grafo)
    print(f"  Total de rutas en el dataset: {len(rutas)}")

    # Guardar CSV junto al script
    dir_base = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(dir_base, "dataset_rutas.csv")
    guardar_csv(rutas, ruta_csv)

    # ── Paso 2: Preparar vectores ────────────────────────────
    print("\nPASO 2 — Extracción y normalización de features")
    print("─" * 50)
    print(f"  Features usadas: {', '.join(FEATURES)}")
    vectores_raw  = extraer_vectores(rutas)
    vectores_norm, minimos, maximos = normalizar(vectores_raw)
    print(f"  Vectores normalizados: {len(vectores_norm)} × {len(FEATURES)}")
    print(f"  Normalización min-max aplicada por columna")

    # ── Paso 3: Método del codo ──────────────────────────────
    print("\nPASO 3 — Selección de k (Método del Codo)")
    print("─" * 50)
    resultados_codo = metodo_del_codo(vectores_norm, k_max=7)
    k_optimo = elegir_k_optimo(resultados_codo)
    print(f"\n  >>> k óptimo detectado automáticamente: k = {k_optimo}")

    # ── Paso 4: Entrenar K-means ─────────────────────────────
    print(f"\nPASO 4 — Entrenamiento K-means con k = {k_optimo}")
    print("─" * 50)
    comparar_con_sklearn(vectores_norm, k_optimo)

    etiquetas, centroides, inercia, iters = kmeans(vectores_norm, k_optimo)
    print(f"    Inercia (SSE)  : {inercia:.4f}")
    print(f"    Iteraciones    : {iters}")

    # Distribución de clusters
    conteo = defaultdict(int)
    for e in etiquetas:
        conteo[e] += 1
    print(f"\n  Distribución de clusters:")
    for c in sorted(conteo):
        pct = conteo[c] / len(rutas) * 100
        barra = "▓" * int(pct / 2)
        print(f"    Cluster {c}: {conteo[c]:>4} rutas  ({pct:5.1f}%)  {barra}")

    # ── Paso 5: Análisis de clusters ─────────────────────────
    print("\nPASO 5 — Interpretación de clusters")
    print("─" * 50)
    perfiles = analizar_clusters(rutas, etiquetas, k_optimo, centroides, minimos, maximos)

    # ── Paso 6: Visualización ────────────────────────────────
    print(f"\nPASO 6 — Visualización")
    print("─" * 50)
    visualizar_clusters(rutas, etiquetas, k_optimo)

    # ── Resumen final ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  CONCLUSIONES")
    print(f"{'═'*60}")
    print(f"  • Dataset: {len(rutas)} rutas, {len(FEATURES)} features por ruta.")
    print(f"  • Algoritmo: K-means con inicialización K-means++.")
    print(f"  • Parámetro k = {k_optimo} elegido con el Método del Codo.")
    print(f"  • Sin supervisión: el modelo encontró {k_optimo} grupos naturales")
    print(f"    basándose únicamente en la estructura de los datos.")
    print(f"  • Los clusters reflejan la topología real de la red:")
    print(f"    rutas cortas intra-troncal vs. rutas largas con transbordos.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
