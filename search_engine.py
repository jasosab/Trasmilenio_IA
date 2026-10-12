"""
MOTOR DE BÚSQUEDA — Sistema Inteligente de Rutas
=================================================
Implementa dos algoritmos para encontrar la mejor ruta
entre dos estaciones de TransMilenio:

  1. BFS  — Búsqueda en amplitud (menos transbordos / menos paradas)
  2. A*   — Búsqueda heurística  (menor tiempo total)

El algoritmo A* es el recomendado para uso real, ya que
optimiza el tiempo de viaje usando la heurística de distancia.

Curso: Inteligencia Artificial / Análisis de Datos
Universidad Ibero - Facultad de Ingeniería
"""

import heapq
from collections import deque
from knowledge_base import construir_grafo, heuristica, ESTACIONES


# ==============================================================
# ALGORITMO 1: BFS — Búsqueda en Amplitud
# Encuentra la ruta con MENOS PARADAS (no considera tiempo)
# ==============================================================

def bfs(grafo, origen, destino):
    """
    Recorre el grafo nivel por nivel.
    Garantiza encontrar la ruta con el menor número de paradas.

    Retorna: (ruta, tiempo_total, transbordos) o None si no hay ruta.
    """
    if origen not in grafo:
        return None
    if destino not in grafo:
        return None
    if origen == destino:
        return [origen], 0, 0

    # Cola de la forma: (ruta_hasta_aquí, tiempo_acumulado, transbordos)
    cola = deque()
    cola.append(([origen], 0, 0))

    visitados = set()
    visitados.add(origen)

    while cola:
        ruta, tiempo, transbordos = cola.popleft()
        estacion_actual = ruta[-1]

        for vecino, t, es_transbordo in grafo[estacion_actual]:
            if vecino not in visitados:
                nueva_ruta       = ruta + [vecino]
                # Se suma +5 min por transbordo (igual que A*) para que
                # ambos algoritmos reporten el mismo tipo de tiempo real
                nuevo_tiempo     = tiempo + t + (5 if es_transbordo else 0)
                nuevos_transbord = transbordos + (1 if es_transbordo else 0)

                if vecino == destino:
                    return nueva_ruta, nuevo_tiempo, nuevos_transbord

                visitados.add(vecino)
                cola.append((nueva_ruta, nuevo_tiempo, nuevos_transbord))

    return None  # No hay ruta posible


# ==============================================================
# ALGORITMO 2: A* — Búsqueda Heurística
# Encuentra la ruta con el MENOR TIEMPO usando f(n) = g(n) + h(n)
#   g(n) = tiempo acumulado desde el origen
#   h(n) = estimación del tiempo restante hasta el destino
# ==============================================================

def astar(grafo, origen, destino):
    """
    Usa una cola de prioridad (min-heap).
    Cada nodo tiene prioridad f = g + h.

    Retorna: (ruta, tiempo_total, transbordos) o None si no hay ruta.
    """
    if origen not in grafo:
        return None
    if destino not in grafo:
        return None
    if origen == destino:
        return [origen], 0, 0

    # Heap: (f, g, estacion, ruta, transbordos)
    h_inicial = heuristica(origen, destino)
    heap = [(h_inicial, 0, origen, [origen], 0)]

    # Costo mínimo conocido para llegar a cada estación
    costo_minimo = {origen: 0}

    while heap:
        f, g, estacion_actual, ruta, transbordos = heapq.heappop(heap)

        # Si llegamos al destino, retornamos la solución
        if estacion_actual == destino:
            return ruta, g, transbordos

        # Si ya encontramos un camino más barato a esta estación, saltamos
        if g > costo_minimo.get(estacion_actual, float('inf')):
            continue

        for vecino, t, es_transbordo in grafo[estacion_actual]:
            nuevo_g = g + t
            # Penalización por transbordo: +5 minutos de espera
            if es_transbordo:
                nuevo_g += 5

            if nuevo_g < costo_minimo.get(vecino, float('inf')):
                costo_minimo[vecino] = nuevo_g
                h = heuristica(vecino, destino)
                nuevo_f = nuevo_g + h
                nueva_ruta = ruta + [vecino]
                nuevos_t   = transbordos + (1 if es_transbordo else 0)
                heapq.heappush(heap, (nuevo_f, nuevo_g, vecino, nueva_ruta, nuevos_t))

    return None  # No hay ruta posible


# ==============================================================
# PRESENTACIÓN DE RESULTADOS
# ==============================================================

def mostrar_ruta(resultado, algoritmo):
    """Imprime la ruta de forma clara y legible."""
    separador = "─" * 52

    if resultado is None:
        print(f"\n  [{algoritmo}] No se encontró ruta posible.")
        return

    ruta, tiempo, transbordos = resultado

    print(f"\n  ┌{separador}┐")
    print(f"  │  Algoritmo : {algoritmo:<37}│")
    print(f"  │  Paradas   : {len(ruta) - 1:<37}│")
    print(f"  │  Transbord.: {transbordos:<37}│")
    print(f"  │  Tiempo est: {str(tiempo) + ' min':<37}│")
    print(f"  └{separador}┘")

    print(f"\n  Recorrido paso a paso:")
    for i, estacion in enumerate(ruta):
        troncal = ESTACIONES[estacion]["troncal"]
        if i == 0:
            prefijo = "  🟢 INICIO"
        elif i == len(ruta) - 1:
            prefijo = "  🔴 LLEGADA"
        else:
            prefijo = f"  ⬜ Paso {i:>2}"
        print(f"  {prefijo} → {estacion}  [{troncal}]")


# ==============================================================
# COMPARACIÓN DE ALGORITMOS
# ==============================================================

def comparar_algoritmos(origen, destino):
    """
    Ejecuta BFS y A* sobre el mismo par origen-destino
    y muestra una comparación de resultados.
    """
    import time
    grafo = construir_grafo()

    print(f"\n{'='*54}")
    print(f"  RUTA: {origen}  →  {destino}")
    print(f"{'='*54}")

    # BFS
    t0 = time.time()
    res_bfs = bfs(grafo, origen, destino)
    t1 = time.time()
    mostrar_ruta(res_bfs, "BFS — Menos paradas")
    print(f"  ⏱ Tiempo de cómputo: {(t1-t0)*1000:.3f} ms")

    # A*
    t0 = time.time()
    res_astar = astar(grafo, origen, destino)
    t1 = time.time()
    mostrar_ruta(res_astar, "A* — Menor tiempo")
    print(f"  ⏱ Tiempo de cómputo: {(t1-t0)*1000:.3f} ms")

    # Conclusión
    if res_bfs and res_astar:
        print(f"\n  📊 CONCLUSIÓN:")
        if res_astar[1] < res_bfs[1]:
            ahorro = res_bfs[1] - res_astar[1]
            print(f"  A* encontró una ruta {ahorro} min más rápida que BFS.")
        elif res_bfs[1] < res_astar[1]:
            print(f"  BFS encontró la misma ruta o similar.")
        else:
            print(f"  Ambos algoritmos encontraron rutas de igual tiempo.")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    # Casos donde ambos algoritmos encuentran la misma ruta
    print("\n📌 CASO 1 — Misma ruta, mismo tiempo (A* confirma que BFS ya era óptimo)")
    comparar_algoritmos("Portal Norte", "Portal Américas")

    # Casos donde A* supera a BFS evitando transbordos innecesarios
    print("\n📌 CASO 2 — A* ahorra 17 min evitando 2 transbordos (ruta diferente)")
    comparar_algoritmos("Portal 80", "Portal Sur")

    print("\n📌 CASO 3 — A* ahorra 4 min con 1 parada extra pero 2 transbordos menos")
    comparar_algoritmos("Mártires", "Restrepo")

    print("\n📌 CASO 4 — A* ahorra 4 min: BFS zigzaguea entre 4 troncales en 4 paradas")
    comparar_algoritmos("NQS Calle 30", "Av. Jiménez")
