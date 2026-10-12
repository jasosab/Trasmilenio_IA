"""
BASE DE CONOCIMIENTO — Sistema TransMilenio (Bogotá)
=====================================================
Representa la red de TransMilenio como un conjunto de
HECHOS y REGLAS lógicas al estilo de un sistema experto.

Estructura:
  - Hechos (facts): estaciones y líneas que existen
  - Reglas (rules): cómo se conectan las estaciones

Curso: Inteligencia Artificial / Análisis de Datos
Universidad Ibero - Facultad de Ingeniería
"""

# ==============================================================
# HECHOS: Estaciones por troncal
# Cada entrada es un hecho del tipo: estacion(nombre, troncal)
# ==============================================================

ESTACIONES = {
    # Troncal Caracas (Norte-Sur, la más larga)
    "Portal Norte":        {"troncal": "Caracas", "x": 4,  "y": 10},
    "Toberín":             {"troncal": "Caracas", "x": 4,  "y": 9},
    "Calle 127":           {"troncal": "Caracas", "x": 4,  "y": 8},
    "Calle 100":           {"troncal": "Caracas", "x": 4,  "y": 7},
    "Calle 72":            {"troncal": "Caracas", "x": 4,  "y": 6},
    "Calle 45":            {"troncal": "Caracas", "x": 4,  "y": 5},
    "Mártires":            {"troncal": "Caracas", "x": 4,  "y": 4},
    "Ricaurte":            {"troncal": "Caracas", "x": 4,  "y": 3},
    "General Santander":   {"troncal": "Caracas", "x": 4,  "y": 2},
    "Portal Sur":          {"troncal": "Caracas", "x": 4,  "y": 1},

    # Troncal NQS (Norte-Quito-Sur)
    "Portal Tunal":        {"troncal": "NQS",     "x": 3,  "y": 1},
    "Olaya":               {"troncal": "NQS",     "x": 3,  "y": 2},
    "Nariño":              {"troncal": "NQS",     "x": 3,  "y": 3},
    "Santa Lucía":         {"troncal": "NQS",     "x": 3,  "y": 4},
    "Pradera":             {"troncal": "NQS",     "x": 3,  "y": 5},
    "Restrepo":            {"troncal": "NQS",     "x": 3,  "y": 5},
    "NQS Calle 30":        {"troncal": "NQS",     "x": 3,  "y": 5},
    "NQS Calle 75":        {"troncal": "NQS",     "x": 3,  "y": 6},
    "NQS Calle 92":        {"troncal": "NQS",     "x": 3,  "y": 7},
    "Portal El Dorado":    {"troncal": "NQS",     "x": 2,  "y": 5},

    # Troncal Calle 80
    "Portal 80":           {"troncal": "Calle80", "x": 1,  "y": 6},
    "Minuto de Dios":      {"troncal": "Calle80", "x": 2,  "y": 6},
    "Granja-Carrera 77":   {"troncal": "Calle80", "x": 3,  "y": 6},
    "Carrera 90":          {"troncal": "Calle80", "x": 1,  "y": 6},

    # Troncal Suba
    "Portal Suba":         {"troncal": "Suba",    "x": 1,  "y": 10},
    "Suba-Calle 95":       {"troncal": "Suba",    "x": 2,  "y": 9},
    "Niza-Calle 127":      {"troncal": "Suba",    "x": 2,  "y": 8},
    "Shaio":               {"troncal": "Suba",    "x": 2,  "y": 8},
    "Puentelargo":         {"troncal": "Suba",    "x": 3,  "y": 8},

    # Troncal Américas
    "Portal Américas":     {"troncal": "Americas","x": 1,  "y": 3},
    "Banderas":            {"troncal": "Americas","x": 2,  "y": 3},
    "Marsella":            {"troncal": "Americas","x": 3,  "y": 3},
    "Av. Jiménez":         {"troncal": "Americas","x": 4,  "y": 4},

    # Nodo central
    "Biblioteca Tintal":   {"troncal": "Americas","x": 1,  "y": 4},
    "Terminal":            {"troncal": "Caracas",  "x": 5,  "y": 4},
}


# ==============================================================
# REGLAS: Conexiones entre estaciones
# Cada regla: conecta(origen, destino, tiempo_minutos, transbordo)
#   - tiempo_minutos: costo real del trayecto
#   - transbordo: True si hay que cambiar de bus/línea
# ==============================================================

CONEXIONES = [
    # --- Troncal Caracas (Norte → Sur) ---
    ("Portal Norte",       "Toberín",           4,  False),
    ("Toberín",            "Calle 127",          3,  False),
    ("Calle 127",          "Calle 100",          4,  False),
    ("Calle 100",          "Calle 72",           5,  False),
    ("Calle 72",           "Calle 45",           5,  False),
    ("Calle 45",           "Mártires",           4,  False),
    ("Mártires",           "Ricaurte",           3,  False),
    ("Ricaurte",           "General Santander",  4,  False),
    ("General Santander",  "Portal Sur",         6,  False),

    # --- Troncal NQS (Norte → Sur) ---
    ("Portal Tunal",       "Olaya",              5,  False),
    ("Olaya",              "Nariño",             4,  False),
    ("Nariño",             "Santa Lucía",        3,  False),
    ("Santa Lucía",        "Pradera",            3,  False),
    ("Pradera",            "Restrepo",           3,  False),
    ("Restrepo",           "NQS Calle 30",       4,  False),
    ("NQS Calle 30",       "NQS Calle 75",       7,  False),
    ("NQS Calle 75",       "NQS Calle 92",       4,  False),

    # --- Troncal Calle 80 (Occidente → Centro) ---
    ("Portal 80",          "Carrera 90",         3,  False),
    ("Carrera 90",         "Minuto de Dios",     4,  False),
    ("Minuto de Dios",     "Granja-Carrera 77",  4,  False),
    ("Granja-Carrera 77",  "NQS Calle 75",       5,  True),   # transbordo a NQS

    # --- Troncal Suba (Occidente Norte → Centro) ---
    ("Portal Suba",        "Suba-Calle 95",      5,  False),
    ("Suba-Calle 95",      "Niza-Calle 127",     4,  False),
    ("Niza-Calle 127",     "Shaio",              3,  False),
    ("Shaio",              "Puentelargo",        4,  False),
    ("Puentelargo",        "Calle 100",          5,  True),    # transbordo a Caracas

    # --- Troncal Américas (Occidente Sur → Centro) ---
    ("Portal Américas",    "Banderas",           5,  False),
    ("Banderas",           "Marsella",           4,  False),
    ("Marsella",           "Mártires",           6,  True),    # transbordo a Caracas
    ("Biblioteca Tintal",  "Banderas",           4,  False),

    # --- Conexiones transversales (transbordos clave) ---
    ("Calle 100",          "Niza-Calle 127",     6,  True),    # Caracas ↔ Suba
    ("Calle 72",           "NQS Calle 75",       7,  True),    # Caracas ↔ NQS
    ("Mártires",           "Av. Jiménez",        3,  True),    # Caracas ↔ Américas
    ("Ricaurte",           "Marsella",           8,  True),    # Caracas ↔ Américas
    ("Portal El Dorado",   "NQS Calle 30",       5,  True),    # Eldorado ↔ NQS
    ("Portal El Dorado",   "Marsella",           6,  True),    # Eldorado ↔ Américas
]


# ==============================================================
# MOTOR DE HECHOS: construye el grafo a partir de las reglas
# Las conexiones son BIDIRECCIONALES (se puede ir y volver)
# ==============================================================

def construir_grafo():
    """
    Regla general: si conecta(A, B, tiempo, transbordo),
    entonces también conecta(B, A, tiempo, transbordo).
    Retorna un diccionario: { estacion: [(vecino, tiempo, transbordo), ...] }
    """
    grafo = {estacion: [] for estacion in ESTACIONES}

    for origen, destino, tiempo, transbordo in CONEXIONES:
        # Verificar que las estaciones existen en la base de conocimiento
        if origen not in ESTACIONES:
            print(f"[AVISO] Estación desconocida: '{origen}'")
            continue
        if destino not in ESTACIONES:
            print(f"[AVISO] Estación desconocida: '{destino}'")
            continue

        # Regla: la conexión es bidireccional
        grafo[origen].append((destino, tiempo, transbordo))
        grafo[destino].append((origen, tiempo, transbordo))

    return grafo


def listar_estaciones():
    """Retorna la lista de todas las estaciones conocidas, ordenadas."""
    return sorted(ESTACIONES.keys())


def heuristica(estacion_actual, estacion_meta):
    """
    Función heurística para A*: distancia Manhattan entre
    las coordenadas (x, y) de dos estaciones.
    Estima el tiempo mínimo restante para llegar al destino.
    """
    x1 = ESTACIONES[estacion_actual]["x"]
    y1 = ESTACIONES[estacion_actual]["y"]
    x2 = ESTACIONES[estacion_meta]["x"]
    y2 = ESTACIONES[estacion_meta]["y"]
    # Cada unidad de distancia ≈ 4 minutos en promedio
    return (abs(x1 - x2) + abs(y1 - y2)) * 4


if __name__ == "__main__":
    grafo = construir_grafo()
    print(f"Base de conocimiento cargada.")
    print(f"  Estaciones registradas : {len(ESTACIONES)}")
    print(f"  Conexiones (reglas)    : {len(CONEXIONES)}")
    print(f"\nEstaciones disponibles:")
    for e in listar_estaciones():
        print(f"  - {e} ({ESTACIONES[e]['troncal']})")
