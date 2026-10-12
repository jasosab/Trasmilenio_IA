"""
SISTEMA INTELIGENTE DE RUTAS — TransMilenio Bogotá
====================================================
Punto de entrada del programa.
El usuario ingresa una estación de origen y una de destino,
y el sistema encuentra la mejor ruta usando BFS y A*.

Uso:
    python main.py

Curso: Inteligencia Artificial / Análisis de Datos
Universidad Ibero - Facultad de Ingeniería
"""

from knowledge_base import construir_grafo, listar_estaciones
from search_engine import comparar_algoritmos


BANNER = """
╔══════════════════════════════════════════════════════╗
║     SISTEMA INTELIGENTE DE RUTAS — TransMilenio      ║
║         Universidad Ibero · IA / Análisis Datos      ║
╚══════════════════════════════════════════════════════╝
"""


def seleccionar_estacion(prompt, estaciones):
    """
    Muestra la lista de estaciones y pide al usuario que elija una.
    Acepta el número de la lista o el nombre directo.
    """
    print(f"\n  Estaciones disponibles:")
    for i, nombre in enumerate(estaciones, 1):
        print(f"    {i:>2}. {nombre}")

    while True:
        entrada = input(f"\n  {prompt}: ").strip()

        # Si ingresó un número
        if entrada.isdigit():
            idx = int(entrada) - 1
            if 0 <= idx < len(estaciones):
                return estaciones[idx]
            else:
                print("  ⚠ Número fuera de rango. Intenta de nuevo.")

        # Si ingresó un nombre (búsqueda parcial, sin importar mayúsculas)
        elif entrada:
            coincidencias = [e for e in estaciones
                             if entrada.lower() in e.lower()]
            if len(coincidencias) == 1:
                return coincidencias[0]
            elif len(coincidencias) > 1:
                print(f"  Varias coincidencias encontradas:")
                for c in coincidencias:
                    print(f"    - {c}")
                print("  Sé más específico.")
            else:
                print("  ⚠ No se encontró esa estación. Intenta de nuevo.")


def main():
    print(BANNER)

    # Cargar base de conocimiento
    grafo = construir_grafo()
    estaciones = listar_estaciones()

    print(f"  Base de conocimiento cargada:")
    print(f"    • {len(estaciones)} estaciones registradas")
    print(f"    • Algoritmos disponibles: BFS y A*")

    while True:
        print("\n" + "─" * 54)
        print("  Ingresa el número o nombre de las estaciones.")
        print("  Escribe 'salir' para terminar.\n")

        # Selección de origen
        origen = seleccionar_estacion("Estación de ORIGEN", estaciones)
        if origen.lower() == "salir":
            break

        # Selección de destino
        destino = seleccionar_estacion("Estación de DESTINO", estaciones)
        if destino.lower() == "salir":
            break

        if origen == destino:
            print("\n  ⚠ El origen y el destino son la misma estación.")
            continue

        # Ejecutar y comparar los dos algoritmos
        comparar_algoritmos(origen, destino)

        # Preguntar si quiere buscar otra ruta
        otra = input("  ¿Buscar otra ruta? (s/n): ").strip().lower()
        if otra != "s":
            break

    print("\n  ¡Hasta pronto!\n")


if __name__ == "__main__":
    main()
