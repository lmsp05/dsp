"""
graficar_viscosidad.py
======================

Crea un grafico de dispersion (scatter plot) de la viscosidad promedio (mu_prom)
en funcion de la velocidad de rotacion, usando:
  * Color: para representar el tipo de aceite (ISO 32/46/68)
  * Forma de marcador: para representar el nivel de desbalanceo (dsb)

Lee la salida de mu_temperatura.py (que contiene las columnas mu1, mu2, mu_prom
anadidas a los datos originales).

Uso:
    python graficar_viscosidad.py --entrada ensayos_mu.txt --salida ./graficos
    python graficar_viscosidad.py --entrada ensayos_mu.txt --salida ./graficos --figsize 10,6 --dpi 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Colores para cada grado ISO
ISO_COLORES = {
    32: "#1f77b4",    # azul
    46: "#2ca02c",    # verde
    68: "#d62728",    # rojo
}

# Marcadores para cada valor de desbalanceo (dsb)
DSB_MARCADORES = {
    1: "o",           # circulo
    2: "s",           # cuadrado
    3: "^",           # triangulo
    4: "D",           # diamante
}


def _split(linea: str, sep: str | None):
    return linea.split(sep) if sep else linea.split()


def _es_nan(txt: str) -> bool:
    return txt is None or txt.strip() == "" or txt.strip().lower() in ("nan", "na", "-")


def _a_float(txt: str) -> float | None:
    if _es_nan(txt):
        return None
    try:
        return float(txt.strip().replace(",", "."))
    except ValueError:
        return None


def _a_int(txt: str) -> int | None:
    if _es_nan(txt):
        return None
    try:
        return int(txt.strip())
    except ValueError:
        return None


def procesar(args) -> int:
    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_file():
        print(f"ERROR: no existe la entrada {entrada}")
        return 1

    sep = None if args.sep == "auto" else ("\t" if args.sep == "tab" else args.sep)

    lineas = entrada.read_text(encoding="utf-8", errors="ignore").splitlines()
    # primera linea no vacia = cabecera
    h = next((i for i, ln in enumerate(lineas) if ln.strip()), None)
    if h is None:
        print("ERROR: archivo vacio.")
        return 1
    cabecera = _split(lineas[h], sep)
    ncol = len(cabecera)
    low = [c.strip().lower() for c in cabecera]

    def col(nombre):
        return low.index(nombre) if nombre in low else None

    i_vel, i_iso, i_dsb, i_mu_prom = (
        col("velocidad"), col("iso"), col("dsb"), col("mu_prom")
    )
    faltan = [n for n, i in (("velocidad", i_vel), ("iso", i_iso),
                             ("dsb", i_dsb), ("mu_prom", i_mu_prom))
              if i is None]
    if faltan:
        print(f"ERROR: la cabecera no tiene las columnas {faltan}. Cabecera: {cabecera}")
        return 1

    salida = (Path(args.salida).expanduser().resolve() if args.salida
              else entrada.parent)
    salida.mkdir(parents=True, exist_ok=True)

    # Lectura de datos
    datos = {}  # {(iso, dsb): [(vel, mu_prom), ...]}
    n_filas = 0
    n_validas = 0

    for ln in lineas[h + 1:]:
        if not ln.strip():
            continue
        n_filas += 1
        campos = _split(ln, sep)
        campos = (campos + ["NaN"] * (ncol - len(campos)))[:ncol]

        vel = _a_float(campos[i_vel])
        iso_v = _a_int(campos[i_iso])
        dsb_v = _a_int(campos[i_dsb])
        mu_prom = _a_float(campos[i_mu_prom])

        # Saltamos filas con datos faltantes
        if vel is None or iso_v is None or dsb_v is None or mu_prom is None:
            continue

        n_validas += 1
        clave = (iso_v, dsb_v)
        if clave not in datos:
            datos[clave] = []
        datos[clave].append((vel, mu_prom))

    if n_validas == 0:
        print(f"ERROR: no hay filas con datos validos (leidas {n_filas})")
        return 1

    # Crear figura
    figsize = tuple(map(float, args.figsize.split(","))) if args.figsize else (12, 8)
    fig, ax = plt.subplots(figsize=figsize)

    # Graficar cada combinacion (iso, dsb)
    for (iso, dsb), puntos in sorted(datos.items()):
        vel_list = [p[0] for p in puntos]
        mu_list = [p[1] for p in puntos]

        color = ISO_COLORES.get(iso, "#000000")
        marcador = DSB_MARCADORES.get(dsb, "o")

        label = f"ISO {iso}, DSB {dsb}"
        ax.scatter(vel_list, mu_list, c=color, marker=marcador, s=100,
                   alpha=0.7, label=label, edgecolors="black", linewidth=0.5)

    # Configurar ejes y leyenda
    ax.set_xlabel("Velocidad de rotacion (RPM)", fontsize=12)
    ax.set_ylabel("Viscosidad promedio μ_prom (cSt)", fontsize=12)
    ax.set_title("Viscosidad vs Velocidad de Rotacion", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Guardar figura
    nombre_salida = f"viscosidad_scatter.png"
    ruta_salida = salida / nombre_salida
    fig.savefig(ruta_salida, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafico guardado: {ruta_salida}")
    print(f"Filas leidas: {n_filas} | Puntos graficados: {n_validas}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crea un scatter plot de mu_prom vs velocidad, con colores por ISO "
                    "y marcadores por DSB.")
    p.add_argument("--entrada", required=True, help="Archivo de salida de mu_temperatura.py")
    p.add_argument("--salida", default="", help="Carpeta de salida (def: misma que entrada)")
    p.add_argument("--sep", default="tab",
                   help="Separador: 'tab' (def), 'auto' (espacios) o un caracter")
    p.add_argument("--figsize", default="12,8",
                   help="Tamaño de figura (ancho,alto) en pulgadas")
    p.add_argument("--dpi", type=int, default=150,
                   help="Resolucion en puntos por pulgada")
    return p


if __name__ == "__main__":
    sys.exit(procesar(construir_parser().parse_args()))
