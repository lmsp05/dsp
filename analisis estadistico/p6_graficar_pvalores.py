"""
PASO 6 — Graficas de valores p (significancia de cada factor)
=============================================================

Genera, con el mismo reparto 2x2 por sensor que `p5_contribucion.png`, una
figura en la que el eje Y es el **valor p** de cada factor en lugar de su
contribucion a la suma de cuadrados. Sobre cada barra se escribe el valor p
exacto y una linea horizontal marca el umbral de significancia (0.05 por
defecto).

Sirve para responder de un vistazo a "¿es la viscosidad un factor relevante?":
si su barra queda POR DEBAJO de la linea del umbral, lo es, y el numero escrito
sobre la barra dice con cuanto margen.

--------------------------------------------------------------------------
COMO SE LEE
--------------------------------------------------------------------------
El eje Y es el valor p en ESCALA LINEAL NORMAL, de 0 a 1 (el rango natural de
una probabilidad). La altura de cada barra ES su valor p, sin transformar.

    barra POR DEBAJO de la linea roja  ->  factor SIGNIFICATIVO
    barra POR ENCIMA de la linea roja  ->  factor no significativo

Es decir: cuanto MAS BAJA la barra, mas fuerte la evidencia. Sobre cada barra
va escrito el valor p exacto, asi que las barras muy pequenas (p muy por debajo
del umbral) se distinguen igualmente por su etiqueta.

Con --ymax se puede acotar el eje (p.ej. --ymax 0.1) para acercarse a la zona
del umbral; las barras que sobresalgan se recortan y se marcan con una flecha.

--------------------------------------------------------------------------
ENTRADA
--------------------------------------------------------------------------
Lee los `anova_split_plot*.csv` que ya escribio `p5_anova_split_plot.py` en
<salida>/p5_estadistica. Detecta solo las variantes que existan: la global, las
de tramo de velocidad (--grupos-velocidad) y las de nivel de desbalanceo
(--grupos-desbalanceo).

SALIDAS (en la misma carpeta)
  p6_pvalores<sufijo>.png    una por variante encontrada
  p6_pvalores_viscosidad.png resumen: el p de la viscosidad en TODAS las
                             variantes, por sensor
  p6_pvalores.csv            los mismos numeros en tabla

Uso:
    python p6_graficar_pvalores.py --salida <carpeta_resultados>
    python p6_graficar_pvalores.py --salida <res> --alfa 0.01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import config
from p5_anova_split_plot import ETIQUETAS

# Suelo numerico: por debajo de esto el valor p ya no es representable en doble
# precision y los CSV traen 0. Se escribe como "<1e-300".
P_MINIMO = 1e-300

COLOR_SIG = "#27ae60"       # significativo
COLOR_NO_SIG = "#bdc3c7"    # no significativo
COLOR_UMBRAL = "#c0392b"


# ============================================================
# UTILIDADES
# ============================================================

def fmt_p(p):
    """Valor p en texto legible."""
    if p is None or not np.isfinite(p):
        return "—"
    if p <= P_MINIMO:
        return "<1e-300"
    if p >= 1e-3:
        return f"{p:.3g}"
    return f"{p:.0e}"


def _leyenda(fig, alfa):
    """
    Leyenda comun. Va fuera de los paneles: dentro chocaria con las etiquetas
    de las barras, y el sentido de la escala (p pequeno = significativo) es
    justo lo que no puede quedar ambiguo.
    """
    handles = [Patch(facecolor=config.COLOR_SENSOR[s], edgecolor="black",
                     linewidth=0.4, label=s) for s in config.SENSORES]
    handles += [
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label=f"NO significativo (p ≥ {alfa:g})"),
        Patch(facecolor=COLOR_SIG, alpha=0.25, edgecolor="none",
              label=f"zona SIGNIFICATIVA — el factor SI influye (p < {alfa:g})"),
        Line2D([0], [0], color=COLOR_UMBRAL, ls="--", lw=1.8,
               label=f"umbral p = {alfa:g}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.05))


def _barras(ax, categorias, valores, alfa, ymax, resaltar=None):
    """
    Barras agrupadas: para cada categoria del eje X, una barra por proximitor.

    Eje lineal: la altura de la barra ES el valor p. Por debajo de la linea del
    umbral = significativo, y esas barras van rellenas; las no significativas
    van rayadas, para que la lectura no dependa solo de la posicion.

      valores : {sensor: [p por categoria]}
    """
    sensores = [s for s in config.SENSORES if s in valores]
    n = len(sensores)
    x = np.arange(len(categorias), dtype=float)
    ancho = 0.82 / max(n, 1)

    for k, sensor in enumerate(sensores):
        desplazamiento = (k - (n - 1) / 2) * ancho
        for i, p in enumerate(valores[sensor]):
            p = float(p)
            sig = p < alfa
            ax.bar(x[i] + desplazamiento, min(p, ymax), ancho * 0.9,
                   color=config.COLOR_SENSOR.get(sensor, "#888888"),
                   edgecolor="black", linewidth=0.4,
                   hatch=None if sig else "///", zorder=2)
            if p > ymax:      # la barra sigue por encima del limite del eje
                ax.annotate("", (x[i] + desplazamiento, ymax),
                            xytext=(x[i] + desplazamiento, ymax * 0.93),
                            arrowprops=dict(arrowstyle="-|>", color="#555555",
                                            lw=1.1), zorder=4)
            ax.annotate(fmt_p(p), (x[i] + desplazamiento, min(p, ymax)),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        va="bottom", fontsize=6.5, rotation=90, zorder=4,
                        color="black" if sig else "#777777")

    ax.set_xticks(x)
    ax.set_xticklabels(categorias, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-0.5, len(categorias) - 0.5)
    if resaltar in categorias:
        i = list(categorias).index(resaltar)
        ax.get_xticklabels()[i].set_color(COLOR_UMBRAL)
        ax.get_xticklabels()[i].set_fontweight("bold")

    # Eje lineal normal, en unidades de valor p.
    ax.set_ylim(0, ymax)
    # Banda sombreada por DEBAJO del umbral: zona en la que el factor SI influye.
    ax.axhspan(0, alfa, color=COLOR_SIG, alpha=0.10, zorder=0)
    ax.axhline(alfa, color=COLOR_UMBRAL, ls="--", lw=1.8, zorder=3)
    ax.text(0.995, alfa + 0.012 * ymax, f"umbral p = {alfa:g}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8.5, color=COLOR_UMBRAL, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.grid(axis="y", ls="--", alpha=0.35)


# ============================================================
# FIGURAS
# ============================================================

def figura_variante(tab, ruta, alfa, etiqueta, ymax):
    """2x2 por sensor: valor p de cada factor en una variante del analisis."""
    contrastables = tab[tab["p"].notna()
                        & (tab["Fuente"] != "Repeticion (bloque)")]
    fuentes = list(dict.fromkeys(contrastables["Fuente"]))
    etiquetas = [ETIQUETAS.get(f, f) for f in fuentes]

    valores = {}
    for sensor in config.SENSORES:
        t = contrastables[contrastables["Sensor"] == sensor].set_index("Fuente")
        valores[sensor] = [float(t.loc[f, "p"]) for f in fuentes]

    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    _barras(ax, etiquetas, valores, alfa, ymax, resaltar="Visc")
    ax.set_ylabel("valor p", fontsize=11)

    fig.suptitle(
        f"Significancia de cada factor — {etiqueta}\n"
        f"los 4 proximitores superpuestos · cuanto MAS PEQUENO el valor p, MAS "
        f"fuerte la evidencia de que el factor influye",
        fontsize=14, fontweight="bold")
    _leyenda(fig, alfa)
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def figura_viscosidad(resumen, ruta, alfa, ymax):
    """2x2 por sensor: el p de la VISCOSIDAD en todas las variantes."""
    nombres = [n for n, _ in resumen]
    valores = {}
    for sensor in config.SENSORES:
        ps = []
        for _, t in resumen:
            fila = t[(t["Sensor"] == sensor) & (t["Fuente"] == "Viscosidad")]
            ps.append(float(fila["p"].iloc[0]) if len(fila) else np.nan)
        valores[sensor] = ps

    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    _barras(ax, nombres, valores, alfa, ymax)
    ax.set_ylabel("valor p de la viscosidad", fontsize=11)

    n_sig = sum(1 for _, t in resumen
                for _, f in t[t["Fuente"] == "Viscosidad"].iterrows()
                if f["p"] < alfa)
    n_tot = len(resumen) * len(config.SENSORES)
    fig.suptitle(
        f"¿Es la viscosidad un factor relevante? — valor p en cada analisis\n"
        f"significativa en {n_sig} de {n_tot} casos (sensor x analisis) · "
        f"los 4 proximitores superpuestos · barra dentro de la banda verde = "
        f"la viscosidad SI influye",
        fontsize=14, fontweight="bold")
    _leyenda(fig, alfa)
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROGRAMA
# ============================================================

def _orden(nombre):
    """Global primero, luego los tramos de velocidad, luego los desbalanceos."""
    if nombre == "global":
        return (0, nombre)
    return (1 if nombre.startswith("Grupo") else 2, nombre)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paso 6: graficas del valor p de cada factor, con umbral de "
                    "significancia.")
    p.add_argument("--entrada", default="",
                   help="Carpeta con los anova_split_plot*.csv "
                        "(def: <salida>/p5_estadistica)")
    p.add_argument("--salida", default=config.DIR_RESULTADOS)
    p.add_argument("--alfa", type=float, default=0.05,
                   help="Umbral de significancia (def: 0.05)")
    p.add_argument("--ymax", type=float, default=1.0,
                   help="Limite superior del eje de valor p (def: 1.0, el rango "
                        "completo de una probabilidad). Usa p.ej. --ymax 0.1 "
                        "para acercarte a la zona del umbral")
    args = p.parse_args()

    salida = Path(args.salida).expanduser()
    carpeta = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.DIR_ESTADISTICA
    if not carpeta.is_dir():
        print(f"ERROR: no existe {carpeta}. Ejecuta antes p5_anova_split_plot.py")
        return 1

    archivos = sorted(carpeta.glob("anova_split_plot*.csv"))
    if not archivos:
        print(f"ERROR: no hay ningun anova_split_plot*.csv en {carpeta}.\n"
              f"       Ejecuta antes p5_anova_split_plot.py")
        return 1

    variantes = []
    for ruta in archivos:
        nombre = ruta.stem.replace("anova_split_plot", "").lstrip("_") or "global"
        variantes.append((nombre, pd.read_csv(ruta)))
    variantes.sort(key=lambda v: _orden(v[0]))

    print(f"Carpeta: {carpeta}")
    print(f"Variantes encontradas: {len(variantes)} "
          f"({', '.join(n for n, _ in variantes)})")
    print(f"Umbral de significancia: p < {args.alfa}")
    print(f"Eje de valor p: lineal, de 0 a {args.ymax:g}"
          + ("   (usa --ymax 0.1 para acercarte al umbral)"
             if args.ymax > 0.2 else "") + "\n")

    filas = []
    for nombre, tab in variantes:
        sufijo = "" if nombre == "global" else f"_{nombre}"
        figura_variante(tab, carpeta / f"p6_pvalores{sufijo}.png", args.alfa,
                        nombre.replace("_", " "), args.ymax)
        sub = tab[tab["p"].notna() & (tab["Fuente"] != "Repeticion (bloque)")]
        for _, f in sub.iterrows():
            filas.append({"Analisis": nombre, "Sensor": f["Sensor"],
                          "Fuente": f["Fuente"], "GL": f["GL"],
                          "GL_denominador": f["GL_denominador"], "F": f["F"],
                          "p": f["p"], "Significativa": "SI" if f["p"] < args.alfa else "no"})
        print(f"  p6_pvalores{sufijo}.png")

    figura_viscosidad(variantes, carpeta / "p6_pvalores_viscosidad.png",
                      args.alfa, args.ymax)
    print(f"  p6_pvalores_viscosidad.png")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "p6_pvalores.csv", index=False)

    # ---- veredicto sobre la viscosidad ----
    visc = tabla[tabla["Fuente"] == "Viscosidad"]
    n_sig = int((visc["Significativa"] == "SI").sum())
    print("\n" + "=" * 78)
    print("VISCOSIDAD")
    print("=" * 78)
    print(visc.pivot(index="Analisis", columns="Sensor", values="p")
          .reindex([n for n, _ in variantes])
          .to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nSignificativa (p < {args.alfa}) en {n_sig} de {len(visc)} casos "
          f"(sensor x analisis).")
    if n_sig == len(visc):
        print("La viscosidad resulta relevante en TODOS los analisis y sensores.")
    elif n_sig == 0:
        print("La viscosidad NO alcanza significancia en ningun caso.")
    else:
        no_sig = visc[visc["Significativa"] == "no"]
        print("No alcanza significancia en: "
              + ", ".join(f"{r.Analisis}/{r.Sensor}" for r in no_sig.itertuples()))

    print(f"\nTabla: {carpeta / 'p6_pvalores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
