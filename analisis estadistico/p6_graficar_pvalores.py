"""
STEP 6 — p-value figures (significance of each factor)
======================================================

Same per-probe layout options as step 5, but the Y axis is the **p value** of
each factor instead of its share of the sum of squares. The exact p is written
on every bar and a horizontal line marks the significance threshold (0.05 by
default).

It answers "is viscosity a relevant factor?" at a glance: if its bar sits BELOW
the threshold line it is, and the number on the bar says by how much.

--------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------
The Y axis is the p value on an ORDINARY LINEAR SCALE from 0 to 1 (the natural
range of a probability). The bar height IS the p value, untransformed.

    bar BELOW the red line  ->  factor is SIGNIFICANT
    bar ABOVE the red line  ->  factor is not significant

So the LOWER the bar, the stronger the evidence. The exact p is printed on each
bar, so bars far below the threshold are still told apart by their label.

--ymax bounds the axis (e.g. --ymax 0.1) to zoom into the threshold region;
bars that exceed it are clipped and flagged with an arrow.

--------------------------------------------------------------------------
LAYOUT
--------------------------------------------------------------------------
By default every probe's bars go SIDE BY SIDE in one panel, in the aspect ratio
of a figure that spans the full page width with no column division, and the bar
groups are spaced so consecutive categories do not run together.
--paneles-por-sensor gives each probe its own panel instead, --eje-y-comun puts
every panel on the same Y limits, and --tamano-letra also thins out the Y axis
divisions so the tick labels never crowd at large type.

--------------------------------------------------------------------------
INPUT
--------------------------------------------------------------------------
Reads the `anova_split_plot*.csv` files already written by
`p5_anova_split_plot.py` in <salida>/p5_statistics. Only the variants that
exist are picked up: the global one, the speed bands (--grupos-velocidad) and
the unbalance levels (--grupos-desbalanceo).

OUTPUTS (same folder)
  p6_pvalues<suffix>.png     one per variant found
  p6_pvalues_viscosity.png   summary: the viscosity p across ALL variants
  p6_pvalues.csv             the same numbers as a table

Usage:
    python p6_graficar_pvalores.py --salida <results_folder>
    python p6_graficar_pvalores.py --salida <res> --formato double --ymax 0.1
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
import comun
from p5_anova_split_plot import ETIQUETAS, reparto_paneles

# Suelo numerico: por debajo de esto el valor p ya no es representable en doble
# precision y los CSV traen 0. Se escribe como "<1e-300".
P_MINIMO = 1e-300

# Shared with step 5, so a p value looks the same wherever it is drawn.
COLOR_SIG = config.COLOR_SIGNIFICATIVO
COLOR_NO_SIG = config.COLOR_NO_SIGNIFICATIVO
COLOR_UMBRAL = config.COLOR_UMBRAL

# Layout switches, set from the command line.
PANELES_POR_SENSOR = False      # --paneles-por-sensor
EJE_Y_COMUN = False             # --eje-y-comun


# ============================================================
# HELPERS
# ============================================================

def _rejilla(fmt, relacion=0.85, leyenda=()):
    """
    One panel per probe, side by side while they fit the target width.

    Returns (fig, axes, cols) with exactly one axes per probe; leftover cells of
    the grid are hidden.
    """
    n = len(config.SENSORES)
    filas, cols = reparto_paneles(fmt, n)
    fig, axes = plt.subplots(filas, cols,
                             figsize=comun.tam_figura(fmt, relacion * filas / cols,
                                                      leyenda=leyenda),
                             constrained_layout=True, squeeze=False)
    ejes = list(np.array(axes).ravel())
    for ax in ejes[n:]:
        ax.set_visible(False)
    return fig, ejes[:n], cols


def fmt_p(p):
    """p value as readable text."""
    if p is None or not np.isfinite(p):
        return "—"
    if p <= P_MINIMO:
        return "<1e-300"
    if p >= 1e-3:
        return f"{p:.3g}"
    return f"{p:.0e}"


def _manijas_leyenda(alfa):
    """Legend handles, built before the figure so its height can allow for them."""
    handles = []
    if not PANELES_POR_SENSOR:
        handles += [Patch(facecolor=config.COLOR_SENSOR[s], edgecolor="black",
                          label=s) for s in config.SENSORES]
    handles += [
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label=f"not significant (p >= {alfa:g})"),
        Patch(facecolor=COLOR_SIG, alpha=0.25, edgecolor="none",
              label=f"significant zone (p < {alfa:g})"),
        Line2D([0], [0], color=COLOR_UMBRAL, ls="--",
               label=f"threshold p = {alfa:g}"),
    ]
    return handles


def _leyenda(fig, alfa, fmt=None):
    """
    Shared legend, placed outside the panels: inside it would collide with the
    bar labels, and the direction of the scale (small p = significant) is
    exactly what must not stay ambiguous.
    """
    handles = _manijas_leyenda(alfa)
    etiquetas = [h.get_label() for h in handles]
    # 'outside lower center' is the placement constrained_layout reserves space
    # for; a negative bbox_to_anchor is ignored by the layout and the legend
    # ends up printed over the rotated tick labels.
    fig.legend(handles=handles, loc="outside lower center",
               ncol=(comun.columnas_leyenda(etiquetas, fmt) if fmt else len(handles)),
               frameon=False)


def _barras(ax, categorias, valores, alfa, ymax, resaltar=None, fmt=None):
    """
    Grouped bars: one bar per probe for every category on the X axis. In
    per-panel mode `valores` holds a single sensor and the grouping collapses.

    Linear axis: the bar height IS the p value. Below the threshold line means
    significant, and those bars are filled; non-significant ones are hatched,
    so the reading does not depend on position alone.

      valores : {sensor: [p per category]}
    """
    sensores = [s for s in config.SENSORES if s in valores]
    n = len(sensores)
    x = np.arange(len(categorias), dtype=float)
    # With few probes the group is narrowed so the white space between category
    # blocks grows: spread across a page-wide figure, the groups must not touch.
    ancho = comun.ancho_grupo_barras(n) / max(n, 1)

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
                        va="bottom", fontsize=matplotlib.rcParams["font.size"] * 0.72,
                        rotation=90, zorder=4,
                        color="black" if sig else "#777777")

    ax.set_xticks(x)
    ax.set_xticklabels(categorias, rotation=45, ha="right")
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
    # The threshold is named in the legend; repeating it inside the axes only
    # collides with the right-hand bars and their rotated p labels.
    ax.grid(axis="y", ls="--", alpha=0.35)
    if fmt is not None:
        comun.divisiones_y(ax, fmt)


# ============================================================
# FIGURES
# ============================================================

def figura_variante(tab, ruta, alfa, etiqueta, ymax, fmt):
    """p value of every factor in one variant of the analysis."""
    contrastables = tab[tab["p"].notna()
                        & (tab["Source"] != "Repetition (block)")]
    fuentes = list(dict.fromkeys(contrastables["Source"]))
    etiquetas = [ETIQUETAS.get(f, f) for f in fuentes]

    valores = {}
    for sensor in config.SENSORES:
        t = contrastables[contrastables["Sensor"] == sensor].set_index("Source")
        valores[sensor] = [float(t.loc[f, "p"]) for f in fuentes]

    etiq_leyenda = [h.get_label() for h in _manijas_leyenda(alfa)]
    if PANELES_POR_SENSOR:
        fig, axes, cols = _rejilla(fmt, 0.85, etiq_leyenda)
        for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
            _barras(ax, etiquetas, {sensor: valores[sensor]}, alfa, ymax,
                    resaltar="Visc", fmt=fmt)
            ax.set_title(sensor, fontweight="bold")
            if not EJE_Y_COMUN or j % cols == 0:
                ax.set_ylabel("p value")
    else:
        # ONE panel with every probe's bars side by side, in the aspect ratio of
        # a figure that spans the FULL PAGE WIDTH with no column division.
        fig, ax = plt.subplots(
            figsize=comun.tam_figura(fmt, comun.RELACION_PAGINA,
                                     leyenda=etiq_leyenda),
            constrained_layout=True)
        _barras(ax, etiquetas, valores, alfa, ymax, resaltar="Visc", fmt=fmt)
        ax.set_ylabel("p value")

    fig.suptitle(comun.envolver_titulo(
        f"Significance of each factor — {etiqueta}\n"
        f"the SMALLER the p value, the stronger the evidence that the factor "
        f"influences the response", fmt), fontweight="bold")
    _leyenda(fig, alfa, fmt)
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def figura_viscosidad(resumen, ruta, alfa, ymax, fmt):
    """The VISCOSITY p value across every variant of the analysis."""
    nombres = [n for n, _ in resumen]
    valores = {}
    for sensor in config.SENSORES:
        ps = []
        for _, t in resumen:
            fila = t[(t["Sensor"] == sensor) & (t["Source"] == "Viscosity")]
            ps.append(float(fila["p"].iloc[0]) if len(fila) else np.nan)
        valores[sensor] = ps

    etiq_leyenda = [h.get_label() for h in _manijas_leyenda(alfa)]
    if PANELES_POR_SENSOR:
        fig, axes, cols = _rejilla(fmt, 0.85, etiq_leyenda)
        for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
            _barras(ax, nombres, {sensor: valores[sensor]}, alfa, ymax, fmt=fmt)
            ax.set_title(sensor, fontweight="bold")
            if not EJE_Y_COMUN or j % cols == 0:
                ax.set_ylabel("viscosity p value")
    else:
        fig, ax = plt.subplots(
            figsize=comun.tam_figura(fmt, comun.RELACION_PAGINA,
                                     leyenda=etiq_leyenda),
            constrained_layout=True)
        _barras(ax, nombres, valores, alfa, ymax, fmt=fmt)
        ax.set_ylabel("viscosity p value")

    n_sig = sum(1 for _, t in resumen
                for _, f in t[t["Source"] == "Viscosity"].iterrows()
                if f["p"] < alfa)
    n_tot = len(resumen) * len(config.SENSORES)
    fig.suptitle(comun.envolver_titulo(
        f"Is viscosity a relevant factor? — p value in every analysis\n"
        f"significant in {n_sig} of {n_tot} cases (probe x analysis) · a bar "
        f"inside the green band means viscosity DOES influence the response",
        fmt), fontweight="bold")
    _leyenda(fig, alfa, fmt)
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


# ============================================================
# PROGRAMA
# ============================================================

def _orden(nombre):
    """Global first, then the speed bands, then the unbalance levels."""
    if nombre == "global":
        return (0, nombre)
    return (1 if nombre.startswith("G") else 2, nombre)


def main() -> int:
    global PANELES_POR_SENSOR, EJE_Y_COMUN
    p = argparse.ArgumentParser(
        description="Step 6: p value of each factor, with a significance threshold.")
    p.add_argument("--entrada", default="",
                   help="folder holding the anova_split_plot*.csv files "
                        "(default: <salida>/p5_statistics)")
    p.add_argument("--salida", default=config.DIR_RESULTADOS)
    p.add_argument("--alfa", type=float, default=0.05,
                   help="significance threshold (default: 0.05)")
    p.add_argument("--ymax", type=float, default=1.0,
                   help="upper bound of the p axis (default: 1.0, the full "
                        "range of a probability). Use e.g. --ymax 0.1 to zoom "
                        "into the threshold region")
    p.add_argument("--formato", default=config.FORMATO_POR_DEFECTO,
                   choices=sorted(config.FORMATOS_FIGURA),
                   help="figure size/typography preset (default: screen)")
    p.add_argument("--paneles-por-sensor", dest="paneles", action="store_true",
                   help="grid with one panel per probe instead of the probes "
                        "on shared axes")
    p.add_argument("--tamano-letra", dest="tam_letra", type=float, default=None,
                   help="base font size in points at final size; every other "
                        "element follows it and the figure grows taller")
    p.add_argument("--tamano-titulo", dest="tam_titulo", type=float, default=None,
                   help="title font size, set independently of --tamano-letra")
    p.add_argument("--cojinete", default="todos", choices=["todos", "P1", "P2"],
                   help="probes to plot: both bearings (default), only P1 "
                        "(P1Y, P1X) or only P2")
    p.add_argument("--eje-y-comun", dest="eje_y", action="store_true",
                   help="all panels of one figure share the same Y limits "
                        "(default: each panel uses its own scale)")
    args = p.parse_args()

    PANELES_POR_SENSOR = args.paneles
    EJE_Y_COMUN = args.eje_y
    fmt = comun.aplicar_formato(args.formato, args.tam_letra, args.tam_titulo)
    config.SENSORES = comun.seleccionar_sensores(args.cojinete)

    salida = Path(args.salida).expanduser()
    carpeta = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.DIR_ESTADISTICA
    if not carpeta.is_dir():
        print(f"ERROR: {carpeta} does not exist. Run p5_anova_split_plot.py first.")
        return 1

    archivos = sorted(carpeta.glob("anova_split_plot*.csv"))
    if not archivos:
        print(f"ERROR: no anova_split_plot*.csv found in {carpeta}.\n"
              f"       Run p5_anova_split_plot.py first.")
        return 1

    variantes, incompletas = [], []
    for ruta in archivos:
        nombre = ruta.stem.replace("anova_split_plot", "").lstrip("_") or "global"
        tab = pd.read_csv(ruta)
        # A leftover CSV from an earlier run with a different --cojinete holds
        # other probes than the ones being plotted now. Skipping it with a note
        # is better than a KeyError halfway through the figures.
        if not set(config.SENSORES) <= set(tab["Sensor"]):
            incompletas.append((nombre, sorted(set(tab["Sensor"]))))
            continue
        variantes.append((nombre, tab))
    variantes.sort(key=lambda v: _orden(v[0]))
    if incompletas:
        print("!! Skipped, they do not hold every requested probe (likely left "
              "over from a run with a different --cojinete; delete them or "
              "re-run step 5):")
        for nombre, tiene in incompletas:
            print(f"     anova_split_plot_{nombre}.csv  has {', '.join(tiene)}")
        print()
    if not variantes:
        print("ERROR: no variant holds all of "
              f"{', '.join(config.SENSORES)}. Re-run p5_anova_split_plot.py.")
        return 1

    print(f"Folder  : {carpeta}")
    print(f"Variants: {len(variantes)} ({', '.join(n for n, _ in variantes)})")
    print(f"Threshold: p < {args.alfa}")
    print(f"p axis  : linear, 0 to {args.ymax:g}"
          + ("   (use --ymax 0.1 to zoom into the threshold)"
             if args.ymax > 0.2 else ""))
    print(f"Figures : format '{args.formato}' ({fmt['ancho']:g} in, "
          f"base font {fmt['base']:g} pt, title {fmt['titulo']:g} pt), layout "
          f"{'one panel per probe' if args.paneles else 'probes on shared axes'}"
          f", Y axis {'shared across panels' if args.eje_y else 'per panel'}")
    if fmt.get("escala_texto", 1.0) > 1.01:
        efectivo = fmt["ancho"] * fmt["escala_texto"] ** 0.85
        print(f"         NOTE: the larger font makes the figures ~{efectivo:.1f} in "
              f"wide instead of {fmt['ancho']:g} in. Placed in a "
              f"{fmt['ancho']:g} in column they scale to "
              f"{fmt['base'] * fmt['ancho'] / efectivo:.1f} pt effective type "
              f"(vs {config.FORMATOS_FIGURA[args.formato]['base']:g} pt at "
              f"--tamano-letra default).")
    print(f"Probes  : {', '.join(config.SENSORES)}\n")

    filas = []
    for nombre, tab in variantes:
        sufijo = "" if nombre == "global" else f"_{nombre}"
        figura_variante(tab, carpeta / f"p6_pvalues{sufijo}.png", args.alfa,
                        nombre, args.ymax, fmt)
        sub = tab[tab["p"].notna() & (tab["Source"] != "Repetition (block)")
                  & tab["Sensor"].isin(config.SENSORES)]
        for _, f in sub.iterrows():
            filas.append({"Analysis": nombre, "Sensor": f["Sensor"],
                          "Source": f["Source"], "DF": f["DF"],
                          "DF_denominator": f["DF_denominator"], "F": f["F"],
                          "p": f["p"],
                          "Significant": "yes" if f["p"] < args.alfa else "no"})
        print(f"  p6_pvalues{sufijo}.png")

    figura_viscosidad(variantes, carpeta / "p6_pvalues_viscosity.png",
                      args.alfa, args.ymax, fmt)
    print(f"  p6_pvalues_viscosity.png")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "p6_pvalues.csv", index=False)

    # ---- verdict on viscosity ----
    visc = tabla[tabla["Source"] == "Viscosity"]
    n_sig = int((visc["Significant"] == "yes").sum())
    print("\n" + "=" * 78)
    print("VISCOSITY")
    print("=" * 78)
    print(visc.pivot(index="Analysis", columns="Sensor", values="p")
          .reindex([n for n, _ in variantes])
          .to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nSignificant (p < {args.alfa}) in {n_sig} of {len(visc)} cases "
          f"(probe x analysis).")
    if n_sig == len(visc):
        print("Viscosity is relevant in EVERY analysis and at every probe.")
    elif n_sig == 0:
        print("Viscosity does not reach significance in any case.")
    else:
        no_sig = visc[visc["Significant"] == "no"]
        print("Not significant in: "
              + ", ".join(f"{r.Analysis}/{r.Sensor}" for r in no_sig.itertuples()))

    print(f"\nTable: {carpeta / 'p6_pvalues.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
