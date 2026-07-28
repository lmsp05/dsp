"""
PASO 6 — Graficas de valores p (significancia de cada factor)
=============================================================

Genera, con el mismo reparto 2x2 por sensor que `p5_contribucion.png`, una
figura en la que el eje Y es el **valor p** de cada factor en lugar de su
contribucion a la suma de cuadrados. Sobre cada barra se escribe el valor p
exacto y una linea horizontal marca el umbral de significancia (0.05 por
defecto).

Sirve para responder de un vistazo a "¿es la viscosidad un factor relevante?":
si su barra supera la linea del umbral, lo es, y el numero sobre la barra dice
con cuanto margen.

--------------------------------------------------------------------------
POR QUE EL EJE NO ES LINEAL
--------------------------------------------------------------------------
Los valores p de este ensayo van desde ~0.007 hasta 1e-300: en un eje lineal
las barras interesantes (las que rondan 0.05) serian invisibles. Por eso la
ALTURA de cada barra es -log10(p) -- mas alta = mas significativa -- pero las
MARCAS DEL EJE estan rotuladas con el valor p correspondiente (1, 0.05, 0.01,
0.001, 1e-10 ...). Es decir: se lee el eje en valores p, pero la escala los
comprime para que todos quepan. Ademas la zona decisiva (p entre 1 y 0.001) se
dibuja en escala lineal y solo por encima se comprime (symlog).

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

import config
from p5_anova_split_plot import ETIQUETAS

# Suelo numerico: por debajo de esto el valor p ya no es representable en doble
# precision y los CSV traen 0. Se dibuja como "<1e-300".
P_MINIMO = 1e-300

# Marcas del eje: (valor p, etiqueta). La posicion real es -log10(p).
MARCAS_P = [(1.0, "1"), (0.05, "0.05"), (0.01, "0.01"), (1e-3, "0.001"),
            (1e-10, "1e-10"), (1e-100, "1e-100")]

COLOR_SIG = "#27ae60"       # significativo
COLOR_NO_SIG = "#bdc3c7"    # no significativo
COLOR_UMBRAL = "#c0392b"


# ============================================================
# UTILIDADES
# ============================================================

def altura(p):
    """-log10(p), con suelo para los p que llegan como 0 por desbordamiento."""
    return -np.log10(max(float(p), P_MINIMO))


def fmt_p(p):
    """Valor p en texto legible."""
    if p is None or not np.isfinite(p):
        return "—"
    if p <= P_MINIMO:
        return "<1e-300"
    if p >= 1e-3:
        return f"{p:.3g}"
    return f"{p:.0e}"


def _eje_p(ax, alfa, techo):
    """Configura el eje Y: geometria en -log10(p), rotulos en valor p."""
    ax.set_yscale("symlog", linthresh=3.0, linscale=1.6)
    ax.set_ylim(0, techo)
    marcas = [(v, t) for v, t in MARCAS_P if altura(v) <= techo]
    ax.set_yticks([altura(v) for v, _ in marcas])
    ax.set_yticklabels([t for _, t in marcas], fontsize=9)
    ax.axhline(altura(alfa), color=COLOR_UMBRAL, ls="--", lw=1.6, zorder=3)
    ax.grid(axis="y", ls="--", alpha=0.35)


def _barras(ax, etiquetas, ps, alfa, techo, resaltar=None):
    """Dibuja las barras de un panel y las anota con su valor p."""
    x = np.arange(len(ps))
    alturas = [altura(p) for p in ps]
    colores = [COLOR_SIG if p < alfa else COLOR_NO_SIG for p in ps]
    bordes = ["black" if e == resaltar else "#555555" for e in etiquetas]
    grosor = [2.0 if e == resaltar else 0.5 for e in etiquetas]

    ax.bar(x, alturas, 0.62, color=colores, edgecolor=bordes, linewidth=grosor,
           zorder=2)
    for xi, h, p in zip(x, alturas, ps):
        # En las barras altas la etiqueta va DENTRO: el eje esta comprimido
        # arriba, asi que un texto por encima se saldria de la figura.
        dentro = h > 0.55 * techo
        ax.annotate(fmt_p(p), (xi, h), textcoords="offset points",
                    xytext=(0, -6 if dentro else 4),
                    ha="center", va="top" if dentro else "bottom",
                    fontsize=7.5, rotation=90, zorder=4,
                    color="white" if dentro else
                          ("black" if p < alfa else "#666666"))

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=9)
    if resaltar in etiquetas:
        ax.get_xticklabels()[list(etiquetas).index(resaltar)].set_color(COLOR_UMBRAL)
        ax.get_xticklabels()[list(etiquetas).index(resaltar)].set_fontweight("bold")
    _eje_p(ax, alfa, techo)


# ============================================================
# FIGURAS
# ============================================================

def figura_variante(tab, ruta, alfa, etiqueta):
    """2x2 por sensor: valor p de cada factor en una variante del analisis."""
    contrastables = tab[tab["p"].notna()
                        & (tab["Fuente"] != "Repeticion (bloque)")]
    fuentes = list(dict.fromkeys(contrastables["Fuente"]))
    etiquetas = [ETIQUETAS.get(f, f) for f in fuentes]

    techo = max(4.0, 1.35 * max(altura(p) for p in contrastables["p"]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, sensor in zip(axes.ravel(), config.SENSORES):
        t = contrastables[contrastables["Sensor"] == sensor].set_index("Fuente")
        ps = [float(t.loc[f, "p"]) for f in fuentes]
        _barras(ax, etiquetas, ps, alfa, techo, resaltar="Visc")
        ax.set_title(f"Sensor {sensor}", fontweight="bold")
        ax.set_ylabel("valor p")

    fig.suptitle(
        f"Significancia de cada factor — {etiqueta}\n"
        f"altura = evidencia · linea roja = umbral p = {alfa} · verde = "
        f"significativo · eje comprimido (symlog)",
        fontsize=14, fontweight="bold")
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def figura_viscosidad(resumen, ruta, alfa):
    """2x2 por sensor: el p de la VISCOSIDAD en todas las variantes."""
    nombres = [n for n, _ in resumen]
    techo = max(4.0, 1.35 * max(altura(t[t["Fuente"] == "Viscosidad"]["p"].min())
                                for _, t in resumen))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, sensor in zip(axes.ravel(), config.SENSORES):
        ps = []
        for _, t in resumen:
            fila = t[(t["Sensor"] == sensor) & (t["Fuente"] == "Viscosidad")]
            ps.append(float(fila["p"].iloc[0]) if len(fila) else np.nan)
        _barras(ax, nombres, ps, alfa, techo)
        ax.set_title(f"Sensor {sensor}", fontweight="bold")
        ax.set_ylabel("valor p de la viscosidad")

    n_sig = sum(1 for _, t in resumen
                for _, f in t[t["Fuente"] == "Viscosidad"].iterrows()
                if f["p"] < alfa)
    n_tot = len(resumen) * len(config.SENSORES)
    fig.suptitle(
        f"¿Es la viscosidad un factor relevante? — valor p en cada analisis\n"
        f"significativa en {n_sig} de {n_tot} casos (sensor x analisis) · "
        f"umbral p = {alfa} · contrastada siempre con Error(a)",
        fontsize=14, fontweight="bold")
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
    print(f"Umbral de significancia: p < {args.alfa}\n")

    filas = []
    for nombre, tab in variantes:
        sufijo = "" if nombre == "global" else f"_{nombre}"
        figura_variante(tab, carpeta / f"p6_pvalores{sufijo}.png", args.alfa,
                        nombre.replace("_", " "))
        sub = tab[tab["p"].notna() & (tab["Fuente"] != "Repeticion (bloque)")]
        for _, f in sub.iterrows():
            filas.append({"Analisis": nombre, "Sensor": f["Sensor"],
                          "Fuente": f["Fuente"], "GL": f["GL"],
                          "GL_denominador": f["GL_denominador"], "F": f["F"],
                          "p": f["p"], "Significativa": "SI" if f["p"] < args.alfa else "no"})
        print(f"  p6_pvalores{sufijo}.png")

    figura_viscosidad(variantes, carpeta / "p6_pvalores_viscosidad.png", args.alfa)
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
