"""
STEP 5 — Statistical analysis matched to the experimental design
================================================================

THE PROBLEM
-----------
The experiment is NOT completely randomised. It is a SPLIT-SPLIT-PLOT design
in blocks:

    block          -> repetition  (r = 3)
    whole plot     -> viscosity   (a = 3)   changing the oil is expensive
    subplot        -> unbalance   (b = 3)   the mass is changed within an oil
    sub-subplot    -> speed       (c = 15)  sweep within a single mounting

Every randomisation level has its OWN error term. Testing viscosity against the
global residual (as an ordinary factorial ANOVA does) treats it as if the rotor
had been re-mounted 405 independent times, when there are only 9 whole-plot
units (3 blocks x 3 oils). The denominator is then far too small, the F ratio
is inflated and the p value is unreal.

    Effect             tested against                 denominator df
    ------------------------------------------------------------------
    Viscosity          Error(a) = Rep x Visc                  4
    Unbalance          Error(b) = Rep x Unb within Visc      12
    Visc x Unb         Error(b)                              12
    Speed              Error(c) = residual                  252
    Visc x Speed       Error(c)                             252
    Unb x Speed        Error(c)                             252
    Visc x Unb x Speed Error(c)                             252

In other words: viscosity is judged with 4 degrees of freedom, not 252. This
script computes the full decomposition, applies each F with its correct
denominator and, so the difference is visible, also computes the NAIVE ANOVA
and prints both side by side.

FIGURE LAYOUT
-------------
--formato picks the final figure width and sizes the type for it, so figures
can be dropped into a two-column letter-size page at 100 %:

    screen  wide, for reviewing on a monitor (default)
    column  one column of a two-column page   (3.5 in / 88.9 mm)
    double  spanning both columns             (7.16 in / 181.6 mm)

--paneles-por-sensor draws the bar figures as a grid with one panel per probe
instead of overlaying the probes on shared axes. With few, short category labels
the shared-axes version is more compact; with many categories the per-probe
panels stay legible at column width.

--tamano-letra sets the base font size and everything else follows it, including
the NUMBER OF Y AXIS DIVISIONS, which is reduced as the type grows so the tick
labels never crowd. --tamano-titulo sets the title size independently.

--eje-y-comun forces every panel of a figure onto the same Y limits, making the
probes comparable by height; without it each panel uses its own scale.

The bar figures (contribution, naive vs correct) put every probe in ONE panel
with an aspect ratio for a figure spanning the FULL PAGE WIDTH, and the bar
groups are spaced so the category blocks stay visually separate. The curve
figures use one panel per probe, side by side, with the title on top.

OUTPUTS (in <salida>/p5_statistics)
-----------------------------------
  anova_split_plot.csv         full table per sensor, with correct F and p
  anova_naive.csv              the ordinary factorial ANOVA, for comparison
  viscosity_comparison.csv     correct vs naive test, summarised
  posthoc_viscosity.csv        Tukey between oils using the whole-plot error
  viscosity_means.csv          every number behind p5_viscosity_effect.png:
                               mean per oil, n, MS/df of Error(a) and the 95 % CI
  variance_components.csv      variance attributable to each stratum
  p5_contribution.png          share of the total SS taken by each source
  p5_naive_vs_correct.png      evidence under both models
  p5_viscosity_effect.png      mean per oil, one panel per probe
  p5_interaction_speed.png     viscosity x speed, one panel per probe
  p5_interaction_unbalance.png viscosity x unbalance, one panel per probe
  p5_variance_strata.png       where the random variability comes from
  (--grupos-velocidad)         anova_split_plot_<G1..G4>.csv per speed band
                               p5_viscosity_by_speed_band_means.png
                               p5_viscosity_by_speed_band_evidence.png
  (--grupos-desbalanceo)       anova_split_plot_<U1..U3>.csv per unbalance level
                               p5_viscosity_by_unbalance_means.png
                               p5_viscosity_by_unbalance_evidence.png

Usage:
    python p5_anova_split_plot.py --salida <results_folder>
    python p5_anova_split_plot.py --salida <res> --formato double --paneles-por-sensor
    python p5_anova_split_plot.py --salida <res> --cojinete P1 --tamano-letra 14 --eje-y-comun
    python p5_anova_split_plot.py --salida <res> --grupos-velocidad --grupos-desbalanceo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import config
import comun


# Speed bands for the optional per-band analysis (see config.GRUPOS_VELOCIDAD).
GRUPOS_VELOCIDAD = {k: v["rpm"] for k, v in config.GRUPOS_VELOCIDAD.items()}

# Layout switches, set from the command line.
PANELES_POR_SENSOR = False      # --paneles-por-sensor
EJE_Y_COMUN = False             # --eje-y-comun
MODO_ERROR = "repeticiones"     # --barras-error
ESCALA_P = "log"                # --escala-p
YMAX_P = 1.0                    # --ymax-p

# Significance threshold used by the partition figures. It matches the 5 % that
# the "Significant" columns of the CSV tables already apply.
ALFA = 0.05

ETIQUETAS = {
    "Repetition (block)": "Block",
    "Viscosity": "Visc",
    "Error(a) = Rep x Visc": "Error(a)",
    "Unbalance": "Unb",
    "Viscosity x Unbalance": "Visc×Unb",
    "Error(b) = Rep x Unb | Visc": "Error(b)",
    "Speed": "Speed",
    "Viscosity x Speed": "Visc×Speed",
    "Unbalance x Speed": "Unb×Speed",
    "Viscosity x Unbalance x Speed": "Visc×Unb×Speed",
    "Error(c)": "Error(c)",
    # diseno reducido (un solo desbalanceo): split-plot de dos estratos
    "Error(b) = residual": "Error(b)",
}


# ============================================================
# ANOVA DE PARCELAS SUBDIVIDIDAS
# ============================================================

def _tensor(df, respuesta, reps, viscs, desbs, vels):
    """
    Construye Y[i,j,k,l] = respuesta en (rep_i, visc_j, desb_k, vel_l).

    Exige el diseno COMPLETO y balanceado: las formulas de descomposicion de
    sumas de cuadrados que siguen solo son validas con datos balanceados.
    """
    Y = np.full((len(reps), len(viscs), len(desbs), len(vels)), np.nan)
    ir = {v: i for i, v in enumerate(reps)}
    ij = {v: i for i, v in enumerate(viscs)}
    ik = {v: i for i, v in enumerate(desbs)}
    il = {v: i for i, v in enumerate(vels)}
    for rep, vi, de, ve, y in zip(df["Repetition"], df["Viscosity"],
                                  df["Unbalance"], df["Speed"],
                                  df[respuesta]):
        Y[ir[rep], ij[vi], ik[de], il[ve]] = y
    faltan = int(np.isnan(Y).sum())
    if faltan:
        raise ValueError(
            f"El diseno no esta completo: faltan {faltan} de {Y.size} celdas "
            f"para '{respuesta}'. El ANOVA de parcelas subdivididas requiere "
            f"datos balanceados.")
    return Y


def anova_split_split_plot(Y):
    """
    Descomposicion completa de un split-split-plot en bloques.

    Y[i,j,k,l] con i=repeticion (bloque), j=viscosidad (parcela grande),
    k=desbalanceo (subparcela), l=velocidad (sub-subparcela); una observacion
    por celda.

    Devuelve un DataFrame con SS, gl, MS, el termino de error de cada fuente,
    F y p.
    """
    r, a, b, c = Y.shape
    mu = Y.mean()

    m_i = Y.mean(axis=(1, 2, 3))          # bloque
    m_j = Y.mean(axis=(0, 2, 3))          # viscosidad
    m_k = Y.mean(axis=(0, 1, 3))          # desbalanceo
    m_l = Y.mean(axis=(0, 1, 2))          # velocidad
    m_ij = Y.mean(axis=(2, 3))
    m_jk = Y.mean(axis=(0, 3))
    m_jl = Y.mean(axis=(0, 2))
    m_kl = Y.mean(axis=(0, 1))
    m_ijk = Y.mean(axis=3)
    m_jkl = Y.mean(axis=0)

    SS_tot = float(((Y - mu) ** 2).sum())

    SS_R = a * b * c * float(((m_i - mu) ** 2).sum())
    SS_A = r * b * c * float(((m_j - mu) ** 2).sum())
    SS_Ea = b * c * float(((m_ij - m_i[:, None] - m_j[None, :] + mu) ** 2).sum())

    SS_B = r * a * c * float(((m_k - mu) ** 2).sum())
    SS_AB = r * c * float(((m_jk - m_j[:, None] - m_k[None, :] + mu) ** 2).sum())
    # Error(b) = todo lo que queda entre las celdas rep x visc x desb.
    SS_celdas_ijk = c * float(((m_ijk - mu) ** 2).sum())
    SS_Eb = SS_celdas_ijk - SS_R - SS_A - SS_Ea - SS_B - SS_AB

    SS_C = r * a * b * float(((m_l - mu) ** 2).sum())
    SS_AC = r * b * float(((m_jl - m_j[:, None] - m_l[None, :] + mu) ** 2).sum())
    SS_BC = r * a * float(((m_kl - m_k[:, None] - m_l[None, :] + mu) ** 2).sum())
    abc = (m_jkl
           - m_jk[:, :, None] - m_jl[:, None, :] - m_kl[None, :, :]
           + m_j[:, None, None] + m_k[None, :, None] + m_l[None, None, :]
           - mu)
    SS_ABC = r * float((abc ** 2).sum())

    SS_Ec = SS_tot - (SS_R + SS_A + SS_Ea + SS_B + SS_AB + SS_Eb
                      + SS_C + SS_AC + SS_BC + SS_ABC)

    filas = [
        ("Repetition (block)",                    SS_R,   r - 1,                 "Error(a)"),
        ("Viscosity",                             SS_A,   a - 1,                 "Error(a)"),
        ("Error(a) = Rep x Visc",                  SS_Ea,  (r - 1) * (a - 1),     ""),
        ("Unbalance",                            SS_B,   b - 1,                 "Error(b)"),
        ("Viscosity x Unbalance",               SS_AB,  (a - 1) * (b - 1),     "Error(b)"),
        ("Error(b) = Rep x Unb | Visc",           SS_Eb,  a * (r - 1) * (b - 1), ""),
        ("Speed",                              SS_C,   c - 1,                 "Error(c)"),
        ("Viscosity x Speed",                 SS_AC,  (a - 1) * (c - 1),     "Error(c)"),
        ("Unbalance x Speed",                SS_BC,  (b - 1) * (c - 1),     "Error(c)"),
        ("Viscosity x Unbalance x Speed",   SS_ABC, (a - 1) * (b - 1) * (c - 1), "Error(c)"),
        ("Error(c)",                               SS_Ec,  a * b * (r - 1) * (c - 1), ""),
    ]

    gl_total = sum(f[2] for f in filas)
    assert gl_total == Y.size - 1, f"gl inconsistentes: {gl_total} != {Y.size - 1}"
    return _completar_tabla(filas, SS_tot)


def anova_split_plot_bloques(Y):
    """
    Descomposicion de un SPLIT-PLOT en bloques (dos estratos de error).

    Es el diseno que queda cuando se analiza UN SOLO nivel de desbalanceo: al
    fijar el desbalanceo, ese factor y sus interacciones desaparecen del modelo
    y con ellos el estrato de subparcela. Quedan:

        bloque         -> repeticion  (r)
        parcela grande -> viscosidad  (a)   contrastada con Error(a) = Rep x Visc
        subparcela     -> velocidad   (c)   contrastada con Error(b) = residual

    Y[i,j,l] con i=repeticion, j=viscosidad, l=velocidad; una observacion por
    celda. La viscosidad sigue juzgandose con (r-1)(a-1) = 4 gl: fijar el
    desbalanceo NO aumenta la informacion sobre el aceite, porque el numero de
    montajes de parcela grande es el mismo.
    """
    r, a, c = Y.shape
    mu = Y.mean()
    m_i = Y.mean(axis=(1, 2))
    m_j = Y.mean(axis=(0, 2))
    m_l = Y.mean(axis=(0, 1))
    m_ij = Y.mean(axis=2)
    m_jl = Y.mean(axis=0)

    SS_tot = float(((Y - mu) ** 2).sum())
    SS_R = a * c * float(((m_i - mu) ** 2).sum())
    SS_A = r * c * float(((m_j - mu) ** 2).sum())
    SS_Ea = c * float(((m_ij - m_i[:, None] - m_j[None, :] + mu) ** 2).sum())
    SS_C = r * a * float(((m_l - mu) ** 2).sum())
    SS_AC = r * float(((m_jl - m_j[:, None] - m_l[None, :] + mu) ** 2).sum())
    SS_Eb = SS_tot - (SS_R + SS_A + SS_Ea + SS_C + SS_AC)

    filas = [
        ("Repetition (block)",      SS_R,  r - 1,                 "Error(a)"),
        ("Viscosity",               SS_A,  a - 1,                 "Error(a)"),
        ("Error(a) = Rep x Visc",    SS_Ea, (r - 1) * (a - 1),     ""),
        ("Speed",                SS_C,  c - 1,                 "Error(b)"),
        ("Viscosity x Speed",   SS_AC, (a - 1) * (c - 1),     "Error(b)"),
        ("Error(b) = residual",      SS_Eb, a * (r - 1) * (c - 1), ""),
    ]
    gl_total = sum(f[2] for f in filas)
    assert gl_total == Y.size - 1, f"gl inconsistentes: {gl_total} != {Y.size - 1}"
    return _completar_tabla(filas, SS_tot)


def _completar_tabla(filas, SS_tot):
    """Anade MS, F, p, contribucion y omega2 a una lista de (fuente, SS, gl, error)."""
    tab = pd.DataFrame(filas, columns=["Source", "SS", "DF", "Error"])
    tab["MS"] = tab["SS"] / tab["DF"]
    errores = {f.split(" =")[0].strip(): (ms, gl) for f, ms, gl, e
               in zip(tab["Source"], tab["MS"], tab["DF"], tab["Error"]) if not e}

    F, P, GLD, OM = [], [], [], []
    for _, fila in tab.iterrows():
        e = fila["Error"]
        if not e or e not in errores or errores[e][0] <= 0:
            F.append(np.nan), P.append(np.nan), GLD.append(np.nan), OM.append(np.nan)
            continue
        ms_e, gl_e = errores[e]
        f = fila["MS"] / ms_e
        F.append(f)
        P.append(float(stats.f.sf(f, fila["DF"], gl_e)))
        GLD.append(gl_e)
        OM.append(100.0 * max(fila["SS"] - fila["DF"] * ms_e, 0.0) / (SS_tot + ms_e))
    tab["DF_denominator"], tab["F"], tab["p"] = GLD, F, P
    tab["Contribution_SS_%"] = 100.0 * tab["SS"] / SS_tot
    tab["omega2_%"] = OM
    return tab


def anova_ingenuo(tab):
    """
    Reproduce el ANOVA factorial corriente (el de `f3_s1_v1.py`): ignora los
    bloques y contrasta TODO contra un unico residual, que resulta de juntar
    todos los estratos de error mas el bloque.

    Funciona con cualquiera de los dos disenos: identifica los estratos de error
    como las filas sin termino de error asignado.

    Sirve unicamente para cuantificar cuanto se infla la significancia.
    """
    es_error = tab["Error"].fillna("") == ""
    es_bloque = tab["Source"] == "Repetition (block)"
    agrupadas = es_error | es_bloque

    ss_res = float(tab.loc[agrupadas, "SS"].sum())
    gl_res = int(tab.loc[agrupadas, "DF"].sum())
    ms_res = ss_res / gl_res

    filas = []
    for _, fila in tab.loc[~agrupadas].iterrows():
        F = fila["MS"] / ms_res
        filas.append({"Source": fila["Source"], "SS": fila["SS"], "DF": fila["DF"],
                      "MS": fila["MS"], "DF_denominator": gl_res, "F": F,
                      "p": float(stats.f.sf(F, fila["DF"], gl_res))})
    filas.append({"Source": "Residual (pooled)", "SS": ss_res, "DF": gl_res,
                  "MS": ms_res, "DF_denominator": np.nan, "F": np.nan, "p": np.nan})
    return pd.DataFrame(filas)


def componentes_varianza(tab, forma):
    """
    Varianza atribuible a cada estrato de aleatorizacion (metodo de los momentos
    sobre los cuadrados medios esperados).

    `forma` = (r, a, b, c) en el diseno completo o (r, a, c) cuando se analiza un
    solo desbalanceo. El divisor de cada estrato es el numero de observaciones
    que hay bajo el, es decir el producto de los niveles de los factores anidados
    por debajo.
    """
    idx = {f: i for i, f in enumerate(tab["Source"])}
    if len(forma) == 4:
        r, a, b, c = forma
        estratos = [("Error(a) = Rep x Visc", "Whole plot (oil mounting)", b * c),
                    ("Error(b) = Rep x Unb | Visc", "Subplot (unbalance mounting)", c),
                    ("Error(c)", "Sub-subplot (speed sweep)", 1)]
    else:
        r, a, c = forma
        estratos = [("Error(a) = Rep x Visc", "Whole plot (oil mounting)", c),
                    ("Error(b) = residual", "Subplot (speed sweep)", 1)]

    ms = [tab.loc[idx[f], "MS"] for f, _, _ in estratos]
    var = []
    for k in range(len(estratos)):
        if k < len(estratos) - 1:
            var.append(max((ms[k] - ms[k + 1]) / estratos[k][2], 0.0))
        else:
            var.append(max(ms[k], 0.0))
    tot = sum(var) or 1.0
    return pd.DataFrame([
        {"Stratum": nombre, "MS": ms[k], "Variance": var[k],
         "Percentage": 100 * var[k] / tot}
        for k, (_, nombre, _) in enumerate(estratos)])


def tukey_viscosidad(Y, tab, viscs, alfa=0.05):
    """
    Comparaciones entre aceites con el termino de error CORRECTO (Error(a)).

    HSD = q(alfa, a, gl_Ea) * sqrt(MS_Ea / n_por_media). `n_por_media` es el
    numero de observaciones que hay detras de la media de cada aceite: r*b*c en
    el diseno completo, r*c cuando se analiza un solo desbalanceo.
    """
    if Y.ndim == 4:
        r, a, b, c = Y.shape
        n = r * b * c
        medias = Y.mean(axis=(0, 2, 3))
    else:
        r, a, c = Y.shape
        n = r * c
        medias = Y.mean(axis=(0, 2))
    idx = {f: i for i, f in enumerate(tab["Source"])}
    ms_e = tab.loc[idx["Error(a) = Rep x Visc"], "MS"]
    gl_e = tab.loc[idx["Error(a) = Rep x Visc"], "DF"]
    q = float(stats.studentized_range.ppf(1 - alfa, a, gl_e))
    hsd = q * np.sqrt(ms_e / n)

    filas = []
    for i in range(a):
        for j in range(i + 1, a):
            d = float(medias[i] - medias[j])
            t = abs(d) / np.sqrt(2 * ms_e / n)
            p = float(stats.studentized_range.sf(t * np.sqrt(2), a, gl_e))
            filas.append({
                "Comparison": f"ISO {viscs[i]} - ISO {viscs[j]}",
                "Difference": d, "HSD_95": hsd,
                "CI_low": d - hsd, "CI_high": d + hsd,
                "p_Tukey": p, "Significant": "yes" if abs(d) > hsd else "no",
            })
    return pd.DataFrame(filas), medias, ms_e, gl_e, n


# ============================================================
# FIGURAS
# ============================================================

def _fmt_p(p):
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# A panel narrower than this cannot hold rotated tick labels and an axis label
# without matplotlib collapsing the layout.
ANCHO_MINIMO_PANEL = 2.0


def reparto_paneles(fmt, n, filas_por_sensor=1):
    """
    (rows, cols) for `n` per-probe panels, given the final figure width.

    Panels are laid side by side while each still gets at least
    ANCHO_MINIMO_PANEL inches; otherwise they are STACKED VERTICALLY, which is
    what keeps a column-width figure usable when the font is large. The
    threshold scales with the font size, since bigger type needs more room.
    """
    minimo = ANCHO_MINIMO_PANEL * fmt.get("escala_texto", 1.0) ** 0.7
    cols = max(1, min(n, int(fmt["ancho"] // minimo)))
    if n > 2 and cols >= 2:
        cols = 2
    filas = int(np.ceil(n / cols)) * filas_por_sensor
    return filas, cols


def _rejilla_sensores(fmt, relacion=0.80, ancho_rel=1.0, leyenda=()):
    """
    One panel per probe, SIDE BY SIDE while they fit the target width.

    Returns (fig, axes, cols): `axes` holds exactly one axes per probe, in
    config.SENSORES order, and `cols` is how many of them sit in a row — needed
    to decide which panels carry the Y axis label. Any leftover cell of the grid
    is hidden, so an odd number of probes does not leave an empty frame.

    `leyenda` is the list of labels of the legend that will be placed below the
    figure, so the canvas can be made tall enough to hold it.
    """
    n = len(config.SENSORES)
    filas, cols = reparto_paneles(fmt, n)
    fig, axes = plt.subplots(filas, cols,
                             figsize=comun.tam_figura(
                                 fmt, relacion * filas / cols, ancho_rel,
                                 leyenda),
                             constrained_layout=True, squeeze=False)
    ejes = list(np.array(axes).ravel())
    for ax in ejes[n:]:
        ax.set_visible(False)
    return fig, ejes[:n], cols


def _CAPSIZE(fmt):
    """Error bar cap, scaled with the line width so it stays visible at any font."""
    return max(2.0, fmt["linea"] * 2.2)


def _TEXTO_ERROR(n_rep):
    """How the error bars must be described in the figure title."""
    if MODO_ERROR == "total":
        return "±1 SD of all the observations behind each point"
    return f"±1 SD across the {n_rep} repetitions"


def _dispersion(df, col, claves):
    """
    +-1 standard deviation for a mean plotted over `claves`.

    Which standard deviation is NOT a detail, because every point on these
    figures averages over factors that the experiment moves on purpose:

      'repeticiones' (default) — the values are first averaged WITHIN each
        repetition, and the SD is taken between those repetition means. It
        therefore measures repeatability: how much the same nominal condition
        moves when the rotor is re-mounted.

      'total' — the SD of every observation behind the point. It also contains
        the designed spread of the averaged factors: an unbalance level changes
        the amplitude by roughly a factor of three, so a bar computed this way
        is dominated by that effect and says almost nothing about experimental
        error.
    """
    if MODO_ERROR == "total":
        return df.groupby(claves)[col].std(ddof=1)
    por_rep = df.groupby(list(claves) + ["Repetition"])[col].mean()
    return por_rep.groupby(list(claves)).std(ddof=1)


def _desviacion_por_viscosidad(Y):
    """
    SD per viscosity for the group figures, from the response tensor.

    Y is (rep, visc, unb, speed) or (rep, visc, speed). In 'repeticiones' mode
    everything below the repetition is collapsed first, leaving one value per
    (repetition, oil), and the SD is taken across repetitions.
    """
    if MODO_ERROR == "total":
        ejes = tuple(i for i in range(Y.ndim) if i != 1)
        return Y.std(axis=ejes, ddof=1)
    por_rep = Y.mean(axis=tuple(range(2, Y.ndim)))       # (rep, visc)
    return por_rep.std(axis=0, ddof=1)


def _limites_error(ax, medias, errores, margen=0.08):
    """
    Y limits that leave the WHOLE error bar inside the axes.

    Autoscaling from the markers alone clips the caps of the outermost bars,
    which is precisely the information the bar was added to show.
    """
    m = np.concatenate([np.asarray(v, float).ravel() for v in medias])
    e = np.nan_to_num(
        np.concatenate([np.asarray(v, float).ravel() for v in errores]))
    finito = np.isfinite(m)
    if not finito.any():
        return
    bajo, alto = float((m - e)[finito].min()), float((m + e)[finito].max())
    pad = margen * (alto - bajo) if alto > bajo else max(abs(alto), 1.0) * margen
    ax.set_ylim(bajo - pad, alto + pad)


def _ticks_velocidad(ax, x, vels, fmt, ancho_panel):
    """
    Speed axis: a tick at EVERY speed, a label on as many as actually fit.

    Thinning the speeds themselves would hide that the sweep is not uniformly
    spaced, so the unlabelled speeds stay as MINOR ticks. The reader can count
    the divisions between two labelled values and place the intermediate points
    approximately, which a bare label every third speed does not allow.
    """
    # Horizontal room a 60 degree rotated numeric label needs, in inches.
    por_etiqueta = fmt["ticks"] * 1.30 / 72.0
    paso = max(1, int(np.ceil(len(vels) * por_etiqueta / max(ancho_panel, 0.1))))
    ax.set_xticks(x[::paso])
    ax.set_xticklabels([str(v) for v in vels[::paso]], rotation=60)
    if paso > 1:
        ax.set_xticks(x, minor=True)
        ax.tick_params(axis="x", which="minor",
                       length=max(2.0, fmt["ticks"] * 0.30))
        ax.grid(axis="x", which="minor", ls=":", alpha=0.30)
    return paso


def _etiqueta_y(ax, j, cols, texto):
    """
    Y axis label: on every panel when each has its own scale, only on the first
    column when --eje-y-comun forces a single shared scale (repeating the same
    label next to identical ticks is just noise).
    """
    if not EJE_Y_COMUN or j % cols == 0:
        ax.set_ylabel(texto)


def barras_por_sensor(ax, categorias, valores, rayado=None, ancho_grupo=None):
    """
    Grouped bars with the four probes on the SAME axes: one bar per probe for
    every category on the X axis.

      categorias : X axis labels
      valores    : {sensor: [value per category]}
      rayado     : {sensor: [bool]} -> hatched bars, used to flag
                   "not significant"

    With few probes the group is made narrower so the gap between categories
    grows: two fat bars touching the next pair read as one block of four.
    """
    sensores = [s for s in config.SENSORES if s in valores]
    n = len(sensores)
    if ancho_grupo is None:
        ancho_grupo = 0.82 if n >= 4 else (0.62 if n == 2 else 0.72)
    x = np.arange(len(categorias), dtype=float)
    ancho = ancho_grupo / max(n, 1)

    for k, sensor in enumerate(sensores):
        desplazamiento = (k - (n - 1) / 2) * ancho
        tramas = (rayado or {}).get(sensor, [False] * len(categorias))
        for i, (v, t) in enumerate(zip(valores[sensor], tramas)):
            ax.bar(x[i] + desplazamiento, v, ancho * 0.92,
                   color=config.COLOR_SENSOR.get(sensor, "#888888"),
                   edgecolor="black", hatch="///" if t else None, zorder=2)

    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(categorias) - 0.5)
    ax.grid(axis="y", ls="--", alpha=0.4)
    return x


def leyenda_figura(fig, handles, fmt=None):
    """
    Legend BELOW the figure, in space reserved by the layout engine.

    'outside lower center' is what makes constrained_layout shrink the axes to
    make room. A plain 'lower center' with a negative bbox_to_anchor is not
    accounted for by the layout, so at large font sizes it lands on top of the
    rotated tick labels.
    """
    etiquetas = [h.get_label() for h in handles]
    fig.legend(handles=handles, loc="outside lower center", frameon=False,
               ncol=(comun.columnas_leyenda(etiquetas, fmt) if fmt
                     else len(handles)))


def leyenda_sensores(fig, extra_handles=(), ax=None, loc="upper right",
                     fmt=None):
    """
    Shared legend for the probes.

    With `ax` it goes INSIDE that axes, which keeps it clear of long rotated
    tick labels; otherwise it sits below the figure.
    """
    handles = [Patch(facecolor=config.COLOR_SENSOR[s], edgecolor="black",
                     label=s) for s in config.SENSORES]
    handles += list(extra_handles)
    if ax is not None:
        etiquetas = [h.get_label() for h in handles]
        ncol = (comun.columnas_leyenda(etiquetas, fmt) if fmt else len(handles))
        ax.legend(handles=handles, loc=loc, ncol=max(1, min(2, ncol)),
                  framealpha=0.9)
    else:
        leyenda_figura(fig, handles, fmt)


# ============================================================
# FIGURES
# ============================================================

def fig_contribucion(tablas, ruta, fmt, extra=""):
    """Share of the total sum of squares taken by each source."""
    # Sources are read from the table itself, so the figure works for the full
    # design (11 sources) and for the reduced one-unbalance design (6).
    fuentes = [f for f in tablas[config.SENSORES[0]]["Source"]
               if f != "Repetition (block)"]
    etiquetas = [ETIQUETAS.get(f, f) for f in fuentes]

    valores, rayado = {}, {}
    for sensor in config.SENSORES:
        t = tablas[sensor].set_index("Source")
        valores[sensor] = [float(t.loc[f, "Contribution_SS_%"]) for f in fuentes]
        rayado[sensor] = [not (np.isfinite(t.loc[f, "p"]) and t.loc[f, "p"] < 0.05)
                          for f in fuentes]

    trama = Patch(facecolor="white", edgecolor="black", hatch="///",
                  label="not significant")
    diseno = "split-split-plot" if "Error(c)" in fuentes else "split-plot"

    if PANELES_POR_SENSOR:
        ymax = max(max(v) for v in valores.values()) * 1.12
        fig, axes, cols = _rejilla_sensores(fmt, 0.85,
                                            leyenda=[trama.get_label()])
        for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
            ax.bar(range(len(fuentes)), valores[sensor],
                   color=config.COLOR_SENSOR[sensor], edgecolor="black",
                   hatch=None, zorder=2)
            for i, t in enumerate(rayado[sensor]):
                if t:
                    ax.patches[i].set_hatch("///")
                    ax.patches[i].set_facecolor("white")
            ax.set_title(sensor, fontweight="bold")
            ax.set_ylim(0, ymax)
            ax.set_xticks(range(len(fuentes)))
            ax.set_xticklabels(etiquetas, rotation=45, ha="right")
            ax.grid(axis="y", ls="--", alpha=0.4)
            comun.divisiones_y(ax, fmt)
            _etiqueta_y(ax, j, cols, "Contribution to total SS (%)")
        leyenda_figura(fig, [trama], fmt)
    else:
        # ONE panel, aspect ratio for a figure that spans the FULL PAGE WIDTH
        # with no column division: the bar groups then spread across the page
        # with real white space between them instead of touching.
        fig, ax = plt.subplots(
            figsize=comun.tam_figura(fmt, comun.RELACION_PAGINA),
            constrained_layout=True)
        barras_por_sensor(ax, etiquetas, valores, rayado,
                          ancho_grupo=comun.ancho_grupo_barras(
                              len(config.SENSORES)))
        ax.set_xticklabels(etiquetas, rotation=45, ha="right")
        ax.set_ylabel("Contribution to total SS (%)")
        comun.divisiones_y(ax, fmt)
        leyenda_sensores(fig, [trama], ax=ax, fmt=fmt)

    fig.suptitle(comun.envolver_titulo(
        f"Variability decomposition — {diseno}{extra}\n"
        "hatched bar = not significant at 5 % with its own error term", fmt),
        fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def fig_comparacion(tablas, ingenuos, ruta, fmt, extra=""):
    """What changes when the design is respected: evidence under both models."""
    fuentes = [f for f in ingenuos[config.SENSORES[0]]["Source"]
               if f != "Residual (pooled)"]
    et = [ETIQUETAS.get(f, f) for f in fuentes]

    def _ejes(ax):
        ax.axhline(-np.log10(0.05), color="black", ls="--")
        # symlog: the decisive band (p from 1 to 0.001) stays linear and legible,
        # and the astronomical p values of the naive model do not squash it.
        ax.set_yscale("symlog", linthresh=3.0, linscale=1.4)
        ax.set_ylim(0, 400)
        ax.set_yticks([0, 1, 2, 3, 10, 100])
        ax.set_yticklabels(["0", "1", "2", "3", "10", "100"])
        ax.set_xticklabels(et, rotation=45, ha="right")
        ax.grid(axis="y", ls="--", alpha=0.4)

    # Short panel titles: side by side, a one-line explanation per panel is
    # wider than the panel and the two titles print over each other. The full
    # explanation lives in the figure title instead.
    modelos = [("Naive ANOVA\n(all against one residual)", ingenuos),
               ("Split-plot\n(own error per effect)", tablas)]

    if PANELES_POR_SENSOR:
        n = len(config.SENSORES)
        _, cols = reparto_paneles(fmt, n)
        bloques = int(np.ceil(n / cols))
        fig, axes = plt.subplots(2 * bloques, cols,
                                 figsize=comun.tam_figura(fmt, 0.55 * 2 * bloques / cols),
                                 constrained_layout=True, squeeze=False)
        for fila, (titulo, fuente) in enumerate(modelos):
            for k, sensor in enumerate(config.SENSORES):
                ax = axes[fila * bloques + k // cols, k % cols]
                col = k % cols
                t = fuente[sensor].set_index("Source")
                ax.bar(range(len(fuentes)),
                       [-np.log10(max(float(t.loc[f, "p"]), 1e-300)) for f in fuentes],
                       color=config.COLOR_SENSOR[sensor], edgecolor="black",
                       zorder=2)
                ax.set_xticks(range(len(fuentes)))
                _ejes(ax)
                if fila == 0:
                    ax.set_title(sensor, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{'naive' if fila == 0 else 'correct'}\n"
                                  "evidence  -log10(p)")
    else:
        # The two models go side by side only if each half is wide enough;
        # otherwise they stack, which is what a narrow column needs.
        _, cols = reparto_paneles(fmt, 2)
        filas = 2 // cols
        fig, axes = plt.subplots(filas, cols,
                                 figsize=comun.tam_figura(
                                     fmt, 0.85 * filas / cols,
                                     leyenda=list(config.SENSORES)),
                                 constrained_layout=True, sharey=True,
                                 squeeze=False)
        axes = list(np.array(axes).ravel())
        for ax, (titulo, fuente) in zip(axes, modelos):
            valores = {}
            for sensor in config.SENSORES:
                t = fuente[sensor].set_index("Source")
                valores[sensor] = [-np.log10(max(float(t.loc[f, "p"]), 1e-300))
                                   for f in fuentes]
            barras_por_sensor(ax, et, valores,
                              ancho_grupo=comun.ancho_grupo_barras(
                                  len(config.SENSORES)))
            _ejes(ax)
            ax.set_title(titulo, fontweight="bold")
        axes[0].set_ylabel("evidence  -log10(p)")
        leyenda_sensores(fig, fmt=fmt)

    fig.suptitle(comun.envolver_titulo(
        f"Effect of respecting the split-plot design{extra}\n"
        "the right-hand side is the REAL evidence; the left-hand side is what "
        "you get by ignoring the design", fmt), fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def fig_efecto_viscosidad(resultados, viscs, ruta, fmt, unidad, extra=""):
    """
    Mean per oil with the confidence interval of the whole-plot error, ONE PANEL
    PER PROBE.

    Separate panels rather than one shared axis: the probes measure different
    directions and their amplitudes are not on the same scale, so overlaying
    them hides the effect inside whichever probe happens to read largest.
    """
    fig, axes, cols = _rejilla_sensores(fmt, 0.88)
    x = np.arange(len(viscs), dtype=float)

    for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
        medias, ms_e, gl_e, n = (resultados[sensor][c] for c in
                                 ("medias", "ms_e", "gl_e", "n"))
        semi = float(stats.t.ppf(0.975, gl_e)) * np.sqrt(ms_e / n)
        for k, v in enumerate(viscs):
            ax.bar(x[k], float(medias[k]), 0.62, yerr=semi, capsize=3,
                   color=comun.color_condicion(v, 3), edgecolor="black",
                   label=f"ISO {v}" if j == 0 else None, zorder=2)
        p = resultados[sensor]["tabla"].set_index("Source").loc["Viscosity", "p"]
        ax.set_title(f"{sensor}   (p = {_fmt_p(p)})", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"ISO {v}" for v in viscs])
        ax.grid(axis="y", ls="--", alpha=0.4)
        comun.divisiones_y(ax, fmt)
        _etiqueta_y(ax, j, cols, unidad)

    comun.igualar_ejes_y(axes, EJE_Y_COMUN)
    # No legend: the X axis already names each oil, and repeating it below the
    # panels only costs vertical space.
    fig.suptitle(comun.envolver_titulo(
        f"Effect of viscosity{extra} — mean per oil at each probe, "
        "95 % CI based on Error(a)", fmt), fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def _leyenda_aceites(fig, viscs, fmt, extra_handles=()):
    handles = [Line2D([0], [0], color=comun.color_condicion(v, 3),
                      label=f"ISO {v}") for v in viscs]
    handles += list(extra_handles)
    leyenda_figura(fig, handles, fmt)


def fig_interaccion_velocidad(df, respuestas, viscs, desbs, vels, ruta, fmt,
                              unidad, extra=""):
    """
    VISCOSITY x SPEED, one panel per probe, panels side by side and the title on
    top.

    This used to be the upper row of a two-row 'interactions' figure. The two
    rows answered different questions on different X axes, so they are now
    separate figures: each can be placed, sized and captioned on its own.
    """
    x = np.arange(len(vels))
    etiq_leyenda = [f"ISO {v}" for v in viscs]
    fig, axes, cols = _rejilla_sensores(fmt, 0.82, leyenda=etiq_leyenda)

    ancho_panel = fig.get_size_inches()[0] / cols
    for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
        col = respuestas[sensor]
        medias, errores = [], []
        for v in viscs:
            sub = df[df["Viscosity"] == v]
            m = sub.groupby("Speed")[col].mean().reindex(vels).to_numpy()
            s = _dispersion(sub, col, ["Speed"]).reindex(vels).to_numpy()
            ax.errorbar(x, m, yerr=s, color=comun.color_condicion(v, 3),
                        label=f"ISO {v}", capsize=_CAPSIZE(fmt),
                        elinewidth=fmt["linea"] * 0.7)
            medias.append(m), errores.append(s)
        ax.set_title(sensor, fontweight="bold")
        _ticks_velocidad(ax, x, vels, fmt, ancho_panel)
        ax.grid(alpha=0.3, ls="--")
        ax.set_xlabel("Speed [rpm]")
        _limites_error(ax, medias, errores)
        comun.divisiones_y(ax, fmt)
        _etiqueta_y(ax, j, cols, f"mean {unidad}")

    comun.igualar_ejes_y(axes, EJE_Y_COMUN)
    _leyenda_aceites(fig, viscs, fmt)
    n_rep = len(df["Repetition"].unique())
    fig.suptitle(comun.envolver_titulo(
        f"Viscosity x speed interaction{extra}\n"
        f"each point averages {n_rep} repetitions x {len(desbs)} unbalance "
        f"level(s); error bars {_TEXTO_ERROR(n_rep)}", fmt), fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def fig_interaccion_desbalanceo(df, respuestas, viscs, desbs, vels, ruta, fmt,
                                unidad, extra=""):
    """VISCOSITY x UNBALANCE, one panel per probe, panels side by side."""
    x = np.arange(len(desbs))
    etiq_leyenda = [f"ISO {v}" for v in viscs]
    fig, axes, cols = _rejilla_sensores(fmt, 0.82, leyenda=etiq_leyenda)

    for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
        col = respuestas[sensor]
        medias, errores = [], []
        for v in viscs:
            sub = df[df["Viscosity"] == v]
            m = sub.groupby("Unbalance")[col].mean().reindex(desbs).to_numpy()
            s = _dispersion(sub, col, ["Unbalance"]).reindex(desbs).to_numpy()
            ax.errorbar(x, m, yerr=s, fmt="o-",
                        color=comun.color_condicion(v, 3), label=f"ISO {v}",
                        capsize=_CAPSIZE(fmt), elinewidth=fmt["linea"] * 0.7)
            medias.append(m), errores.append(s)
        ax.set_title(sensor, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"U{d}" for d in desbs])
        ax.set_xlabel("Unbalance level")
        ax.grid(alpha=0.3, ls="--")
        _limites_error(ax, medias, errores)
        comun.divisiones_y(ax, fmt)
        _etiqueta_y(ax, j, cols, f"mean {unidad}")

    comun.igualar_ejes_y(axes, EJE_Y_COMUN)
    _leyenda_aceites(fig, viscs, fmt)
    n_rep = len(df["Repetition"].unique())
    fig.suptitle(comun.envolver_titulo(
        f"Viscosity x unbalance interaction{extra}\n"
        f"each point averages {n_rep} repetitions x {len(vels)} speeds; "
        f"error bars {_TEXTO_ERROR(n_rep)}", fmt), fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def fig_grupo_medias(resumen, viscs, ruta, fmt, unidad, que="speed band",
                     n_rep=0):
    """
    Mean per oil across the levels of `que`, ONE PANEL PER PROBE side by side.

    Formerly the upper row of `p5_viscosity_by_*`; it is a figure of its own
    because it carries the measured magnitude, while the companion figure
    carries the evidence. Mixing units and -log10(p) in one figure forced the
    reader to switch scales between rows.

    The band meanings are deliberately NOT written into the title: they are
    declared once elsewhere in the document instead of on every figure.
    """
    nombres = [n for n, _, _, _ in resumen]
    x = np.arange(len(resumen), dtype=float)
    etiq_leyenda = [f"ISO {v}" for v in viscs]
    fig, axes, cols = _rejilla_sensores(fmt, 0.82, leyenda=etiq_leyenda)

    for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
        medias, errores = [], []
        for iv, v in enumerate(viscs):
            m = np.array([me[sensor][iv] for _, me, _, _ in resumen], float)
            s = np.array([de[sensor][iv] for _, _, de, _ in resumen], float)
            ax.errorbar(x, m, yerr=s, fmt="o-",
                        color=comun.color_condicion(v, 3), label=f"ISO {v}",
                        capsize=_CAPSIZE(fmt), elinewidth=fmt["linea"] * 0.7)
            medias.append(m), errores.append(s)
        ax.set_title(sensor, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(nombres)
        ax.set_xlabel(que.capitalize())
        ax.grid(alpha=0.3, ls="--")
        _limites_error(ax, medias, errores)
        comun.divisiones_y(ax, fmt)
        _etiqueta_y(ax, j, cols, f"mean {unidad}")

    comun.igualar_ejes_y(axes, EJE_Y_COMUN)
    _leyenda_aceites(fig, viscs, fmt)
    fig.suptitle(comun.envolver_titulo(
        f"Mean response per oil by {que} — error bars {_TEXTO_ERROR(n_rep)}",
        fmt), fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def _fmt_p_barra(p):
    """Exact p value as a short bar label, keeping resolution for tiny values."""
    if not np.isfinite(p):
        return "—"
    if p <= 1e-300:
        return "<1e-300"
    return f"{p:.3g}" if p >= 1e-3 else f"{p:.0e}"


def _barras_evidencia_log(ax, x, ps, sensor, fmt):
    """
    -log10(p): the bar GROWS with the evidence.

    The transform is what keeps a p of 1e-8 and a p of 0.04 on the same axis at
    all; the price is that the height is no longer a probability, so the
    threshold has to be read off the dashed line.
    """
    ax.bar(x, -np.log10(ps), 0.58,
           color=[config.COLOR_SENSOR[sensor] if q < ALFA else "white"
                  for q in ps],
           edgecolor="black",
           hatch=["" if q < ALFA else "///" for q in ps], zorder=2)
    ax.axhline(-np.log10(ALFA), color=config.COLOR_UMBRAL, ls="--", lw=1.6,
               zorder=3)
    return "evidence  -log10(p)"


def _barras_evidencia_lineal(ax, x, ps, sensor, fmt):
    """
    The p value itself, on an ordinary linear axis from 0 to YMAX_P.

    The bar height IS the probability, so the reading is direct — but the
    direction INVERTS with respect to -log10(p): here the SHORTER the bar, the
    stronger the evidence. That reversal is exactly what must not stay
    ambiguous, so the significant region is shaded green, non-significant bars
    are hatched, and the exact p is written on every bar.
    """
    for i, p in enumerate(ps):
        sig = p < ALFA
        ax.bar(x[i], min(p, YMAX_P), 0.58,
               color=config.COLOR_SENSOR[sensor] if sig else "white",
               edgecolor="black", hatch=None if sig else "///", zorder=2)
        if p > YMAX_P:
            # The bar runs off the top of the axis. Its label has to go INSIDE
            # the panel: printed above the axis it would climb over the panel
            # title, which is exactly what --ymax-p makes happen.
            ax.annotate("", (x[i], YMAX_P), xytext=(x[i], YMAX_P * 0.93),
                        arrowprops=dict(arrowstyle="-|>", color="#555555",
                                        lw=1.1), zorder=4)
            ax.annotate(_fmt_p_barra(p), (x[i], YMAX_P * 0.90), ha="center",
                        va="top", fontsize=fmt["anotacion"], rotation=90,
                        zorder=5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.75))
        else:
            ax.annotate(_fmt_p_barra(p), (x[i], p),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        va="bottom", fontsize=fmt["anotacion"], rotation=90,
                        zorder=4, color="black" if sig else "#777777")
    ax.set_ylim(0, YMAX_P)
    ax.axhspan(0, ALFA, color=config.COLOR_SIGNIFICATIVO, alpha=0.10, zorder=0)
    ax.axhline(ALFA, color=config.COLOR_UMBRAL, ls="--", lw=1.6, zorder=3)
    return "p value"


def fig_grupo_evidencia(resumen, ruta, fmt, que="speed band"):
    """
    Evidence of the viscosity effect across the levels of `que`, ONE PANEL PER
    PROBE side by side. Formerly the lower row of `p5_viscosity_by_*`.

    --escala-p picks how the p value is drawn: 'log' plots -log10(p) (taller =
    more evidence) and 'lineal' plots the raw probability (shorter = more
    evidence). See the two helpers above.
    """
    nombres = [n for n, _, _, _ in resumen]
    x = np.arange(len(resumen), dtype=float)
    lineal = ESCALA_P == "lineal"

    etiq = [f"not significant (p >= {ALFA:g})"]
    if lineal:
        etiq += [f"significant zone (p < {ALFA:g})"]
    etiq += [f"threshold p = {ALFA:g}"]
    fig, axes, cols = _rejilla_sensores(fmt, 0.82, leyenda=etiq)

    for j, (ax, sensor) in enumerate(zip(axes, config.SENSORES)):
        ps = np.array([max(float(pp[sensor]), 1e-300) for _, _, _, pp in resumen])
        dibujar = _barras_evidencia_lineal if lineal else _barras_evidencia_log
        titulo_y = dibujar(ax, x, ps, sensor, fmt)
        ax.set_title(sensor, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(nombres)
        ax.set_xlabel(que.capitalize())
        ax.grid(axis="y", alpha=0.3, ls="--")
        comun.divisiones_y(ax, fmt)
        _etiqueta_y(ax, j, cols, titulo_y)

    # On the linear scale every panel already spans 0..YMAX_P, so equalising is
    # a no-op there; it only matters for -log10(p).
    comun.igualar_ejes_y(axes, EJE_Y_COMUN)

    handles = [Patch(facecolor="white", edgecolor="black", hatch="///",
                     label=f"not significant (p >= {ALFA:g})")]
    if lineal:
        handles.append(Patch(facecolor=config.COLOR_SIGNIFICATIVO, alpha=0.25,
                             edgecolor="none",
                             label=f"significant zone (p < {ALFA:g})"))
    handles.append(Line2D([0], [0], color=config.COLOR_UMBRAL, ls="--",
                          label=f"threshold p = {ALFA:g}"))
    leyenda_figura(fig, handles, fmt)

    n_sig = sum(1 for _, _, _, pp in resumen
                for s in config.SENSORES if pp[s] < ALFA)
    n_tot = len(resumen) * len(config.SENSORES)
    # The band meanings are NOT spelled out here on purpose: they are declared
    # once elsewhere in the document rather than repeated on every figure.
    como = ("a bar inside the green band means p < "
            f"{ALFA:g} — the SHORTER the bar, the stronger the evidence"
            if lineal else
            f"a bar above the dashed line means p < {ALFA:g} — the TALLER the "
            f"bar, the stronger the evidence")
    fig.suptitle(comun.envolver_titulo(
        f"Evidence of the viscosity effect by {que} — significant in {n_sig} "
        f"of {n_tot} cases (probe x level); {como}", fmt), fontweight="bold")
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def fig_estratos(comps, ruta, fmt, extra=""):
    """Where the random variability comes from, stratum by stratum."""
    fig, ax = plt.subplots(figsize=comun.tam_figura(fmt, 0.48),
                           constrained_layout=True)
    estratos = comps[config.SENSORES[0]]["Stratum"].tolist()
    x = np.arange(len(config.SENSORES))
    abajo = np.zeros(len(config.SENSORES))
    colores = (["#8e44ad", "#2980b9", "#95a5a6"] if len(estratos) == 3
               else ["#8e44ad", "#95a5a6"])
    for i, e in enumerate(estratos):
        v = np.array([comps[s].iloc[i]["Percentage"] for s in config.SENSORES])
        ax.bar(x, v, 0.55, bottom=abajo, label=e, color=colores[i],
               edgecolor="black")
        abajo += v
    ax.set_xticks(x)
    ax.set_xticklabels(config.SENSORES)
    ax.set_ylabel("% of the random variance")
    comun.divisiones_y(ax, fmt)
    ax.set_title(comun.envolver_titulo(
        f"Variance split across the design strata{extra} — the larger the "
        "purple share, the more each viscosity observation costs", fmt),
        fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.savefig(ruta, dpi=fmt["dpi"])
    plt.close(fig)


def validar_diseno(reps, viscs, desbs, vels):
    """
    Check there are enough levels for the three error strata to exist. With
    fewer than 2 repetitions there is no Error(a) nor Error(b) and the
    split-plot analysis is impossible.
    """
    # The unbalance factor may have a single level: it then drops out of the
    # model and the design reduces to a two-stratum split-plot.
    problemas = []
    for nombre, niveles, minimo in (("repetitions (blocks)", reps, 2),
                                    ("viscosities", viscs, 2),
                                    ("unbalance levels", desbs, 1),
                                    ("speeds", vels, 2)):
        if len(niveles) < minimo:
            problemas.append(f"{len(niveles)} {nombre} (need >= {minimo})")
    if problemas:
        raise ValueError(
            "the design cannot support a split-plot ANOVA: " + "; ".join(problemas)
            + ". With fewer than 2 repetitions the Error(a) and Error(b) terms "
              "do not exist.")


def analizar(df, respuestas, unidad, destino: Path, fmt, sufijo: str = "",
             figuras: bool = True):
    """Run the full analysis and write tables and figures."""
    reps = sorted(df["Repetition"].unique())
    viscs = sorted(df["Viscosity"].unique())
    desbs = sorted(df["Unbalance"].unique())
    vels = sorted(df["Speed"].unique())
    validar_diseno(reps, viscs, desbs, vels)

    # With >= 2 unbalance levels the design is a split-split-plot; with a single
    # level the factor disappears and a two-stratum split-plot remains.
    completo = len(desbs) >= 2

    tablas, ingenuos, comps, resultados, posthoc = {}, {}, {}, {}, []
    for sensor in config.SENSORES:
        Y = _tensor(df, respuestas[sensor], reps, viscs, desbs, vels)
        if completo:
            tab = anova_split_split_plot(Y)
        else:
            Y = Y[:, :, 0, :]
            tab = anova_split_plot_bloques(Y)
        tablas[sensor] = tab
        ingenuos[sensor] = anova_ingenuo(tab)
        comps[sensor] = componentes_varianza(tab, Y.shape)
        ph, medias, ms_e, gl_e, n = tukey_viscosidad(Y, tab, viscs)
        ph.insert(0, "Sensor", sensor)
        posthoc.append(ph)
        resultados[sensor] = {"medias": medias, "ms_e": ms_e, "gl_e": gl_e,
                              "n": n, "tabla": tab,
                              "desv": _desviacion_por_viscosidad(Y)}

    destino.mkdir(parents=True, exist_ok=True)
    s = f"_{sufijo}" if sufijo else ""

    def _guardar(dic, nombre):
        out = pd.concat([d.assign(Sensor=k) for k, d in dic.items()],
                        ignore_index=True)
        out = out[["Sensor"] + [c for c in out.columns if c != "Sensor"]]
        out.to_csv(destino / nombre, index=False)
        return out

    _guardar(tablas, f"anova_split_plot{s}.csv")
    _guardar(ingenuos, f"anova_naive{s}.csv")
    _guardar(comps, f"variance_components{s}.csv")
    pd.concat(posthoc, ignore_index=True).to_csv(
        destino / f"posthoc_viscosity{s}.csv", index=False)

    # Every number behind p5_viscosity_effect.png, so the figure can be quoted
    # without recomputing it. The bar height is the plain mean over the
    # collapsed factors; the error bar is the 95 % CI of a whole-plot mean.
    filas_medias = []
    for sensor in config.SENSORES:
        rr = resultados[sensor]
        t_c = float(stats.t.ppf(0.975, rr["gl_e"]))
        semi = t_c * float(np.sqrt(rr["ms_e"] / rr["n"]))
        for k, v in enumerate(viscs):
            media = float(rr["medias"][k])
            filas_medias.append({
                "Sensor": sensor, "Viscosity": int(v), "Mean": media,
                "N_observations": int(rr["n"]),
                "MS_Error_a": float(rr["ms_e"]), "DF_Error_a": int(rr["gl_e"]),
                "t_crit_95": t_c, "CI95_half_width": semi,
                "CI95_low": media - semi, "CI95_high": media + semi,
                "SD": float(rr["desv"][k]), "SD_mode": MODO_ERROR,
            })
    pd.DataFrame(filas_medias).to_csv(
        destino / f"viscosity_means{s}.csv", index=False)

    # condensed viscosity comparison
    filas = []
    for sensor in config.SENSORES:
        c = tablas[sensor].set_index("Source").loc["Viscosity"]
        n_ = ingenuos[sensor].set_index("Source").loc["Viscosity"]
        filas.append({
            "Sensor": sensor,
            "F_correct": c["F"], "DF_den_correct": c["DF_denominator"],
            "p_correct": c["p"],
            "F_naive": n_["F"], "DF_den_naive": n_["DF_denominator"],
            "p_naive": n_["p"],
            "F_inflation_factor": n_["F"] / c["F"] if c["F"] else np.nan,
            "significant_correct": "yes" if c["p"] < 0.05 else "no",
            "significant_naive": "yes" if n_["p"] < 0.05 else "no",
        })
    comparacion = pd.DataFrame(filas)
    comparacion.to_csv(destino / f"viscosity_comparison{s}.csv", index=False)

    if figuras:
        extra = f" — {sufijo}" if sufijo else ""
        fig_contribucion(tablas, destino / f"p5_contribution{s}.png", fmt, extra)
        fig_comparacion(tablas, ingenuos,
                        destino / f"p5_naive_vs_correct{s}.png", fmt, extra)
        fig_efecto_viscosidad(resultados, viscs,
                              destino / f"p5_viscosity_effect{s}.png", fmt,
                              unidad, extra)
        fig_interaccion_velocidad(
            df, respuestas, viscs, desbs, vels,
            destino / f"p5_interaction_speed{s}.png", fmt, unidad, extra)
        # With a single unbalance level the curves would be one point each.
        if len(desbs) >= 2:
            fig_interaccion_desbalanceo(
                df, respuestas, viscs, desbs, vels,
                destino / f"p5_interaction_unbalance{s}.png", fmt, unidad, extra)
        fig_estratos(comps, destino / f"p5_variance_strata{s}.png", fmt, extra)

    medias = {sen: resultados[sen]["medias"] for sen in config.SENSORES}
    desvs = {sen: resultados[sen]["desv"] for sen in config.SENSORES}
    return (tablas, ingenuos, comparacion, comps,
            pd.concat(posthoc, ignore_index=True), medias, desvs)


def main() -> int:
    global PANELES_POR_SENSOR, EJE_Y_COMUN, MODO_ERROR, ESCALA_P, YMAX_P
    p = argparse.ArgumentParser(
        description="Step 5: split-plot ANOVA matched to the experimental design.")
    p.add_argument("--entrada", default="",
                   help="default: <salida>/p3_phasors_compensated.txt")
    p.add_argument("--salida", default=config.DIR_RESULTADOS)
    p.add_argument("--respuesta", default="amp", choices=["amp", "phase"],
                   help="analysed magnitude (default: amp)")
    p.add_argument("--formato", default=config.FORMATO_POR_DEFECTO,
                   choices=sorted(config.FORMATOS_FIGURA),
                   help="figure size/typography preset (default: screen)")
    p.add_argument("--paneles-por-sensor", dest="paneles", action="store_true",
                   help="bar figures as a grid with one panel per probe, "
                        "instead of the probes on shared axes")
    p.add_argument("--tamano-letra", dest="tam_letra", type=float, default=None,
                   help="base font size in points at final size; tick labels, "
                        "legend, annotations and line widths follow it, and the "
                        "figure grows taller to make room")
    p.add_argument("--tamano-titulo", dest="tam_titulo", type=float, default=None,
                   help="title font size, set independently of --tamano-letra")
    p.add_argument("--cojinete", default="todos", choices=["todos", "P1", "P2"],
                   help="probes to plot: both bearings (default), only P1 "
                        "(P1Y, P1X) or only P2")
    p.add_argument("--barras-error", dest="barras_error",
                   default="repeticiones", choices=["repeticiones", "total"],
                   help="what the error bars of the mean-response figures show: "
                        "'repeticiones' (default) +-1 SD between the repetition "
                        "means, i.e. repeatability; 'total' +-1 SD of every "
                        "observation behind the point, which also contains the "
                        "designed spread of the averaged factors")
    p.add_argument("--escala-p", dest="escala_p", default="log",
                   choices=["log", "lineal"],
                   help="how the p value is drawn in the partition figures "
                        "(p5_viscosity_by_*_evidence): 'log' (default) plots "
                        "-log10(p), taller = more evidence; 'lineal' plots the "
                        "raw probability on an ordinary 0..1 axis, shorter = "
                        "more evidence")
    p.add_argument("--ymax-p", dest="ymax_p", type=float, default=1.0,
                   help="upper bound of the linear p axis (default 1.0, the "
                        "full range of a probability). Use e.g. --ymax-p 0.1 "
                        "to zoom into the threshold; only affects "
                        "--escala-p lineal")
    p.add_argument("--eje-y-comun", dest="eje_y", action="store_true",
                   help="all panels of one figure share the same Y limits, so "
                        "the probes are comparable by bar or curve height "
                        "(default: each panel uses its own scale)")
    p.add_argument("--grupos-velocidad", dest="grupos", action="store_true",
                   help="repeat the analysis inside each speed band")
    p.add_argument("--grupos-desbalanceo", dest="grupos_desb", action="store_true",
                   help="repeat the analysis for each unbalance level separately "
                        "(the design reduces to a split-plot)")
    args = p.parse_args()

    PANELES_POR_SENSOR = args.paneles
    EJE_Y_COMUN = args.eje_y
    MODO_ERROR = args.barras_error
    ESCALA_P, YMAX_P = args.escala_p, args.ymax_p
    fmt = comun.aplicar_formato(args.formato, args.tam_letra, args.tam_titulo)
    # Every figure and table reads config.SENSORES, so restricting it here
    # propagates the bearing selection through the whole step.
    config.SENSORES = comun.seleccionar_sensores(args.cojinete)

    salida = Path(args.salida).expanduser()
    entrada = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.ARCHIVO_FASORES_COMP
    if not entrada.is_file():
        print(f"ERROR: {entrada} does not exist. Run p3_compensar_runout.py first.")
        return 1

    df = comun.leer_tabla(entrada)
    respuestas = {s: f"{s}_{args.respuesta}" for s in config.SENSORES}
    unidad = "1X amplitude [um]" if args.respuesta == "amp" else "phase [deg]"

    if args.respuesta == "phase":
        print("!! WARNING: phase is a CIRCULAR variable. An ordinary ANOVA on\n"
              "   degrees is not valid in general (359 and 1 are 2 degrees\n"
              "   apart, not 358). Use it only after checking that no condition\n"
              "   crosses 0/360.\n")

    destino = salida / config.DIR_ESTADISTICA
    try:
        tablas, ingenuos, comparacion, comps, posthoc, _, _ = analizar(
            df, respuestas, unidad, destino, fmt)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    r = len(df["Repetition"].unique()); a = len(df["Viscosity"].unique())
    b = len(df["Unbalance"].unique()); c = len(df["Speed"].unique())
    print(f"Input  : {entrada}  ({len(df)} rows)")
    print(f"Design : {r} blocks x {a} viscosities x {b} unbalance levels x "
          f"{c} speeds = {r*a*b*c} observations")
    print(f"Figures: format '{args.formato}' ({fmt['ancho']:g} in wide, "
          f"{fmt['dpi']} dpi), base font {fmt['base']:g} pt, "
          f"title {fmt['titulo']:g} pt")
    if fmt.get("escala_texto", 1.0) > 1.01:
        efectivo = fmt["ancho"] * fmt["escala_texto"] ** 0.85
        print(f"         NOTE: the larger font makes the figures ~{efectivo:.1f} in "
              f"wide instead of {fmt['ancho']:g} in. Placed in a "
              f"{fmt['ancho']:g} in column they scale to "
              f"{fmt['base'] * fmt['ancho'] / efectivo:.1f} pt effective type "
              f"(vs {config.FORMATOS_FIGURA[args.formato]['base']:g} pt at "
              f"--tamano-letra default).")
    print(f"         layout "
          f"{'one panel per probe' if args.paneles else 'probes on shared axes'}"
          f", Y axis {'shared across panels' if args.eje_y else 'per panel'}"
          f", probes: {', '.join(config.SENSORES)}")
    print(f"         error bars: {_TEXTO_ERROR(len(df['Repetition'].unique()))}")
    print(f"         p axis in the partition figures: "
          + ("-log10(p), taller = more evidence" if args.escala_p == "log"
             else f"linear p from 0 to {args.ymax_p:g}, shorter = more evidence")
          + "\n")

    print("=" * 78)
    print(f"SPLIT-PLOT ANOVA — sensor {config.SENSORES[0]} (example)")
    print("=" * 78)
    t = tablas[config.SENSORES[0]]
    print(t[["Source", "SS", "DF", "MS", "DF_denominator", "F", "p",
             "Contribution_SS_%"]]
          .to_string(index=False, float_format=lambda v: f"{v:10.4g}"))

    print("\n" + "=" * 78)
    print("VISCOSITY EFFECT: correct test vs naive test")
    print("=" * 78)
    print(comparacion.to_string(index=False, float_format=lambda v: f"{v:9.4g}"))

    n_c = (comparacion["significant_correct"] == "yes").sum()
    n_i = (comparacion["significant_naive"] == "yes").sum()
    n_s = len(config.SENSORES)
    print(f"\nSensors with a significant viscosity effect: {n_c}/{n_s} with the "
          f"correct error term, {n_i}/{n_s} with the naive ANOVA.")
    infl = comparacion["F_inflation_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(infl):
        print(f"The naive ANOVA multiplies the viscosity F by a factor of "
              f"{infl.min():.1f} to {infl.max():.1f}.")
    print(f"Viscosity is judged with "
          f"{int(comparacion['DF_den_correct'].iloc[0])} denominator df, "
          f"not {int(comparacion['DF_den_naive'].iloc[0])}.")

    print("\n" + "=" * 78)
    print("PAIRWISE COMPARISONS BETWEEN OILS (Tukey with Error(a))")
    print("=" * 78)
    print(posthoc.to_string(index=False, float_format=lambda v: f"{v:9.4g}"))

    # ---- optional: per speed band ----
    if args.grupos:
        print("\n" + "=" * 78)
        print("ANALYSIS BY SPEED BAND")
        print("=" * 78)
        print(config.leyenda_grupos() + "\n")
        resumen = []
        for nombre, vels in GRUPOS_VELOCIDAD.items():
            sub = df[df["Speed"].isin(vels)]
            disponibles = sorted(sub["Speed"].unique())
            if len(disponibles) < 2:
                print(f"  {nombre}: fewer than 2 speeds available, skipped.")
                continue
            try:
                _, _, cmp_g, _, _, medias_g, desv_g = analizar(
                    sub, respuestas, unidad, destino, fmt, nombre)
            except ValueError as e:
                print(f"  {nombre}: {e}")
                continue
            sig = (cmp_g["significant_correct"] == "yes").sum()
            print(f"  {nombre} ({len(disponibles)} rpm): viscosity significant "
                  f"in {sig}/{len(config.SENSORES)} sensors  "
                  f"(min p = {cmp_g['p_correct'].min():.4g})")
            resumen.append((nombre, medias_g, desv_g,
                            dict(zip(cmp_g["Sensor"], cmp_g["p_correct"]))))

        if len(resumen) >= 2:
            viscs = sorted(df["Viscosity"].unique())
            fig_grupo_medias(resumen, viscs,
                             destino / "p5_viscosity_by_speed_band_means.png",
                             fmt, unidad, que="speed band", n_rep=r)
            fig_grupo_evidencia(
                resumen, destino / "p5_viscosity_by_speed_band_evidence.png",
                fmt, que="speed band")
            print("\n  Cross-band figures: "
                  "p5_viscosity_by_speed_band_means.png, "
                  "p5_viscosity_by_speed_band_evidence.png")

    # ---- optional: per unbalance level ----
    if args.grupos_desb:
        print("\n" + "=" * 78)
        print("ANALYSIS BY UNBALANCE LEVEL")
        print("=" * 78)
        print("Fixing the unbalance removes that factor from the model: the\n"
              "design goes from split-split-plot to split-plot (block ->\n"
              "viscosity -> speed). Viscosity is still tested against\n"
              "Error(a) = Rep x Visc with 4 df: splitting by unbalance adds no\n"
              "degrees of freedom for the oil.\n")
        resumen_d = []
        for d in sorted(df["Unbalance"].unique()):
            sub = df[df["Unbalance"] == d]
            nombre = f"U{int(d)}"
            try:
                _, _, cmp_d, _, _, medias_d, desv_d = analizar(
                    sub, respuestas, unidad, destino, fmt, nombre)
            except ValueError as e:
                print(f"  {nombre}: {e}")
                continue
            sig = (cmp_d["significant_correct"] == "yes").sum()
            print(f"  {nombre}: viscosity significant in "
                  f"{sig}/{len(config.SENSORES)} sensors  "
                  f"(min p = {cmp_d['p_correct'].min():.4g})")
            resumen_d.append((nombre, medias_d, desv_d,
                              dict(zip(cmp_d["Sensor"], cmp_d["p_correct"]))))

        if len(resumen_d) >= 2:
            viscs = sorted(df["Viscosity"].unique())
            fig_grupo_medias(resumen_d, viscs,
                             destino / "p5_viscosity_by_unbalance_means.png",
                             fmt, unidad, que="unbalance level", n_rep=r)
            fig_grupo_evidencia(
                resumen_d, destino / "p5_viscosity_by_unbalance_evidence.png",
                fmt, que="unbalance level")
            print("\n  Cross-level figures: "
                  "p5_viscosity_by_unbalance_means.png, "
                  "p5_viscosity_by_unbalance_evidence.png")

    # ---- what ran and what did not ----
    print("\n" + "=" * 78)
    print("OPTIONAL ANALYSES")
    print("=" * 78)
    for activo, etiqueta, bandera in (
            (args.grupos, "by speed band", "--grupos-velocidad"),
            (args.grupos_desb, "by unbalance level", "--grupos-desbalanceo")):
        if activo:
            print(f"  [x] {etiqueta:22s} done")
        else:
            print(f"  [ ] {etiqueta:22s} NOT done — add {bandera}")

    n = len(list(destino.glob("*")))
    print(f"\n{n} files in: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
