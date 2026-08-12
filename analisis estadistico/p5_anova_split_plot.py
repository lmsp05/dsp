"""
PASO 5 — Analisis estadistico acorde al diseno del ensayo
=========================================================

EL PROBLEMA
-----------
El ensayo NO es un experimento completamente aleatorizado. Es un diseno de
PARCELAS SUBDIVIDIDAS (split-split-plot) en bloques:

    bloque          -> repeticion   (r = 3)
    parcela grande  -> viscosidad   (a = 3)   se cambia el aceite: dificil
    subparcela      -> desbalanceo  (b = 3)   se cambia la masa dentro del aceite
    sub-subparcela  -> velocidad    (c = 15)  barrido dentro de cada montaje

Cada nivel de aleatorizacion tiene su PROPIO termino de error. Contrastar la
viscosidad contra el residual global (como hace un ANOVA factorial corriente)
la trata como si se hubiese re-montado el rotor 405 veces de forma
independiente, cuando en realidad solo hay 9 unidades de parcela grande
(3 bloques x 3 aceites). El resultado es un denominador demasiado pequeno,
una F inflada y un valor p irreal.

    Efecto           se contrasta contra          gl del denominador
    ---------------------------------------------------------------
    Viscosidad       Error(a) = Rep x Visc                4
    Desbalanceo      Error(b) = Rep x Desb dentro de Visc  12
    Visc x Desb      Error(b)                             12
    Velocidad        Error(c) = residual                 252
    Visc x Vel       Error(c)                            252
    Desb x Vel       Error(c)                            252
    Visc x Desb x Vel Error(c)                           252

Es decir: la viscosidad se juzga con 4 grados de libertad, no con 252. Este
script calcula la descomposicion completa, aplica cada F con su denominador
correcto y, para que la diferencia sea visible, calcula TAMBIEN el ANOVA
ingenuo (el que se venia usando) y los pone lado a lado.

SALIDAS (en <salida>/p5_estadistica)
------------------------------------
  anova_split_plot.csv         tabla completa por sensor, con F y p correctos
  anova_ingenuo.csv            el ANOVA factorial corriente, para comparar
  comparacion_viscosidad.csv   resumen del contraste correcto vs ingenuo
  posthoc_viscosidad.csv       Tukey entre aceites con el error de parcela grande
  componentes_varianza.csv     varianza atribuible a cada estrato
  p5_*.png                     figuras que explican el efecto y el diseno
  (opcional --grupos-velocidad) anova_split_plot_<grupo>.csv por tramo de rpm

Uso:
    python p5_anova_split_plot.py --salida <carpeta_resultados>
    python p5_anova_split_plot.py --salida <res> --entrada <res>/p1_fasores.txt
    python p5_anova_split_plot.py --salida <res> --grupos-velocidad
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


# Tramos de velocidad para el analisis opcional por grupos (como f3_s1_v2.py).
GRUPOS_VELOCIDAD = {
    "Grupo_1_bajas":     [600, 1300, 2000],
    "Grupo_2_pre_1a":    [2500, 2600, 2700, 2800],
    "Grupo_3_1a_critica": [3400, 3500, 3600, 3700],
    "Grupo_4_altas":     [4000, 4500, 5000, 5500],
}

ETIQUETAS = {
    "Repeticion (bloque)": "Rep",
    "Viscosidad": "Visc",
    "Error(a) = Rep x Visc": "Error(a)",
    "Desbalanceo": "Desb",
    "Viscosidad x Desbalanceo": "Visc×Desb",
    "Error(b) = Rep x Desb | Visc": "Error(b)",
    "Velocidad": "Vel",
    "Viscosidad x Velocidad": "Visc×Vel",
    "Desbalanceo x Velocidad": "Desb×Vel",
    "Viscosidad x Desbalanceo x Velocidad": "Visc×Desb×Vel",
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
    for rep, vi, de, ve, y in zip(df["Repeticion"], df["Viscosidad"],
                                  df["Desbalanceo"], df["Velocidad"],
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
        ("Repeticion (bloque)",                    SS_R,   r - 1,                 "Error(a)"),
        ("Viscosidad",                             SS_A,   a - 1,                 "Error(a)"),
        ("Error(a) = Rep x Visc",                  SS_Ea,  (r - 1) * (a - 1),     ""),
        ("Desbalanceo",                            SS_B,   b - 1,                 "Error(b)"),
        ("Viscosidad x Desbalanceo",               SS_AB,  (a - 1) * (b - 1),     "Error(b)"),
        ("Error(b) = Rep x Desb | Visc",           SS_Eb,  a * (r - 1) * (b - 1), ""),
        ("Velocidad",                              SS_C,   c - 1,                 "Error(c)"),
        ("Viscosidad x Velocidad",                 SS_AC,  (a - 1) * (c - 1),     "Error(c)"),
        ("Desbalanceo x Velocidad",                SS_BC,  (b - 1) * (c - 1),     "Error(c)"),
        ("Viscosidad x Desbalanceo x Velocidad",   SS_ABC, (a - 1) * (b - 1) * (c - 1), "Error(c)"),
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
        ("Repeticion (bloque)",      SS_R,  r - 1,                 "Error(a)"),
        ("Viscosidad",               SS_A,  a - 1,                 "Error(a)"),
        ("Error(a) = Rep x Visc",    SS_Ea, (r - 1) * (a - 1),     ""),
        ("Velocidad",                SS_C,  c - 1,                 "Error(b)"),
        ("Viscosidad x Velocidad",   SS_AC, (a - 1) * (c - 1),     "Error(b)"),
        ("Error(b) = residual",      SS_Eb, a * (r - 1) * (c - 1), ""),
    ]
    gl_total = sum(f[2] for f in filas)
    assert gl_total == Y.size - 1, f"gl inconsistentes: {gl_total} != {Y.size - 1}"
    return _completar_tabla(filas, SS_tot)


def _completar_tabla(filas, SS_tot):
    """Anade MS, F, p, contribucion y omega2 a una lista de (fuente, SS, gl, error)."""
    tab = pd.DataFrame(filas, columns=["Fuente", "SS", "GL", "Error"])
    tab["MS"] = tab["SS"] / tab["GL"]
    errores = {f.split(" =")[0].strip(): (ms, gl) for f, ms, gl, e
               in zip(tab["Fuente"], tab["MS"], tab["GL"], tab["Error"]) if not e}

    F, P, GLD, OM = [], [], [], []
    for _, fila in tab.iterrows():
        e = fila["Error"]
        if not e or e not in errores or errores[e][0] <= 0:
            F.append(np.nan), P.append(np.nan), GLD.append(np.nan), OM.append(np.nan)
            continue
        ms_e, gl_e = errores[e]
        f = fila["MS"] / ms_e
        F.append(f)
        P.append(float(stats.f.sf(f, fila["GL"], gl_e)))
        GLD.append(gl_e)
        OM.append(100.0 * max(fila["SS"] - fila["GL"] * ms_e, 0.0) / (SS_tot + ms_e))
    tab["GL_denominador"], tab["F"], tab["p"] = GLD, F, P
    tab["Contribucion_SS_%"] = 100.0 * tab["SS"] / SS_tot
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
    es_bloque = tab["Fuente"] == "Repeticion (bloque)"
    agrupadas = es_error | es_bloque

    ss_res = float(tab.loc[agrupadas, "SS"].sum())
    gl_res = int(tab.loc[agrupadas, "GL"].sum())
    ms_res = ss_res / gl_res

    filas = []
    for _, fila in tab.loc[~agrupadas].iterrows():
        F = fila["MS"] / ms_res
        filas.append({"Fuente": fila["Fuente"], "SS": fila["SS"], "GL": fila["GL"],
                      "MS": fila["MS"], "GL_denominador": gl_res, "F": F,
                      "p": float(stats.f.sf(F, fila["GL"], gl_res))})
    filas.append({"Fuente": "Residual (agrupado)", "SS": ss_res, "GL": gl_res,
                  "MS": ms_res, "GL_denominador": np.nan, "F": np.nan, "p": np.nan})
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
    idx = {f: i for i, f in enumerate(tab["Fuente"])}
    if len(forma) == 4:
        r, a, b, c = forma
        estratos = [("Error(a) = Rep x Visc", "Parcela grande (montaje del aceite)", b * c),
                    ("Error(b) = Rep x Desb | Visc", "Subparcela (montaje del desbalanceo)", c),
                    ("Error(c)", "Sub-subparcela (barrido de velocidad)", 1)]
    else:
        r, a, c = forma
        estratos = [("Error(a) = Rep x Visc", "Parcela grande (montaje del aceite)", c),
                    ("Error(b) = residual", "Subparcela (barrido de velocidad)", 1)]

    ms = [tab.loc[idx[f], "MS"] for f, _, _ in estratos]
    var = []
    for k in range(len(estratos)):
        if k < len(estratos) - 1:
            var.append(max((ms[k] - ms[k + 1]) / estratos[k][2], 0.0))
        else:
            var.append(max(ms[k], 0.0))
    tot = sum(var) or 1.0
    return pd.DataFrame([
        {"Estrato": nombre, "MS": ms[k], "Varianza": var[k],
         "Porcentaje": 100 * var[k] / tot}
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
    idx = {f: i for i, f in enumerate(tab["Fuente"])}
    ms_e = tab.loc[idx["Error(a) = Rep x Visc"], "MS"]
    gl_e = tab.loc[idx["Error(a) = Rep x Visc"], "GL"]
    q = float(stats.studentized_range.ppf(1 - alfa, a, gl_e))
    hsd = q * np.sqrt(ms_e / n)

    filas = []
    for i in range(a):
        for j in range(i + 1, a):
            d = float(medias[i] - medias[j])
            t = abs(d) / np.sqrt(2 * ms_e / n)
            p = float(stats.studentized_range.sf(t * np.sqrt(2), a, gl_e))
            filas.append({
                "Comparacion": f"ISO {viscs[i]} - ISO {viscs[j]}",
                "Diferencia": d, "HSD_95": hsd,
                "IC_inf": d - hsd, "IC_sup": d + hsd,
                "p_Tukey": p, "Significativa": "SI" if abs(d) > hsd else "no",
            })
    return pd.DataFrame(filas), medias, ms_e, gl_e, n


# ============================================================
# FIGURAS
# ============================================================

def _fmt_p(p):
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def barras_por_sensor(ax, categorias, valores, rayado=None, ancho_grupo=0.82):
    """
    Barras agrupadas con los 4 sensores en unos MISMOS ejes.

    Para cada categoria del eje X se dibuja una barra por sensor, de modo que
    los 4 proximitores se comparan directamente en la misma figura en vez de
    repartirse en un panel cada uno.

      categorias : etiquetas del eje X
      valores    : {sensor: [valor por categoria]}
      rayado     : {sensor: [bool]} -> las marcadas se dibujan con trama, que se
                   usa para senalar "no significativo"

    Devuelve las posiciones de los grupos.
    """
    sensores = [s for s in config.SENSORES if s in valores]
    n = len(sensores)
    x = np.arange(len(categorias), dtype=float)
    ancho = ancho_grupo / max(n, 1)

    for k, sensor in enumerate(sensores):
        desplazamiento = (k - (n - 1) / 2) * ancho
        tramas = (rayado or {}).get(sensor, [False] * len(categorias))
        for i, (v, t) in enumerate(zip(valores[sensor], tramas)):
            ax.bar(x[i] + desplazamiento, v, ancho * 0.92,
                   color=config.COLOR_SENSOR.get(sensor, "#888888"),
                   edgecolor="black", linewidth=0.4,
                   hatch="///" if t else None, zorder=2)

    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(categorias) - 0.5)
    ax.grid(axis="y", ls="--", alpha=0.4)
    return x


def leyenda_sensores(fig, extra_handles=(), ncol=None, y=-0.02):
    """Leyenda comun de los 4 sensores, fuera de los ejes."""
    handles = [Patch(facecolor=config.COLOR_SENSOR[s], edgecolor="black",
                     linewidth=0.4, label=s) for s in config.SENSORES]
    handles += list(extra_handles)
    fig.legend(handles=handles, loc="lower center",
               ncol=ncol or len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, y))


def fig_contribucion(tablas, ruta, extra=""):
    """Contribucion de cada fuente a la SS, con los 4 sensores en unos ejes."""
    # Las fuentes se leen de la propia tabla: asi la figura vale igual para el
    # diseno completo (11 fuentes) y para el reducido de un solo desbalanceo (6).
    fuentes = [f for f in tablas[config.SENSORES[0]]["Fuente"]
               if f != "Repeticion (bloque)"]
    etiquetas = [ETIQUETAS.get(f, f) for f in fuentes]

    valores, rayado = {}, {}
    for sensor in config.SENSORES:
        t = tablas[sensor].set_index("Fuente")
        valores[sensor] = [float(t.loc[f, "Contribucion_SS_%"]) for f in fuentes]
        # Trama en las barras que NO son significativas (y en los estratos de
        # error, que no se contrastan contra nada).
        rayado[sensor] = [not (np.isfinite(t.loc[f, "p"]) and t.loc[f, "p"] < 0.05)
                          for f in fuentes]

    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    barras_por_sensor(ax, etiquetas, valores, rayado)
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Contribucion a la SS total (%)", fontsize=11)

    diseno = "split-split-plot" if "Error(c)" in fuentes else "split-plot"
    fig.suptitle(f"Descomposicion de la variabilidad — {diseno}{extra}\n"
                 "los 4 proximitores superpuestos · barra rayada = no "
                 "significativa al 5 % con su termino de error",
                 fontsize=14, fontweight="bold")
    leyenda_sensores(fig, [Patch(facecolor="white", edgecolor="black",
                                 hatch="///", label="no significativo")])
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def fig_comparacion(tablas, ingenuos, ruta, extra=""):
    """Lo que cambia al respetar el diseno: F y p de cada efecto, ambos modelos."""
    fuentes = [f for f in ingenuos[config.SENSORES[0]]["Fuente"]
               if f != "Residual (agrupado)"]
    et = [ETIQUETAS.get(f, f) for f in fuentes]
    x = np.arange(len(fuentes))

    # Un panel por MODELO (no por sensor): en cada uno los 4 proximitores
    # aparecen juntos, que es lo que permite compararlos de un vistazo.
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), constrained_layout=True,
                             sharey=True)
    for ax, (titulo, fuente) in zip(
            axes, [("ANOVA ingenuo — todo contra el residual", ingenuos),
                   ("split-plot — cada efecto con SU error", tablas)]):
        valores = {}
        for sensor in config.SENSORES:
            t = fuente[sensor].set_index("Fuente")
            valores[sensor] = [-np.log10(max(float(t.loc[f, "p"]), 1e-300))
                               for f in fuentes]
        barras_por_sensor(ax, et, valores)
        ax.axhline(-np.log10(0.05), color="black", ls="--", lw=1.4)
        ax.text(-0.45, -np.log10(0.05), "p = 0.05", va="bottom", fontsize=9)
        # Escala symlog: la zona decisiva (p entre 1 y 0.001) queda lineal y
        # legible, y los p astronomicos del modelo ingenuo no la aplastan.
        ax.set_yscale("symlog", linthresh=3.0, linscale=1.4)
        ax.set_ylim(0, 400)
        ax.set_yticks([0, 1, 2, 3, 10, 100])
        ax.set_yticklabels(["0", "1", "2", "3", "10", "100"])
        ax.set_title(titulo, fontweight="bold")
        ax.set_xticklabels(et, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("evidencia   -log10(p)   [escala symlog]")

    fig.suptitle(f"Consecuencia de respetar el diseno de parcelas subdivididas{extra}\n"
                 "el panel derecho es la evidencia REAL; el izquierdo, la que se "
                 "obtiene si se ignora la estructura del ensayo",
                 fontsize=14, fontweight="bold")
    leyenda_sensores(fig)
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def fig_efecto_viscosidad(resultados, viscs, ruta, unidad, extra=""):
    """
    Medias por aceite con el intervalo del error de parcela, los 4 sensores en
    unos mismos ejes: X = sensor, una barra por aceite dentro de cada sensor.
    """
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    n_v = len(viscs)
    ancho = 0.80 / n_v
    x = np.arange(len(config.SENSORES), dtype=float)

    for k, v in enumerate(viscs):
        alturas, errores = [], []
        for sensor in config.SENSORES:
            medias, ms_e, gl_e, n = (resultados[sensor][c] for c in
                                     ("medias", "ms_e", "gl_e", "n"))
            alturas.append(float(medias[k]))
            errores.append(float(stats.t.ppf(0.975, gl_e)) * np.sqrt(ms_e / n))
        ax.bar(x + (k - (n_v - 1) / 2) * ancho, alturas, ancho * 0.9,
               yerr=errores, capsize=4, color=comun.color_condicion(v, 3),
               edgecolor="black", linewidth=0.5, label=f"ISO {v}", zorder=2)

    etiquetas = [f"{s}\n(p = {_fmt_p(resultados[s]['tabla'].set_index('Fuente').loc['Viscosidad', 'p'])})"
                 for s in config.SENSORES]
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=11)
    ax.set_ylabel(unidad, fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=10, title="aceite")
    fig.suptitle(f"Efecto de la viscosidad{extra} — media por aceite en cada "
                 f"proximitor,\ncon IC 95 % basado en Error(a)",
                 fontsize=13, fontweight="bold")
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def fig_interacciones(df, respuestas, viscs, desbs, vels, ruta, unidad, extra=""):
    """
    Interaccion viscosidad x velocidad (arriba) y viscosidad x desbalanceo
    (abajo), con los 4 proximitores superpuestos en unos mismos ejes.

    Cada sensor se NORMALIZA por su propia media antes de superponerlo: P1Y y
    P2Y difieren en un factor ~6 en micrometros, asi que sin normalizar el
    sensor pequeno quedaria aplastado contra el eje y no se veria su forma. Al
    normalizar, lo que se compara es la FORMA de la respuesta -- que es de lo
    que habla una interaccion -- y no su magnitud absoluta (esa esta, en
    micrometros, en p5_efecto_viscosidad.png).

    Color = viscosidad · estilo de linea = sensor.
    """
    x = np.arange(len(vels))
    # Con un solo desbalanceo la fila viscosidad x desbalanceo seria un unico
    # punto por curva: no se dibuja.
    dos_filas = len(desbs) >= 2
    estilos = dict(zip(config.SENSORES, ["-", "--", "-.", ":"]))

    fig, axes = plt.subplots(2 if dos_filas else 1, 1,
                             figsize=(15, 9 if dos_filas else 5),
                             constrained_layout=True, squeeze=False)

    for sensor in config.SENSORES:
        col = respuestas[sensor]
        escala = float(df[col].mean()) or 1.0        # media global del sensor
        for v in viscs:
            sub = df[df["Viscosidad"] == v]
            m = sub.groupby("Velocidad")[col].mean().reindex(vels).to_numpy()
            axes[0, 0].plot(x, 100.0 * m / escala, lw=1.7,
                            color=comun.color_condicion(v, 3),
                            linestyle=estilos[sensor])
            if dos_filas:
                m2 = sub.groupby("Desbalanceo")[col].mean().reindex(desbs).to_numpy()
                axes[1, 0].plot(range(len(desbs)), 100.0 * m2 / escala, "o-",
                                lw=1.7, color=comun.color_condicion(v, 3),
                                linestyle=estilos[sensor])

    ax = axes[0, 0]
    paso = 1 if len(vels) <= 8 else 1
    ax.set_xticks(x[::paso])
    ax.set_xticklabels([str(v) for v in vels[::paso]], rotation=60, fontsize=8)
    ax.set_xlabel("Velocidad [rpm]", fontsize=10)
    ax.set_ylabel("media normalizada (% de la media del sensor)", fontsize=10)
    ax.set_title("Viscosidad x velocidad", fontweight="bold")
    ax.grid(alpha=0.3, ls="--")

    if dos_filas:
        ax = axes[1, 0]
        ax.set_xticks(range(len(desbs)))
        ax.set_xticklabels([f"desbalanceo {d}" for d in desbs], fontsize=10)
        ax.set_ylabel("media normalizada (% de la media del sensor)", fontsize=10)
        ax.set_title("Viscosidad x desbalanceo", fontweight="bold")
        ax.grid(alpha=0.3, ls="--")

    n_rep = len(df["Repeticion"].unique())
    handles = [Line2D([0], [0], color=comun.color_condicion(v, 3), lw=2.4,
                      label=f"ISO {v}") for v in viscs]
    handles += [Line2D([0], [0], color="#444444", lw=1.7, linestyle=estilos[s],
                       label=s) for s in config.SENSORES]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    titulo = (f"Interacciones{extra} — los 4 proximitores superpuestos\n"
              f"arriba: cada punto promedia {n_rep} rep x {len(desbs)} "
              f"desbalanceo(s) = {n_rep * len(desbs)} valores")
    if dos_filas:
        titulo += (f"   |   abajo: {n_rep} rep x {len(vels)} velocidades = "
                   f"{n_rep * len(vels)} valores")
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def fig_viscosidad_por_grupo(resumen, viscs, ruta, unidad, que="tramo de velocidad"):
    """
    Como cambia el efecto de la viscosidad al recorrer los niveles de `que`.

    `resumen` = lista de (nombre, {sensor: medias_por_viscosidad},
                          {sensor: p_viscosidad}).

    Arriba: media por aceite en cada nivel, con los 4 sensores superpuestos y
    normalizados a su propia media (ver la nota de `fig_interacciones`).
    Abajo: evidencia del efecto, barras agrupadas por sensor.
    """
    nombres = [n.replace("_", " ") for n, _, _ in resumen]
    x = np.arange(len(resumen), dtype=float)
    estilos = dict(zip(config.SENSORES, ["-", "--", "-.", ":"]))

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)

    ax = axes[0]
    for sensor in config.SENSORES:
        todas = [m[sensor][iv] for _, m, _ in resumen for iv in range(len(viscs))]
        escala = float(np.mean(todas)) or 1.0
        for iv, v in enumerate(viscs):
            ax.plot(x, [100.0 * m[sensor][iv] / escala for _, m, _ in resumen],
                    "o-", lw=1.8, ms=4, color=comun.color_condicion(v, 3),
                    linestyle=estilos[sensor])
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=10)
    ax.set_ylabel("media normalizada (% de la media del sensor)", fontsize=10)
    ax.set_title("Media por aceite en cada nivel", fontweight="bold")
    ax.grid(alpha=0.3, ls="--")

    ax = axes[1]
    valores, rayado = {}, {}
    for sensor in config.SENSORES:
        ps = [max(float(pp[sensor]), 1e-300) for _, _, pp in resumen]
        valores[sensor] = [-np.log10(q) for q in ps]
        rayado[sensor] = [q >= 0.05 for q in ps]
    barras_por_sensor(ax, nombres, valores, rayado)
    ax.axhline(-np.log10(0.05), color="black", ls="--", lw=1.4)
    ax.text(-0.45, -np.log10(0.05), "p = 0.05", va="bottom", fontsize=9)
    ax.set_xticklabels(nombres, fontsize=10)
    ax.set_ylabel("evidencia  -log10(p)", fontsize=10)
    ax.set_title("Evidencia del efecto de la viscosidad "
                 "(barra rayada = no significativa)", fontweight="bold")

    handles = [Line2D([0], [0], color=comun.color_condicion(v, 3), lw=2.4,
                      label=f"ISO {v}") for v in viscs]
    handles += [Patch(facecolor=config.COLOR_SENSOR[s], edgecolor="black",
                      linewidth=0.4, label=s) for s in config.SENSORES]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    n_sig = sum(1 for _, _, pp in resumen for s in config.SENSORES
                if pp[s] < 0.05)
    n_tot = len(resumen) * len(config.SENSORES)
    fig.suptitle(f"Efecto de la viscosidad segun el {que}\n"
                 f"significativa en {n_sig} de {n_tot} casos "
                 f"(sensor x nivel) · los 4 proximitores superpuestos",
                 fontsize=14, fontweight="bold")
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def fig_estratos(comps, ruta, extra=""):
    """De donde viene la variabilidad: montaje del aceite / del desbalanceo / ruido."""
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    estratos = comps[config.SENSORES[0]]["Estrato"].tolist()
    x = np.arange(len(config.SENSORES))
    abajo = np.zeros(len(config.SENSORES))
    colores = (["#8e44ad", "#2980b9", "#95a5a6"] if len(estratos) == 3
               else ["#8e44ad", "#95a5a6"])
    for i, e in enumerate(estratos):
        v = np.array([comps[s].iloc[i]["Porcentaje"] for s in config.SENSORES])
        ax.bar(x, v, 0.55, bottom=abajo, label=e, color=colores[i],
               edgecolor="black", linewidth=0.4)
        abajo += v
    ax.set_xticks(x)
    ax.set_xticklabels(config.SENSORES)
    ax.set_ylabel("% de la varianza aleatoria")
    ax.set_title(f"Reparto de la variabilidad entre los estratos del diseno{extra}\n"
                 "(cuanto mayor es el morado, mas caro es cada dato de viscosidad)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROGRAMA
# ============================================================

def validar_diseno(reps, viscs, desbs, vels):
    """
    Comprueba que hay niveles suficientes para que existan los tres estratos de
    error. Sin al menos 2 repeticiones no hay Error(a) ni Error(b) y el analisis
    de parcelas subdivididas es imposible.
    """
    # El desbalanceo puede tener un solo nivel: en ese caso sale del modelo y el
    # diseno se reduce a un split-plot de dos estratos (ver anova_split_plot_bloques).
    problemas = []
    for nombre, niveles, minimo in (("repeticiones (bloques)", reps, 2),
                                    ("viscosidades", viscs, 2),
                                    ("desbalanceos", desbs, 1),
                                    ("velocidades", vels, 2)):
        if len(niveles) < minimo:
            problemas.append(f"{len(niveles)} {nombre} (hacen falta >= {minimo})")
    if problemas:
        raise ValueError(
            "el diseno no da para un ANOVA de parcelas subdivididas: "
            + "; ".join(problemas)
            + ". Con menos de 2 repeticiones no existen los terminos de error "
              "Error(a) ni Error(b).")


def analizar(df, respuestas, unidad, destino: Path, sufijo: str = "",
             figuras: bool = True):
    """Ejecuta el analisis completo y escribe tablas y figuras."""
    reps = sorted(df["Repeticion"].unique())
    viscs = sorted(df["Viscosidad"].unique())
    desbs = sorted(df["Desbalanceo"].unique())
    vels = sorted(df["Velocidad"].unique())
    validar_diseno(reps, viscs, desbs, vels)

    # Con >= 2 desbalanceos el diseno es split-split-plot; con uno solo el factor
    # desaparece y queda un split-plot de dos estratos.
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
                              "n": n, "tabla": tab}

    destino.mkdir(parents=True, exist_ok=True)
    s = f"_{sufijo}" if sufijo else ""

    def _guardar(dic, nombre):
        out = pd.concat([d.assign(Sensor=k) for k, d in dic.items()], ignore_index=True)
        out = out[["Sensor"] + [c for c in out.columns if c != "Sensor"]]
        out.to_csv(destino / nombre, index=False)
        return out

    _guardar(tablas, f"anova_split_plot{s}.csv")
    _guardar(ingenuos, f"anova_ingenuo{s}.csv")
    _guardar(comps, f"componentes_varianza{s}.csv")
    pd.concat(posthoc, ignore_index=True).to_csv(
        destino / f"posthoc_viscosidad{s}.csv", index=False)

    # comparacion resumida para la viscosidad
    filas = []
    for sensor in config.SENSORES:
        c = tablas[sensor].set_index("Fuente").loc["Viscosidad"]
        n_ = ingenuos[sensor].set_index("Fuente").loc["Viscosidad"]
        filas.append({
            "Sensor": sensor,
            "F_correcto": c["F"], "gl_den_correcto": c["GL_denominador"],
            "p_correcto": c["p"],
            "F_ingenuo": n_["F"], "gl_den_ingenuo": n_["GL_denominador"],
            "p_ingenuo": n_["p"],
            "veces_F_inflada": n_["F"] / c["F"] if c["F"] else np.nan,
            "significativa_correcto": "SI" if c["p"] < 0.05 else "no",
            "significativa_ingenuo": "SI" if n_["p"] < 0.05 else "no",
        })
    comparacion = pd.DataFrame(filas)
    comparacion.to_csv(destino / f"comparacion_viscosidad{s}.csv", index=False)

    if figuras:
        extra = f" — {sufijo.replace('_', ' ')}" if sufijo else ""
        fig_contribucion(tablas, destino / f"p5_contribucion{s}.png", extra)
        fig_comparacion(tablas, ingenuos,
                        destino / f"p5_ingenuo_vs_correcto{s}.png", extra)
        fig_efecto_viscosidad(resultados, viscs,
                              destino / f"p5_efecto_viscosidad{s}.png", unidad, extra)
        fig_interacciones(df, respuestas, viscs, desbs, vels,
                          destino / f"p5_interacciones{s}.png", unidad, extra)
        fig_estratos(comps, destino / f"p5_estratos_varianza{s}.png", extra)

    medias = {sen: resultados[sen]["medias"] for sen in config.SENSORES}
    return (tablas, ingenuos, comparacion, comps,
            pd.concat(posthoc, ignore_index=True), medias)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paso 5: ANOVA de parcelas subdivididas acorde al diseno.")
    p.add_argument("--entrada", default="",
                   help="def: <salida>/p3_fasores_compensados.txt")
    p.add_argument("--salida", default=config.DIR_RESULTADOS)
    p.add_argument("--respuesta", default="amp", choices=["amp", "fase"],
                   help="Magnitud analizada (def: amp)")
    p.add_argument("--grupos-velocidad", dest="grupos", action="store_true",
                   help="Repite el analisis dentro de cada tramo de velocidad")
    p.add_argument("--grupos-desbalanceo", dest="grupos_desb", action="store_true",
                   help="Repite el analisis para cada nivel de desbalanceo por "
                        "separado (el diseno se reduce a un split-plot)")
    args = p.parse_args()

    salida = Path(args.salida).expanduser()
    entrada = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.ARCHIVO_FASORES_COMP
    if not entrada.is_file():
        print(f"ERROR: no existe {entrada}. Ejecuta antes p3_compensar_runout.py")
        return 1

    df = comun.leer_tabla(entrada)
    respuestas = {s: f"{s}_{args.respuesta}" for s in config.SENSORES}
    unidad = "amplitud 1X [um]" if args.respuesta == "amp" else "fase [grados]"

    if args.respuesta == "fase":
        print("!! AVISO: la fase es una variable CIRCULAR. Un ANAVA corriente sobre\n"
              "   grados no es valido en general (359 y 1 distan 2 grados, no 358).\n"
              "   Uselo solo si ha comprobado que ninguna condicion cruza el 0/360.\n")

    destino = salida / config.DIR_ESTADISTICA
    try:
        tablas, ingenuos, comparacion, comps, posthoc, _ = analizar(
            df, respuestas, unidad, destino)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # ---- informe ----
    r = len(df["Repeticion"].unique()); a = len(df["Viscosidad"].unique())
    b = len(df["Desbalanceo"].unique()); c = len(df["Velocidad"].unique())
    print(f"Entrada : {entrada}  ({len(df)} filas)")
    print(f"Diseno  : {r} bloques x {a} viscosidades x {b} desbalanceos x "
          f"{c} velocidades = {r*a*b*c} observaciones\n")

    print("=" * 78)
    print(f"ANOVA split-split-plot — sensor {config.SENSORES[0]} (ejemplo)")
    print("=" * 78)
    t = tablas[config.SENSORES[0]]
    print(t[["Fuente", "SS", "GL", "MS", "GL_denominador", "F", "p",
             "Contribucion_SS_%"]]
          .to_string(index=False, float_format=lambda v: f"{v:10.4g}"))

    print("\n" + "=" * 78)
    print("EFECTO DE LA VISCOSIDAD: contraste correcto frente al ingenuo")
    print("=" * 78)
    print(comparacion.to_string(index=False, float_format=lambda v: f"{v:9.4g}"))

    n_c = (comparacion["significativa_correcto"] == "SI").sum()
    n_i = (comparacion["significativa_ingenuo"] == "SI").sum()
    print(f"\nSensores con viscosidad significativa: {n_c}/4 con el error correcto, "
          f"{n_i}/4 con el ANOVA ingenuo.")
    infl = comparacion["veces_F_inflada"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(infl):
        print(f"El ANOVA ingenuo multiplica la F de la viscosidad por un factor de "
              f"{infl.min():.1f} a {infl.max():.1f}.")
    print("La viscosidad se juzga con "
          f"{int(comparacion['gl_den_correcto'].iloc[0])} gl en el denominador, "
          f"no con {int(comparacion['gl_den_ingenuo'].iloc[0])}.")

    print("\n" + "=" * 78)
    print("COMPARACIONES ENTRE ACEITES (Tukey con Error(a))")
    print("=" * 78)
    print(posthoc.to_string(index=False, float_format=lambda v: f"{v:9.4g}"))

    # ---- opcional: por tramos de velocidad ----
    if args.grupos:
        print("\n" + "=" * 78)
        print("ANALISIS POR TRAMOS DE VELOCIDAD")
        print("=" * 78)
        resumen = []
        for nombre, vels in GRUPOS_VELOCIDAD.items():
            sub = df[df["Velocidad"].isin(vels)]
            disponibles = sorted(sub["Velocidad"].unique())
            if len(disponibles) < 2:
                print(f"  {nombre}: menos de 2 velocidades disponibles, se omite.")
                continue
            try:
                _, _, cmp_g, _, _, medias_g = analizar(
                    sub, respuestas, unidad, destino, nombre)
            except ValueError as e:
                print(f"  {nombre}: {e}")
                continue
            sig = (cmp_g["significativa_correcto"] == "SI").sum()
            print(f"  {nombre} ({len(disponibles)} rpm): viscosidad significativa "
                  f"en {sig}/4 sensores  (p min = {cmp_g['p_correcto'].min():.4g})")
            resumen.append((nombre, medias_g,
                            dict(zip(cmp_g["Sensor"], cmp_g["p_correcto"]))))

        if len(resumen) >= 2:
            viscs = sorted(df["Viscosidad"].unique())
            fig_viscosidad_por_grupo(
                resumen, viscs, destino / "p5_viscosidad_por_grupo.png", unidad)
            print(f"\n  Figura comparativa entre tramos: "
                  f"{destino / 'p5_viscosidad_por_grupo.png'}")

    # ---- opcional: cada desbalanceo por separado ----
    if args.grupos_desb:
        print("\n" + "=" * 78)
        print("ANALISIS POR NIVEL DE DESBALANCEO")
        print("=" * 78)
        print("Al fijar el desbalanceo, ese factor sale del modelo: el diseno pasa\n"
              "de split-split-plot a split-plot (bloque -> viscosidad -> velocidad).\n"
              "La viscosidad sigue contrastandose con Error(a) = Rep x Visc y 4 gl:\n"
              "separar por desbalanceo NO aporta grados de libertad al aceite.\n")
        resumen_d = []
        for d in sorted(df["Desbalanceo"].unique()):
            sub = df[df["Desbalanceo"] == d]
            nombre = f"Desbalanceo_{int(d)}"
            try:
                _, _, cmp_d, _, _, medias_d = analizar(
                    sub, respuestas, unidad, destino, nombre)
            except ValueError as e:
                print(f"  {nombre}: {e}")
                continue
            sig = (cmp_d["significativa_correcto"] == "SI").sum()
            print(f"  {nombre}: viscosidad significativa en {sig}/4 sensores  "
                  f"(p min = {cmp_d['p_correcto'].min():.4g})")
            resumen_d.append((nombre, medias_d,
                              dict(zip(cmp_d["Sensor"], cmp_d["p_correcto"]))))

        if len(resumen_d) >= 2:
            viscs = sorted(df["Viscosidad"].unique())
            fig_viscosidad_por_grupo(
                resumen_d, viscs, destino / "p5_viscosidad_por_desbalanceo.png",
                unidad, que="nivel de desbalanceo")
            print(f"\n  Figura comparativa entre desbalanceos: "
                  f"{destino / 'p5_viscosidad_por_desbalanceo.png'}")

    # ---- resumen de lo que se ha hecho y de lo que no ----
    print("\n" + "=" * 78)
    print("ANALISIS OPCIONALES")
    print("=" * 78)
    for activo, etiqueta, bandera in (
            (args.grupos, "por tramos de velocidad", "--grupos-velocidad"),
            (args.grupos_desb, "por nivel de desbalanceo", "--grupos-desbalanceo")):
        if activo:
            print(f"  [x] {etiqueta:26s} ejecutado")
        else:
            print(f"  [ ] {etiqueta:26s} NO ejecutado — anade {bandera}")

    n = len(list(destino.glob("*")))
    print(f"\n{n} archivos en: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
