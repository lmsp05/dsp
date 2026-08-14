"""
comun.py — helpers shared by steps 1..6.
========================================

This module holds the correct handling of the phase angle, which is the
delicate point of the whole pipeline:

  * Angles arrive wrapped into [0, 360). A phasor that physically barely moves
    (say from 358 to 2 degrees) looks like a 356 degree jump. Averaging,
    subtracting or plotting those numbers directly gives meaningless results.

  * The correct fix is NOT to average angles: it is to work with the COMPLEX
    PHASOR  z = A * exp(i*phi). The vector mean  mean(z)  is immune to wrapping
    because it never treats the angle as a scalar. The mean amplitude
    (|mean(z)|) and mean phase (angle(mean(z))) follow from it.

  * To study the time evolution (stabilisation detection) and to draw
    continuous curves when wanted, `desenvolver_fase` undoes the +-360 degree
    jumps by adding the multiple of 360 that minimises the difference with the
    previous sample.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import config


# ============================================================
# FILE NAMING CONVENTION
# ============================================================

# rep<R>_<viscosity>_<unbalance>_<p|ac>_<date>.xlsx
PATRON_ARCHIVO = re.compile(
    r"^rep(\d+)_(\d+)_(\d+)_(p|ac)_([^.]*)\.xlsx$", re.IGNORECASE)


def parsear_nombre(nombre: str) -> dict | None:
    """Return {rep, visc, desb, tipo, fecha}, or None if the name does not match."""
    m = PATRON_ARCHIVO.match(nombre)
    if not m:
        return None
    return {
        "rep": int(m.group(1)),
        "visc": int(m.group(2)),
        "desb": int(m.group(3)),
        "tipo": m.group(4).lower(),
        "fecha": m.group(5),
    }


def listar_excels(directorio: Path, tipo: str = "p"):
    """
    List the .xlsx files in `directorio` that match the naming convention and
    are of the requested `tipo` ('p' = proximity probes, 'ac' = accelerometers).

    Returns (valid, ignored), where `valid` is a list of dicts with an extra
    'ruta' key. Any other file in the folder is ignored.
    """
    validos, ignorados = [], []
    for ruta in sorted(directorio.iterdir()):
        if not ruta.is_file():
            continue
        if ruta.name.startswith("~$"):          # Excel lock files
            continue
        info = parsear_nombre(ruta.name)
        if info is None:
            ignorados.append(ruta.name)
            continue
        if info["tipo"] != tipo.lower():
            ignorados.append(ruta.name)
            continue
        info["ruta"] = ruta
        validos.append(info)
    return validos, ignorados


# ============================================================
# PHASE ANGLE HANDLING
# ============================================================

def envolver_0_360(grados):
    """Wrap any angle into [0, 360)."""
    return np.mod(np.asarray(grados, dtype=float), 360.0)


def envolver_180(grados):
    """Wrap any angle into (-180, 180]. Useful for differences."""
    return (np.asarray(grados, dtype=float) + 180.0) % 360.0 - 180.0


def desenvolver_fase(grados):
    """
    Undo the artificial +-360 degree jumps in a series of angles.

    It walks the series adding to each sample the multiple of 360 that leaves
    it as close as possible to the previous one, so 359 -> 1 becomes
    359 -> 361 and the series stays continuous. Equivalent to np.unwrap on
    radians.
    """
    g = np.asarray(grados, dtype=float)
    if g.size == 0:
        return g
    return np.rad2deg(np.unwrap(np.deg2rad(g)))


def a_fasor(amp, fase_grados):
    """Convert (amplitude, phase in degrees) into the complex phasor A*exp(i*phi)."""
    return np.asarray(amp, dtype=float) * np.exp(1j * np.deg2rad(np.asarray(fase_grados, dtype=float)))


def de_fasor(z):
    """Convert a complex phasor into (amplitude, phase in [0, 360))."""
    z = np.asarray(z)
    return np.abs(z), envolver_0_360(np.rad2deg(np.angle(z)))


def promedio_vectorial(amp, fase_grados):
    """
    VECTOR mean of a set of phasors -> (amplitude, phase).

    Immune to angle wrapping. This is the physically correct way to average a
    1X vector: it averages real and imaginary parts, not modulus and angle
    separately.
    """
    z = a_fasor(amp, fase_grados)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan, np.nan
    m = z.mean()
    return float(np.abs(m)), float(envolver_0_360(np.rad2deg(np.angle(m))))


def promedio_escalar_fase(fase_grados):
    """
    Circular mean of the ANGLE only (mean of unit directions) and its
    concentration R in [0, 1].

    R ~ 1  -> the angles are tightly grouped (the mean is representative).
    R ~ 0  -> the angles are scattered (the mean means nothing).
    """
    f = np.asarray(fase_grados, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return np.nan, np.nan
    u = np.exp(1j * np.deg2rad(f)).mean()
    return float(envolver_0_360(np.rad2deg(np.angle(u)))), float(np.abs(u))


def dispersion_robusta(x):
    """Robust standard deviation (scaled MAD). Insensitive to outliers."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def dispersion_circular(fase_grados):
    """
    CIRCULAR standard deviation of a set of angles [degrees].

        s = sqrt(-2 * ln R)      with R = |mean(exp(i*phi))|

    Unlike the standard deviation of the unwrapped phase, this measure is
    BOUNDED and does not blow up when the phasor passes near the origin and the
    angle spins without physical meaning. It saturates at 180 degrees.
    """
    f = np.asarray(fase_grados, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return np.nan
    R = float(np.abs(np.exp(1j * np.deg2rad(f)).mean()))
    if R <= 1e-12:
        return 180.0
    return float(min(180.0, np.rad2deg(np.sqrt(-2.0 * np.log(min(R, 1.0))))))


def diferencia_angular(a, b):
    """
    Difference a - b between two angles, wrapped into (-180, 180].

    This is the correct way to compare two phases: the distance between 359 and
    1 is 2 degrees, not 358.
    """
    return float(envolver_180(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# ============================================================
# FIGURE COLOURS
# ============================================================

def color_condicion(visc: int, desb: int):
    """
    Curve colour: the HUE is set by viscosity (32 red, 46 blue, 68 green) and
    the SATURATION/darkness by the unbalance level (1 the most washed out,
    3 the most intense).
    """
    r, g, b = config.COLOR_BASE_VISCOSIDAD[int(visc)]
    f = config.ATENUACION_DESBALANCEO[int(desb)]      # 0 = full colour
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


def estilo_repeticion(rep: int) -> str:
    return config.ESTILO_REPETICION.get(int(rep), "-")


# ============================================================
# PHASOR TABLE I/O
# ============================================================

COLUMNAS_FACTOR = ["Repetition", "Viscosity", "Unbalance", "Speed"]


def columnas_fasor(sensores=None):
    """['P1Y_amp', 'P1Y_phase', 'P1X_amp', ...] in config.SENSORES order."""
    sensores = sensores or config.SENSORES
    cols = []
    for s in sensores:
        cols += [f"{s}_amp", f"{s}_phase"]
    return cols


def leer_tabla(ruta: Path):
    """Read a tab-separated phasor table into a DataFrame."""
    import pandas as pd
    df = pd.read_csv(ruta, sep="\t")
    for c in COLUMNAS_FACTOR:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def escribir_tabla(df, ruta: Path, decimales: int = 4):
    """Write the table as tab-separated text with a uniform number format."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, sep="\t", index=False, float_format=f"%.{decimales}f")
    return ruta


# ============================================================
# FIGURE FORMAT (publication sizing)
# ============================================================

def aplicar_formato(nombre: str, base: float | None = None,
                    titulo: float | None = None) -> dict:
    """
    Apply one of `config.FORMATOS_FIGURA` to matplotlib and return its settings.

    Type is sized for the FINAL printed width, so the figure is placed at 100 %
    with no scaling. Shrinking a wide figure into a narrow column is exactly
    what makes the labels unreadable.

    `base` overrides the base font size and `titulo` the title size, both in
    points at final size. Everything else follows `base`: tick labels, legend,
    bar annotations, line widths and marker sizes, so raising it does not leave
    the rest of the figure behind. The figure also grows TALLER (see
    `tam_figura`), because at a fixed column width the extra type needs extra
    vertical room for rotated tick labels.
    """
    import matplotlib

    f = dict(config.FORMATOS_FIGURA[nombre])
    f["escala_texto"] = 1.0

    if base:
        f["escala_texto"] = base / f["base"]
        f["base"] = base
        f["ejes"] = base
        f["ticks"] = base * 0.86
        f["leyenda"] = base * 0.92
        f["anotacion"] = base * 0.78
        # Strokes grow more slowly than type: matching them 1:1 would make the
        # figure look heavy at large font sizes.
        f["linea"] = f["linea"] * f["escala_texto"] ** 0.6

    f["titulo"] = titulo if titulo else f["titulo"] * f["escala_texto"]

    matplotlib.rcParams.update({
        "font.size": f["base"],
        "axes.titlesize": f["titulo"],
        "axes.labelsize": f["ejes"],
        "xtick.labelsize": f["ticks"],
        "ytick.labelsize": f["ticks"],
        "legend.fontsize": f["leyenda"],
        "figure.titlesize": f["titulo"],
        "lines.linewidth": f["linea"],
        "lines.markersize": max(2.5, f["linea"] * 2.6),
        "axes.linewidth": max(0.5, f["linea"] * 0.5),
        "grid.linewidth": max(0.4, f["linea"] * 0.4),
        "patch.linewidth": max(0.3, f["linea"] * 0.3),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return f


def tam_figura(f: dict, relacion: float, ancho_rel: float = 1.0):
    """
    Figure size in inches: `ancho_rel` of the preset width, `relacion` = h/w.

    Raising the font size GROWS THE WHOLE FIGURE rather than squeezing the plot
    area: at a fixed width the extra type eventually leaves no room for the axes
    and matplotlib collapses the layout. The figure then exceeds the target
    column and has to be scaled down when placed, but because it was drawn with
    large type it survives that scaling — which is the point.

    Width grows more slowly than the type (exponent 0.85) so the net effect is
    still a gain in relative type size, and height a little more slowly still,
    keeping the figure from turning into a tall strip.
    """
    e = f.get("escala_texto", 1.0)
    ancho = f["ancho"] * ancho_rel * e ** 0.85
    return (ancho, ancho * relacion * e ** 0.10)


def seleccionar_sensores(cojinete: str) -> list[str]:
    """
    Probes to plot: 'todos', 'P1' (P1Y, P1X) or 'P2' (P2Y, P2X).

    The scripts assign the result to `config.SENSORES`, which every figure and
    table reads, so the selection propagates without threading it through
    every call.
    """
    if cojinete == "todos":
        return list(config.SENSORES)
    return [s for s in config.SENSORES if s.startswith(cojinete)]


def envolver_titulo(texto: str, f: dict, ancho_rel: float = 1.0) -> str:
    """
    Wrap a title to the figure width.

    Without this a long one-line title forces `bbox_inches="tight"` to widen the
    canvas past the target column, which silently defeats the whole point of
    fixing the width. Each existing line break is honoured and re-wrapped.
    """
    import textwrap

    # ~0.5 * font size is a good average glyph advance for this font.
    n = max(20, int(f["ancho"] * ancho_rel * 72.0 / (f["titulo"] * 0.5)))
    return "\n".join(textwrap.fill(linea, n) for linea in texto.split("\n"))


def columnas_leyenda(etiquetas, f: dict, ancho_rel: float = 1.0) -> int:
    """
    How many legend columns fit across the figure without widening it.

    Same reasoning as `envolver_titulo`: a single row of long labels is what
    stretches the canvas.
    """
    if not etiquetas:
        return 1
    mas_larga = max(len(str(e)) for e in etiquetas)
    ancho_pt = f["ancho"] * ancho_rel * 72.0
    # label text + marker + padding, in points
    por_entrada = mas_larga * f["leyenda"] * 0.55 + f["leyenda"] * 3.2
    return max(1, min(len(etiquetas), int(ancho_pt // max(por_entrada, 1.0))))
