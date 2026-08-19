"""
config.py — shared settings for the statistical analysis module.
================================================================

Edit the default paths and the experiment constants here. Every script also
accepts command-line arguments, which take precedence over these values.
"""

from __future__ import annotations

# ============================================================
# DEFAULT PATHS  (override with --entrada / --salida)
# ============================================================

# Folder holding the original .xlsx files (rep1_32_1_p_0424.xlsx, ...)
DIR_EXCEL = r"C:\Users\Owner\Documents\BD\All data"

# Folder where every result of this module is written
DIR_RESULTADOS = r"C:\Users\Owner\Documents\BD\resultados_analisis"


# ============================================================
# EXPERIMENTAL DESIGN
# ============================================================

REPETICIONES = [1, 2, 3]        # blocks
VISCOSIDADES = [32, 46, 68]     # whole plot   (ISO VG grade)
DESBALANCEOS = [1, 2, 3]        # subplot
VELOCIDADES = [600, 1300, 2000, 2500, 2600, 2700, 2800,
               3400, 3500, 3600, 3700, 4000, 4500, 5000, 5500]  # sub-subplot

# Expected number of .xlsx files = rep x visc x unbalance
N_ARCHIVOS_ESPERADOS = len(REPETICIONES) * len(VISCOSIDADES) * len(DESBALANCEOS)  # 27

# Speed used as "slow roll" for the runout compensation (step 3)
RPM_SLOW_ROLL = 600


# ============================================================
# SPEED GROUPS  (optional per-band analysis, step 5)
# ============================================================
# Short keys (G1..G4) are what the figures print on the axes: long names become
# unreadable once a figure is scaled down to a journal column. The description
# is meant for the figure caption and is printed in the subtitle.

GRUPOS_VELOCIDAD = {
    "G1": {"rpm": [600, 1300, 2000],        "desc": "low speed"},
    "G2": {"rpm": [2500, 2600, 2700, 2800], "desc": "below 1st critical"},
    "G3": {"rpm": [3400, 3500, 3600, 3700], "desc": "near 1st critical"},
    "G4": {"rpm": [4000, 4500, 5000, 5500], "desc": "above 1st critical"},
}


def leyenda_grupos() -> str:
    """One-line description of the speed bands, for figure subtitles."""
    return " · ".join(
        f"{k}: {v['desc']} ({min(v['rpm'])}-{max(v['rpm'])} rpm)"
        for k, v in GRUPOS_VELOCIDAD.items())


# ============================================================
# SENSORS
# ============================================================

SENSORES = ["P1Y", "P1X", "P2Y", "P2X"]
COJINETES = ["P1", "P2"]
DIRECCIONES = ["Y", "X"]        # column order in the 2x2 phasor figures

# Colour of each sensor in the statistical figures, where the four proximity
# probes may share one set of axes. Bearing 1 in blues, bearing 2 in warm
# tones; the Y direction dark and the X direction light.
COLOR_SENSOR = {
    "P1Y": "#1f4e79",   # dark blue
    "P1X": "#6fa8dc",   # light blue
    "P2Y": "#7f3f00",   # brown
    "P2X": "#e69138",   # orange
}

# Significance colours, shared by steps 5 and 6 so that a p value always looks
# the same wherever it is drawn: green marks the region where the factor DOES
# influence the response, red marks the threshold itself.
COLOR_SIGNIFICATIVO = "#27ae60"
COLOR_NO_SIGNIFICATIVO = "#bdc3c7"
COLOR_UMBRAL = "#c0392b"


# ============================================================
# FIGURE FORMATS  (--formato)
# ============================================================
# A figure is legible in print when its font size is right AT THE SIZE IT WILL
# BE PRINTED. Drawing a 16 in wide figure and shrinking it into a 3.5 in column
# scales the text down by 4.6x, which is why large on-screen figures come out
# unreadable on paper. Each preset therefore fixes the FINAL width and sizes
# the type for that width, so the figure is placed at 100 % with no scaling.
#
#   screen : wide, for reviewing on a monitor (default)
#   column : one column of a two-column letter-size page  (3.5 in / 88.9 mm)
#   double : spanning both columns of that page           (7.16 in / 181.6 mm)

FORMATOS_FIGURA = {
    "screen": {"ancho": 16.0, "base": 11.0, "titulo": 13.0, "ejes": 11.0,
               "ticks": 9.0, "leyenda": 10.0, "anotacion": 8.0,
               "linea": 1.7, "dpi": 200},
    "column": {"ancho": 3.5, "base": 8.0, "titulo": 8.5, "ejes": 8.0,
               "ticks": 6.5, "leyenda": 6.5, "anotacion": 5.5,
               "linea": 1.0, "dpi": 600},
    "double": {"ancho": 7.16, "base": 9.0, "titulo": 10.0, "ejes": 9.0,
               "ticks": 7.5, "leyenda": 8.0, "anotacion": 6.5,
               "linea": 1.3, "dpi": 600},
}

FORMATO_POR_DEFECTO = "screen"


# ============================================================
# PHASOR FIGURES (steps 2 and 4)
# ============================================================
# Colour = viscosity ; colour saturation = unbalance ; line style = repetition.

COLOR_BASE_VISCOSIDAD = {
    32: (0.85, 0.08, 0.08),   # red
    46: (0.10, 0.25, 0.85),   # blue
    68: (0.02, 0.52, 0.14),   # green
}

# Blend towards white: unbalance 1 = most washed out, 3 = full colour.
ATENUACION_DESBALANCEO = {1: 0.58, 2: 0.30, 3: 0.0}

ESTILO_REPETICION = {1: "-", 2: "--", 3: ":"}

FIGSIZE = (15, 9)
DPI = 200

NOMBRE_DIR_SIN_COMP = "phasors without runout compensation"
NOMBRE_DIR_CON_COMP = "phasors with runout compensation"


# ============================================================
# STABILISATION DETECTION (step 1)
# ============================================================
# Each measurement lasts ~3 min. The script looks for the longest final stretch
# in which amplitude and phase have already settled (no drift, low scatter).

FRAC_MINIMA_ESTABLE = 0.30   # the settled stretch must cover >= 30 % of the record

# Primary criterion: the phasor NO LONGER DRIFTS (the transient is over).
TOL_DERIVA_AMP = 0.05        # max amplitude change between halves (fraction)
TOL_DERIVA_FASE = 5.0        # max phase change between halves [degrees]

# Secondary criterion: scatter inside the stretch must not be excessive. Real
# scatter grows with speed, so these tolerances are deliberately loose.
TOL_CV_AMP = 0.15            # max robust coefficient of variation of amplitude
TOL_DISP_FASE = 12.0         # max CIRCULAR standard deviation of phase [degrees]

N_CANDIDATOS = 120           # start positions evaluated per sensor


# ============================================================
# INTERMEDIATE FILE NAMES
# ============================================================

ARCHIVO_FASORES = "p1_phasors.txt"
ARCHIVO_DIAGNOSTICO = "p1_stabilisation_diagnostics.txt"
ARCHIVO_FASORES_COMP = "p3_phasors_compensated.txt"
ARCHIVO_SLOW_ROLL = "p3_slow_roll_vectors.txt"
DIR_ESTADISTICA = "p5_statistics"
