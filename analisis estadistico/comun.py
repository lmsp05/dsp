"""
comun.py — utilidades compartidas por los pasos 1..5.
=====================================================

Incluye el tratamiento correcto del angulo de fase, que es el punto delicado de
todo el modulo:

  * Los angulos vienen envueltos en [0, 360). Un fasor que fisicamente cambia
    poco (p.ej. de 358 a 2 grados) aparenta un salto de 356 grados. Promediar,
    restar o graficar esos numeros directamente produce resultados sin sentido.

  * La solucion correcta NO es promediar angulos: es trabajar con el FASOR
    COMPLEJO  z = A * exp(i*phi). El promedio vectorial  mean(z)  es inmune al
    envolvimiento porque nunca manipula el angulo como escalar. De ahi salen la
    amplitud (|mean(z)|) y la fase (angle(mean(z))) promedio.

  * Para analizar la evolucion temporal (deteccion de estabilizacion) y para
    graficar curvas continuas si se desea, se usa `desenvolver_fase`, que
    deshace los saltos de +-360 grados sumando el multiplo de 360 que minimiza
    la diferencia con la muestra anterior.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import config


# ============================================================
# NOMENCLATURA DE LOS ARCHIVOS
# ============================================================

# rep<R>_<viscosidad>_<desbalanceo>_<p|ac>_<fecha>.xlsx
PATRON_ARCHIVO = re.compile(
    r"^rep(\d+)_(\d+)_(\d+)_(p|ac)_([^.]*)\.xlsx$", re.IGNORECASE)


def parsear_nombre(nombre: str) -> dict | None:
    """Devuelve {rep, visc, desb, tipo, fecha} o None si no cumple la nomenclatura."""
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
    Lista los .xlsx de `directorio` que cumplen la nomenclatura y son del `tipo`
    pedido ('p' = proximitores, 'ac' = acelerometros).

    Devuelve (validos, ignorados) donde `validos` es una lista de dicts con la
    clave extra 'ruta'. Cualquier otro archivo de la carpeta se ignora.
    """
    validos, ignorados = [], []
    for ruta in sorted(directorio.iterdir()):
        if not ruta.is_file():
            continue
        if ruta.name.startswith("~$"):          # temporales de Excel
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
# TRATAMIENTO DEL ANGULO DE FASE
# ============================================================

def envolver_0_360(grados):
    """Lleva cualquier angulo al rango [0, 360)."""
    return np.mod(np.asarray(grados, dtype=float), 360.0)


def envolver_180(grados):
    """Lleva cualquier angulo al rango (-180, 180]. Util para diferencias."""
    return (np.asarray(grados, dtype=float) + 180.0) % 360.0 - 180.0


def desenvolver_fase(grados):
    """
    Deshace los saltos artificiales de +-360 grados de una serie de angulos.

    Es el "algoritmo para deshacer ese cambio" pedido en el paso 3: recorre la
    serie y a cada muestra le suma el multiplo de 360 que la deja lo mas cerca
    posible de la anterior, de modo que 359 -> 1 se convierte en 359 -> 361 y la
    serie queda continua. Equivale a np.unwrap sobre radianes.
    """
    g = np.asarray(grados, dtype=float)
    if g.size == 0:
        return g
    return np.rad2deg(np.unwrap(np.deg2rad(g)))


def a_fasor(amp, fase_grados):
    """Convierte (amplitud, fase en grados) al fasor complejo A*exp(i*phi)."""
    return np.asarray(amp, dtype=float) * np.exp(1j * np.deg2rad(np.asarray(fase_grados, dtype=float)))


def de_fasor(z):
    """Convierte un fasor complejo a (amplitud, fase en [0, 360))."""
    z = np.asarray(z)
    return np.abs(z), envolver_0_360(np.rad2deg(np.angle(z)))


def promedio_vectorial(amp, fase_grados):
    """
    Promedio VECTORIAL de un conjunto de fasores -> (amplitud, fase).

    Inmune al envolvimiento del angulo. Es el promedio fisicamente correcto de
    un 1X: promedia parte real e imaginaria, no el modulo y el angulo por
    separado.
    """
    z = a_fasor(amp, fase_grados)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan, np.nan
    m = z.mean()
    return float(np.abs(m)), float(envolver_0_360(np.rad2deg(np.angle(m))))


def promedio_escalar_fase(fase_grados):
    """
    Promedio circular SOLO del angulo (media de direcciones unitarias) y su
    concentracion R en [0, 1].

    R ~ 1  -> los angulos estan muy agrupados (el promedio es representativo).
    R ~ 0  -> los angulos estan dispersos (el promedio no significa nada).
    """
    f = np.asarray(fase_grados, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return np.nan, np.nan
    u = np.exp(1j * np.deg2rad(f)).mean()
    return float(envolver_0_360(np.rad2deg(np.angle(u)))), float(np.abs(u))


def dispersion_robusta(x):
    """Desviacion tipica robusta (MAD escalada). Insensible a valores atipicos."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def dispersion_circular(fase_grados):
    """
    Desviacion tipica CIRCULAR de un conjunto de angulos [grados].

        s = sqrt(-2 * ln R)      con R = |mean(exp(i*phi))|

    A diferencia de la desviacion tipica sobre la fase desenvuelta, esta medida
    esta ACOTADA y no se dispara cuando el fasor pasa cerca del origen y el
    angulo gira sin sentido fisico. Se satura en 180 grados.
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
    Diferencia a - b entre dos angulos, llevada a (-180, 180].

    Es la forma correcta de comparar dos fases: la distancia entre 359 y 1 es
    2 grados, no 358.
    """
    return float(envolver_180(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# ============================================================
# COLORES DE LAS GRAFICAS
# ============================================================

def color_condicion(visc: int, desb: int):
    """
    Color de una curva: el TONO lo fija la viscosidad (32 rojo, 46 azul,
    68 verde) y la SATURACION/oscuridad la fija el desbalanceo (1 el mas
    atenuado, 3 el mas intenso).
    """
    r, g, b = config.COLOR_BASE_VISCOSIDAD[int(visc)]
    f = config.ATENUACION_DESBALANCEO[int(desb)]      # 0 = color pleno
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


def estilo_repeticion(rep: int) -> str:
    return config.ESTILO_REPETICION.get(int(rep), "-")


# ============================================================
# LECTURA / ESCRITURA DE LA TABLA DE FASORES
# ============================================================

COLUMNAS_FACTOR = ["Repeticion", "Viscosidad", "Desbalanceo", "Velocidad"]


def columnas_fasor(sensores=None):
    """['P1Y_amp', 'P1Y_fase', 'P1X_amp', ...] en el orden de config.SENSORES."""
    sensores = sensores or config.SENSORES
    cols = []
    for s in sensores:
        cols += [f"{s}_amp", f"{s}_fase"]
    return cols


def leer_tabla(ruta: Path):
    """Lee una tabla de fasores separada por tabuladores como DataFrame."""
    import pandas as pd
    df = pd.read_csv(ruta, sep="\t")
    for c in COLUMNAS_FACTOR:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def escribir_tabla(df, ruta: Path, decimales: int = 4):
    """Escribe la tabla separada por tabuladores con formato uniforme."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, sep="\t", index=False, float_format=f"%.{decimales}f")
    return ruta
