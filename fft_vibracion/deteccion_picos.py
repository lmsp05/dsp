"""
deteccion_picos.py
==================

Lee los resultados de la FFT generados por ``procesar_fft.py`` (el .txt tabular
o su version .npy) y detecta los picos de cada espectro replicando el
procedimiento del script de MATLAB ``x0302_nf_gph_transformations.m``.

--------------------------------------------------------------------------
PROCEDIMIENTO (equivalente al de MATLAB)
--------------------------------------------------------------------------
En el .m, la deteccion de picos "por espectro" es el bucle de *ridge detection*
que se aplica a cada fila de la cascada (cada RPM):

    [~,locs] = findpeaks(row, 'MinPeakProminence', max(row)*0.1, ...
                              'MinPeakDistance', 5);

Es decir: `scipy.signal.find_peaks` sobre la AMPLITUD LINEAL del espectro, con

  * prominencia minima = `max(amplitud) * FRACCION_PROMINENCIA`  (0.1 como en el
    bucle de crestas del .m), y
  * distancia minima entre picos = `DIST_BINS` bins (5 como en el .m).

(El .m convierte la cascada de dB a lineal antes de detectar; nuestra amplitud
ya esta en unidades lineales, asi que se usa directamente.)

Para cada espectro (una combinacion rep/iso/dsb/rpm/sensor) se detectan los
picos, se ordenan por frecuencia ascendente y se guardan como omega1, omega2, ...

--------------------------------------------------------------------------
SALIDAS (en la misma carpeta que la entrada, o la indicada con --outdir)
--------------------------------------------------------------------------
  * picos_detectados.txt : tabla con columnas
        rep  iso  dsb  rpm  sensor  omega1  omega2  ...  omegaN
    (una fila por archivo y sensor; celdas vacias donde no hay mas picos).
  * picos_scatter.png     : dispersion frecuencia (Y) vs velocidad (X), con
        color = viscosidad ISO y forma = nivel de desbalance dsb.
  * picos_scatter_<sensor>.png : la misma dispersion, un archivo por sensor.

Uso:
    python deteccion_picos.py --entrada resultados_fft.txt
    python deteccion_picos.py --entrada resultados_fft.npy --outdir figuras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# Parametros del procedimiento (valores del bucle de crestas del .m)
FRACCION_PROMINENCIA = 0.1   # MinPeakProminence = max(espectro) * fraccion
DIST_BINS = 5                # MinPeakDistance en numero de bins

# Codificacion visual del diagrama de dispersion
COLORES_ISO = {32: "#1f77b4", 46: "#ff7f0e", 68: "#2ca02c"}
FORMAS_DSB = {1: "o", 2: "s", 3: "^"}


# ============================================================
# LECTURA DE LOS RESULTADOS DE procesar_fft.py
# ============================================================

def cargar_espectros(path: Path) -> np.ndarray:
    """Carga el .txt tabular o el .npy y devuelve un array estructurado con,
    al menos, los campos rep, iso, dsb, rpm_nominal, sensor, frecuencia_Hz,
    amplitud."""
    if path.suffix.lower() == ".npy":
        return np.load(path)

    with path.open("r", encoding="utf-8") as fh:
        utiles = [ln.rstrip("\n") for ln in fh
                  if ln.strip() and not ln.lstrip().startswith("#")]
    if len(utiles) < 2:
        raise ValueError("El archivo no contiene filas de datos.")
    columnas = utiles[0].split("\t")
    arr = np.genfromtxt(
        utiles[1:], delimiter="\t", names=columnas, dtype=None, encoding="utf-8")
    return np.atleast_1d(arr)


# ============================================================
# DETECCION DE PICOS POR ESPECTRO (procedimiento del .m)
# ============================================================

def detectar_picos(frecuencia: np.ndarray, amplitud: np.ndarray,
                   fraccion: float, dist_bins: int,
                   fmin: float | None, fmax: float | None) -> np.ndarray:
    """Devuelve las frecuencias de los picos, ordenadas ascendentemente."""
    orden = np.argsort(frecuencia)
    f = np.asarray(frecuencia)[orden]
    a = np.asarray(amplitud)[orden]

    if fmin is not None or fmax is not None:
        m = np.ones(len(f), dtype=bool)
        if fmin is not None:
            m &= f >= fmin
        if fmax is not None:
            m &= f <= fmax
        f, a = f[m], a[m]

    if a.size == 0 or a.max() <= 0:
        return np.array([])

    prominencia = a.max() * fraccion
    picos, _ = find_peaks(a, prominence=prominencia, distance=max(1, dist_bins))
    return f[picos]


# ============================================================
# AGRUPAMIENTO POR (rep, iso, dsb, rpm, sensor)
# ============================================================

def procesar(arr: np.ndarray, args) -> list[dict]:
    rep = arr["rep"].astype(int)
    iso = arr["iso"].astype(int)
    dsb = arr["dsb"].astype(int)
    rpm = arr["rpm_nominal"].astype(float)
    sensor = np.asarray(arr["sensor"]).astype(str)
    freq = arr["frecuencia_Hz"].astype(float)
    amp = arr["amplitud"].astype(float)

    claves = list(zip(rep, iso, dsb, rpm, sensor))
    vistos: dict[tuple, None] = {}
    for c in claves:
        vistos.setdefault(c, None)

    claves_arr = np.array(claves, dtype=object)
    filas = []
    for clave in vistos:
        m = np.array([c == clave for c in claves], dtype=bool)
        omegas = detectar_picos(
            freq[m], amp[m], args.fraccion, args.dist_bins, args.fmin, args.fmax)
        r, i_, d, v, s = clave
        filas.append({
            "rep": int(r), "iso": int(i_), "dsb": int(d), "rpm": float(v),
            "sensor": str(s), "omegas": np.sort(omegas),
        })
    filas.sort(key=lambda x: (x["rep"], x["iso"], x["dsb"], x["rpm"], x["sensor"]))
    return filas


# ============================================================
# TABLA DE RESULTADOS
# ============================================================

def escribir_tabla(filas: list[dict], salida: Path) -> int:
    n_max = max((len(f["omegas"]) for f in filas), default=0)
    cols = ["rep", "iso", "dsb", "rpm", "sensor"] + [f"omega{k+1}" for k in range(n_max)]
    lineas = ["\t".join(cols)]
    for f in filas:
        celdas = [str(f["rep"]), str(f["iso"]), str(f["dsb"]),
                  f"{f['rpm']:g}", f["sensor"]]
        celdas += [f"{w:.4f}" for w in f["omegas"]]
        celdas += [""] * (n_max - len(f["omegas"]))
        lineas.append("\t".join(celdas))
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return n_max


# ============================================================
# DIAGRAMA DE DISPERSION frecuencia vs velocidad
# ============================================================

def _dibujar_scatter(filas, ax, titulo, con_ordenes=True):
    isos = sorted({f["iso"] for f in filas})
    dsbs = sorted({f["dsb"] for f in filas})
    rpms = sorted({f["rpm"] for f in filas})

    for f in filas:
        if len(f["omegas"]) == 0:
            continue
        color = COLORES_ISO.get(f["iso"], "gray")
        marker = FORMAS_DSB.get(f["dsb"], "x")
        ax.scatter([f["rpm"]] * len(f["omegas"]), f["omegas"],
                   c=color, marker=marker, s=45, alpha=0.75,
                   edgecolors="k", linewidths=0.4)

    # lineas de orden 1X, 2X, 3X como referencia (freq = n * rpm/60)
    if con_ordenes and rpms:
        xr = np.array([min(rpms), max(rpms)], dtype=float)
        for n in (1, 2, 3):
            ax.plot(xr, n * xr / 60.0, ls="--", lw=0.8, color="gray", alpha=0.5)
            ax.text(xr[-1], n * xr[-1] / 60.0, f" {n}X", fontsize=7,
                    color="gray", va="center")

    ax.set_xlabel("Velocidad (RPM)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)

    # dos leyendas: color = ISO, forma = dsb
    leg_iso = [Line2D([0], [0], marker="o", ls="", mfc=COLORES_ISO.get(i, "gray"),
                      mec="k", ms=8, label=f"ISO {i}") for i in isos]
    leg_dsb = [Line2D([0], [0], marker=FORMAS_DSB.get(d, "x"), ls="", mfc="gray",
                      mec="k", ms=8, label=f"dsb {d}") for d in dsbs]
    l1 = ax.legend(handles=leg_iso, title="Viscosidad", loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=leg_dsb, title="Desbalance", loc="upper right", fontsize=8)


def graficar(filas: list[dict], outdir: Path):
    # 1) diagrama combinado (todos los sensores)
    fig, ax = plt.subplots(figsize=(11, 7))
    _dibujar_scatter(filas, ax, "Frecuencias detectadas vs velocidad (todos los sensores)")
    png = outdir / "picos_scatter.png"
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generados = [png]

    # 2) un diagrama por sensor
    sensores = sorted({f["sensor"] for f in filas})
    for s in sensores:
        sub = [f for f in filas if f["sensor"] == s]
        fig, ax = plt.subplots(figsize=(11, 7))
        _dibujar_scatter(sub, ax, f"Frecuencias detectadas vs velocidad — sensor {s}")
        png = outdir / f"picos_scatter_{s}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generados.append(png)
    return generados


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main(args) -> int:
    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_file():
        print(f"ERROR: no existe la entrada {entrada}")
        return 1
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else entrada.parent
    outdir.mkdir(parents=True, exist_ok=True)

    arr = cargar_espectros(entrada)
    faltan = [c for c in ("rep", "iso", "dsb", "rpm_nominal", "sensor",
                          "frecuencia_Hz", "amplitud") if c not in arr.dtype.names]
    if faltan:
        print(f"ERROR: a la entrada le faltan columnas: {faltan}")
        return 1

    filas = procesar(arr, args)
    if not filas:
        print("No se pudo formar ningun espectro.")
        return 1

    tabla = outdir / "picos_detectados.txt"
    n_max = escribir_tabla(filas, tabla)
    generados = graficar(filas, outdir)

    total_picos = sum(len(f["omegas"]) for f in filas)
    print(f"Espectros procesados: {len(filas)}")
    print(f"Picos detectados en total: {total_picos} (max por espectro: {n_max})")
    print(f"Tabla: {tabla}")
    for g in generados:
        print(f"Figura: {g}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deteccion de picos sobre los espectros de procesar_fft.py "
                    "(procedimiento de x0302_nf_gph_transformations.m) + tabla y "
                    "diagrama frecuencia-vs-velocidad.")
    p.add_argument("--entrada", default="resultados_fft.txt",
                   help="Archivo .txt o .npy generado por procesar_fft.py")
    p.add_argument("--outdir", default="",
                   help="Carpeta de salida (def: la misma de la entrada)")
    p.add_argument("--fraccion", type=float, default=FRACCION_PROMINENCIA,
                   help="MinPeakProminence = max(espectro) * fraccion (def: 0.1)")
    p.add_argument("--dist-bins", dest="dist_bins", type=int, default=DIST_BINS,
                   help="Distancia minima entre picos, en bins (def: 5)")
    p.add_argument("--fmin", type=float, default=None, help="Frecuencia minima a considerar [Hz]")
    p.add_argument("--fmax", type=float, default=None, help="Frecuencia maxima a considerar [Hz]")
    return p


if __name__ == "__main__":
    sys.exit(main(construir_parser().parse_args()))
