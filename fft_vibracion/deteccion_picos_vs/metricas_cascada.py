"""
metricas_cascada.py
==================

Deteccion de frecuencias naturales por METRICAS SOBRE LA CASCADA (freq x RPM),
portado de ``x0302_nf_gph_transformations.m``. En vez de promediar espectros, para
cada condicion se apila la cascada Z (filas = RPM, columnas = frecuencia comun) y
se calculan tres metricas ATRAVESANDO las RPM, aprovechando que una frecuencia
natural es FIJA con la velocidad mientras que el 1X y sus armonicos se MUEVEN:

  1. RMS por frecuencia:    RMS(f)   = sqrt(mean_rpm( Z^2 ))
  2. Kurtosis por frecuencia: Kurt(f) = kurtosis_rpm( Z )   (Pearson, como MATLAB)
  3. Ridge / persistencia de crestas: para cada RPM se detectan sus picos
     (findpeaks con prominencia = max(fila)*0.1 y distancia 5), y ridge(f) cuenta
     en cuantas RPM hay un pico en esa frecuencia. Una natural aparece en casi
     todas las RPM -> ridge alto; un orden aparece en una sola RPM -> ridge bajo.

Sobre cada metrica se detectan picos con findpeaks (prominencia relativa al
maximo, igual que el .m: RMS 5 %, Kurtosis 10 %, Ridge 20 %).

Salida (POR CONDICION, todo en la carpeta --outdir):
  * metricas_<condicion>.txt : frecuencias detectadas por cada metodo.
  * metricas_<condicion>.png : las 3 metricas con sus picos + la cascada con las
    lineas de las frecuencias detectadas (estilo del .m).

Condicion = rep · iso · dsb · sensor (una cascada por sensor, como el
Cascade_P1Y del .m). Necesita >= 2 RPM por condicion. Funciona igual sobre el
full spectrum complejo (frecuencias negativas incluidas).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import kurtosis as sp_kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deteccion_picos import (  # noqa: E402
    cargar_espectros, filtrar_condiciones, parse_niveles,
)

# Fracciones de prominencia POCO severas (el .m usaba 0.05/0.10/0.20; se bajan
# para dejar pasar picos poco perceptibles). Configurables por CLI.
FRAC_RMS = 0.015
FRAC_KURT = 0.03
FRAC_RIDGE = 0.05
PROM_RIDGE_FILA = 0.04   # MinPeakProminence por fila para el ridge (mas bajo = mas crestas)
DIST_RIDGE = 3           # MinPeakDistance (bins) del ridge


# ============================================================
# CASCADA Y METRICAS
# ============================================================

def _cascada(espectros, fmin, fmax):
    """espectros: lista de (rpm, f, amp). Devuelve (grid, rpms, Z[n_rpm, n_freq])."""
    espectros = sorted(espectros, key=lambda e: e[0])
    lo = max(f[0] for _, f, _ in espectros)
    hi = min(f[-1] for _, f, _ in espectros)
    if fmin is not None:
        lo = max(lo, fmin)
    if fmax is not None:
        hi = min(hi, fmax)
    dft = float(np.median([np.median(np.diff(f)) for _, f, _ in espectros if len(f) > 1]))
    grid = np.arange(lo, hi, dft)
    Z = np.vstack([np.interp(grid, f, a) for _, f, a in espectros])
    rpms = np.array([r for r, _, _ in espectros], dtype=float)
    return grid, rpms, Z


def _metricas(Z):
    rms = np.sqrt(np.mean(Z ** 2, axis=0))
    kurt = sp_kurtosis(Z, axis=0, fisher=False, bias=True)   # Pearson, como MATLAB
    kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)
    ridge_map = np.zeros_like(Z)
    for i, fila in enumerate(Z):
        mx = float(np.max(fila))
        if mx <= 0:
            continue
        loc, _ = find_peaks(fila, prominence=mx * PROM_RIDGE_FILA, distance=DIST_RIDGE)
        ridge_map[i, loc] = 1
    ridge = ridge_map.sum(axis=0)
    return rms, kurt, ridge


def _picos(v, frac):
    v = np.asarray(v, dtype=float)
    if v.size == 0 or not np.isfinite(v).any() or np.nanmax(v) <= 0:
        return np.array([], dtype=int)
    loc, _ = find_peaks(v, prominence=float(np.nanmax(v)) * frac)
    return loc


# ============================================================
# SALIDAS POR CONDICION
# ============================================================

def escribir_txt(salida: Path, etiqueta, rpms, grid, res):
    lineas = [
        f"# Metricas RMS/Kurtosis/Ridge sobre cascada — {etiqueta}",
        f"# n_rpm={len(rpms)}  rpms={','.join(f'{r:g}' for r in rpms)}",
        f"# n_frecuencias={len(grid)}  df={np.median(np.diff(grid)):.4f} Hz",
        "# ------------------------------------------------------------",
        "metodo\tfrecuencia_Hz",
    ]
    for metodo in ("RMS", "Kurtosis", "Ridge"):
        for fr in res[metodo]:
            lineas.append(f"{metodo}\t{fr:.4f}")
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def graficar(salida: Path, etiqueta, grid, rpms, Z, met, res):
    rms, kurt, ridge = met
    fig, axes = plt.subplots(4, 1, figsize=(13, 12))
    fig.suptitle(f"Frecuencias naturales por metricas de cascada — {etiqueta}", fontsize=13)

    for ax, (nombre, v, color) in zip(
            axes[:3],
            [("RMS por frecuencia", rms, "steelblue"),
             ("Kurtosis por frecuencia", kurt, "seagreen"),
             ("Ridge strength (persistencia de crestas)", ridge, "purple")]):
        ax.plot(grid, v, lw=1.2, color=color)
        loc = res["_loc"][nombre.split()[0]]   # "RMS" | "Kurtosis" | "Ridge"
        if len(loc):
            ax.plot(grid[loc], v[loc], "o", color="red", ms=6)
            for L in loc:
                ax.annotate(f"{grid[L]:.1f}", (grid[L], v[L]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=7, color="darkred")
        ax.set_title(nombre)
        ax.set_ylabel("valor")
        ax.grid(True, alpha=0.3)

    # 4) cascada (freq x rpm) + lineas de las frecuencias detectadas por ridge
    axc = axes[3]
    if len(rpms) >= 2:
        pcm = axc.pcolormesh(grid, rpms, Z, shading="auto", cmap="turbo")
        fig.colorbar(pcm, ax=axc, label="amplitud")
    for fr in res["Ridge"]:
        axc.axvline(fr, color="cyan", lw=1.4, alpha=0.9)
    for fr in res["RMS"]:
        axc.axvline(fr, color="red", lw=0.9, ls="--", alpha=0.7)
    for fr in res["Kurtosis"]:
        axc.axvline(fr, color="lime", lw=0.9, ls=":", alpha=0.7)
    axc.set_title("Cascada + frecuencias detectadas (cyan=Ridge, rojo=RMS, verde=Kurtosis)")
    axc.set_xlabel("Frecuencia (Hz)")
    axc.set_ylabel("RPM")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Deteccion de naturales por metricas de cascada (RMS/Kurtosis/Ridge).")
    p.add_argument("--entrada", default="resultados_fft.txt")
    p.add_argument("--outdir", default="")
    p.add_argument("--rep", default="")
    p.add_argument("--iso", default="")
    p.add_argument("--dsb", default="")
    p.add_argument("--rpm", default="")
    p.add_argument("--sensor", default="")
    p.add_argument("--fmin", type=float, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--frac-rms", dest="frac_rms", type=float, default=FRAC_RMS)
    p.add_argument("--frac-kurt", dest="frac_kurt", type=float, default=FRAC_KURT)
    p.add_argument("--frac-ridge", dest="frac_ridge", type=float, default=FRAC_RIDGE)
    args, _ = p.parse_known_args()   # ignora argumentos extra reenviados por el comparador

    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_file():
        print(f"ERROR: no existe la entrada {entrada}")
        return 1
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else entrada.parent
    outdir.mkdir(parents=True, exist_ok=True)

    arr = cargar_espectros(entrada)
    arr = filtrar_condiciones(
        arr, parse_niveles(args.rep, int), parse_niveles(args.iso, int),
        parse_niveles(args.dsb, int), parse_niveles(args.rpm, float),
        parse_niveles(args.sensor, str))
    if len(arr) == 0:
        print("Ninguna condicion pasa el filtro indicado.")
        return 1

    rep = arr["rep"].astype(int)
    iso = arr["iso"].astype(int)
    dsb = arr["dsb"].astype(int)
    rpm = arr["rpm_nominal"].astype(float)
    sensor = np.asarray(arr["sensor"]).astype(str)
    freq = arr["frecuencia_Hz"].astype(float)
    amp = arr["amplitud"].astype(float)

    # espectros individuales por (rep,iso,dsb,rpm,sensor)
    idx = defaultdict(list)
    for i, c in enumerate(zip(rep, iso, dsb, rpm, sensor)):
        idx[c].append(i)
    espectros = {}
    for clave, ii in idx.items():
        sub = np.array(ii)
        f = freq[sub]
        o = np.argsort(f)
        espectros[clave] = (f[o], amp[sub][o])

    # agrupar por condicion (rep,iso,dsb,sensor); cada grupo = varias RPM
    grupos = defaultdict(list)
    for (r, i_, d, v, s), (f, a) in espectros.items():
        grupos[(r, i_, d, s)].append((v, f, a))

    n_ok = 0
    for (r, i_, d, s), lista in sorted(grupos.items()):
        etiqueta = f"rep{r}_iso{i_}_dsb{d}_{s}"
        if len(lista) < 2:
            print(f"  [aviso] {etiqueta}: solo {len(lista)} RPM (se necesitan >=2), se omite")
            continue
        grid, rpms, Z = _cascada(lista, args.fmin, args.fmax)
        rms, kurt, ridge = _metricas(Z)
        loc_rms = _picos(rms, args.frac_rms)
        loc_kurt = _picos(kurt, args.frac_kurt)
        loc_ridge = _picos(ridge, args.frac_ridge)
        res = {
            "RMS": grid[loc_rms], "Kurtosis": grid[loc_kurt], "Ridge": grid[loc_ridge],
            "_loc": {"RMS": loc_rms, "Kurtosis": loc_kurt, "Ridge": loc_ridge},
        }
        escribir_txt(outdir / f"metricas_{etiqueta}.txt", etiqueta, rpms, grid, res)
        graficar(outdir / f"metricas_{etiqueta}.png", etiqueta, grid, rpms, Z,
                 (rms, kurt, ridge), res)
        print(f"  OK {etiqueta} ({len(rpms)} RPM): "
              f"RMS={np.round(res['RMS'],1).tolist()} "
              f"Kurt={np.round(res['Kurtosis'],1).tolist()} "
              f"Ridge={np.round(res['Ridge'],1).tolist()}")
        n_ok += 1

    print(f"\nListo. {n_ok} condicion(es) analizadas. Resultados en: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
