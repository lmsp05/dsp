"""
metricas_versiones.py
====================

Aplica las tres metricas de deteccion de naturales (RMS, Kurtosis, Ridge del
``x0302_nf_gph_transformations.m``) en las MISMAS 5 configuraciones de
promediado/combinacion que las variantes v0..v4, seleccionadas con ``--modo``.

Las metricas agregan un CONJUNTO de espectros. En cada modo, el conjunto es lo
que esa version combina, y el resultado se agrupa asi:

    modo  conjunto de la metrica          se detecta por            grafica
    ----  ------------------------------  ------------------------  -------------------
    v0    las RPM (cascada)               (condicion, sensor)       cuadricula 5x4 / cond
    v1    los 4 sensores                  (condicion, velocidad)    Campbell (freq vs RPM)
    v2    sensores + velocidades          (condicion)               freq vs condicion
    v3    sensores + velocidades          (condicion)               freq vs condicion
    v4    todos los espectros             (global)                  curva metrica vs freq

Eliminacion de 1X/armonicos ANTES de agregar: v0, v1, v2 -> SI ; v3, v4 -> NO.

Cada metrica se calcula sobre la matriz Z (filas = miembros del conjunto,
columnas = frecuencia comun) y sus picos se detectan con findpeaks y las mismas
fracciones de prominencia del .m (RMS 5 %, Kurtosis 10 %, Ridge 20 %).

Salida en ``--outdir`` (por version): un .txt con las frecuencias detectadas por
cada metrica y grupo, y las graficas (una por metrica + una combinada; en v0 una
cuadricula por condicion). Funciona igual sobre el full spectrum complejo.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.stats import kurtosis as sp_kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deteccion_picos import (  # noqa: E402
    cargar_espectros, filtrar_condiciones, parse_niveles, analizar_espectro,
    _dibujar_scatter, COLORES_ISO, FORMAS_DSB,
    FRACCION_PROMINENCIA, DIST_BINS, N_ARMONICOS, TOL_ORDEN, BAND_1X, ANCHO_BINS_NOTCH,
)
from _nucleo_vs import _plot_condicion  # noqa: E402

# Fracciones de prominencia POCO severas (dejan pasar picos poco perceptibles).
# El .m usaba 0.05 / 0.10 / 0.20; aqui se bajan bastante. Configurables por CLI.
FRAC = {"RMS": 0.015, "Kurtosis": 0.03, "Ridge": 0.05}
PROM_RIDGE_FILA = 0.04   # prominencia por fila del ridge (mas baja = mas crestas)
DIST_RIDGE = 3
COL_MET = {"RMS": "tab:blue", "Kurtosis": "tab:green", "Ridge": "purple"}
MRK_MET = {"RMS": "o", "Kurtosis": "s", "Ridge": "^"}
METRICAS = ("RMS", "Kurtosis", "Ridge")


@dataclass
class ConfigM:
    modo: str
    out_key: tuple      # agrupamiento de la deteccion
    quitar: bool        # eliminar 1X/armonicos antes de agregar
    plot: str           # "campbell" | "condicion" | "curva" | "v0grid"
    etiqueta: str


CONFIGS = {
    "v0": ConfigM("v0", ("rep", "iso", "dsb", "sensor"), True, "v0grid",
                  "v0 · metrica sobre RPM (por condicion y sensor)"),
    "v1": ConfigM("v1", ("rep", "iso", "dsb", "rpm"), True, "campbell",
                  "v1 · metrica sobre sensores (por condicion y velocidad)"),
    "v2": ConfigM("v2", ("rep", "iso", "dsb"), True, "condicion",
                  "v2 · metrica sobre sensores+velocidades (por condicion)"),
    "v3": ConfigM("v3", ("rep", "iso", "dsb"), False, "condicion",
                  "v3 · metrica sobre sensores+velocidades SIN eliminacion (por condicion)"),
    "v4": ConfigM("v4", (), False, "curva",
                  "v4 · metrica sobre todos los espectros (global) SIN eliminacion"),
    "v5": ConfigM("v5", ("rep", "iso", "dsb", "sensor"), False, "v0grid",
                  "v5 · metrica sobre RPM SIN eliminacion (por condicion y sensor)"),
}


# ============================================================
# ESPECTROS INDIVIDUALES + METRICAS
# ============================================================

def _individuales(arr, quitar, args):
    rep = arr["rep"].astype(int); iso = arr["iso"].astype(int); dsb = arr["dsb"].astype(int)
    rpm = arr["rpm_nominal"].astype(float); sensor = np.asarray(arr["sensor"]).astype(str)
    freq = arr["frecuencia_Hz"].astype(float); amp = arr["amplitud"].astype(float)
    tiene_med = "rpm_medida" in arr.dtype.names
    rpm_med = arr["rpm_medida"].astype(float) if tiene_med else rpm

    idx = defaultdict(list)
    for i, c in enumerate(zip(rep, iso, dsb, rpm, sensor)):
        idx[c].append(i)
    out = []
    for clave, ii in idx.items():
        sub = np.array(ii)
        f = freq[sub]; a = amp[sub]
        o = np.argsort(f); f, a = f[o], a[o]
        rm = float(np.median(rpm_med[sub])) if tiene_med else float(clave[3])
        det = analizar_espectro(f, a, args.fraccion, args.dist_bins, args.fmin, args.fmax,
                                rpm=rm, quitar_armonicos=quitar, n_armonicos=args.n_armonicos,
                                tol_orden=args.tol_orden, band_1x=args.band_1x,
                                ancho_bins=args.ancho_bins)
        out.append({"rep": int(clave[0]), "iso": int(clave[1]), "dsb": int(clave[2]),
                    "rpm": float(clave[3]), "sensor": str(clave[4]),
                    "f": det["f"], "amp": det["a_det"], "picos": det["picos_freq"]})
    return out


def _cascada(miembros, fmin, fmax):
    lo = max(f[0] for f, _ in miembros); hi = min(f[-1] for f, _ in miembros)
    if fmin is not None:
        lo = max(lo, fmin)
    if fmax is not None:
        hi = min(hi, fmax)
    dft = float(np.median([np.median(np.diff(f)) for f, _ in miembros if len(f) > 1]))
    if not np.isfinite(dft) or dft <= 0 or hi <= lo:
        return miembros[0][0], np.atleast_2d(miembros[0][1])
    grid = np.arange(lo, hi, dft)
    Z = np.vstack([np.interp(grid, f, a) for f, a in miembros])
    return grid, Z


def _metricas(Z, prom_ridge_fila):
    rms = np.sqrt(np.mean(Z ** 2, axis=0))
    kurt = np.nan_to_num(sp_kurtosis(Z, axis=0, fisher=False, bias=True), nan=0.0,
                         posinf=0.0, neginf=0.0)
    ridge_map = np.zeros_like(Z)
    for i, fila in enumerate(Z):
        mx = float(np.max(fila))
        if mx <= 0:
            continue
        loc, _ = find_peaks(fila, prominence=mx * prom_ridge_fila, distance=DIST_RIDGE)
        ridge_map[i, loc] = 1
    ridge = ridge_map.sum(axis=0)
    return {"RMS": rms, "Kurtosis": kurt, "Ridge": ridge}


def _picos(grid, curva, frac):
    curva = np.asarray(curva, dtype=float)
    if curva.size == 0 or np.nanmax(curva) <= 0:
        return np.array([])
    loc, _ = find_peaks(curva, prominence=float(np.nanmax(curva)) * frac)
    return grid[loc]


def _agrupar(individuales, cfg: ConfigM, args):
    fracs = {"RMS": args.frac_rms, "Kurtosis": args.frac_kurt, "Ridge": args.frac_ridge}
    grupos = defaultdict(list)
    for e in individuales:
        grupos[tuple(e[k] for k in cfg.out_key)].append(e)
    filas = []
    for clave, miembros in grupos.items():
        grid, Z = _cascada([(m["f"], m["amp"]) for m in miembros], args.fmin, args.fmax)
        curvas = _metricas(Z, args.prom_ridge_fila)
        picos = {M: _picos(grid, curvas[M], fracs[M]) for M in METRICAS}
        fila = {k: v for k, v in zip(cfg.out_key, clave)}
        fila.setdefault("iso", miembros[0]["iso"])
        fila.setdefault("dsb", miembros[0]["dsb"])
        fila.update(grid=grid, curvas=curvas, picos=picos, miembros=miembros, n=len(miembros))
        filas.append(fila)
    filas.sort(key=lambda d: tuple(d.get(k, 0) for k in ("rep", "iso", "dsb", "rpm", "sensor")))
    return filas


# ============================================================
# TABLA
# ============================================================

def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def escribir_tabla(filas, cfg: ConfigM, salida: Path):
    cols = list(cfg.out_key) + ["metodo", "frecuencia_Hz"]
    lineas = ["\t".join(cols)]
    for f in filas:
        base = [_fmt(f[k]) for k in cfg.out_key]
        for M in METRICAS:
            for fr in f["picos"][M]:
                lineas.append("\t".join(base + [M, f"{fr:.4f}"]))
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ============================================================
# GRAFICAS
# ============================================================

def _guardar(fig, salida):
    fig.tight_layout()
    fig.savefig(salida, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _leyenda_metricas(ax):
    handles = [Line2D([0], [0], marker=MRK_MET[M], ls="", mfc=COL_MET[M], mec="k",
                      ms=8, label=M) for M in METRICAS]
    ax.legend(handles=handles, title="Metrica", loc="upper right", fontsize=8)


def graficar(filas, cfg: ConfigM, outdir: Path):
    generados = []

    if cfg.plot == "campbell":
        for M in METRICAS:
            fig, ax = plt.subplots(figsize=(11, 7))
            pseudo = [{"iso": f["iso"], "dsb": f["dsb"], "rpm": f["rpm"], "omegas": f["picos"][M]}
                      for f in filas]
            _dibujar_scatter(pseudo, ax, f"{cfg.etiqueta} — {M}")
            png = outdir / f"picos_{M}.png"; _guardar(fig, png); generados.append(png)
        # combinado (color/forma por metrica)
        fig, ax = plt.subplots(figsize=(11, 7))
        for M in METRICAS:
            xs, ys = [], []
            for f in filas:
                for fr in f["picos"][M]:
                    xs.append(f["rpm"]); ys.append(fr)
            ax.scatter(xs, ys, c=COL_MET[M], marker=MRK_MET[M], s=45, alpha=0.75,
                       edgecolors="k", linewidths=0.4)
        ax.set_xlabel("Velocidad (RPM)"); ax.set_ylabel("Frecuencia (Hz)")
        ax.set_title(f"{cfg.etiqueta} — combinado"); ax.grid(True, alpha=0.3)
        _leyenda_metricas(ax)
        png = outdir / "picos_combinado.png"; _guardar(fig, png); generados.append(png)

    elif cfg.plot == "condicion":
        for M in METRICAS:
            fig, ax = plt.subplots(figsize=(12, 7))
            pseudo = [{"rep": f["rep"], "iso": f["iso"], "dsb": f["dsb"], "omegas": f["picos"][M]}
                      for f in filas]
            _plot_condicion(pseudo, ax, f"{cfg.etiqueta} — {M}")
            png = outdir / f"picos_{M}.png"; _guardar(fig, png); generados.append(png)
        # combinado
        fig, ax = plt.subplots(figsize=(12, 7))
        fs = sorted(filas, key=lambda f: (f["rep"], f["iso"], f["dsb"]))
        labels = [f"r{f['rep']}·i{f['iso']}·d{f['dsb']}" for f in fs]
        for x, f in enumerate(fs):
            for M in METRICAS:
                if len(f["picos"][M]):
                    ax.scatter([x] * len(f["picos"][M]), f["picos"][M], c=COL_MET[M],
                               marker=MRK_MET[M], s=40, alpha=0.75, edgecolors="k", linewidths=0.4)
        ax.set_xticks(range(len(fs))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_xlabel("Condicion (rep·iso·dsb)"); ax.set_ylabel("Frecuencia (Hz)")
        ax.set_title(f"{cfg.etiqueta} — combinado"); ax.grid(True, alpha=0.3)
        _leyenda_metricas(ax)
        png = outdir / "picos_combinado.png"; _guardar(fig, png); generados.append(png)

    elif cfg.plot == "curva":
        f = filas[0]  # global
        for M in METRICAS:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(f["grid"], f["curvas"][M], lw=1.2, color=COL_MET[M])
            for fr in f["picos"][M]:
                j = int(np.argmin(np.abs(f["grid"] - fr)))
                ax.plot(fr, f["curvas"][M][j], "o", color="red", ms=6)
                ax.annotate(f"{fr:.1f}", (fr, f["curvas"][M][j]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=7, color="darkred")
            ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel(M)
            ax.set_title(f"{cfg.etiqueta} — {M}"); ax.grid(True, alpha=0.3)
            png = outdir / f"picos_{M}.png"; _guardar(fig, png); generados.append(png)
        # combinado: lineas verticales de las frecuencias detectadas por cada metrica
        fig, ax = plt.subplots(figsize=(12, 5))
        for M in METRICAS:
            for fr in f["picos"][M]:
                ax.axvline(fr, color=COL_MET[M], lw=1.3, alpha=0.8)
        ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("frecuencias detectadas")
        ax.set_title(f"{cfg.etiqueta} — combinado"); ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        handles = [Line2D([0], [0], color=COL_MET[M], lw=2, label=M) for M in METRICAS]
        ax.legend(handles=handles, title="Metrica", loc="upper right", fontsize=8)
        png = outdir / "picos_combinado.png"; _guardar(fig, png); generados.append(png)

    elif cfg.plot == "v0grid":
        # una cuadricula 5x4 (metodos x sensores) por condicion (rep,iso,dsb)
        conds = defaultdict(dict)
        for f in filas:
            conds[(f["rep"], f["iso"], f["dsb"])][f["sensor"]] = f
        sensores = sorted({f["sensor"] for f in filas})
        filas_metodo = ["prominencia"] + list(METRICAS) + ["combinado"]
        for (r, i_, d), porsensor in sorted(conds.items()):
            fig, axes = plt.subplots(len(filas_metodo), len(sensores),
                                     figsize=(4.2 * len(sensores), 2.6 * len(filas_metodo)),
                                     squeeze=False)
            fig.suptitle(f"v0 · metricas por sensor — rep{r}·iso{i_}·dsb{d}", fontsize=13)
            for ci, s in enumerate(sensores):
                fdat = porsensor.get(s)
                rpms = [m["rpm"] for m in fdat["miembros"]] if fdat else []
                for ri, metodo in enumerate(filas_metodo):
                    ax = axes[ri][ci]
                    if ci == 0:
                        ax.set_ylabel(f"{metodo}\nFrec (Hz)", fontsize=8)
                    if ri == 0:
                        ax.set_title(s, fontsize=9)
                    if fdat is None:
                        continue
                    if metodo == "prominencia":
                        for m in fdat["miembros"]:
                            ax.scatter([m["rpm"]] * len(m["picos"]), m["picos"], s=18,
                                       c="tab:blue", alpha=0.7, edgecolors="k", linewidths=0.3)
                    elif metodo == "combinado":
                        for M in METRICAS:
                            for fr in fdat["picos"][M]:
                                ax.axhline(fr, color=COL_MET[M], lw=1.1, alpha=0.8)
                    else:
                        for fr in fdat["picos"][metodo]:
                            ax.axhline(fr, color=COL_MET[metodo], lw=1.3, alpha=0.85)
                    if rpms:
                        ax.set_xlim(min(rpms) - 20, max(rpms) + 20)
                    ax.grid(True, alpha=0.3)
                    if ri == len(filas_metodo) - 1:
                        ax.set_xlabel("RPM", fontsize=8)
            png = outdir / f"v0grid_rep{r}_iso{i_}_dsb{d}.png"
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(png, dpi=140, bbox_inches="tight")
            plt.close(fig)
            generados.append(png)

    return generados


# ============================================================
# PROGRAMA
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Metricas RMS/Kurtosis/Ridge aplicadas en las 5 configuraciones v0..v4.")
    p.add_argument("--modo", required=True, choices=list(CONFIGS.keys()))
    p.add_argument("--entrada", default="resultados_fft.txt")
    p.add_argument("--outdir", default="")
    p.add_argument("--rep", default=""); p.add_argument("--iso", default="")
    p.add_argument("--dsb", default=""); p.add_argument("--rpm", default="")
    p.add_argument("--sensor", default="")
    p.add_argument("--fmin", type=float, default=None); p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--fraccion", type=float, default=FRACCION_PROMINENCIA)
    p.add_argument("--dist-bins", dest="dist_bins", type=int, default=DIST_BINS)
    p.add_argument("--n-armonicos", dest="n_armonicos", type=int, default=N_ARMONICOS)
    p.add_argument("--tol-orden", dest="tol_orden", type=float, default=TOL_ORDEN)
    p.add_argument("--band-1x", dest="band_1x", type=float, default=BAND_1X)
    p.add_argument("--ancho-bins", dest="ancho_bins", type=float, default=ANCHO_BINS_NOTCH)
    # criterios de las metricas (mas bajos = menos severos, dejan pasar picos debiles)
    p.add_argument("--frac-rms", dest="frac_rms", type=float, default=FRAC["RMS"])
    p.add_argument("--frac-kurt", dest="frac_kurt", type=float, default=FRAC["Kurtosis"])
    p.add_argument("--frac-ridge", dest="frac_ridge", type=float, default=FRAC["Ridge"])
    p.add_argument("--prom-ridge-fila", dest="prom_ridge_fila", type=float, default=PROM_RIDGE_FILA)
    args, _ = p.parse_known_args()

    cfg = CONFIGS[args.modo]
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

    individuales = _individuales(arr, cfg.quitar, args)
    filas = _agrupar(individuales, cfg, args)
    if not filas:
        print("No se formo ningun grupo.")
        return 1

    escribir_tabla(filas, cfg, outdir / "picos_detectados.txt")
    generados = graficar(filas, cfg, outdir)

    print(f"[{cfg.etiqueta}]")
    print(f"  eliminacion de 1X/armonicos: {'SI' if cfg.quitar else 'NO'}")
    print(f"  grupos: {len(filas)} | miembros/grupo (medio): "
          f"{np.mean([f['n'] for f in filas]):.1f}")
    for M in METRICAS:
        tot = sum(len(f['picos'][M]) for f in filas)
        print(f"  {M}: {tot} picos")
    print(f"  Tabla: {outdir / 'picos_detectados.txt'}")
    for g in generados:
        print(f"  Grafica: {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
