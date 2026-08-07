"""
_nucleo_vs.py
=============

Nucleo compartido por las variantes de deteccion de picos (v1..v4) de la carpeta
``deteccion_picos_vs``. NO se ejecuta solo: cada script-version lo configura con
un ``Config`` y llama a ``ejecutar(cfg)``.

Idea comun a todas las versiones:

  1. Se toman los espectros individuales (una combinacion rep/iso/dsb/rpm/sensor).
  2. Opcionalmente se eliminan el 1X y sus armonicos de CADA espectro (con la
     misma logica de ``deteccion_picos.py``: ``analizar_espectro`` por hombros).
  3. Se PROMEDIAN los espectros del grupo indicado por ``cfg.nivel`` (interpolando
     a una grilla de frecuencia comun, porque distintas RPM tienen distinto df).
  4. Se detectan los picos sobre el espectro PROMEDIO (prominencia relativa a su
     maximo + distancia minima), y se genera la tabla .txt y la grafica.

Lo unico que cambia entre versiones es: sobre que se promedia (``nivel``), si se
eliminan o no los sincronos (``quitar``) y el tipo de grafica (``plot``).
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

# Permite importar deteccion_picos.py de la carpeta padre (fft_vibracion).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deteccion_picos import (  # noqa: E402
    cargar_espectros, filtrar_condiciones, parse_niveles, analizar_espectro,
    _dibujar_scatter, COLORES_ISO, FORMAS_DSB,
    FRACCION_PROMINENCIA, DIST_BINS, N_ARMONICOS, TOL_ORDEN, BAND_1X, ANCHO_BINS_NOTCH,
)


@dataclass
class Config:
    nivel: tuple      # claves de agrupamiento, p.ej. ("rep","iso","dsb","rpm")
    quitar: bool      # eliminar 1X + armonicos ANTES de promediar
    plot: str         # "campbell" | "condicion" | "espectro"
    etiqueta: str     # titulo legible de la variante


# ============================================================
# ESPECTROS INDIVIDUALES (limpios o crudos)
# ============================================================

def _individuales(arr, quitar, args) -> list[dict]:
    rep = arr["rep"].astype(int)
    iso = arr["iso"].astype(int)
    dsb = arr["dsb"].astype(int)
    rpm = arr["rpm_nominal"].astype(float)
    sensor = np.asarray(arr["sensor"]).astype(str)
    freq = arr["frecuencia_Hz"].astype(float)
    amp = arr["amplitud"].astype(float)
    tiene_med = "rpm_medida" in arr.dtype.names
    rpm_med = arr["rpm_medida"].astype(float) if tiene_med else rpm

    claves = list(zip(rep, iso, dsb, rpm, sensor))
    idx = defaultdict(list)
    for i, c in enumerate(claves):
        idx[c].append(i)

    out = []
    for clave, ii in idx.items():
        sub = np.array(ii)
        f = freq[sub]
        a = amp[sub]
        o = np.argsort(f)
        f, a = f[o], a[o]
        rm = float(np.median(rpm_med[sub])) if tiene_med else float(clave[3])
        if quitar:
            det = analizar_espectro(
                f, a, args.fraccion, args.dist_bins, args.fmin, args.fmax,
                rpm=rm, quitar_armonicos=True, n_armonicos=args.n_armonicos,
                tol_orden=args.tol_orden, band_1x=args.band_1x, ancho_bins=args.ancho_bins)
            fc, ac = det["f"], det["a_det"]
        else:
            m = np.ones(len(f), dtype=bool)
            if args.fmin is not None:
                m &= f >= args.fmin
            if args.fmax is not None:
                m &= f <= args.fmax
            fc, ac = f[m], a[m]
        out.append({"rep": int(clave[0]), "iso": int(clave[1]), "dsb": int(clave[2]),
                    "rpm": float(clave[3]), "sensor": str(clave[4]), "f": fc, "amp": ac})
    return out


# ============================================================
# PROMEDIADO SOBRE UNA GRILLA COMUN Y DETECCION
# ============================================================

def _promediar(miembros, fmin, fmax):
    """Promedia varios espectros (f, amp) interpolandolos a una grilla comun."""
    lo = max(f[0] for f, _ in miembros)
    hi = min(f[-1] for f, _ in miembros)
    if fmin is not None:
        lo = max(lo, fmin)
    if fmax is not None:
        hi = min(hi, fmax)
    dft = float(np.median([np.median(np.diff(f)) for f, _ in miembros if len(f) > 1]))
    if not np.isfinite(dft) or dft <= 0 or hi <= lo:
        f0, a0 = miembros[0]
        return f0, a0
    grid = np.arange(lo, hi, dft)
    M = np.vstack([np.interp(grid, f, a) for f, a in miembros])
    return grid, M.mean(axis=0)


def _agrupar_y_detectar(individuales, nivel, args) -> list[dict]:
    grupos = defaultdict(list)
    for e in individuales:
        clave = tuple(e[k] for k in nivel)
        grupos[clave].append(e)

    filas = []
    for clave, miembros in grupos.items():
        grid, avg = _promediar([(e["f"], e["amp"]) for e in miembros], args.fmin, args.fmax)
        det = analizar_espectro(grid, avg, args.fraccion, args.dist_bins,
                                None, None, quitar_armonicos=False)
        fila = {k: v for k, v in zip(nivel, clave)}
        etq = "·".join(f"{k}{fila[k]:g}" if isinstance(fila[k], float) else f"{k}{fila[k]}"
                       for k in nivel) or "promedio global"
        fila.setdefault("iso", miembros[0]["iso"])
        fila.setdefault("dsb", miembros[0]["dsb"])
        fila.update(omegas=det["picos_freq"], f=grid, amp=avg, n=len(miembros), _etq=etq)
        filas.append(fila)
    filas.sort(key=lambda d: tuple(d.get(k, 0) for k in ("rep", "iso", "dsb", "rpm")))
    return filas


# ============================================================
# TABLA DE SALIDA
# ============================================================

def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def escribir_tabla(filas, nivel, salida: Path) -> int:
    nmax = max((len(f["omegas"]) for f in filas), default=0)
    cols = list(nivel) + [f"omega{k+1}" for k in range(nmax)]
    lineas = ["\t".join(cols) if cols else "omega (ninguna)"]
    for f in filas:
        celdas = [_fmt(f[k]) for k in nivel] + [f"{w:.4f}" for w in f["omegas"]]
        celdas += [""] * (nmax - len(f["omegas"]))
        lineas.append("\t".join(celdas))
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return nmax


# ============================================================
# GRAFICAS
# ============================================================

def _leyendas_iso_dsb(ax, filas):
    isos = sorted({f["iso"] for f in filas})
    dsbs = sorted({f["dsb"] for f in filas})
    leg_iso = [Line2D([0], [0], marker="o", ls="", mfc=COLORES_ISO.get(i, "gray"),
                      mec="k", ms=8, label=f"ISO {i}") for i in isos]
    leg_dsb = [Line2D([0], [0], marker=FORMAS_DSB.get(d, "x"), ls="", mfc="gray",
                      mec="k", ms=8, label=f"dsb {d}") for d in dsbs]
    l1 = ax.legend(handles=leg_iso, title="Viscosidad", loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=leg_dsb, title="Desbalance", loc="upper right", fontsize=8)


def _plot_condicion(filas, ax, titulo):
    filas = sorted(filas, key=lambda f: (f["rep"], f["iso"], f["dsb"]))
    labels = [f"r{f['rep']}·i{f['iso']}·d{f['dsb']}" for f in filas]
    ymax = 0.0
    ymin = 0.0
    for x, f in enumerate(filas):
        if len(f["omegas"]) == 0:
            continue
        ax.scatter([x] * len(f["omegas"]), f["omegas"],
                   c=COLORES_ISO.get(f["iso"], "gray"), marker=FORMAS_DSB.get(f["dsb"], "x"),
                   s=45, alpha=0.75, edgecolors="k", linewidths=0.4)
        ymax = max(ymax, float(np.max(f["omegas"])))
        ymin = min(ymin, float(np.min(f["omegas"])))
    ax.set_xticks(range(len(filas)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    if ymin < 0:
        ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    if ymax > 0 or ymin < 0:
        ax.set_ylim(ymin * 1.08 if ymin < 0 else 0, ymax * 1.08)
    ax.set_xlabel("Condicion (rep·iso·dsb)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    _leyendas_iso_dsb(ax, filas)


def _plot_espectro(filas, ax, titulo):
    for f in filas:
        ax.plot(f["f"], f["amp"], lw=1.0, alpha=0.85, label=f.get("_etq", "promedio"))
        if len(f["omegas"]):
            yy = [f["amp"][int(np.argmin(np.abs(f["f"] - w)))] for w in f["omegas"]]
            ax.plot(f["omegas"], yy, "o", color="red", ms=6)
            for w, y in zip(f["omegas"], yy):
                ax.annotate(f"{w:.1f}", (w, y), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=7, color="darkred")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Amplitud (promedio)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    if len(filas) <= 12:
        ax.legend(fontsize=7, loc="upper right")


def graficar(filas, cfg: Config, salida: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    if cfg.plot == "campbell":
        _dibujar_scatter(filas, ax, cfg.etiqueta)   # reutiliza el scatter 1X..20X
    elif cfg.plot == "condicion":
        _plot_condicion(filas, ax, cfg.etiqueta)
    else:
        _plot_espectro(filas, ax, cfg.etiqueta)
    fig.tight_layout()
    fig.savefig(salida, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROGRAMA (parametrizado por Config)
# ============================================================

def construir_parser(cfg: Config) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Deteccion de picos — {cfg.etiqueta}")
    p.add_argument("--entrada", default="resultados_fft.txt",
                   help="Archivo .txt o .npy generado por procesar_fft.py")
    p.add_argument("--outdir", default="", help="Carpeta de salida (def: la de la entrada)")
    # filtros por niveles (limitan el conjunto a promediar)
    p.add_argument("--rep", default="", help="Niveles de repeticion, ej. 1,3 (vacio = todos)")
    p.add_argument("--iso", default="", help="Niveles de viscosidad ISO, ej. 32,68")
    p.add_argument("--dsb", default="", help="Niveles de desbalance, ej. 1,2")
    p.add_argument("--rpm", default="", help="RPM nominales, ej. 600,900")
    p.add_argument("--sensor", default="", help="Sensores, ej. P1.Y,P2.X")
    # parametros de deteccion / eliminacion (iguales a deteccion_picos.py)
    p.add_argument("--fraccion", type=float, default=FRACCION_PROMINENCIA)
    p.add_argument("--dist-bins", dest="dist_bins", type=int, default=DIST_BINS)
    p.add_argument("--fmin", type=float, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--conservar-armonicos", dest="conservar_armonicos", action="store_true",
                   help="Fuerza NO eliminar sincronos (solo aplica a las versiones con eliminacion)")
    p.add_argument("--n-armonicos", dest="n_armonicos", type=int, default=N_ARMONICOS)
    p.add_argument("--tol-orden", dest="tol_orden", type=float, default=TOL_ORDEN)
    p.add_argument("--band-1x", dest="band_1x", type=float, default=BAND_1X)
    p.add_argument("--ancho-bins", dest="ancho_bins", type=float, default=ANCHO_BINS_NOTCH)
    return p


def ejecutar(cfg: Config) -> int:
    args = construir_parser(cfg).parse_args()
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

    arr = filtrar_condiciones(
        arr, parse_niveles(args.rep, int), parse_niveles(args.iso, int),
        parse_niveles(args.dsb, int), parse_niveles(args.rpm, float),
        parse_niveles(args.sensor, str))
    if len(arr) == 0:
        print("Ninguna condicion pasa el filtro indicado.")
        return 1

    quitar = cfg.quitar and not args.conservar_armonicos
    individuales = _individuales(arr, quitar, args)
    filas = _agrupar_y_detectar(individuales, cfg.nivel, args)

    tabla = outdir / "picos_detectados.txt"
    n_max = escribir_tabla(filas, cfg.nivel, tabla)
    grafico = outdir / "picos_grafico.png"
    graficar(filas, cfg, grafico)

    total = sum(len(f["omegas"]) for f in filas)
    print(f"[{cfg.etiqueta}]")
    print(f"  eliminacion de 1X/armonicos: {'SI' if quitar else 'NO'}")
    print(f"  espectros promediados en {len(filas)} grupo(s); picos totales: {total} "
          f"(max por grupo: {n_max})")
    print(f"  Tabla:   {tabla}")
    print(f"  Grafica: {grafico}")
    return 0
