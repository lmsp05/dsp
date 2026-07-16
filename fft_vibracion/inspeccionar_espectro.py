"""
inspeccionar_espectro.py
========================

Inspeccion visual del proceso de deteccion de picos.

Se filtran las condiciones por NIVELES de cada propiedad (repeticion, viscosidad
ISO, desbalance, RPM nominal y sensor), indicados como listas separadas por comas
(p. ej. --rep 1,3 --iso 32,68); una propiedad sin niveles admite cualquier valor.
Se genera UNA figura por cada combinacion que pasa el filtro, con 3 paneles:

  1. Espectro ORIGINAL (amplitud vs frecuencia). Se marca cada pico SINCRONO
     detectado (punta dentro del margen de un orden entero) y sus HOMBROS (primer
     minimo local a cada lado): el intervalo entre hombros es lo que se elimina.
  2. Espectro DESPUES de eliminar el 1X y sus armonicos por el metodo de hombros
     (recorte del pico completo entre sus hombros + interpolacion de la base).
  3. Espectro limpio con los PICOS DETECTADOS marcados y las METRICAS de
     deteccion: prominencia minima (umbral), distancia minima, y la prominencia
     real de cada pico dibujada como una barra vertical.

Reutiliza exactamente el mismo pipeline que ``deteccion_picos.py`` (funcion
``analizar_espectro``), asi que lo que se ve es lo que ese script realmente hace.

Uso:
    # una sola condicion
    python inspeccionar_espectro.py --entrada resultados_fft.txt \
        --rep 1 --iso 46 --dsb 1 --rpm 600 --sensor P1.Y
    # varios niveles -> una figura por combinacion (rep{1,3} x iso{32,68} x ...)
    python inspeccionar_espectro.py --entrada resultados_fft.txt \
        --rep 1,3 --iso 32,68 --sensor P1.Y

Si ningun filtro coincide, el script lista las condiciones disponibles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

from deteccion_picos import (
    cargar_espectros, analizar_espectro, parse_niveles, filtrar_condiciones,
    FRACCION_PROMINENCIA, DIST_BINS, N_ARMONICOS, TOL_ORDEN, BAND_1X, ANCHO_BINS_NOTCH,
)


def _seleccionar(arr, rep, iso, dsb, rpm, sensor):
    """Devuelve (freq, amp, rpm_medida) de la condicion elegida, o None."""
    m = ((arr["rep"].astype(int) == rep) & (arr["iso"].astype(int) == iso) &
         (arr["dsb"].astype(int) == dsb) &
         (np.abs(arr["rpm_nominal"].astype(float) - rpm) < 1e-6) &
         (np.asarray(arr["sensor"]).astype(str) == sensor))
    if not m.any():
        return None
    f = arr["frecuencia_Hz"].astype(float)[m]
    a = arr["amplitud"].astype(float)[m]
    if "rpm_medida" in arr.dtype.names:
        rpm_med = float(np.median(arr["rpm_medida"].astype(float)[m]))
    else:
        rpm_med = float(rpm)
    return f, a, rpm_med


def _listar_disponibles(arr):
    combos = sorted({(int(r), int(i), int(d), float(v), str(s)) for r, i, d, v, s in zip(
        arr["rep"], arr["iso"], arr["dsb"], arr["rpm_nominal"], arr["sensor"])})
    print("Combinaciones disponibles (rep, iso, dsb, rpm, sensor):")
    for c in combos:
        print(f"  rep{c[0]}  iso{c[1]}  dsb{c[2]}  rpm{c[3]:g}  {c[4]}")


def graficar(det, info, salida: Path):
    f, a, a_det, sinc = det["f"], det["a"], det["a_det"], det["sinc"]
    f1, prom, dist = det["f1"], det["prominencia"], det["dist_bins"]
    pidx, pf, pa = det["picos_idx"], det["picos_freq"], det["picos_amp"]
    intervalos = det.get("sinc_intervalos", [])

    df = float(np.median(np.diff(f))) if len(f) > 1 else 0.0
    ymax = float(a.max()) * 1.1

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    titulo = (f"rep{info['rep']} · iso{info['iso']} · dsb{info['dsb']} · "
              f"rpm{info['rpm']:g} · {info['sensor']}")
    fig.suptitle(f"Inspeccion de deteccion de picos — {titulo}", fontsize=13)

    # ---- sombreado de bandas sincronas (bloques contiguos de la mascara) ----
    def sombrear(ax):
        if not sinc.any():
            return
        bordes = np.diff(sinc.astype(int))
        ini = list(np.where(bordes == 1)[0] + 1)
        fin = list(np.where(bordes == -1)[0] + 1)
        if sinc[0]:
            ini = [0] + ini
        if sinc[-1]:
            fin = fin + [len(f)]
        for i0, i1 in zip(ini, fin):
            ax.axvspan(f[i0], f[min(i1, len(f) - 1)], color="red", alpha=0.12,
                       label="_nolegend_")

    # ---- Panel 1: espectro original ----
    ax1.plot(f, a, color="steelblue", lw=1.0)
    sombrear(ax1)
    if f1:
        for n in range(1, 100):
            if n * f1 > f[-1]:
                break
            ax1.axvline(n * f1, color="purple", ls=":", lw=0.8, alpha=0.5)
        ax1.axvline(f1, color="purple", ls="--", lw=1.2, label=f"1X ≈ {f1:.2f} Hz")
    # puntas sincronas detectadas y sus hombros (lo que define el intervalo a quitar)
    if intervalos:
        pf_s = [f[p] for (_, p, _, _) in intervalos]
        pa_s = [a[p] for (_, p, _, _) in intervalos]
        hx = [f[l] for (l, _, _, _) in intervalos] + [f[r] for (_, _, r, _) in intervalos]
        hy = [a[l] for (l, _, _, _) in intervalos] + [a[r] for (_, _, r, _) in intervalos]
        ax1.plot(pf_s, pa_s, "v", color="purple", ms=7, label="punta síncrona")
        ax1.plot(hx, hy, "|", color="black", ms=10, mew=1.5, label="hombros")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("Amplitud")
    ax1.set_title("1) Espectro original  (▽ punta síncrona · | hombros · banda roja = intervalo eliminado)")
    ax1.set_ylim(0, ymax)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: espectro sin sincronos ----
    ax2.plot(f, a_det, color="seagreen", lw=1.0)
    sombrear(ax2)
    ax2.set_ylabel("Amplitud")
    ax2.set_title("2) Espectro tras eliminar 1X y armónicos (recorte por hombros + interpolación)")
    ax2.set_ylim(0, ymax)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: espectro limpio + picos + metricas ----
    ax3.plot(f, a_det, color="seagreen", lw=1.0, alpha=0.7)

    # todos los maximos locales candidatos y su prominencia (gris)
    cand, _ = find_peaks(a_det)
    if len(cand):
        prom_cand = peak_prominences(a_det, cand)[0]
        ax3.plot(f[cand], a_det[cand], "x", color="gray", ms=4, alpha=0.5,
                 label="máximos locales")

    # picos aceptados (rojo) con barra de prominencia
    if len(pidx):
        prom_pf = peak_prominences(a_det, pidx)[0]
        for x, y, pr in zip(pf, pa, prom_pf):
            ax3.vlines(x, y - pr, y, color="red", lw=1.5, alpha=0.7)
            ax3.annotate(f"{x:.2f} Hz", (x, y), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8, color="darkred")
        ax3.plot(pf, pa, "o", color="red", ms=7, label="picos detectados")

    # umbral de prominencia como referencia (altura del umbral desde 0)
    ax3.axhline(prom, color="orange", ls="--", lw=1.2,
                label=f"prominencia mín = {prom:.4g}")
    ax3.set_xlabel("Frecuencia (Hz)")
    ax3.set_ylabel("Amplitud")
    ax3.set_title("3) Picos detectados y métricas (escala propia del espectro limpio)")
    # escala propia: el espectro limpio es mucho menor que el original (sin 1X)
    ymax3 = float(a_det.max()) * 1.35 if a_det.size and a_det.max() > 0 else ymax
    ax3.set_ylim(0, ymax3)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # caja de metricas
    dist_hz = dist * df
    linea_f1 = f"1X estimado = {f1:.3f} Hz\n" if f1 else "1X estimado = (no)\n"
    txt = (
        f"MÉTRICAS DE DETECCIÓN\n"
        f"fracción prominencia = {info['fraccion']:g}\n"
        f"prominencia mínima = {prom:.4g}\n"
        f"  (= máx · fracción)\n"
        f"distancia mínima = {dist} bins ({dist_hz:.2f} Hz)\n"
        + linea_f1 +
        f"nº armónicos quitados = {info['n_armonicos']}\n"
        f"piso notch (hombros) = {info['ancho_bins']:g} bins\n"
        f"df = {df:.3f} Hz\n"
        f"nº picos detectados = {len(pf)}"
    )
    ax3.text(1.012, 1.0, txt, transform=ax3.transAxes, va="top", ha="left",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", fc="whitesmoke", ec="gray"))

    fig.tight_layout(rect=[0, 0, 0.83, 0.97])
    fig.savefig(salida, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(args) -> int:
    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_file():
        print(f"ERROR: no existe la entrada {entrada}")
        return 1
    arr = cargar_espectros(entrada)

    # filtro por niveles: solo las combinaciones de los niveles indicados
    sub = filtrar_condiciones(
        arr, parse_niveles(args.rep, int), parse_niveles(args.iso, int),
        parse_niveles(args.dsb, int), parse_niveles(args.rpm, float),
        parse_niveles(args.sensor, str))
    if len(sub) == 0:
        print("Ninguna condicion pasa el filtro indicado (rep/iso/dsb/rpm/sensor).\n")
        _listar_disponibles(arr)
        return 1

    # enumera todas las combinaciones (rep, iso, dsb, rpm, sensor) que pasan
    combos = sorted({(int(r), int(i), int(d), float(v), str(s)) for r, i, d, v, s in zip(
        sub["rep"], sub["iso"], sub["dsb"], sub["rpm_nominal"], sub["sensor"])})
    if args.limit:
        combos = combos[:args.limit]

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else entrada.parent
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Condiciones que pasan el filtro: {len(combos)}")

    n_ok = 0
    for rep, iso, dsb, rpm, sensor in combos:
        sel = _seleccionar(sub, rep, iso, dsb, rpm, sensor)
        if sel is None:
            continue
        f, a, rpm_med = sel
        det = analizar_espectro(
            f, a, args.fraccion, args.dist_bins, args.fmin, args.fmax,
            rpm=rpm_med, quitar_armonicos=not args.conservar_armonicos,
            n_armonicos=args.n_armonicos, tol_orden=args.tol_orden,
            band_1x=args.band_1x, ancho_bins=args.ancho_bins)

        nombre = (f"inspeccion_rep{rep}_iso{iso}_dsb{dsb}_rpm{rpm:g}_{sensor}.png")
        salida = outdir / nombre
        info = {"rep": rep, "iso": iso, "dsb": dsb, "rpm": rpm, "sensor": sensor,
                "fraccion": args.fraccion, "n_armonicos": args.n_armonicos,
                "ancho_bins": args.ancho_bins}
        graficar(det, info, salida)
        picos = ", ".join(f"{w:.2f}" for w in det["picos_freq"]) or "(ninguno)"
        f1txt = f"{det['f1']:.3f}" if det["f1"] else "n/a"
        print(f"  OK rep{rep}_iso{iso}_dsb{dsb}_rpm{rpm:g}_{sensor}: "
              f"1X={f1txt} Hz | picos={picos} -> {salida.name}")
        n_ok += 1

    print(f"\nListo. {n_ok} figuras en: {outdir}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspeccion visual de la deteccion de picos. Filtra por niveles "
                    "(listas separadas por comas) y genera una figura por condicion.")
    p.add_argument("--entrada", default="resultados_fft.txt",
                   help="Archivo .txt o .npy de procesar_fft.py")
    # filtros por niveles (listas separadas por comas; vacio = cualquiera)
    p.add_argument("--rep", default="", help="Niveles de repeticion, ej. 1,3 (vacio = todos)")
    p.add_argument("--iso", default="", help="Niveles de viscosidad ISO, ej. 32,68 (vacio = todos)")
    p.add_argument("--dsb", default="", help="Niveles de desbalance, ej. 1,2 (vacio = todos)")
    p.add_argument("--rpm", default="", help="RPM nominales, ej. 600,900 (vacio = todas)")
    p.add_argument("--sensor", default="", help="Sensores, ej. P1.Y,P2.X (vacio = todos)")
    p.add_argument("--outdir", default="", help="Carpeta de salida (def: la de la entrada)")
    p.add_argument("--limit", type=int, default=0, help="Maximo de figuras a generar (0 = sin limite)")
    # parametros de deteccion (mismos que deteccion_picos.py)
    p.add_argument("--fraccion", type=float, default=FRACCION_PROMINENCIA)
    p.add_argument("--dist-bins", dest="dist_bins", type=int, default=DIST_BINS)
    p.add_argument("--fmin", type=float, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--conservar-armonicos", dest="conservar_armonicos", action="store_true")
    p.add_argument("--n-armonicos", dest="n_armonicos", type=int, default=N_ARMONICOS)
    p.add_argument("--tol-orden", dest="tol_orden", type=float, default=TOL_ORDEN)
    p.add_argument("--band-1x", dest="band_1x", type=float, default=BAND_1X)
    p.add_argument("--ancho-bins", dest="ancho_bins", type=float, default=ANCHO_BINS_NOTCH)
    return p


if __name__ == "__main__":
    sys.exit(main(construir_parser().parse_args()))
