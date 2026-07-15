"""
revision_rpms.py
================

Herramienta de DIAGNOSTICO de la RPM y de la segmentacion por estacionariedad,
para entender por que la RPM promedio medida se aparta de la RPM nominal.

Para cada archivo de la carpeta genera una figura que muestra:

  * La RPM instantanea (pulso a pulso) frente al tiempo.
  * La RPM media por bloque de `bloque_s` segundos (la señal que alimenta la
    segmentacion) y los puntos donde el cambio supera el umbral y separa tramos,
    con el % de cambio anotado.
  * Todos los tramos candidatos, sombreados y etiquetados como ACEPTADO (verde),
    RECHAZADO (rojo, con el motivo) y el tramo finalmente SELECCIONADO (marco
    dorado), con su duracion, media, mediana y coeficiente de variacion.
  * La distribucion de la RPM instantanea (histograma) con las metricas de
    dispersion, resaltando la diferencia MEDIA vs MEDIANA y los outliers.

Ademas escribe un CSV (`revision_rpms_metricas.csv`) con las metricas por tramo
de todos los archivos, para revisarlas juntas.

--------------------------------------------------------------------------
POR QUE MEDIA vs MEDIANA IMPORTA
--------------------------------------------------------------------------
`procesar_fft.py` reporta la RPM promedio como la MEDIA de la RPM instantanea.
La media se dispara con muy pocos pulsos espurios o perdidos del keyphasor: un
solo intervalo con un pulso doble da una RPM instantanea de decenas de miles de
RPM, que arrastra la media aunque el 99.9 % de los pulsos esten perfectos. La
MEDIANA (y metricas robustas como el MAD) practicamente no se ven afectadas.
Por eso este diagnostico reporta ambas: si media y mediana difieren mucho, el
problema es de deteccion de pulsos (outliers), no de la velocidad real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reutiliza el pipeline real para que la revision vea lo mismo que el analisis.
from procesar_fft import (
    FS, ANALISIS_BLOQUE_S, UMBRAL_RPM_PCT, MIN_DURACION_TRAMO_S,
    leer_archivo, detectar_pulsos, rpm_instantanea,
    segmentar_estacionario, seleccionar_tramo,
    descubrir_archivos, ArchivoInfo,
)


# ============================================================
# METRICAS DE DISPERSION
# ============================================================

def metricas(rpm: np.ndarray) -> dict:
    rpm = np.asarray(rpm, dtype=float)
    if rpm.size == 0:
        return {}
    media = float(rpm.mean())
    mediana = float(np.median(rpm))
    std = float(rpm.std())
    mad = float(np.median(np.abs(rpm - mediana)))
    # Umbral de outlier robusto (MAD escalado a sigma-equivalente).
    sigma_rob = 1.4826 * mad
    if sigma_rob > 0:
        n_out = int(np.sum(np.abs(rpm - mediana) > 3 * sigma_rob))
    else:
        n_out = 0
    return {
        "n": int(rpm.size),
        "media": media,
        "mediana": mediana,
        "std": std,
        "cv_pct": 100 * std / media if media else float("nan"),
        "min": float(rpm.min()),
        "max": float(rpm.max()),
        "rango": float(rpm.max() - rpm.min()),
        "p5": float(np.percentile(rpm, 5)),
        "p95": float(np.percentile(rpm, 95)),
        "iqr": float(np.percentile(rpm, 75) - np.percentile(rpm, 25)),
        "mad": mad,
        "n_outliers": n_out,
        "pct_outliers": 100 * n_out / rpm.size,
    }


# ============================================================
# SEGMENTACION DETALLADA (incluye los tramos RECHAZADOS)
# ============================================================

def bloques_rpm(t_pulso, rpm, bloque_s):
    """RPM media por bloque temporal (misma cuenta que segmentar_estacionario)."""
    tmax = float(t_pulso.max())
    bordes = np.arange(0, tmax + bloque_s, bloque_s)
    rpm_b = []
    for i in range(len(bordes) - 1):
        m = (t_pulso >= bordes[i]) & (t_pulso < bordes[i + 1])
        rpm_b.append(np.mean(rpm[m]) if m.sum() > 0 else np.nan)
    return bordes, np.array(rpm_b), tmax


def grupos_detallados(t_pulso, rpm, bordes, rpm_b, tmax, umbral_pct, min_dur_s):
    """Replica el agrupamiento de segmentar_estacionario pero CONSERVA los grupos
    rechazados, anotando el motivo. Devuelve (grupos, splits)."""

    def cerrar(a, b):
        t0 = float(bordes[a])
        t1 = float(min(bordes[b], tmax))
        dur = t1 - t0
        m = (t_pulso >= t0) & (t_pulso < t1)
        n = int(m.sum())
        g = {"t0": t0, "t1": t1, "dur": dur, "n": n}
        if n < 2:
            g.update(aceptado=False, motivo="rechazado: <2 pulsos",
                     media=np.nan, mediana=np.nan, std=np.nan, cv=np.nan)
            return g
        r = rpm[m]
        aceptado = dur >= min_dur_s
        g.update(
            aceptado=aceptado,
            motivo="aceptado" if aceptado else f"rechazado: dur {dur:.1f}s < {min_dur_s:g}s",
            media=float(r.mean()), mediana=float(np.median(r)),
            std=float(r.std()), cv=float(100 * r.std() / r.mean()) if r.mean() else np.nan,
        )
        return g

    grupos, splits = [], []
    inicio = 0
    for i in range(1, len(rpm_b)):
        if np.isnan(rpm_b[i]) or np.isnan(rpm_b[inicio]):
            continue
        cambio = 100 * abs(rpm_b[i] - rpm_b[inicio]) / rpm_b[inicio]
        if cambio > umbral_pct:
            grupos.append(cerrar(inicio, i))
            splits.append((float(bordes[i]), cambio))
            inicio = i
    grupos.append(cerrar(inicio, len(rpm_b)))
    return grupos, splits


def _mismo_tramo(g, tr, eps=1e-6):
    return abs(g["t0"] - tr.t0) < eps and abs(g["t1"] - tr.t1) < eps


# ============================================================
# FIGURA POR ARCHIVO
# ============================================================

def graficar(info: ArchivoInfo, t_pulso, rpm, bordes, rpm_b, grupos, splits,
             seleccionado, met_global, salida_png):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.4]})

    etiqueta = f"rep{info.rep}_iso{info.iso}_dsb{info.dsb}_rpm{info.rpm:g}"

    # ---- Panel superior: RPM vs tiempo ----
    ax1.plot(t_pulso, rpm, ".", ms=2.5, alpha=0.35, color="steelblue",
             label="RPM instantanea (pulso a pulso)")

    # RPM media por bloque (escalones centrados en cada bloque)
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    ax1.step(centros, rpm_b, where="mid", color="darkorange", lw=1.6,
             label=f"RPM media por bloque ({ANALISIS_BLOQUE_S:g}s)")

    # Lineas de referencia: nominal y mediana global
    ax1.axhline(info.rpm, color="black", ls="--", lw=1.2,
                label=f"RPM nominal = {info.rpm:g}")
    ax1.axhline(met_global["mediana"], color="green", ls=":", lw=1.4,
                label=f"mediana global = {met_global['mediana']:.1f}")
    ax1.axhline(met_global["media"], color="red", ls=":", lw=1.2,
                label=f"media global = {met_global['media']:.1f}")

    # Sombreado de tramos candidatos
    for g in grupos:
        if g["aceptado"]:
            color, alpha = ("gold" if seleccionado and _mismo_tramo(g, seleccionado) else "green"), 0.14
        else:
            color, alpha = "red", 0.10
        ax1.axvspan(g["t0"], g["t1"], color=color, alpha=alpha)
        # marco distintivo para el seleccionado
        if seleccionado and _mismo_tramo(g, seleccionado):
            for x in (g["t0"], g["t1"]):
                ax1.axvline(x, color="goldenrod", lw=2.0)

        if np.isnan(g["media"]):
            txt = f"{g['motivo']}\ndur={g['dur']:.1f}s"
        else:
            sel = "  [SELECCIONADO]" if (seleccionado and _mismo_tramo(g, seleccionado)) else ""
            txt = (f"{g['motivo']}{sel}\n"
                   f"dur={g['dur']:.1f}s\n"
                   f"media={g['media']:.1f}\nmediana={g['mediana']:.1f}\n"
                   f"cv={g['cv']:.2f}%")
        ax1.text(0.5 * (g["t0"] + g["t1"]), 0.02, txt, transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.5,
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    # Puntos de separacion (splits) con el % de cambio
    for ts, cambio in splits:
        ax1.axvline(ts, color="purple", ls="--", lw=1.0, alpha=0.7)
        ax1.text(ts, 0.98, f"split {cambio:.1f}%\n(umbral {UMBRAL_RPM_PCT:g}%)",
                 transform=ax1.get_xaxis_transform(), rotation=90,
                 ha="right", va="top", fontsize=7, color="purple")

    d_media = met_global["media"] - info.rpm
    d_mediana = met_global["mediana"] - info.rpm
    ax1.set_title(
        f"{etiqueta}   |   Δmedia vs nominal = {d_media:+.1f} RPM   "
        f"Δmediana vs nominal = {d_mediana:+.1f} RPM   "
        f"(media-mediana = {met_global['media']-met_global['mediana']:+.1f})")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("RPM")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # ---- Panel inferior: distribucion de la RPM instantanea ----
    # recorta el eje para que los outliers extremos no aplasten el histograma
    lo = met_global["mediana"] - max(5 * met_global["mad"], 3 * met_global["std"], 15)
    hi = met_global["mediana"] + max(5 * met_global["mad"], 3 * met_global["std"], 15)
    ax2.hist(np.clip(rpm, lo, hi), bins=80, color="steelblue", alpha=0.7)
    ax2.axvline(met_global["mediana"], color="green", ls=":", lw=1.6, label="mediana")
    ax2.axvline(met_global["media"], color="red", ls=":", lw=1.4, label="media")
    ax2.axvline(info.rpm, color="black", ls="--", lw=1.2, label="nominal")
    ax2.set_xlim(lo, hi)
    ax2.set_xlabel("RPM instantanea (recortada al rango central)")
    ax2.set_ylabel("cuentas")
    ax2.legend(loc="upper right", fontsize=8)

    txt = (
        f"n pulsos = {met_global['n']}\n"
        f"media = {met_global['media']:.2f}\n"
        f"mediana = {met_global['mediana']:.2f}\n"
        f"std = {met_global['std']:.2f}   cv = {met_global['cv_pct']:.2f}%\n"
        f"MAD = {met_global['mad']:.3f}\n"
        f"IQR = {met_global['iqr']:.2f}\n"
        f"min/max = {met_global['min']:.1f} / {met_global['max']:.1f}\n"
        f"p5/p95 = {met_global['p5']:.1f} / {met_global['p95']:.1f}\n"
        f"outliers (>3·MADn) = {met_global['n_outliers']} "
        f"({met_global['pct_outliers']:.3f}%)"
    )
    ax2.text(1.01, 1.0, txt, transform=ax2.transAxes, va="top", ha="left",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", fc="whitesmoke", ec="gray"))

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig(salida_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CSV DE METRICAS
# ============================================================

CSV_COLS = [
    "rep", "iso", "dsb", "rpm_nominal", "tramo_idx", "t0_s", "t1_s", "dur_s",
    "aceptado", "seleccionado", "motivo", "n_pulsos",
    "media", "mediana", "std", "cv_pct", "min", "max", "mad", "n_outliers",
]


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def procesar(args) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: la carpeta de datos no existe: {data_dir}")
        return 1

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    archivos = descubrir_archivos(data_dir)
    if args.limit:
        archivos = archivos[:args.limit]
    if not archivos:
        print(f"No se encontraron archivos validos en {data_dir}")
        return 1

    print(f"Archivos a revisar: {len(archivos)}")
    print(f"Figuras en: {outdir}\n")

    csv_path = outdir / "revision_rpms_metricas.csv"
    filas_csv = ["\t".join(CSV_COLS)]

    n_ok = 0
    for info in archivos:
        etiqueta = f"rep{info.rep}_iso{info.iso}_dsb{info.dsb}_rpm{info.rpm:g}"
        try:
            matriz, canales = leer_archivo(info.path)
            try:
                col_sync = next(i for i, c in enumerate(canales) if c == "sync")
            except StopIteration:
                col_sync = 0
            pulsos = detectar_pulsos(matriz[:, col_sync])
            t_pulso, rpm = rpm_instantanea(pulsos, FS)
        except Exception as e:
            print(f"  [ERROR] {etiqueta}: {e}")
            continue

        met_global = metricas(rpm)

        bordes, rpm_b, tmax = bloques_rpm(t_pulso, rpm, args.bloque_s)
        grupos, splits = grupos_detallados(
            t_pulso, rpm, bordes, rpm_b, tmax, args.umbral_rpm, args.min_dur)
        tramos_acc = segmentar_estacionario(
            t_pulso, rpm, args.bloque_s, args.umbral_rpm, args.min_dur)
        seleccionado = seleccionar_tramo(tramos_acc, info.rpm)

        png = outdir / f"rpm_{etiqueta}.png"
        graficar(info, t_pulso, rpm, bordes, rpm_b, grupos, splits,
                 seleccionado, met_global, png)

        # filas del CSV (una por tramo candidato + una fila 'GLOBAL')
        mg = met_global
        filas_csv.append("\t".join(str(x) for x in [
            info.rep, info.iso, info.dsb, f"{info.rpm:g}", "GLOBAL", 0.0, tmax, tmax,
            "", "", "registro_completo", mg["n"],
            f"{mg['media']:.3f}", f"{mg['mediana']:.3f}", f"{mg['std']:.3f}",
            f"{mg['cv_pct']:.3f}", f"{mg['min']:.2f}", f"{mg['max']:.2f}",
            f"{mg['mad']:.3f}", mg["n_outliers"],
        ]))
        for k, g in enumerate(grupos):
            m = (t_pulso >= g["t0"]) & (t_pulso < g["t1"])
            mt = metricas(rpm[m]) if m.sum() > 0 else {}
            sel = bool(seleccionado and _mismo_tramo(g, seleccionado))
            filas_csv.append("\t".join(str(x) for x in [
                info.rep, info.iso, info.dsb, f"{info.rpm:g}", k,
                f"{g['t0']:.2f}", f"{g['t1']:.2f}", f"{g['dur']:.2f}",
                g["aceptado"], sel, g["motivo"], g["n"],
                f"{g['media']:.3f}" if not np.isnan(g["media"]) else "nan",
                f"{g['mediana']:.3f}" if not np.isnan(g["mediana"]) else "nan",
                f"{g['std']:.3f}" if not np.isnan(g["std"]) else "nan",
                f"{g['cv']:.3f}" if not np.isnan(g["cv"]) else "nan",
                f"{mt.get('min', float('nan')):.2f}", f"{mt.get('max', float('nan')):.2f}",
                f"{mt.get('mad', float('nan')):.3f}", mt.get("n_outliers", 0),
            ]))

        sel_txt = (f"sel=[{seleccionado.t0:.1f},{seleccionado.t1:.1f}]s "
                   f"media={seleccionado.rpm_media:.1f}" if seleccionado else "sin tramo")
        print(f"  OK {etiqueta}: media={mg['media']:.1f} mediana={mg['mediana']:.1f} "
              f"(Δnom media={mg['media']-info.rpm:+.1f} mediana={mg['mediana']-info.rpm:+.1f}) "
              f"outliers={mg['n_outliers']} | {sel_txt}")
        n_ok += 1

    csv_path.write_text("\n".join(filas_csv) + "\n", encoding="utf-8")
    print(f"\nListo. {n_ok} archivos revisados.")
    print(f"Metricas por tramo: {csv_path}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnostico de RPM y segmentacion por estacionariedad, con "
                    "una figura por archivo y un CSV de metricas.")
    p.add_argument("--data-dir", default="datos",
                   help="Carpeta raiz con las carpetas repN_isoVG_dsbM (def: ./datos)")
    p.add_argument("--outdir", default="revision_rpms",
                   help="Carpeta de salida para las figuras y el CSV")
    p.add_argument("--bloque-s", dest="bloque_s", type=float, default=ANALISIS_BLOQUE_S,
                   help="Bloque temporal para medir estacionariedad [s]")
    p.add_argument("--umbral-rpm", dest="umbral_rpm", type=float, default=UMBRAL_RPM_PCT,
                   help="%% de variacion de RPM que separa tramos")
    p.add_argument("--min-dur", dest="min_dur", type=float, default=MIN_DURACION_TRAMO_S,
                   help="Duracion minima de un tramo aceptado [s]")
    p.add_argument("--limit", type=int, default=0, help="Revisar solo los primeros N archivos")
    return p


if __name__ == "__main__":
    args = construir_parser().parse_args()
    sys.exit(procesar(args))
