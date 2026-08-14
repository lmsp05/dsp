"""
PASO 3 — Compensacion de runout (resta vectorial del fasor de slow roll)
========================================================================

Toma la tabla de fasores del paso 1 y le resta VECTORIALMENTE el fasor de slow
roll medido a baja velocidad (600 rpm por defecto):

    z_compensado(rpm) = z_medido(rpm) - z_slowroll

El fasor de slow roll se obtiene promediando, para cada sensor, los fasores a
600 rpm de todas las demas condiciones. Con --modo se elige el agrupamiento:

  * viscosidad (por defecto) : un slow roll por viscosidad y sensor
                               (promedia las 3 repeticiones x 3 desbalanceos)
  * global                   : un unico slow roll por sensor
  * rep_viscosidad           : un slow roll por repeticion x viscosidad y sensor

--------------------------------------------------------------------------
EL PROBLEMA DEL ANGULO Y COMO SE RESUELVE
--------------------------------------------------------------------------
Las fases vienen envueltas en [0, 360). Dos medidas casi iguales pueden salir
como 359 y 1 grados, y su media aritmetica daria 180: exactamente lo contrario
del valor correcto (0). El promedio aritmetico de angulos NO ES VALIDO.

Aqui se promedia SIEMPRE de forma vectorial: se pasa cada medida a su fasor
complejo z = A*exp(i*phi), se promedian parte real e imaginaria, y del promedio
se recuperan amplitud y fase. Como el angulo nunca se manipula como escalar, el
envolvimiento deja de existir: el metodo es exacto por construccion y no
necesita "arreglar" nada a posteriori.

El script informa ademas, para cada grupo, la media INGENUA de la fase (media
aritmetica de los valores 0-360) junto a la correcta, para que se vea cuando y
cuanto se equivocaria el metodo ingenuo. Para las series en las que si interesa
un angulo continuo (graficas, derivas) se usa `comun.desenvolver_fase`.

Salidas en --salida:
  * p3_fasores_compensados.txt   tabla con los fasores ya compensados
  * p3_vectores_slow_roll.txt    los vectores de slow roll y sus diagnosticos
  * p3_slow_roll.png             diagramas polares de los fasores a 600 rpm

Uso:
    python p3_compensar_runout.py --salida <carpeta_resultados>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import comun


MODOS = {
    "viscosidad": ["Viscosity"],
    "global": [],
    "rep_viscosidad": ["Repetition", "Viscosity"],
}


# ============================================================
# VECTORES DE SLOW ROLL
# ============================================================

def calcular_slow_roll(df, claves, rpm_sr):
    """
    Promedia vectorialmente los fasores a `rpm_sr` para cada grupo de `claves`.

    Devuelve (vectores, diagnostico):
      vectores[(clave..., sensor)] = complejo
      diagnostico = lista de dicts con el detalle de cada grupo
    """
    base = df[df["Speed"] == rpm_sr]
    if base.empty:
        raise SystemExit(f"ERROR: no hay filas a {rpm_sr} rpm para el slow roll.")

    vectores, diag = {}, []
    grupos = [((), base)] if not claves else list(base.groupby(claves, sort=True))

    for clave, g in grupos:
        clave = (clave,) if not isinstance(clave, tuple) else clave
        for sensor in config.SENSORES:
            amp = g[f"{sensor}_amp"].to_numpy(dtype=float)
            fase = g[f"{sensor}_phase"].to_numpy(dtype=float)
            ok = np.isfinite(amp) & np.isfinite(fase)
            amp, fase = amp[ok], fase[ok]
            if amp.size == 0:
                continue

            z = comun.a_fasor(amp, fase)
            z_medio = z.mean()
            vectores[clave + (sensor,)] = z_medio

            a_sr, f_sr = comun.de_fasor(z_medio)
            _, R = comun.promedio_escalar_fase(fase)
            # Media aritmetica ingenua de la fase, solo para comparar.
            f_ingenua = float(np.mean(fase))
            # Dispersion de los fasores individuales alrededor de su promedio.
            disp = float(np.mean(np.abs(z - z_medio)))

            diag.append({
                **{k: v for k, v in zip(claves, clave)},
                "Sensor": sensor, "n": int(amp.size),
                "SR_amp_um": float(a_sr), "SR_fase_deg": float(f_sr),
                "fase_media_ingenua_deg": f_ingenua,
                "error_metodo_ingenuo_deg": abs(comun.diferencia_angular(f_ingenua, f_sr)),
                "concentracion_R": R,
                "dispersion_intra_um": disp,
                "dispersion_intra_rel": disp / max(abs(z_medio), 1e-12),
            })
    return vectores, pd.DataFrame(diag)


def comparar_viscosidades(df, rpm_sr):
    """
    ¿Tiene sentido que el slow roll dependa de la viscosidad?

    Compara, por sensor, la dispersion DENTRO de cada viscosidad (ruido de
    medida entre repeticiones y desbalanceos) con la separacion ENTRE las
    medias de las tres viscosidades. Si la separacion entre viscosidades no
    supera claramente al ruido interno, la diferencia observada no es real y
    conviene usar --modo global.
    """
    base = df[df["Speed"] == rpm_sr]
    filas = []
    for sensor in config.SENSORES:
        medias, intra = {}, []
        for visc, g in base.groupby("Viscosity"):
            z = comun.a_fasor(g[f"{sensor}_amp"], g[f"{sensor}_phase"])
            z = z[np.isfinite(z)]
            if z.size == 0:
                continue
            medias[int(visc)] = z.mean()
            intra.append(np.mean(np.abs(z - z.mean())))
        if len(medias) < 2:
            continue
        vs = list(medias.values())
        entre = max(abs(a - b) for i, a in enumerate(vs) for b in vs[i + 1:])
        d_intra = float(np.mean(intra))
        filas.append({
            "Sensor": sensor,
            "separacion_entre_viscosidades_um": float(entre),
            "dispersion_dentro_viscosidad_um": d_intra,
            "razon_entre_dentro": float(entre / max(d_intra, 1e-12)),
        })
    return pd.DataFrame(filas)


# ============================================================
# COMPENSACION
# ============================================================

def compensar(df, vectores, claves):
    """Resta a cada fila el vector de slow roll de su grupo."""
    salida = df.copy()
    n_sin = 0
    for sensor in config.SENSORES:
        ca, cf = f"{sensor}_amp", f"{sensor}_phase"
        z = comun.a_fasor(df[ca], df[cf])
        clave_filas = (list(zip(*[df[k] for k in claves])) if claves
                       else [()] * len(df))
        sr = np.empty(len(df), dtype=complex)
        for i, cl in enumerate(clave_filas):
            cl = tuple(int(x) for x in cl)
            v = vectores.get(cl + (sensor,))
            if v is None:
                sr[i] = np.nan
                n_sin += 1
            else:
                sr[i] = v
        a, f = comun.de_fasor(z - sr)
        salida[ca], salida[cf] = a, f
    if n_sin:
        print(f"  !! AVISO: {n_sin} valores sin vector de slow roll (quedan NaN)")
    return salida


# ============================================================
# GRAFICA POLAR
# ============================================================

def graficar_slow_roll(df, vectores, claves, rpm_sr, ruta):
    """Diagramas polares de los fasores a `rpm_sr`, coloreados por viscosidad."""
    base = df[df["Speed"] == rpm_sr]
    fig, axes = plt.subplots(1, len(config.SENSORES), figsize=(17, 4.8),
                             subplot_kw={"projection": "polar"})
    for ax, sensor in zip(np.atleast_1d(axes), config.SENSORES):
        for visc, g in base.groupby("Viscosity"):
            z = comun.a_fasor(g[f"{sensor}_amp"], g[f"{sensor}_phase"])
            col = comun.color_condicion(int(visc), 3)
            ax.plot(np.angle(z), np.abs(z), "o", ms=4, color=col, alpha=0.65,
                    label=f"ISO {int(visc)}")
            zm = z.mean()
            ax.annotate("", xy=(np.angle(zm), abs(zm)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.set_title(sensor, fontsize=11, fontweight="bold", pad=12)
        ax.set_theta_zero_location("E")
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=7)

    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"Fasores a {rpm_sr} rpm (slow roll) — puntos = condiciones "
                 f"individuales, flechas = promedio vectorial por viscosidad",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ============================================================
# PROGRAMA
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Paso 3: compensacion de runout por resta vectorial del slow roll.")
    p.add_argument("--entrada", default="", help="def: <salida>/p1_fasores.txt")
    p.add_argument("--salida", default=config.DIR_RESULTADOS)
    p.add_argument("--modo", default="viscosidad", choices=sorted(MODOS),
                   help="Agrupamiento del vector de slow roll (def: viscosidad)")
    p.add_argument("--rpm-slow-roll", dest="rpm_sr", type=int,
                   default=config.RPM_SLOW_ROLL)
    args = p.parse_args()

    salida = Path(args.salida).expanduser()
    entrada = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.ARCHIVO_FASORES
    if not entrada.is_file():
        print(f"ERROR: no existe {entrada}. Ejecuta antes p1_extraer_fasores.py")
        return 1

    df = comun.leer_tabla(entrada)
    claves = MODOS[args.modo]

    vectores, diag = calcular_slow_roll(df, claves, args.rpm_sr)
    comp = compensar(df, vectores, claves)

    comun.escribir_tabla(comp, salida / config.ARCHIVO_FASORES_COMP)
    comun.escribir_tabla(diag, salida / config.ARCHIVO_SLOW_ROLL)
    ruta_png = graficar_slow_roll(df, vectores, claves, args.rpm_sr,
                                  salida / "p3_slow_roll.png")

    # ---- informe ----
    print(f"Entrada: {entrada}  ({len(df)} filas)")
    print(f"Slow roll a {args.rpm_sr} rpm, modo '{args.modo}' "
          f"({len(vectores)} vectores)\n")

    cols = [c for c in ["Viscosity", "Repetition"] if c in diag.columns]
    print("Vectores de slow roll:")
    print(diag[cols + ["Sensor", "n", "SR_amp_um", "SR_fase_deg",
                       "fase_media_ingenua_deg", "error_metodo_ingenuo_deg",
                       "concentracion_R"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    peor = diag["error_metodo_ingenuo_deg"].max()
    print(f"\nMedia vectorial vs media aritmetica de la fase: discrepancia maxima "
          f"{peor:.2f} grados.")
    if peor > 1.0:
        print("  -> El promedio aritmetico de angulos SI habria fallado en algun grupo"
              " (envolvimiento 0/360). El promedio vectorial usado aqui lo evita.")
    else:
        print("  -> En estos datos ningun grupo cruza el 0/360, pero el promedio"
              " vectorial sigue siendo el correcto en general.")

    bajo = diag[diag["concentracion_R"] < 0.9]
    if len(bajo):
        print(f"\n!! AVISO: {len(bajo)} grupo(s) con fases poco agrupadas (R < 0.9);"
              " su fase promedio es poco representativa.")

    # ---- ¿depende el slow roll de la viscosidad? ----
    cmp_v = comparar_viscosidades(df, args.rpm_sr)
    print("\n¿Es real la dependencia del slow roll con la viscosidad?")
    if cmp_v.empty:
        print("  (no evaluable: hacen falta al menos 2 viscosidades en los datos)")
    else:
        print(cmp_v.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
        r = cmp_v["razon_entre_dentro"]
        print(f"\n  razon (separacion entre viscosidades / dispersion interna): "
              f"min {r.min():.2f}, max {r.max():.2f}")
        if r.max() < 2:
            print("  -> La separacion NO supera al ruido interno: la diferencia por"
                  " viscosidad no esta respaldada. Considera --modo global.")
        else:
            print("  -> La separacion supera al ruido interno en algun sensor: la"
                  " diferencia por viscosidad es medible (ver la nota del README:"
                  " a 600 rpm no hay solo runout, tambien hay respuesta sincrona).")

    print(f"\nFasores compensados : {salida / config.ARCHIVO_FASORES_COMP}")
    print(f"Vectores slow roll  : {salida / config.ARCHIVO_SLOW_ROLL}")
    print(f"Diagrama polar      : {ruta_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
