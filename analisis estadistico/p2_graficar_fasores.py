"""
PASO 2 (y PASO 4) — Graficas de fasores frente a la velocidad
=============================================================

Genera, a partir de una tabla de fasores, figuras 2x2 con este reparto:

               columna 1 (sensor Y)      columna 2 (sensor X)
    fila 1     amplitud [um]             amplitud [um]
    fila 2     fase [grados]             fase [grados]

Se producen DOS familias de figuras (12 en total):

  A) Una por cada DESBALANCEO x COJINETE  (3 x 2 = 6)
     Cada figura superpone las 3 viscosidades x 3 repeticiones (9 curvas).

  B) Una por cada VISCOSIDAD x COJINETE   (3 x 2 = 6)
     Cada figura superpone los 3 desbalanceos x 3 repeticiones (9 curvas).

Convenios de representacion (constantes en todas las figuras):
  * COLOR   -> viscosidad: ISO 32 rojo, ISO 46 azul, ISO 68 verde.
  * SATURACION/oscuridad del color -> desbalanceo: 1 el mas atenuado, 3 el pleno.
  * ESTILO de linea -> repeticion: 1 continua, 2 discontinua, 3 punteada.
  * Solo linea, sin marcadores. Cada curva une los puntos de una misma
    repeticion al aumentar la velocidad.
  * Eje X: las 15 velocidades EQUIESPACIADAS (posiciones categoricas), no a
    escala numerica de rpm.
  * Eje Y: limites COMUNES a todas las figuras, tomados del maximo y el minimo
    global de toda la tabla (uno para amplitud y otro para fase).

Uso (paso 2, datos sin compensar):
    python p2_graficar_fasores.py --entrada <res>/p1_fasores.txt --salida <res>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
import comun


# ============================================================
# LIMITES GLOBALES
# ============================================================

def _rango(v, margen):
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    d = (hi - lo) or 1.0
    return (lo - margen * d, hi + margen * d)


def limites_globales(df, margen: float = 0.04, desenvolver: bool = False):
    """
    (lim_amp, lim_fase) a partir del maximo y el minimo de TODOS los valores que
    se van a representar.

    Se aplican por igual a las 12 figuras para que sean comparables entre si.
    Si `desenvolver`, los limites de fase se calculan sobre las curvas ya
    desenvueltas, que es lo que realmente se dibuja.
    """
    lim_amp = _rango(df[[f"{s}_amp" for s in config.SENSORES]].to_numpy(dtype=float),
                     margen)

    if not desenvolver:
        return lim_amp, _rango(
            df[[f"{s}_phase" for s in config.SENSORES]].to_numpy(dtype=float), margen)

    velocidades = sorted(int(v) for v in df["Speed"].unique())
    valores = []
    for _, g in df.groupby(["Viscosity", "Unbalance", "Repetition"], sort=False):
        g = g.set_index("Speed").reindex(velocidades)
        for s in config.SENSORES:
            valores.append(comun.desenvolver_fase(g[f"{s}_phase"].to_numpy(dtype=float)))
    return lim_amp, _rango(np.concatenate(valores) if valores else [0.0, 1.0], margen)


# ============================================================
# UNA FIGURA
# ============================================================

def _figura(df, cojinete, curvas, titulo, lim_amp, lim_fase, velocidades, ruta,
            desenvolver=False):
    """
    Dibuja y guarda una figura 2x2.

    `curvas` es la lista de (visc, desb, rep) que se superponen.
    Si `desenvolver`, la fase de cada curva se hace continua a lo largo de la
    velocidad (elimina los saltos artificiales de 0 a 360).
    """
    fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE, sharex=True)
    x = np.arange(len(velocidades))            # posiciones equiespaciadas

    for j, direccion in enumerate(config.DIRECCIONES):      # columna: Y, X
        sensor = f"{cojinete}{direccion}"
        for fila, magnitud in enumerate(["amp", "phase"]):  # row: amplitude, phase
            ax = axes[fila, j]
            col = f"{sensor}_{magnitud}"
            for visc, desb, rep in curvas:
                sub = df[(df["Viscosity"] == visc) & (df["Unbalance"] == desb)
                         & (df["Repetition"] == rep)]
                if sub.empty:
                    continue
                sub = sub.set_index("Speed").reindex(velocidades)
                y = sub[col].to_numpy(dtype=float)
                if magnitud == "phase" and desenvolver:
                    y = comun.desenvolver_fase(y)
                ax.plot(x, y,
                        color=comun.color_condicion(visc, desb),
                        linestyle=comun.estilo_repeticion(rep),
                        linewidth=1.6, solid_capstyle="round")

            ax.set_ylim(*(lim_amp if magnitud == "amp" else lim_fase))
            ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6)
            if fila == 0:
                ax.set_title(sensor, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel("1X amplitude [um]" if magnitud == "amp"
                              else "1X phase [deg]", fontsize=11)
            if fila == 1:
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in velocidades],
                                   rotation=60, fontsize=8)
                ax.set_xlabel("Speed [rpm]  (evenly spaced scale)", fontsize=11)

    axes[1, 0].set_xlim(x[0], x[-1])

    # ---- leyenda ----
    handles = [Line2D([0], [0], color=comun.color_condicion(v, d),
                      linestyle=comun.estilo_repeticion(r), linewidth=2,
                      label=f"ISO {v} · U{d} · rep {r}")
               for v, d, r in curvas]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(5, max(1, len(handles))), fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.085, 1, 0.96])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ============================================================
# LAS 12 FIGURAS
# ============================================================

def generar_figuras(df, destino: Path, etiqueta: str,
                    lim_amp=None, lim_fase=None, desenvolver: bool = False):
    """
    Genera las 12 figuras en `destino`. `etiqueta` se anade al titulo
    ('sin compensacion de runout' / 'con compensacion de runout').

    Si no se pasan limites, se calculan de `df`.
    """
    if lim_amp is None or lim_fase is None:
        lim_amp, lim_fase = limites_globales(df, desenvolver=desenvolver)

    velocidades = sorted(int(v) for v in df["Speed"].unique())
    viscosidades = sorted(int(v) for v in df["Viscosity"].unique())
    desbalanceos = sorted(int(v) for v in df["Unbalance"].unique())
    repeticiones = sorted(int(v) for v in df["Repetition"].unique())

    destino.mkdir(parents=True, exist_ok=True)
    generadas = []

    # ---- A) una figura por desbalanceo x cojinete: compara viscosidades y reps ----
    for desb in desbalanceos:
        curvas = [(v, desb, r) for v in viscosidades for r in repeticiones]
        for coj in config.COJINETES:
            ruta = destino / f"unbalance{desb}_{coj}.png"
            generadas.append(_figura(
                df, coj, curvas,
                f"Bearing {coj} — unbalance level {desb} — {etiqueta}\n"
                f"comparison across viscosities and repetitions",
                lim_amp, lim_fase, velocidades, ruta, desenvolver))

    # ---- B) una figura por viscosidad x cojinete: compara desbalanceos y reps ----
    for visc in viscosidades:
        curvas = [(visc, d, r) for d in desbalanceos for r in repeticiones]
        for coj in config.COJINETES:
            ruta = destino / f"viscosity{visc}_{coj}.png"
            generadas.append(_figura(
                df, coj, curvas,
                f"Bearing {coj} — ISO {visc} oil — {etiqueta}\n"
                f"comparison across unbalance levels and repetitions",
                lim_amp, lim_fase, velocidades, ruta, desenvolver))

    return generadas, lim_amp, lim_fase


# ============================================================
# PROGRAMA
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Paso 2: figuras 2x2 de amplitud y fase frente a la velocidad.")
    p.add_argument("--entrada", default="",
                   help="Tabla de fasores (def: <salida>/p1_fasores.txt)")
    p.add_argument("--salida", default=config.DIR_RESULTADOS,
                   help="Carpeta de resultados")
    p.add_argument("--subcarpeta", default=config.NOMBRE_DIR_SIN_COMP,
                   help="Subcarpeta donde escribir las figuras")
    p.add_argument("--etiqueta", default="without runout compensation",
                   help="Texto que se anade al titulo de cada figura")
    p.add_argument("--desenvolver", action="store_true",
                   help="Dibuja la fase de forma continua a lo largo de la "
                        "velocidad (quita los saltos artificiales 0<->360)")
    args = p.parse_args()

    salida = Path(args.salida).expanduser()
    entrada = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.ARCHIVO_FASORES
    if not entrada.is_file():
        print(f"ERROR: no existe la tabla de fasores {entrada}\n"
              f"       Ejecuta antes p1_extraer_fasores.py")
        return 1

    df = comun.leer_tabla(entrada)
    destino = salida / args.subcarpeta
    generadas, lim_amp, lim_fase = generar_figuras(
        df, destino, args.etiqueta, desenvolver=args.desenvolver)

    print(f"Table   : {entrada}  ({len(df)} rows)")
    if args.desenvolver:
        print("Phase drawn continuously (--desenvolver).")
    print(f"Shared Y range -> amplitude [{lim_amp[0]:.2f}, {lim_amp[1]:.2f}] um  |  "
          f"phase [{lim_fase[0]:.1f}, {lim_fase[1]:.1f}] deg")
    print(f"\n{len(generadas)} figures in: {destino}")
    for r in generadas:
        print(f"  {r.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
