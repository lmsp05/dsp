"""
Run steps 1 to 6 in order, each with its default arguments.

    python ejecutar_todo.py --entrada <xlsx_folder> --salida <results_folder>

Useful arguments:
    --formato double          figure width/typography for publication (steps 5-6)
    --paneles-por-sensor      one panel per probe instead of shared axes (5-6)
    --tamano-letra 14         base font size in points (steps 5-6)
    --tamano-titulo 18        title font size, set independently (steps 5-6)
    --cojinete P1             plot only bearing 1 (P1Y, P1X) in steps 5-6
    --eje-y-comun             same Y limits on every panel of a figure (5-6)
    --barras-error total      error bars over all observations instead of
                              between repetitions (step 5)
    --escala-p lineal         raw p value instead of -log10(p) in the
                              partition figures (step 5)
    --ymax-p 0.1              zoom the linear p axis (step 5)
    --modo-runout global      slow-roll vector grouping (step 3)
    --desenvolver             continuous phase in the phasor figures (steps 2, 4)
    --grupos-velocidad        also run the ANOVA per speed band (step 5)
    --grupos-desbalanceo      also run it per unbalance level (step 5)
    --alfa 0.01               significance threshold (step 6)
    --desde 3                 resume from a given step (earlier ones are skipped)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

AQUI = Path(os.path.dirname(os.path.abspath(__file__)))


def correr(script: str, argumentos: list[str]) -> int:
    print("\n" + "=" * 78)
    print(f"  {script}  {' '.join(argumentos)}")
    print("=" * 78)
    return subprocess.call([sys.executable, str(AQUI / script)] + argumentos)


def main() -> int:
    p = argparse.ArgumentParser(description="Run steps 1..6 of the analysis.")
    p.add_argument("--entrada", default=config.DIR_EXCEL, help="Folder holding the .xlsx files")
    p.add_argument("--salida", default=config.DIR_RESULTADOS, help="Results folder")
    p.add_argument("--tipo", default="p", choices=["p", "ac"])
    p.add_argument("--modo-runout", dest="modo_runout", default="viscosidad",
                   choices=["viscosidad", "global", "rep_viscosidad"])
    p.add_argument("--respuesta", default="amp", choices=["amp", "phase"])
    p.add_argument("--desenvolver", action="store_true")
    p.add_argument("--grupos-velocidad", dest="grupos", action="store_true")
    p.add_argument("--grupos-desbalanceo", dest="grupos_desb", action="store_true")
    p.add_argument("--alfa", type=float, default=0.05,
                   help="Significance threshold for the step 6 figures")
    p.add_argument("--formato", default=config.FORMATO_POR_DEFECTO,
                   choices=sorted(config.FORMATOS_FIGURA),
                   help="Figure size/typography preset for steps 5 and 6")
    p.add_argument("--paneles-por-sensor", dest="paneles", action="store_true",
                   help="One panel per probe in the steps 5 and 6 figures")
    p.add_argument("--tamano-letra", dest="tam_letra", type=float, default=None,
                   help="Base font size for the steps 5 and 6 figures")
    p.add_argument("--tamano-titulo", dest="tam_titulo", type=float, default=None,
                   help="Title font size for the steps 5 and 6 figures")
    p.add_argument("--cojinete", default="todos", choices=["todos", "P1", "P2"],
                   help="Probes plotted in steps 5 and 6")
    p.add_argument("--eje-y-comun", dest="eje_y", action="store_true",
                   help="All panels of a figure share the same Y limits (5-6)")
    p.add_argument("--barras-error", dest="barras_error", default="repeticiones",
                   choices=["repeticiones", "total"],
                   help="What the error bars of step 5 show")
    p.add_argument("--escala-p", dest="escala_p", default="log",
                   choices=["log", "lineal"],
                   help="p axis of the step 5 partition figures")
    p.add_argument("--ymax-p", dest="ymax_p", type=float, default=1.0,
                   help="Upper bound of the linear p axis (step 5)")
    p.add_argument("--desde", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    args = p.parse_args()

    sal = ["--salida", args.salida]
    desenv = ["--desenvolver"] if args.desenvolver else []
    fig = ["--formato", args.formato, "--cojinete", args.cojinete]
    if args.paneles:
        fig.append("--paneles-por-sensor")
    if args.eje_y:
        fig.append("--eje-y-comun")
    if args.tam_letra:
        fig += ["--tamano-letra", str(args.tam_letra)]
    if args.tam_titulo:
        fig += ["--tamano-titulo", str(args.tam_titulo)]

    pasos = [
        (1, "p1_extraer_fasores.py", ["--entrada", args.entrada, "--tipo", args.tipo] + sal),
        (2, "p2_graficar_fasores.py", sal + desenv),
        (3, "p3_compensar_runout.py", sal + ["--modo", args.modo_runout]),
        (4, "p4_graficar_compensados.py", sal + desenv),
        (5, "p5_anova_split_plot.py", sal + fig
            + ["--respuesta", args.respuesta,
               "--barras-error", args.barras_error,
               "--escala-p", args.escala_p,
               "--ymax-p", str(args.ymax_p)]
            + (["--grupos-velocidad"] if args.grupos else [])
            + (["--grupos-desbalanceo"] if args.grupos_desb else [])),
        (6, "p6_graficar_pvalores.py", sal + fig + ["--alfa", str(args.alfa)]),
    ]

    for numero, script, argumentos in pasos:
        if numero < args.desde:
            continue
        codigo = correr(script, argumentos)
        if codigo != 0:
            print(f"\n!! Step {numero} ({script}) failed with code {codigo}. Stopping.")
            return codigo

    print("\n" + "=" * 78)
    print(f"  Pipeline complete. Results in: {args.salida}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
