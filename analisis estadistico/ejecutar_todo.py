"""
Ejecuta los pasos 1 a 5 en orden, cada uno con sus argumentos por defecto.

    python ejecutar_todo.py --entrada <carpeta_xlsx> --salida <carpeta_resultados>

Argumentos utiles:
    --modo-runout global      agrupamiento del vector de slow roll (paso 3)
    --desenvolver             fase continua en las graficas (pasos 2 y 4)
    --grupos-velocidad        ademas, ANOVA por tramos de rpm (paso 5)
    --desde 3                 reanuda a partir de un paso (no repite los previos)
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
    p = argparse.ArgumentParser(description="Ejecuta los pasos 1..5 del analisis.")
    p.add_argument("--entrada", default=config.DIR_EXCEL, help="Carpeta con los .xlsx")
    p.add_argument("--salida", default=config.DIR_RESULTADOS, help="Carpeta de resultados")
    p.add_argument("--tipo", default="p", choices=["p", "ac"])
    p.add_argument("--modo-runout", dest="modo_runout", default="viscosidad",
                   choices=["viscosidad", "global", "rep_viscosidad"])
    p.add_argument("--respuesta", default="amp", choices=["amp", "fase"])
    p.add_argument("--desenvolver", action="store_true")
    p.add_argument("--grupos-velocidad", dest="grupos", action="store_true")
    p.add_argument("--desde", type=int, default=1, choices=[1, 2, 3, 4, 5])
    args = p.parse_args()

    sal = ["--salida", args.salida]
    desenv = ["--desenvolver"] if args.desenvolver else []

    pasos = [
        (1, "p1_extraer_fasores.py", ["--entrada", args.entrada, "--tipo", args.tipo] + sal),
        (2, "p2_graficar_fasores.py", sal + desenv),
        (3, "p3_compensar_runout.py", sal + ["--modo", args.modo_runout]),
        (4, "p4_graficar_compensados.py", sal + desenv),
        (5, "p5_anova_split_plot.py", sal + ["--respuesta", args.respuesta]
            + (["--grupos-velocidad"] if args.grupos else [])),
    ]

    for numero, script, argumentos in pasos:
        if numero < args.desde:
            continue
        codigo = correr(script, argumentos)
        if codigo != 0:
            print(f"\n!! El paso {numero} ({script}) fallo con codigo {codigo}. Se detiene.")
            return codigo

    print("\n" + "=" * 78)
    print(f"  Proceso completo. Resultados en: {args.salida}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
