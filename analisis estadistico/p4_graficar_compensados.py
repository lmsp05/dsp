"""
PASO 4 — Las mismas graficas del paso 2, con los fasores YA COMPENSADOS
=======================================================================

Reutiliza exactamente la logica de `p2_graficar_fasores.py` (mismos convenios de
color, mismo reparto 2x2, mismo eje X equiespaciado) pero partiendo de la tabla
compensada del paso 3, y escribe las 12 figuras en la subcarpeta
"fasores con compensacion runout".

Los limites del eje Y se calculan sobre la tabla compensada, de modo que las 12
figuras de este paso comparten escala entre si. Con --escala-comun se fuerzan
unos limites que engloban ADEMAS los datos sin compensar, para poder comparar
directamente las figuras del paso 2 con las del paso 4.

Uso:
    python p4_graficar_compensados.py --salida <carpeta_resultados>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
import comun
import p2_graficar_fasores as p2


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paso 4: figuras 2x2 a partir de los fasores compensados.")
    p.add_argument("--entrada", default="",
                   help="def: <salida>/p3_fasores_compensados.txt")
    p.add_argument("--salida", default=config.DIR_RESULTADOS)
    p.add_argument("--subcarpeta", default=config.NOMBRE_DIR_CON_COMP)
    p.add_argument("--etiqueta", default="with runout compensation")
    p.add_argument("--desenvolver", action="store_true",
                   help="Dibuja la fase de forma continua (quita saltos 0<->360)")
    p.add_argument("--escala-comun", dest="escala_comun", action="store_true",
                   help="Usa limites Y que engloban tambien la tabla sin compensar, "
                        "para comparar el paso 2 con el paso 4")
    args = p.parse_args()

    salida = Path(args.salida).expanduser()
    entrada = Path(args.entrada).expanduser() if args.entrada \
        else salida / config.ARCHIVO_FASORES_COMP
    if not entrada.is_file():
        print(f"ERROR: no existe {entrada}. Ejecuta antes p3_compensar_runout.py")
        return 1

    df = comun.leer_tabla(entrada)

    lim_amp = lim_fase = None
    if args.escala_comun:
        sin_comp = salida / config.ARCHIVO_FASORES
        if sin_comp.is_file():
            juntas = pd.concat([comun.leer_tabla(sin_comp), df], ignore_index=True)
            lim_amp, lim_fase = p2.limites_globales(
                juntas, desenvolver=args.desenvolver)
            print("Escala Y comun con los datos SIN compensar (--escala-comun).")
        else:
            print(f"AVISO: no se encontro {sin_comp}; se usan los limites propios.")

    destino = salida / args.subcarpeta
    generadas, lim_amp, lim_fase = p2.generar_figuras(
        df, destino, args.etiqueta, lim_amp, lim_fase,
        desenvolver=args.desenvolver)

    print(f"Table   : {entrada}  ({len(df)} rows)")
    if args.desenvolver:
        print("Phase drawn continuously (--desenvolver).")
    print(f"Y range -> amplitude [{lim_amp[0]:.2f}, {lim_amp[1]:.2f}] um  |  "
          f"phase [{lim_fase[0]:.1f}, {lim_fase[1]:.1f}] deg")
    print(f"\n{len(generadas)} figures in: {destino}")
    for r in generadas:
        print(f"  {r.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
