"""
txt_a_npy.py
============

Convierte el .txt tabular generado por ``procesar_fft.py`` en un archivo .npy
(array estructurado de numpy) mucho mas rapido de leer y filtrar desde otros
scripts.

El array estructurado resultante tiene un campo por columna, mas un campo
`complejo` (numero complejo = amplitud * exp(i * fase)) que permite operaciones
vectoriales (promediar, restar el runout, comparar fases entre sensores) sin
perder precision por el redondeo de la fase en grados.

--------------------------------------------------------------------------
SOBRE EL FORMATO DE ALMACENAMIENTO (la otra duda que planteaste)
--------------------------------------------------------------------------
Recomendacion:
  * En el .txt (legible por humanos): AMPLITUD + FASE (grados). Es lo mas
    interpretable para un ingeniero, se compara directo con la lectura de los
    instrumentos y los saltos de fase se ven a simple vista.
  * En el .npy (para maquinas): el NUMERO COMPLEJO. Es sin perdidas -> de el se
    recuperan amplitud (abs) y fase (angle) exactas, y ademas permite promediar
    o restar vectorialmente. Este script guarda AMBAS representaciones para que
    cada script aguas abajo use la que le convenga.

Uso:
    python txt_a_npy.py                       # resultados_fft.txt -> resultados_fft.npy
    python txt_a_npy.py --entrada mi.txt --salida mi.npy

Lectura posterior (ejemplo):
    import numpy as np
    d = np.load("resultados_fft.npy")
    sel = d[(d["rep"] == 1) & (d["iso"] == 32) & (d["sensor"] == "P1.Y")]
    f, amp, fase = sel["frecuencia_Hz"], sel["amplitud"], sel["fase_grados"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# dtype del array estructurado de salida (orden = columnas del .txt + complejo)
DTYPE = np.dtype([
    ("rep", "i4"),
    ("iso", "i4"),
    ("dsb", "i4"),
    ("rpm_nominal", "f8"),
    ("sensor", "U8"),
    ("t_ini_s", "f8"),
    ("t_fin_s", "f8"),
    ("rpm_medida", "f8"),
    ("n_bloques", "i4"),
    ("revs_bloque", "i4"),
    ("df_Hz", "f8"),
    ("frecuencia_Hz", "f8"),
    ("amplitud", "f8"),
    ("fase_grados", "f8"),
    ("complejo", "c16"),
])


def convertir(entrada: Path, salida: Path) -> int:
    if not entrada.is_file():
        print(f"ERROR: no existe {entrada}")
        return 1

    # Filtra comentarios ('#') y lineas vacias; la primera linea util es la
    # cabecera de columnas y el resto son los datos.
    with entrada.open("r", encoding="utf-8") as fh:
        utiles = [ln.rstrip("\n") for ln in fh
                  if ln.strip() and not ln.lstrip().startswith("#")]
    if len(utiles) < 2:
        print("ERROR: el archivo no contiene filas de datos.")
        return 1
    columnas = utiles[0].split("\t")
    filas = np.genfromtxt(
        utiles[1:], delimiter="\t", names=columnas, dtype=None, encoding="utf-8",
    )
    filas = np.atleast_1d(filas)

    n = filas.shape[0]
    out = np.empty(n, dtype=DTYPE)
    for campo in DTYPE.names:
        if campo == "complejo":
            continue
        out[campo] = filas[campo]

    fase_rad = np.deg2rad(out["fase_grados"])
    out["complejo"] = out["amplitud"] * np.exp(1j * fase_rad)

    np.save(salida, out)
    print(f"Guardadas {n} filas en {salida}")
    print(f"Sensores: {sorted(set(out['sensor'].tolist()))}")
    print(f"Condiciones (rep,iso,dsb): "
          f"{sorted(set(zip(out['rep'].tolist(), out['iso'].tolist(), out['dsb'].tolist())))}")
    print(f"RPM nominales: {sorted(set(out['rpm_nominal'].tolist()))}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convierte el .txt de espectros FFT en un .npy estructurado.")
    p.add_argument("--entrada", default="resultados_fft.txt", help="Archivo .txt de entrada")
    p.add_argument("--salida", default="resultados_fft.npy", help="Archivo .npy de salida")
    return p


if __name__ == "__main__":
    args = construir_parser().parse_args()
    entrada = Path(args.entrada).expanduser().resolve()
    salida = Path(args.salida).expanduser().resolve()
    sys.exit(convertir(entrada, salida))
