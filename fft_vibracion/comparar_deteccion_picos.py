"""
comparar_deteccion_picos.py
==========================

Ejecuta TODAS las variantes de deteccion de picos sobre un mismo archivo de FFT:
la original (``deteccion_picos.py``) y las 4 versiones de ``deteccion_picos_vs/``.
Para cada una crea una subcarpeta con el nombre del script dentro de la carpeta de
salida y deja alli sus resultados (tabla + grafica).

Uso:
    python comparar_deteccion_picos.py --entrada resultados_fft.txt --outdir comparacion

Cualquier argumento extra (filtros --iso/--rpm/..., --fraccion, etc.) se reenvia
tal cual a TODAS las variantes:
    python comparar_deteccion_picos.py --entrada resultados_fft.txt --outdir comp \
        --iso 32,68 --fraccion 0.08
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
VS = AQUI / "deteccion_picos_vs"

# (ruta del script, nombre de la subcarpeta de salida)
VARIANTES = [
    (AQUI / "deteccion_picos.py", "deteccion_picos"),
    (VS / "v1_prom_sensores.py", "v1_prom_sensores"),
    (VS / "v2_prom_sensores_velocidades.py", "v2_prom_sensores_velocidades"),
    (VS / "v3_prom_sensores_velocidades_sin_elim.py", "v3_prom_sensores_velocidades_sin_elim"),
    (VS / "v4_prom_global_sin_elim.py", "v4_prom_global_sin_elim"),
    (VS / "metricas_cascada.py", "metricas_cascada"),
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Ejecuta la deteccion de picos original y sus 4 variantes, "
                    "cada una en su propia subcarpeta.")
    p.add_argument("--entrada", required=True, help="Archivo .txt o .npy de procesar_fft.py")
    p.add_argument("--outdir", required=True, help="Carpeta donde crear las subcarpetas por variante")
    args, extra = p.parse_known_args()   # 'extra' se reenvia a cada variante

    entrada = str(Path(args.entrada).expanduser().resolve())
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Entrada: {entrada}")
    print(f"Salida:  {outdir}")
    if extra:
        print(f"Args reenviados a todas las variantes: {' '.join(extra)}")
    print()

    n_ok = 0
    for script, nombre in VARIANTES:
        sub = outdir / nombre
        sub.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--entrada", entrada, "--outdir", str(sub)] + extra
        print(f"=== {nombre} ===")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            n_ok += 1
        else:
            print(f"  [aviso] {nombre} termino con codigo {res.returncode}")
        print()

    print(f"Listo. {n_ok}/{len(VARIANTES)} variantes ejecutadas. Resultados en: {outdir}")
    return 0 if n_ok == len(VARIANTES) else 1


if __name__ == "__main__":
    sys.exit(main())
