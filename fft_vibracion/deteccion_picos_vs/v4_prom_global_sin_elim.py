"""
v4_prom_global_sin_elim.py
=========================

Version 4 (SIN eliminacion de picos): promedia TODO -- todos los sensores, todas
las condiciones y todas las velocidades -- en un unico espectro global, SIN
eliminar el 1X ni los armonicos, y luego detecta los picos.

Como es un solo espectro promedio, la grafica es ese espectro con los picos.
Salidas: picos_detectados.txt (omega1...) y picos_grafico.png.
"""

import sys
import _nucleo_vs as nuc

if __name__ == "__main__":
    sys.exit(nuc.ejecutar(nuc.Config(
        nivel=(),
        quitar=False,
        plot="espectro",
        etiqueta="v4 · promedio global de todo (SIN eliminar 1X/armonicos)")))
