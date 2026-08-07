"""
v1_prom_sensores.py
===================

Version 1: promedia los espectros ENTRE SENSORES (con el 1X y los armonicos ya
eliminados de cada sensor) para cada combinacion rep/iso/dsb/rpm, y luego detecta
los picos sobre ese promedio.

Se conserva la velocidad, asi que la grafica es el diagrama frecuencia-vs-RPM.
Salidas: picos_detectados.txt (rep, iso, dsb, rpm, omega1...) y picos_grafico.png.
"""

import sys
import _nucleo_vs as nuc

if __name__ == "__main__":
    sys.exit(nuc.ejecutar(nuc.Config(
        nivel=("rep", "iso", "dsb", "rpm"),
        quitar=True,
        plot="campbell",
        etiqueta="v1 · promedio entre sensores (1X y armonicos eliminados)")))
