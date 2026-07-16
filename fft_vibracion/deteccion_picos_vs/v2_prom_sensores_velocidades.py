"""
v2_prom_sensores_velocidades.py
==============================

Version 2: para cada condicion (rep/iso/dsb) promedia los espectros de TODAS las
velocidades y sensores (con el 1X y los armonicos ya eliminados de cada uno), y
luego detecta los picos sobre ese promedio.

Como la velocidad se promedia, la grafica es frecuencia-vs-condicion.
Salidas: picos_detectados.txt (rep, iso, dsb, omega1...) y picos_grafico.png.
"""

import sys
import _nucleo_vs as nuc

if __name__ == "__main__":
    sys.exit(nuc.ejecutar(nuc.Config(
        nivel=("rep", "iso", "dsb"),
        quitar=True,
        plot="condicion",
        etiqueta="v2 · promedio entre sensores y velocidades (1X y armonicos eliminados)")))
