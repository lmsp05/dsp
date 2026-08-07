"""
v3_prom_sensores_velocidades_sin_elim.py
=======================================

Version 3 (SIN eliminacion de picos): igual que la v2 -- para cada condicion
(rep/iso/dsb) promedia los espectros de TODAS las velocidades y sensores -- pero
SIN eliminar el 1X ni los armonicos antes de promediar. Luego detecta los picos.

Grafica frecuencia-vs-condicion.
Salidas: picos_detectados.txt (rep, iso, dsb, omega1...) y picos_grafico.png.
"""

import sys
import _nucleo_vs as nuc

if __name__ == "__main__":
    sys.exit(nuc.ejecutar(nuc.Config(
        nivel=("rep", "iso", "dsb"),
        quitar=False,
        plot="condicion",
        etiqueta="v3 · promedio entre sensores y velocidades (SIN eliminar 1X/armonicos)")))
