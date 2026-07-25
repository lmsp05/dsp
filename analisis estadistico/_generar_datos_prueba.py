"""
UTILIDAD OPCIONAL — genera un juego de .xlsx sinteticos con la misma estructura
que los archivos reales, para probar el pipeline completo sin datos de campo.

Reproduce: 27 archivos rep<R>_<visc>_<desb>_p_<fecha>.xlsx, cada uno con 15
hojas de velocidad, 4 sensores (P1Y/P1X/P2Y/P2X), ~3 min de registro con un
transitorio inicial, un runout de slow roll comun y una respuesta al
desbalanceo que pasa por dos criticas.

    python _generar_datos_prueba.py --salida <carpeta>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(20240424)

# Runout de slow roll "verdadero" por sensor (amplitud um, fase grados).
RUNOUT = {"P1Y": (9.0, 12.0), "P1X": (3.4, 121.0),
          "P2Y": (2.7, 117.0), "P2X": (11.3, 196.0)}

# Criticas y amortiguamiento aparente por sensor.
MODOS = {"P1Y": [(3500, 0.06), (5200, 0.09)], "P1X": [(3600, 0.07), (5300, 0.10)],
         "P2Y": [(3400, 0.09), (5100, 0.12)], "P2X": [(2900, 0.05), (5400, 0.08)]}


def _respuesta(rpm, visc, desb, sensor):
    """Fasor de respuesta al desbalanceo (sin runout)."""
    z = 0j
    # La viscosidad desplaza levemente la critica y cambia el amortiguamiento.
    k_wn = {32: 0.99, 46: 1.00, 68: 1.015}[visc]
    k_z = {32: 0.85, 46: 1.00, 68: 1.20}[visc]
    for wn, zeta in MODOS[sensor]:
        wn, zeta = wn * k_wn, zeta * k_z
        r = rpm / wn
        den = (1 - r ** 2) + 2j * zeta * r
        z += (desb * 3.0) * (r ** 2) / den
    return z


# Variabilidad de MONTAJE, la que da sentido al analisis de parcelas divididas.
# Cada vez que se cambia el aceite (rep x visc) el rotor se vuelve a montar y el
# fasor cambia un poco; lo mismo, en menor medida, al cambiar el desbalanceo.
_MONTAJE: dict = {}
rep_actual = 1


def _perturbacion(clave, escala):
    if clave not in _MONTAJE:
        _MONTAJE[clave] = escala * (RNG.normal() + 1j * RNG.normal())
    return _MONTAJE[clave]


def _serie(rpm, visc, desb, sensor, n):
    """Serie temporal (amplitud, fase) con transitorio inicial y ruido."""
    a0, f0 = RUNOUT[sensor]
    z_obj = a0 * np.exp(1j * np.deg2rad(f0)) + _respuesta(rpm, visc, desb, sensor)
    # Parcela grande (montaje del aceite) y subparcela (montaje del desbalanceo).
    z_obj = z_obj * (1 + _perturbacion(("a", rep_actual, visc, sensor), 0.030)
                     + _perturbacion(("b", rep_actual, visc, desb, sensor), 0.012))

    t = np.cumsum(RNG.uniform(1.30, 1.45, n))
    # Transitorio: arranca desplazado y decae con tau ~ 20 s.
    desvio = z_obj * RNG.uniform(0.10, 0.30) * np.exp(1j * RNG.uniform(0, 2 * np.pi))
    z = z_obj + desvio * np.exp(-t / 20.0)
    # Ruido creciente con la velocidad.
    s = abs(z_obj) * (0.004 + 0.010 * rpm / 5500.0)
    z = z + RNG.normal(0, s, n) + 1j * RNG.normal(0, s, n)

    amp = np.abs(z)
    fase = np.mod(np.rad2deg(np.angle(z)), 360.0)
    return t, amp, fase


def generar(destino: Path):
    destino.mkdir(parents=True, exist_ok=True)
    n_por_rpm = {600: 132, 1300: 254, 2000: 388, 2500: 438, 2600: 460, 2700: 463,
                 2800: 586, 3400: 629, 3500: 602, 3600: 602, 3700: 599, 4000: 606,
                 4500: 889, 5000: 902, 5500: 894}

    global rep_actual
    n = 0
    for rep in config.REPETICIONES:
        rep_actual = rep
        for visc in config.VISCOSIDADES:
            for desb in config.DESBALANCEOS:
                ruta = destino / f"rep{rep}_{visc}_{desb}_p_0424.xlsx"
                with pd.ExcelWriter(ruta, engine="openpyxl") as w:
                    for rpm in config.VELOCIDADES:
                        filas = n_por_rpm.get(rpm, 400)
                        cab0 = [None]
                        cab1 = ["Time"]
                        cab2 = ["s"]
                        cols = []
                        for s in config.SENSORES:
                            canal = f"NVD: Prof >Bode> 1X [Pk-Pk]> Mach1.{s[:2]}.{s[2]}"
                            cab0 += [canal, canal]
                            cab1 += ["Displacement", "Phase"]
                            cab2 += ["µm", "°"]
                        t = None
                        for s in config.SENSORES:
                            ts, a, f = _serie(rpm, visc, desb, s, filas)
                            t = ts if t is None else t
                            cols += [np.round(a, 3), np.round(f, 1)]
                        cuerpo = np.column_stack([np.round(t, 3)] + cols)
                        df = pd.DataFrame(
                            np.vstack([cab0, cab1, cab2, cuerpo.astype(object)]))
                        df.to_excel(w, sheet_name=str(rpm), header=False, index=False)
                n += 1
                print(f"  {ruta.name}")
    print(f"\n{n} archivos generados en {destino}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera .xlsx sinteticos de prueba.")
    p.add_argument("--salida", required=True)
    generar(Path(p.parse_args().salida).expanduser())
