"""
generate_synthetic_data.py
==========================

Genera una base de datos sintética con **el formato real del equipo de
adquisición** (el mismo de los archivos ``.txt`` del rotor Jeffcott), para
validar tanto el pipeline de peak picking (``main.py``) como el de SSI
(``ssi_main.py``) sin disponer de los datos originales.

Formato de cada archivo (idéntico al real)::

    BEGIN_HEADER
    END_HEADER
    BEGIN_INFO
    Sampling_frequency<TAB>1.28000e+004
    Number_of_tracks<TAB>6
    Number_of_points<TAB>...
    Track<TAB>1<TAB>2<TAB>3<TAB>4<TAB>5<TAB>6
    Name<TAB>Ext. sync. 1<TAB>Input 1: Mach1.P1.Y<TAB>...<TAB>Input 5: Mach1.S1
    ...
    END_INFO
    BEGIN_DATA
    <tacómetro><TAB><P1Y><TAB><P1X><TAB><P2Y><TAB><P2X><TAB><S1>
    ...
    END_DATA

Es decir, **6 columnas**: la primera es el tacómetro (``Ext. sync.``) y las
cinco siguientes son las cuatro sondas de proximidad y el acelerómetro.

A diferencia de una síntesis canal a canal, aquí se simula un **sistema modal
multi-salida coherente**: dos modos (frecuencia + amortiguamiento + forma
modal sobre las cuatro sondas) excitados por ruido blanco mediante osciladores
de segundo orden. Así las cuatro sondas comparten los mismos polos y formas
modales, que es justo lo que la SSI (Peeters & De Roeck) debe recuperar. Se
añade el armónico 1X del desbalance (excitación forzada) y ruido de medida.

Uso::

    python generate_synthetic_data.py --data-dir data_synthetic --duration 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

FS = 12800.0
# Canales de datos (sin el tacómetro), en el orden real de las columnas 2..6.
DATA_CHANNELS = ["Mach1.P1.Y", "Mach1.P1.X", "Mach1.P2.Y", "Mach1.P2.X", "Mach1.S1"]

# Modos "estructurales" de referencia (independientes de la RPM para poder
# comprobar que la SSI recupera exactamente estas frecuencias):
#   (frecuencia [Hz], amortiguamiento, forma modal en [P1Y, P1X, P2Y, P2X]).
MODES = [
    (44.0, 0.030, np.array([1.00, 0.85, -0.90, -0.70])),
    (83.0, 0.020, np.array([1.00, -0.60, 1.00, -0.55])),
]


def _sdof_response(freq: float, zeta: float, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Respuesta de un oscilador de 2º orden a ruido blanco (coordenada modal).

    Se discretiza el sistema ``H(s) = 1 / (s^2 + 2 zeta wn s + wn^2)`` con la
    transformación bilineal implícita en un filtro IIR de 2 polos y se excita
    con ruido blanco gaussiano. El resultado es una señal aleatoria estacionaria
    cuya densidad espectral tiene un pico en ``freq`` con el ancho de banda que
    corresponde a ``zeta`` (respuesta libre típica de OMA).
    """
    wn = 2.0 * np.pi * freq
    dt = 1.0 / FS
    # Polos discretos del oscilador amortiguado.
    r = np.exp(-zeta * wn * dt)
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 1e-9))
    a = [1.0, -2.0 * r * np.cos(wd * dt), r ** 2]
    b = [dt ** 2]  # ganancia arbitraria (la escala no afecta a la modal)
    return lfilter(b, a, rng.standard_normal(n))


def synth_measurement(rpm: float, duration: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Sintetiza las 6 columnas (tacómetro + 5 canales) de un ensayo.

    Returns
    -------
    np.ndarray
        Matriz ``(N, 6)``: columna 0 = tacómetro, 1..4 = sondas, 5 = acelerómetro.
    """
    n = int(duration * FS)
    t = np.arange(n) / FS
    f_rot = rpm / 60.0

    # Respuesta modal: cada modo aporta su coordenada q_m escalada por la forma.
    probes = np.zeros((n, 4))
    for freq, zeta, shape in MODES:
        q = _sdof_response(freq, zeta, n, rng)
        q /= np.std(q) + 1e-12
        probes += np.outer(q, shape)

    # Armónico 1X del desbalance (excitación forzada síncrona, no es un modo):
    # órbita circular -> Y = sin, X = cos, con niveles distintos por cojinete.
    amp_1x = np.array([1.4, 1.4, 1.0, 1.0])
    phase_1x = np.array([0.0, np.pi / 2, 0.2, np.pi / 2 + 0.2])
    for j in range(4):
        probes[:, j] += amp_1x[j] * np.sin(2 * np.pi * f_rot * t + phase_1x[j])
        probes[:, j] += 0.25 * amp_1x[j] * np.sin(2 * np.pi * 2 * f_rot * t)

    # Ruido de medida independiente por sonda.
    probes += 0.05 * rng.standard_normal((n, 4))

    # Nivel DC (separación media de la sonda), como en los datos reales en µm.
    probes += np.array([-2050.0, -1965.0, -2118.0, -2118.0])

    # Acelerómetro en P1Y: derivada segunda aproximada de la señal P1Y (g).
    accel = np.zeros(n)
    accel[1:-1] = np.diff(probes[:, 0], 2) * (FS ** 2) / 9.81
    accel += 0.02 * rng.standard_normal(n)

    # Tacómetro: pulso 1X (sincronismo), acoplado AC alrededor de -1 V.
    tacho = -1.0 + 2.0 * (np.sin(2 * np.pi * f_rot * t) > 0.98)

    return np.column_stack([tacho, probes, accel])


def write_file(path: Path, rpm: float, iso: int, dsb: int, duration: float,
               rng: np.random.Generator) -> None:
    """Escribe un archivo de medición sintético con el formato real del equipo."""
    data = synth_measurement(rpm, duration, rng)
    n = data.shape[0]

    names = ["Ext. sync. 1"] + [f"Input {k + 1}: {c}"
                                for k, c in enumerate(DATA_CHANNELS)]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("BEGIN_HEADER\n")
        fh.write("END_HEADER\n")
        fh.write("BEGIN_INFO\n")
        fh.write(f"Sampling_frequency\t{FS:.5e}\n")
        fh.write("Number_of_tracks\t6\n")
        fh.write(f"Number_of_points\t{n}\n")
        fh.write("Track\t" + "\t".join(str(i + 1) for i in range(6)) + "\t\n")
        fh.write("Name\t" + "\t".join(names) + "\t\n")
        fh.write("Magnitude\tPotential_Difference\tLength\tLength\tLength\t"
                 "Length\tAcceleration\t\n")
        fh.write("Unit\tV\tµm\tµm\tµm\tµm\tg\t\n")
        fh.write("Type\tEXTERNAL\tANALOG\tANALOG\tANALOG\tANALOG\tANALOG\t\n")
        fh.write("Coupling\tAC\tDC\tDC\tDC\tDC\tICP\t\n")
        fh.write("END_INFO\n")
        fh.write("BEGIN_DATA\n")
        np.savetxt(fh, data, fmt="%.5e", delimiter="\t", newline="\t\n")
        fh.write("END_DATA\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data_synthetic",
                        help="Carpeta de salida de la base de datos sintética.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duración de cada señal en segundos (la SSI "
                             "necesita registros largos: >= 20 s).")
    parser.add_argument("--reps", type=int, default=2,
                        help="Número de repeticiones.")
    parser.add_argument("--rpms", type=float, nargs="+",
                        default=[600, 1200, 1800],
                        help="Velocidades de rotación a generar.")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    root = Path(args.data_dir)

    for rep in range(1, args.reps + 1):
        for iso in (32, 68):
            for dsb in (1, 2):
                folder = root / f"rep{rep}_iso{iso}_dsb{dsb}"
                folder.mkdir(parents=True, exist_ok=True)
                for rpm in args.rpms:
                    name = f"Rec_stb_iso{iso}_dsb(0+8-7)-Rpm{rpm:g}.txt"
                    write_file(folder / name, rpm, iso, dsb, args.duration, rng)
                    print(f"  {folder / name}")

    print(f"\nBase de datos sintética creada en: {root.resolve()}")
    print(f"Modos estructurales inyectados: "
          f"{', '.join(f'{f:.1f} Hz (ζ={100*z:.1f}%)' for f, z, _ in MODES)}")


if __name__ == "__main__":
    main()
