"""
generate_synthetic_data.py
==========================

Genera una base de datos sintética con la misma estructura que la base de
datos experimental real, para poder validar el pipeline completo sin
disponer de los datos originales.

Estructura generada::

    <data_dir>/
        rep1_iso32_dsb1/
            Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt
            Rec_stb_iso32_dsb(0+8-7)-Rpm1200.txt
            ...
        rep1_iso32_dsb2/
        ...

Cada archivo imita el formato real: bloques BEGIN_HEADER/END_HEADER,
BEGIN_INFO/END_INFO y BEGIN_DATA/END_DATA, con cinco columnas de datos
(una por canal). Las señales contienen:

* Dos "modos" con frecuencias que se desplazan ligeramente con la RPM
  (como ocurre con cojinetes hidrodinámicos).
* El armónico 1X del desbalance (frecuencia de rotación).
* Ruido blanco de banda ancha.

Uso::

    python generate_synthetic_data.py --data-dir data_synthetic --duration 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FS = 12800.0
CHANNELS = ["Mach1.P1.Y", "Mach1.P1.X", "Mach1.P2.Y", "Mach1.P2.X", "Mach1.S1"]


def synth_signal(rpm: float, duration: float, rng: np.random.Generator,
                 phase_shift: float = 0.0) -> np.ndarray:
    """Sintetiza la señal de un canal para una velocidad de rotación dada.

    Parameters
    ----------
    rpm : float
        Velocidad de rotación [rpm].
    duration : float
        Duración de la señal [s].
    rng : np.random.Generator
        Generador de números aleatorios (reproducible).
    phase_shift : float
        Desfase entre canales para que no sean idénticos.
    """
    n = int(duration * FS)
    t = np.arange(n) / FS
    f_rot = rpm / 60.0

    # Frecuencias "naturales" que se desplazan levemente con la velocidad
    # (rigidez de la película de aceite dependiente de la velocidad).
    f_mode1 = 42.0 + 0.0015 * rpm
    f_mode2 = 87.0 + 0.0025 * rpm

    x = (
        # Modos excitados de forma aleatoria (amplitud variable).
        0.8 * rng.uniform(0.7, 1.3) * np.sin(2 * np.pi * f_mode1 * t + phase_shift)
        + 0.5 * rng.uniform(0.7, 1.3) * np.sin(2 * np.pi * f_mode2 * t + 2 * phase_shift)
        # Componente 1X del desbalance.
        + 1.2 * np.sin(2 * np.pi * f_rot * t + phase_shift)
        # Componente 2X más débil.
        + 0.3 * np.sin(2 * np.pi * 2 * f_rot * t)
        # Ruido de banda ancha.
        + 0.4 * rng.standard_normal(n)
    )
    return x


def write_file(path: Path, rpm: float, iso: int, dsb: int, duration: float,
               rng: np.random.Generator) -> None:
    """Escribe un archivo de medición sintético con el formato real."""
    n = int(duration * FS)
    t = np.arange(n) / FS
    data = np.column_stack(
        [t] + [synth_signal(rpm, duration, rng, phase_shift=0.6 * k)
               for k in range(len(CHANNELS))]
    )

    with path.open("w", encoding="utf-8") as fh:
        fh.write("BEGIN_HEADER\n")
        fh.write(f"SampleRate\t{FS:g} Hz\n")
        fh.write(f"Channels\tTime\t" + "\t".join(CHANNELS) + "\n")
        fh.write("END_HEADER\n")
        fh.write("BEGIN_INFO\n")
        fh.write(f"Test rig: Jeffcott rotor, ISO VG {iso}, unbalance {dsb}\n")
        fh.write(f"Speed: {rpm:g} rpm\n")
        fh.write("END_INFO\n")
        fh.write("BEGIN_DATA\n")
        np.savetxt(fh, data, fmt="%.6e", delimiter="\t")
        fh.write("END_DATA\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data_synthetic",
                        help="Carpeta de salida de la base de datos sintética.")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="Duración de cada señal en segundos.")
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


if __name__ == "__main__":
    main()
