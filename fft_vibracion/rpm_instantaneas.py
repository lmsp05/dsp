"""
rpm_instantaneas.py
===================

Script sencillo: recorre la carpeta de datos y guarda la RPM instantanea
(pulso a pulso) de cada archivo, junto con su condicion (rep, iso, dsb, rpm
nominal), en un unico .txt filtrable.

Uso:
    python rpm_instantaneas.py                         # ./datos -> rpm_instantaneas.txt
    python rpm_instantaneas.py --data-dir /ruta/datos --salida rpm.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from procesar_fft import (
    FS, leer_archivo, detectar_pulsos, rpm_instantanea, descubrir_archivos,
)


def main(args) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: la carpeta de datos no existe: {data_dir}")
        return 1

    archivos = descubrir_archivos(data_dir)
    if not archivos:
        print(f"No se encontraron archivos validos en {data_dir}")
        return 1

    salida = Path(args.salida).expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with salida.open("w", encoding="utf-8") as fh:
        fh.write("rep\tiso\tdsb\trpm_nominal\tt_s\trpm_instantanea\n")
        for info in archivos:
            etiqueta = f"rep{info.rep}_iso{info.iso}_dsb{info.dsb}_rpm{info.rpm:g}"
            try:
                matriz, canales = leer_archivo(info.path)
                try:
                    col_sync = next(i for i, c in enumerate(canales) if c == "sync")
                except StopIteration:
                    col_sync = 0
                pulsos = detectar_pulsos(matriz[:, col_sync])
                t_pulso, rpm = rpm_instantanea(pulsos, FS)
            except Exception as e:
                print(f"  [ERROR] {etiqueta}: {e}")
                continue

            base = f"{info.rep}\t{info.iso}\t{info.dsb}\t{info.rpm:g}\t"
            fh.writelines(base + f"{t:.5f}\t{r:.2f}\n" for t, r in zip(t_pulso, rpm))
            print(f"  OK {etiqueta}: {len(rpm)} valores de RPM instantanea")
            n_ok += 1

    print(f"\nListo. {n_ok} archivos. Resultado en: {salida}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Guarda la RPM instantanea (pulso a pulso) de cada archivo "
                    "con su condicion, en un .txt.")
    p.add_argument("--data-dir", default="datos", help="Carpeta raiz de datos (def: ./datos)")
    p.add_argument("--salida", default="rpm_instantaneas.txt", help="Archivo .txt de salida")
    sys.exit(main(p.parse_args()))
