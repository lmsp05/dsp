#!/usr/bin/env python3
"""
_generar_datos_prueba.py
========================

Genera una base de datos FALSA con la estructura ``repN_isoVG_dsbM/...RpmXXXX.txt``
para probar ``inventario_combinaciones.py`` sin necesidad de los datos reales.

Los archivos no contienen mediciones utiles: solo relleno del tamano deseado.
Lo que interesa aqui son los nombres y los pesos.

A proposito se introducen los casos que el inventario debe detectar:

  * condiciones sin ningun archivo,
  * combinaciones faltantes (se omiten algunas RPM),
  * combinaciones repetidas (un mismo ensayo guardado dos o tres veces),
  * archivos con peso atipico, bajos y altos, con todas o solo alguna copia,
  * archivos con texto despues del numero de RPM (``...Rpm4500xxxx.txt``),
  * archivos ajenos a la nomenclatura, que no se procesan pero si se cuentan,
  * y un archivo con una RPM fuera del catalogo.

Uso:
    python _generar_datos_prueba.py --outdir datos_prueba
    python inventario_combinaciones.py datos_prueba --outdir inventario_prueba
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

REPS = (1, 2, 3)
ISOS = (32, 46, 68)
DSBS = (1, 2, 3)
RPMS = (600, 1300, 2000, 2500, 2600, 2700, 2800,
        3400, 3500, 3600, 3700, 4000, 4500, 5000, 5500)

#: Peso nominal de un archivo de medicion en la base de datos falsa [MiB].
PESO_NOMINAL_MB = 1.0

#: Condiciones rep_iso_dsb que no se crean: no tienen ni carpeta ni archivos.
#: Deben aparecer como (Ninguno) en la tabla por condicion.
SIN_DATOS = {
    (2, 32, 3),
}

#: Combinaciones que se omiten para simular ensayos faltantes.
FALTANTES = {
    (1, 32, 1, 2600),
    (1, 32, 1, 5500),
    (2, 46, 3, 600),
    (3, 68, 2, 3400),
    (3, 68, 2, 3500),
    (3, 68, 2, 3600),
}

#: Combinaciones duplicadas: cuantas copias se guardan de cada una.
REPETIDAS = {
    (1, 46, 2, 2000): 2,
    (2, 32, 1, 4000): 3,
    (3, 32, 3, 5000): 2,
}

#: Combinaciones con peso anomalo en TODAS sus copias: factor sobre el nominal.
#: La velocidad se pierde por completo -> debe salir en out[...].
OUTLIERS = {
    (1, 68, 1, 2700): 0.15,   # registro truncado
    (2, 68, 2, 3700): 4.00,   # registro mucho mas largo de lo normal
    (3, 46, 1, 1300): 0.40,
    (3, 68, 2, 600): 0.20,    # condicion que ademas tiene velocidades faltantes
    (3, 32, 3, 5000): 0.25,   # duplicada con las DOS copias anomalas
}

#: Combinaciones duplicadas donde solo UNA copia es anomala: {combo: (indice, factor)}.
#: La velocidad sigue disponible gracias a la copia sana -> NO debe salir en out[...].
OUTLIERS_PARCIAL = {
    (1, 46, 2, 2000): (1, 0.20),
}

#: Texto que se agrega DESPUES del numero de RPM en el nombre del archivo.
#: Debe seguir contando como archivo valido de esa velocidad.
SUFIJOS_TRAS_RPM = {
    (1, 32, 2, 4500): "xxxx",
    (2, 46, 1, 2800): "_v2",
}

#: Archivos que no siguen la nomenclatura y deben quedar fuera del conteo de
#: procesados, pero si aparecer en la cuenta de archivos que hay en la carpeta.
BASURA = {
    (1, 32, 1): ["notas.txt", "calibracion.csv"],
    (3, 46, 2): ["README.txt"],
}


def escribir(ruta: Path, mb: float, rng: random.Random) -> None:
    """Crea un archivo de texto del tamano pedido con lineas numericas."""
    objetivo = int(mb * 1024 * 1024)
    linea_ejemplo = "  ".join(f"{rng.uniform(-1, 1):+.6f}" for _ in range(5)) + "\n"
    n_lineas = max(1, objetivo // len(linea_ejemplo))
    with ruta.open("w", encoding="utf-8") as fh:
        fh.write("BEGIN_DATA\n")
        for _ in range(n_lineas):
            fh.write("  ".join(f"{rng.uniform(-1, 1):+.6f}" for _ in range(5)) + "\n")
        fh.write("END_DATA\n")


def generar(outdir: Path, semilla: int) -> int:
    rng = random.Random(semilla)
    outdir.mkdir(parents=True, exist_ok=True)
    n_archivos = 0

    for rep in REPS:
        for iso in ISOS:
            for dsb in DSBS:
                if (rep, iso, dsb) in SIN_DATOS:
                    continue
                carpeta = outdir / f"rep{rep}_iso{iso}_dsb{dsb}"
                carpeta.mkdir(exist_ok=True)
                for rpm in RPMS:
                    combo = (rep, iso, dsb, rpm)
                    if combo in FALTANTES:
                        continue
                    factor = OUTLIERS.get(combo, 1.0)
                    parcial = OUTLIERS_PARCIAL.get(combo)
                    copias = REPETIDAS.get(combo, 1)
                    tras_rpm = SUFIJOS_TRAS_RPM.get(combo, "")
                    for k in range(copias):
                        f = parcial[1] if parcial and parcial[0] == k else factor
                        sufijo = "" if k == 0 else f"_copia{k}"
                        nombre = (f"Rec_stb_iso{iso}_dsb(0+8-7){sufijo}"
                                  f"-Rpm{rpm}{tras_rpm}.txt")
                        # Pequena variacion natural del tamano (+-2 %).
                        escribir(carpeta / nombre,
                                 PESO_NOMINAL_MB * f * rng.uniform(0.98, 1.02), rng)
                        n_archivos += 1

                for nombre in BASURA.get((rep, iso, dsb), []):
                    (carpeta / nombre).write_text("archivo ajeno al inventario\n",
                                                  encoding="utf-8")

    # Un archivo con una RPM que no esta en el catalogo.
    extra = outdir / "rep1_iso32_dsb1" / "Rec_stb_iso32_dsb(0+8-7)-Rpm900.txt"
    escribir(extra, PESO_NOMINAL_MB, rng)
    n_archivos += 1

    print(f"Base de datos de prueba en {outdir.resolve()}")
    print(f"  {n_archivos} archivos generados")
    print(f"  {len(SIN_DATOS)} condiciones sin ningun archivo")
    print(f"  {len(FALTANTES)} combinaciones omitidas")
    print(f"  {len(REPETIDAS)} combinaciones repetidas")
    print(f"  {len(OUTLIERS)} con todas sus copias anomalas, "
          f"{len(OUTLIERS_PARCIAL)} con solo una copia anomala")
    print(f"  {len(SUFIJOS_TRAS_RPM)} con texto despues del numero de RPM")
    print(f"  {sum(len(v) for v in BASURA.values())} archivos ajenos a la nomenclatura")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera una base de datos falsa para "
                                            "probar el inventario de combinaciones.")
    p.add_argument("--outdir", default="datos_prueba", help="Carpeta a crear")
    p.add_argument("--semilla", type=int, default=0, help="Semilla del generador")
    args = p.parse_args()
    raise SystemExit(generar(Path(args.outdir).expanduser(), args.semilla))
