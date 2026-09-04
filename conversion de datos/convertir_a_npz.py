"""
convertir_a_npz.py — de la base de datos .txt a archivos .npz de numpy
=====================================================================

    python "conversion de datos/convertir_a_npz.py" --entrada <carpeta_datos> --salida <carpeta_npz>

Recorre la estructura de carpetas de la base de datos experimental::

    <entrada>/
        rep1_iso32_dsb1/
            Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt
            Rec_stb_iso32_dsb(0+8-7)-Rpm1200.txt
            ...
        rep1_iso32_dsb2/
        ...

y escribe un `.npz` por cada `.txt`, con el mismo contenido numérico y los
metadatos de la medición incorporados::

    <salida>/
        rep1_iso32_dsb1_rpm600.npz
        rep1_iso32_dsb1_rpm1200.npz
        ...

POR QUÉ .npz
------------
Los `.txt` son texto: cada lectura vuelve a parsear delimitadores, separador
decimal y bloques de cabecera, lo que cuesta segundos por archivo. El `.npz`
guarda el array binario ya interpretado, así que la carga es prácticamente
instantánea y siempre devuelve exactamente los mismos números.

QUÉ CONTIENE CADA .npz
----------------------
    datos          matriz float64 (n_muestras x n_columnas)
    columnas       nombres de las columnas de `datos`, en orden
    rep            repetición                     (int)
    iso            grado ISO de viscosidad        (int)
    dsb            nivel de desbalanceo           (int)
    rpm            velocidad de giro              (float)
    fs_hz          frecuencia de muestreo         (float)
    tiempo_origen  'archivo' o 'sintetizado'      (str)
    n_muestras     número de filas                (int)
    archivo_origen nombre del .txt de procedencia (str)

La primera columna de `datos` es siempre `tiempo_s`. Si el archivo original
trae su propia base de tiempos se conserva tal cual (`tiempo_origen` =
'archivo'); si no, se genera a partir de `--fs` (`tiempo_origen` =
'sintetizado'), y en ese caso el eje temporal es una suposición, no un dato
medido.

CÓMO SE VUELVE A ABRIR
----------------------
    import numpy as np, pandas as pd

    d  = np.load("rep1_iso32_dsb1_rpm600.npz", allow_pickle=False)
    df = pd.DataFrame(d["datos"], columns=d["columnas"])
    rpm = float(d["rpm"])

o, con el ayudante de este mismo módulo::

    from convertir_a_npz import cargar
    df, meta = cargar("rep1_iso32_dsb1_rpm600.npz")

`allow_pickle=False` funciona porque no se guarda ningún objeto de Python:
solo arrays numéricos y cadenas. Eso hace los archivos portables y seguros de
abrir.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# El parseo de los .txt (codificaciones, bloques BEGIN_DATA/END_DATA,
# delimitador, coma decimal, detección de canales) ya está resuelto y probado
# en io_utils.py, en la raíz del repositorio. Se reutiliza en lugar de
# duplicarlo: cualquier arreglo allí beneficia también a esta conversión.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

try:
    import io_utils
except ImportError as e:                                  # pragma: no cover
    raise SystemExit(
        f"No se pudo importar io_utils.py desde {RAIZ}.\n"
        f"Este script debe vivir en una subcarpeta del repositorio, junto a "
        f"io_utils.py y config.py.\nDetalle: {e}")

try:
    import config as config_raiz
    FOLDER_PATTERN = config_raiz.FOLDER_PATTERN
    RPM_PATTERN = config_raiz.RPM_PATTERN
    FILE_EXTENSION = config_raiz.FILE_EXTENSION
    CHANNEL_NAMES = list(config_raiz.CHANNEL_NAMES)
    FS = float(config_raiz.FS)
    HAS_TIME_COLUMN = config_raiz.HAS_TIME_COLUMN
except Exception:                                          # pragma: no cover
    # Valores de respaldo si config.py no está disponible o cambia de forma.
    FOLDER_PATTERN = r"rep(?P<rep>\d+)_iso(?P<iso>\d+)_dsb(?P<dsb>\d+)"
    RPM_PATTERN = r"[Rr][Pp][Mm][\s_\-]*(\d+)"
    FILE_EXTENSION = ".txt"
    CHANNEL_NAMES = ["Mach1.P1.Y", "Mach1.P1.X", "Mach1.P2.Y", "Mach1.P2.X",
                     "Mach1.S1"]
    FS = 12800.0
    HAS_TIME_COLUMN = "auto"

COLUMNA_TIEMPO = "tiempo_s"

log = logging.getLogger("convertir_a_npz")


# ============================================================
# NOMBRE DE SALIDA
# ============================================================

def nombre_salida(rep: int, iso: int, dsb: int, rpm: float) -> str:
    """
    Nombre del .npz: rep<R>_iso<V>_dsb<D>_rpm<N>.npz

    Lleva la condición experimental completa en el propio nombre, de modo que
    un archivo suelto sigue siendo identificable sin abrirlo y sin depender de
    la carpeta en la que esté. La RPM se formatea con '%g' para que 600.0 se
    escriba 600 y no arrastre un decimal vacío.
    """
    return f"rep{int(rep)}_iso{int(iso)}_dsb{int(dsb)}_rpm{rpm:g}.npz"


# ============================================================
# CONVERSIÓN DE UN ARCHIVO
# ============================================================

def convertir_uno(info, destino: Path, fs: float, canales: list[str],
                  columna_tiempo, comprimir: bool, sobrescribir: bool) -> dict:
    """
    Convierte un .txt en un .npz y devuelve un resumen de lo hecho.

    Devuelve un dict con 'estado' en {'convertido', 'omitido', 'error'}.
    Un archivo ilegible NO detiene la conversión: se registra y se sigue, que
    es lo que permite lanzar el lote completo sin vigilarlo.
    """
    ruta_npz = destino / nombre_salida(info.rep, info.iso, info.dsb, info.rpm)
    resumen = {"origen": info.path.name, "destino": ruta_npz.name,
               "estado": "convertido", "detalle": ""}

    if ruta_npz.exists() and not sobrescribir:
        resumen["estado"] = "omitido"
        resumen["detalle"] = "ya existe (usa --sobrescribir para rehacerlo)"
        return resumen

    try:
        # matriz completa sin tocar: no se resta la media ni se filtra nada.
        # Este script convierte de formato, no procesa la señal.
        matriz, canales_detectados, offset = io_utils._load_matrix(
            info, canales, columna_tiempo)
    except Exception as e:
        resumen["estado"] = "error"
        resumen["detalle"] = str(e)
        return resumen

    n_columnas_datos = matriz.shape[1] - offset
    nombres = list(canales_detectados[:n_columnas_datos])
    # Si el archivo trae más columnas que nombres conocidos, se nombran por
    # posición en vez de descartarlas: perder una columna al convertir sería
    # peor que darle un nombre genérico.
    while len(nombres) < n_columnas_datos:
        nombres.append(f"canal_{len(nombres) + 1}")

    if offset:
        tiempo = matriz[:, 0].astype(float)
        origen_tiempo = "archivo"
    else:
        tiempo = np.arange(matriz.shape[0], dtype=float) / fs
        origen_tiempo = "sintetizado"

    datos = np.column_stack([tiempo, matriz[:, offset:].astype(float)])
    columnas = [COLUMNA_TIEMPO] + nombres

    guardar = np.savez_compressed if comprimir else np.savez
    destino.mkdir(parents=True, exist_ok=True)
    guardar(
        ruta_npz,
        datos=datos,
        columnas=np.asarray(columnas, dtype="U"),
        rep=np.int64(info.rep),
        iso=np.int64(info.iso),
        dsb=np.int64(info.dsb),
        rpm=np.float64(info.rpm),
        fs_hz=np.float64(fs),
        tiempo_origen=np.asarray(origen_tiempo, dtype="U"),
        n_muestras=np.int64(datos.shape[0]),
        archivo_origen=np.asarray(info.path.name, dtype="U"),
    )
    resumen["detalle"] = (f"{datos.shape[0]} x {datos.shape[1]}  "
                          f"tiempo:{origen_tiempo}")
    return resumen


# ============================================================
# LECTURA DE VUELTA
# ============================================================

def cargar(ruta):
    """
    Abre un .npz generado por este script.

    Returns
    -------
    (DataFrame, dict)
        El DataFrame con una columna por canal (la primera es `tiempo_s`) y
        un diccionario con los metadatos de la medición.

    Requiere pandas. Si solo se necesita el array, `np.load(ruta)["datos"]`
    basta y no hace falta pandas para nada.
    """
    import pandas as pd

    with np.load(ruta, allow_pickle=False) as d:
        df = pd.DataFrame(d["datos"], columns=[str(c) for c in d["columnas"]])
        meta = {
            "rep": int(d["rep"]), "iso": int(d["iso"]), "dsb": int(d["dsb"]),
            "rpm": float(d["rpm"]), "fs_hz": float(d["fs_hz"]),
            "tiempo_origen": str(d["tiempo_origen"]),
            "n_muestras": int(d["n_muestras"]),
            "archivo_origen": str(d["archivo_origen"]),
        }
    return df, meta


# ============================================================
# PROGRAMA
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Convierte la base de datos .txt a archivos .npz de numpy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--entrada", required=True,
                   help="Carpeta raíz que contiene las carpetas repX_isoYY_dsbZ")
    p.add_argument("--salida", required=True,
                   help="Carpeta donde se escriben los .npz (se crea si no existe)")
    p.add_argument("--fs", type=float, default=FS,
                   help="Frecuencia de muestreo [Hz], usada solo para generar "
                        "el eje de tiempo cuando el archivo no trae uno")
    p.add_argument("--patron-carpeta", dest="patron_carpeta",
                   default=FOLDER_PATTERN,
                   help="Regex de la carpeta, con los grupos rep, iso y dsb")
    p.add_argument("--patron-rpm", dest="patron_rpm", default=RPM_PATTERN,
                   help="Regex con un grupo de captura para la RPM del archivo")
    p.add_argument("--extension", default=FILE_EXTENSION,
                   help="Extensión de los archivos de medición")
    p.add_argument("--canales", nargs="+", default=CHANNEL_NAMES,
                   help="Nombres de los canales, en el orden de las columnas. "
                        "Solo se usan si la cabecera del archivo no los declara")
    p.add_argument("--columna-tiempo", dest="columna_tiempo", default="auto",
                   choices=["auto", "si", "no"],
                   help="Si la primera columna del bloque de datos es tiempo")
    p.add_argument("--comprimir", action="store_true",
                   help="Guardar comprimido: archivos mucho más pequeños, "
                        "carga algo más lenta")
    p.add_argument("--sobrescribir", action="store_true",
                   help="Rehacer los .npz que ya existan (por defecto se omiten)")
    p.add_argument("--limite", type=int, default=0,
                   help="Convertir solo los N primeros archivos (0 = todos). "
                        "Útil para probar antes de lanzar el lote completo")
    p.add_argument("--seco", action="store_true",
                   help="No escribe nada: solo lista lo que se convertiría")
    p.add_argument("--verboso", action="store_true",
                   help="Una línea por archivo en vez de una barra de avance")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verboso else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    entrada = Path(args.entrada).expanduser()
    destino = Path(args.salida).expanduser()
    columna_tiempo = {"auto": "auto", "si": True, "no": False}[args.columna_tiempo]

    try:
        archivos = io_utils.discover_files(
            entrada, args.patron_carpeta, args.patron_rpm, args.extension)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    if not archivos:
        print(f"ERROR: no se encontró ningún '*{args.extension}' bajo {entrada}\n"
              f"       en carpetas que cumplan el patrón "
              f"'{args.patron_carpeta}'.")
        return 1

    if args.limite:
        archivos = archivos[:args.limite]

    condiciones = sorted({f.condition_id for f in archivos})
    rpms = sorted({f.rpm for f in archivos})
    print(f"Entrada  : {entrada}")
    print(f"Salida   : {destino}")
    print(f"Archivos : {len(archivos)}  ({len(condiciones)} condiciones x "
          f"{len(rpms)} velocidades)")
    print(f"Formato  : {'comprimido' if args.comprimir else 'sin comprimir'}"
          f"   fs = {args.fs:g} Hz")
    print(f"Canales  : {', '.join(args.canales)}\n")

    if args.seco:
        print("MODO SECO — no se escribe nada.\n")
        for f in archivos:
            print(f"  {f.path.name}"
                  f"  ->  {nombre_salida(f.rep, f.iso, f.dsb, f.rpm)}")
        print(f"\n{len(archivos)} archivos se convertirían en {destino}")
        return 0

    t0 = time.time()
    resumenes = []
    for i, info in enumerate(archivos, 1):
        r = convertir_uno(info, destino, args.fs, args.canales, columna_tiempo,
                          args.comprimir, args.sobrescribir)
        resumenes.append(r)
        if args.verboso:
            print(f"  [{i}/{len(archivos)}] {r['estado']:11s} "
                  f"{r['destino']:38s} {r['detalle']}")
        elif i % 10 == 0 or i == len(archivos):
            print(f"\r  {i}/{len(archivos)} archivos...", end="", flush=True)
    if not args.verboso:
        print()

    hechos = [r for r in resumenes if r["estado"] == "convertido"]
    omitidos = [r for r in resumenes if r["estado"] == "omitido"]
    errores = [r for r in resumenes if r["estado"] == "error"]

    tam = sum(os.path.getsize(destino / r["destino"]) for r in hechos) / 1e6
    print("\n" + "=" * 70)
    print(f"Convertidos: {len(hechos)}   Omitidos: {len(omitidos)}   "
          f"Errores: {len(errores)}")
    if hechos:
        print(f"Escritos {tam:.1f} MB en {time.time() - t0:.1f} s  ->  {destino}")
    if omitidos and not args.sobrescribir:
        print(f"{len(omitidos)} ya existían; usa --sobrescribir para rehacerlos.")
    if errores:
        print("\nArchivos que no se pudieron convertir:")
        for r in errores:
            print(f"  {r['origen']}: {r['detalle']}")
    print("=" * 70)

    if hechos:
        ejemplo = destino / hechos[0]["destino"]
        print(f"\nPara abrir uno:\n"
              f"    import numpy as np, pandas as pd\n"
              f"    d  = np.load(r\"{ejemplo}\", allow_pickle=False)\n"
              f"    df = pd.DataFrame(d[\"datos\"], columns=d[\"columnas\"])")

    return 1 if errores and not hechos else 0


if __name__ == "__main__":
    sys.exit(main())
