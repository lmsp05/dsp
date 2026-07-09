"""
ssi_main.py
===========

Punto de entrada del análisis OMA automatizado por **Identificación
Estocástica de Subespacios basada en referencias** (SSI-COV/ref).

Implementa el procedimiento descrito en la bibliografía compartida:

* **Peeters & De Roeck (2000)** — SSI basado en covarianzas y referencias:
  para cada ensayo se estima el espacio de estados a partir de la matriz
  bloque-Toeplitz de covarianzas y se identifican los polos (frecuencia,
  amortiguamiento y forma modal) para un rango de órdenes del modelo, lo que
  produce el diagrama de estabilización.
* **Dreher et al. (2023)** — automatización mediante *clustering*: los polos
  estables se agrupan automáticamente en modos físicos y los armónicos de la
  velocidad de giro se marcan aparte, sin intervención manual sobre el
  diagrama de estabilización.

Flujo completo:

1.  Descubre la base de datos (``io_utils.discover_files``), reutilizando la
    misma estructura de carpetas ``repX_isoYY_dsbZ`` que el peak picking.
2.  Para cada archivo (paralelizable): lee las 4 sondas de proximidad, ejecuta
    SSI-COV/ref, agrupa los polos estables en modos, marca los armónicos y
    guarda el diagrama de estabilización.
3.  Agrupa los modos por velocidad de rotación (RPM): para cada RPM reúne los
    modos de todas las condiciones, los agrupa por frecuencia y calcula su
    persistencia entre condiciones (robustez frente a viscosidad y desbalance).
4.  Exporta las tablas a CSV y genera el diagrama de Campbell de las
    frecuencias naturales identificadas.

Uso::

    python ssi_main.py                       # usa config.py tal cual
    python ssi_main.py --data-dir otra/ruta --output-dir results_ssi
    python ssi_main.py --no-parallel --limit 10

Todos los parámetros finos (referencias, órdenes, diezmado, tolerancias de
estabilidad y de clustering) se ajustan en la sección 9 de ``config.py``.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config
import plotting
import ssi
from io_utils import FileInfo, discover_files, read_measurement_multi
from peak_detection import refine_rotation_frequency

logger = logging.getLogger("ssi")

try:
    from tqdm import tqdm
except ImportError:  # tqdm es opcional
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Configuración de logging (idéntica a main.py para coherencia)
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    """Configura logging a consola y, opcionalmente, a archivo."""
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if config.LOG_TO_FILE:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(output_dir / "ssi_analysis.log", mode="w",
                                encoding="utf-8")
        )
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Procesamiento de un archivo (ejecutable en un proceso hijo)
# ---------------------------------------------------------------------------

def build_params() -> dict:
    """Empaqueta los parámetros SSI de ``config.py`` para los trabajadores."""
    return {
        "output_sensors": config.SSI_OUTPUT_SENSORS,
        "ref_sensors": config.SSI_REF_SENSORS,
        "channel_names": config.CHANNEL_NAMES,
        "has_time_column": config.HAS_TIME_COLUMN,
        "fs": config.FS,
        "decimation": config.SSI_DECIMATION,
        "block_rows": config.SSI_BLOCK_ROWS,
        "order_min": config.SSI_ORDER_MIN,
        "order_max": config.SSI_ORDER_MAX,
        "order_step": config.SSI_ORDER_STEP,
        "zeta_max": config.SSI_ZETA_MAX,
        "fmin": config.SSI_FMIN,
        "fmax": config.SSI_FMAX,
        "df_tol": config.SSI_DF_TOL,
        "dz_tol": config.SSI_DZ_TOL,
        "mac_tol": config.SSI_MAC_TOL,
        "cluster_freq_weight": config.SSI_CLUSTER_FREQ_WEIGHT,
        "cluster_mac_weight": config.SSI_CLUSTER_MAC_WEIGHT,
        "cluster_distance": config.SSI_CLUSTER_DISTANCE,
        "min_cluster_fraction": config.SSI_MIN_CLUSTER_FRACTION,
        "min_cluster_size": config.SSI_MIN_CLUSTER_SIZE,
        "flag_harmonics": config.SSI_FLAG_HARMONICS,
        "harmonics_max_order": config.SSI_HARMONICS_MAX_ORDER,
        "harmonics_tol_hz": config.SSI_HARMONICS_TOL_HZ,
        "harmonics_tol_rel": config.SSI_HARMONICS_TOL_REL,
        "harmonics_refine_search_rel": config.HARMONICS_REFINE_SEARCH_REL,
        "save_stab_plots": config.SSI_SAVE_STABILIZATION_PLOTS,
        "figure_format": config.FIGURE_FORMAT,
        "figure_dpi": config.FIGURE_DPI,
        "campbell_harmonic_orders": config.CAMPBELL_HARMONIC_ORDERS,
        "output_dir": str(config.OUTPUT_DIR),
    }


def process_file(task: tuple[FileInfo, dict]) -> dict:
    """Ejecuta SSI-COV/ref sobre un archivo y devuelve sus modos.

    Se ejecuta en un proceso trabajador cuando la paralelización está activa.
    El diagrama de estabilización se guarda aquí mismo (matplotlib Agg es
    seguro por proceso) para no transportar objetos pesados con las formas
    modales de vuelta al proceso principal; solo se devuelve la tabla de modos.

    Returns
    -------
    dict
        En éxito: ``{"info", "modes"}`` con ``modes`` un ``DataFrame``.
        En error: ``{"info", "error"}``.
    """
    info, p = task
    try:
        signals = read_measurement_multi(
            info,
            sensors=p["output_sensors"],
            channel_names=p["channel_names"],
            has_time_column=p["has_time_column"],
            remove_mean=False,  # la SSI resta la media internamente
        )
        stab = ssi.run_stabilization(
            signals,
            output_sensors=p["output_sensors"],
            ref_sensors=p["ref_sensors"],
            fs=p["fs"],
            decimation_factor=p["decimation"],
            block_rows=p["block_rows"],
            order_min=p["order_min"],
            order_max=p["order_max"],
            order_step=p["order_step"],
            zeta_max=p["zeta_max"],
            fmin=p["fmin"],
            fmax=p["fmax"],
            df_tol=p["df_tol"],
            dz_tol=p["dz_tol"],
            mac_tol=p["mac_tol"],
            remove_mean=True,
        )
        clusters = ssi.cluster_modes(
            stab.stable_poles,
            n_orders=len(stab.orders),
            freq_weight=p["cluster_freq_weight"],
            mac_weight=p["cluster_mac_weight"],
            distance_threshold=p["cluster_distance"],
            min_cluster_fraction=p["min_cluster_fraction"],
            min_cluster_size=p["min_cluster_size"],
        )

        # Frecuencia de rotación 1X refinada desde el espectro de referencia.
        f_rot = 0.0
        if len(stab.ref_freq) and info.rpm > 0:
            f_rot = refine_rotation_frequency(
                stab.ref_freq, stab.ref_spectrum, info.rpm,
                search_rel=p["harmonics_refine_search_rel"],
            )
        if p["flag_harmonics"] and f_rot > 0:
            ssi.flag_harmonics(
                clusters, f_rot, p["harmonics_max_order"],
                p["harmonics_tol_hz"], p["harmonics_tol_rel"],
            )

        if p["save_stab_plots"]:
            fig_path = (Path(p["output_dir"]) / "figures" / "stabilization"
                        / f"{info.label}.{p['figure_format']}")
            plotting.plot_stabilization_diagram(
                stab, fig_path, clusters=clusters, f_rotation=f_rot,
                harmonic_orders=p["campbell_harmonic_orders"],
                dpi=p["figure_dpi"],
            )

        modes = ssi.modes_to_dataframe(
            clusters, rpm=info.rpm, condition_id=info.condition_id,
            file_name=info.path.name,
        )
        return {"info": info, "modes": modes}
    except Exception as exc:  # noqa: BLE001 - un archivo malo no detiene nada
        return {"info": info, "error": f"{type(exc).__name__}: {exc}"}


def run_processing(
    files: list[FileInfo],
    params: dict,
    parallel: bool,
) -> list[dict]:
    """Procesa todos los archivos, en serie o en paralelo, con progreso."""
    tasks = [(info, params) for info in files]

    if parallel and len(files) > 1:
        workers = config.PARALLEL_WORKERS or mp.cpu_count()
        workers = min(workers, len(files))
        logger.info("Procesando %d archivos con SSI en %d procesos...",
                    len(files), workers)
        with mp.Pool(processes=workers) as pool:
            iterator = pool.imap(process_file, tasks, chunksize=1)
            if config.SHOW_PROGRESS:
                iterator = tqdm(iterator, total=len(tasks), desc="SSI",
                                unit="archivo")
            results = list(iterator)
    else:
        logger.info("Procesando %d archivos con SSI en serie...", len(files))
        iterator = tqdm(tasks, desc="SSI", unit="archivo") \
            if config.SHOW_PROGRESS else tasks
        results = [process_file(t) for t in iterator]

    ok = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            logger.error("Archivo omitido %s: %s", r["info"].path, r["error"])
    logger.info("Archivos procesados correctamente: %d / %d.",
                len(ok), len(results))
    return ok


# ---------------------------------------------------------------------------
# Agregación y exportación
# ---------------------------------------------------------------------------

def export_raw_modes(results: list[dict], output_dir: Path) -> pd.DataFrame:
    """Concatena y exporta todos los modos identificados por ensayo."""
    frames = [r["modes"] for r in results if not r["modes"].empty]
    all_modes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = output_dir / "tables" / "ssi_modes_raw.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    all_modes.to_csv(path, index=False)
    logger.info("Modos por ensayo exportados: %s (%d modos).",
                path, len(all_modes))
    return all_modes


def aggregate_by_rpm(all_modes: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los modos por RPM y calcula la persistencia entre condiciones."""
    if all_modes.empty:
        return pd.DataFrame()

    tables = []
    for rpm in sorted(all_modes["rpm"].unique()):
        sub = all_modes[all_modes["rpm"] == rpm]
        n_conditions = sub["condition"].nunique()
        table = ssi.aggregate_modes_by_rpm(
            sub,
            n_conditions=n_conditions,
            tolerance_hz=config.SSI_CLUSTER_TOLERANCE_HZ,
            include_harmonics=not config.SSI_EXCLUDE_HARMONICS,
        )
        if not table.empty:
            tables.append(table)
        logger.info("RPM %g: %d condiciones, %d modos -> %d frecuencias.",
                    rpm, n_conditions, len(sub), len(table))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def filter_persistent(table: pd.DataFrame) -> pd.DataFrame:
    """Filtra por persistencia y apariciones mínimas."""
    if table.empty:
        return table
    mask = (table["persistence"] >= config.SSI_MIN_PERSISTENCE) & (
        table["n_appearances"] >= config.SSI_MIN_APPEARANCES)
    return table.loc[mask].reset_index(drop=True)


def export_tables(
    global_table: pd.DataFrame,
    persistent_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Exporta la tabla global y la principal de frecuencias naturales."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    global_table.to_csv(tables_dir / "ssi_modes_all.csv", index=False)

    renamed = persistent_table.rename(columns={
        "rpm": "RPM",
        "freq_mean": "Frecuencia_Natural_Hz",
        "freq_std": "Desviacion_Frecuencia_Hz",
        "zeta_mean": "Amortiguamiento",
        "zeta_std": "Desviacion_Amortiguamiento",
        "persistence": "Persistencia",
        "n_appearances": "Numero_apariciones",
        "n_conditions": "Numero_condiciones",
    })
    order = ["RPM", "Frecuencia_Natural_Hz", "Amortiguamiento", "Persistencia",
             "Numero_apariciones", "Numero_condiciones",
             "Desviacion_Frecuencia_Hz", "Desviacion_Amortiguamiento"]
    renamed = renamed[[c for c in order if c in renamed.columns]]
    renamed.to_csv(tables_dir / "ssi_natural_frequencies.csv", index=False)
    logger.info("Tablas SSI exportadas en %s (grupos: %d, persistentes: %d).",
                tables_dir, len(global_table), len(persistent_table))


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos de línea de comandos (anulan valores de ``config.py``)."""
    parser = argparse.ArgumentParser(
        description="OMA automatizado por SSI-COV/ref sobre un rotor Jeffcott "
                    "con cojinetes hidrodinámicos (Peeters & De Roeck 2000, "
                    "Dreher et al. 2023).",
    )
    parser.add_argument("--data-dir", default=None,
                        help="Carpeta raíz de la base de datos.")
    parser.add_argument("--output-dir", default=None,
                        help="Carpeta de resultados.")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Desactiva el procesamiento en paralelo.")
    parser.add_argument("--no-stab-plots", action="store_true",
                        help="No guardar los diagramas de estabilización.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Analizar solo los primeros N archivos.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el análisis SSI completo. Devuelve el código de salida."""
    args = parse_args(argv)

    data_dir = Path(args.data_dir or config.DATA_DIR)
    output_dir = Path(args.output_dir or config.OUTPUT_DIR)
    parallel = config.PARALLEL_ENABLED and not args.no_parallel
    if args.no_stab_plots:
        config.SSI_SAVE_STABILIZATION_PLOTS = False

    setup_logging(output_dir)
    logger.info("=== OMA por SSI-COV/ref · rotor Jeffcott ===")
    logger.info("Datos: %s | Salida: %s", data_dir.resolve(),
                output_dir.resolve())
    logger.info("Salidas: %s | Referencias: %s | Diezmado: %d (fs_ef=%.0f Hz)",
                ", ".join(config.SSI_OUTPUT_SENSORS),
                ", ".join(config.SSI_REF_SENSORS),
                config.SSI_DECIMATION, config.FS / config.SSI_DECIMATION)

    # 1. Descubrimiento de la base de datos.
    try:
        files = discover_files(
            data_dir,
            folder_pattern=config.FOLDER_PATTERN,
            rpm_pattern=config.RPM_PATTERN,
            file_extension=config.FILE_EXTENSION,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    if not files:
        logger.error("No se encontró ningún archivo de medición en %s.", data_dir)
        return 1
    if args.limit:
        files = files[: args.limit]
        logger.info("Limitado a los primeros %d archivos.", len(files))

    index_path = output_dir / "tables" / "files_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"file": str(f.path), "rep": f.rep, "iso": f.iso, "dsb": f.dsb,
         "rpm": f.rpm, "condition": f.condition_id}
        for f in files
    ]).to_csv(index_path, index=False)

    # Actualiza el output_dir en los parámetros (por si viene de --output-dir).
    params = build_params()
    params["output_dir"] = str(output_dir)

    # 2. SSI por archivo (paralelizable).
    results = run_processing(files, params, parallel)
    if not results:
        logger.error("Ningún archivo pudo procesarse; se aborta el análisis.")
        return 1

    # 3. Exportación de los modos por ensayo.
    all_modes = export_raw_modes(results, output_dir)

    # 4. Agrupamiento por RPM y persistencia.
    global_table = aggregate_by_rpm(all_modes)
    persistent_table = filter_persistent(global_table)
    logger.info("Frecuencias naturales: %d grupos, %d superan persistencia >= "
                "%.2f y apariciones >= %d.",
                len(global_table), len(persistent_table),
                config.SSI_MIN_PERSISTENCE, config.SSI_MIN_APPEARANCES)

    # 5. Tablas y Campbell.
    export_tables(global_table, persistent_table, output_dir)
    plotting.plot_ssi_campbell(
        persistent_table,
        path=output_dir / "figures" / f"ssi_campbell.{config.FIGURE_FORMAT}",
        colormap=config.CAMPBELL_COLORMAP,
        show_harmonics=config.CAMPBELL_SHOW_HARMONIC_LINES,
        harmonic_orders=config.CAMPBELL_HARMONIC_ORDERS,
        dpi=config.FIGURE_DPI,
    )
    if not global_table.empty:
        plotting.plot_persistence_histogram(
            global_table,
            path=(output_dir / "figures"
                  / f"ssi_persistence_histogram.{config.FIGURE_FORMAT}"),
            dpi=config.FIGURE_DPI,
        )

    logger.info("Análisis SSI completado. Resultados en: %s",
                output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
