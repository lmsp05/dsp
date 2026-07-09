"""
main.py
=======

Punto de entrada del análisis OMA por Peak Picking.

Flujo completo:

1.  Descubre la base de datos (``io_utils.discover_files``).
2.  Para cada archivo: lee la señal del sensor, calcula la PSD (Welch) y
    detecta los picos. Esta etapa puede ejecutarse en paralelo.
3.  Guarda una figura PSD + picos por archivo (opcional).
4.  Agrupa los resultados por velocidad de rotación: para cada RPM reúne los
    picos de todas las condiciones (repeticiones × viscosidades ×
    desbalances), los agrupa por proximidad en frecuencia y calcula la
    persistencia de cada grupo.
5.  Calcula la PSD promedio (media ± std) de cada RPM.
6.  Exporta todas las tablas a CSV y genera las figuras finales, incluido el
    diagrama tipo Campbell basado en persistencia.

Uso::

    python main.py                       # usa config.py tal cual
    python main.py --data-dir otra/ruta --sensor Mach1.P2.X
    python main.py --no-parallel --no-individual-plots

Todos los parámetros finos se ajustan en ``config.py``.
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
from io_utils import (FileInfo, discover_files, read_measurement,
                      read_measurement_multi)
from peak_detection import (PeakSet, detect_peaks, filter_peaks_by_coherence)
from persistence import PeakObservation, analyze_rpm_group, filter_persistent
from psd import PSDResult, average_psd, compute_cpsd_composite, compute_psd

logger = logging.getLogger("oma")

try:
    from tqdm import tqdm
except ImportError:  # tqdm es opcional: sin él solo se pierde la barra
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    """Configura logging a consola y, opcionalmente, a archivo."""
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if config.LOG_TO_FILE:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(output_dir / "analysis.log", mode="w",
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

def process_file(task: tuple[FileInfo, dict]) -> dict:
    """Lee un archivo, calcula su PSD y detecta los picos.

    Se ejecuta en un proceso trabajador cuando la paralelización está activa,
    por lo que recibe todos los parámetros de forma explícita y devuelve solo
    estructuras serializables.

    Parameters
    ----------
    task : tuple
        ``(FileInfo, params)`` donde ``params`` es el diccionario de
        parámetros construido por :func:`build_params`.

    Returns
    -------
    dict
        En caso de éxito: ``{"info", "freq", "psd", "peaks"}``.
        En caso de error: ``{"info", "error"}`` con el mensaje del fallo.
    """
    info, p = task
    try:
        coherence = None
        if p["mode"] == "cpsd":
            # Espectro cruzado compuesto entre sondas de proximidad.
            signals = read_measurement_multi(
                info,
                sensors=p["proximity_sensors"],
                channel_names=p["channel_names"],
                has_time_column=p["has_time_column"],
                remove_mean=p["remove_mean"],
            )
            psd_res, coherence = compute_cpsd_composite(
                signals,
                pairs=p["cpsd_pairs"],
                fs=p["fs"],
                nperseg=p["nperseg"],
                overlap=p["overlap"],
                window=p["window"],
                detrend=p["detrend"],
                fmin=p["fmin"],
                fmax=p["fmax"],
                smoothing=p["smoothing"],
                smoothing_window_bins=p["smoothing_window_bins"],
                compute_coherence=p["use_coherence"],
            )
        else:
            # Autoespectro (PSD) del sensor único configurado.
            record = read_measurement(
                info,
                sensor=p["sensor"],
                channel_names=p["channel_names"],
                has_time_column=p["has_time_column"],
                remove_mean=p["remove_mean"],
            )
            psd_res = compute_psd(
                record.signal,
                fs=p["fs"],
                nperseg=p["nperseg"],
                overlap=p["overlap"],
                window=p["window"],
                detrend=p["detrend"],
                fmin=p["fmin"],
                fmax=p["fmax"],
                smoothing=p["smoothing"],
                smoothing_window_bins=p["smoothing_window_bins"],
            )
        peaks = detect_peaks(
            psd_res.freq,
            psd_res.psd,
            prominence_mode=p["prominence_mode"],
            prominence_db=p["prominence_db"],
            auto_factor=p["auto_factor"],
            auto_min_db=p["auto_min_db"],
            noise_window_hz=p["noise_window_hz"],
            distance_hz=p["distance_hz"],
            width_hz=p["width_hz"],
            height_db=p["height_db"],
            max_peaks=p["max_peaks"],
            rpm=info.rpm,
            remove_harmonics=p["remove_harmonics"],
            harmonics_max_order=p["harmonics_max_order"],
            harmonics_tol_hz=p["harmonics_tol_hz"],
            harmonics_tol_rel=p["harmonics_tol_rel"],
            harmonics_refine_f1=p["harmonics_refine_f1"],
            harmonics_refine_search_rel=p["harmonics_refine_search_rel"],
        )
        # En modo CPSD, descartar los picos con coherencia baja entre sondas.
        if coherence is not None and p["use_coherence"]:
            peaks = filter_peaks_by_coherence(
                peaks, psd_res.freq, coherence, p["coherence_threshold"]
            )
        return {"info": info, "freq": psd_res.freq, "psd": psd_res.psd,
                "peaks": peaks, "coherence": coherence}
    except Exception as exc:  # noqa: BLE001 - un archivo malo no detiene nada
        return {"info": info, "error": f"{type(exc).__name__}: {exc}"}


def build_params(sensor: str, mode: str) -> dict:
    """Empaqueta los parámetros de ``config.py`` para los trabajadores."""
    return {
        "mode": mode,
        "sensor": sensor,
        "proximity_sensors": config.PROXIMITY_SENSORS,
        "cpsd_pairs": config.CPSD_PAIRS,
        "use_coherence": config.CPSD_USE_COHERENCE,
        "coherence_threshold": config.COHERENCE_THRESHOLD,
        "channel_names": config.CHANNEL_NAMES,
        "has_time_column": config.HAS_TIME_COLUMN,
        "remove_mean": config.REMOVE_MEAN,
        "fs": config.FS,
        "nperseg": config.WELCH_NPERSEG,
        "overlap": config.WELCH_OVERLAP,
        "window": config.WELCH_WINDOW,
        "detrend": config.WELCH_DETREND,
        "fmin": config.FMIN,
        "fmax": config.FMAX,
        "smoothing": config.SMOOTHING_ENABLED,
        "smoothing_window_bins": config.SMOOTHING_WINDOW_BINS,
        "prominence_mode": config.PEAK_PROMINENCE_MODE,
        "prominence_db": config.PEAK_PROMINENCE_DB,
        "auto_factor": config.PEAK_AUTO_FACTOR,
        "auto_min_db": config.PEAK_AUTO_MIN_DB,
        "noise_window_hz": config.PEAK_NOISE_WINDOW_HZ,
        "distance_hz": config.PEAK_DISTANCE_HZ,
        "width_hz": config.PEAK_WIDTH_HZ,
        "height_db": config.PEAK_HEIGHT_DB,
        "max_peaks": config.PEAK_MAX_PEAKS,
        "remove_harmonics": config.REMOVE_HARMONICS,
        "harmonics_max_order": config.HARMONICS_MAX_ORDER,
        "harmonics_tol_hz": config.HARMONICS_TOL_HZ,
        "harmonics_tol_rel": config.HARMONICS_TOL_REL,
        "harmonics_refine_f1": config.HARMONICS_REFINE_F1,
        "harmonics_refine_search_rel": config.HARMONICS_REFINE_SEARCH_REL,
    }


def run_processing(
    files: list[FileInfo],
    params: dict,
    parallel: bool,
) -> list[dict]:
    """Procesa todos los archivos, en serie o en paralelo, con progreso."""
    tasks = [(info, params) for info in files]
    results: list[dict] = []

    if parallel and len(files) > 1:
        workers = config.PARALLEL_WORKERS or mp.cpu_count()
        workers = min(workers, len(files))
        logger.info("Procesando %d archivos con %d procesos...",
                    len(files), workers)
        with mp.Pool(processes=workers) as pool:
            iterator = pool.imap(process_file, tasks, chunksize=1)
            if config.SHOW_PROGRESS:
                iterator = tqdm(iterator, total=len(tasks), desc="PSD + picos",
                                unit="archivo")
            results = list(iterator)
    else:
        logger.info("Procesando %d archivos en serie...", len(files))
        iterator = tasks
        if config.SHOW_PROGRESS:
            iterator = tqdm(tasks, desc="PSD + picos", unit="archivo")
        results = [process_file(t) for t in iterator]

    ok = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            logger.error("Archivo omitido %s: %s", r["info"].path, r["error"])
    logger.info("Archivos procesados correctamente: %d / %d.",
                len(ok), len(results))
    return ok


# ---------------------------------------------------------------------------
# Etapas de agregación y exportación
# ---------------------------------------------------------------------------

def export_raw_peaks(results: list[dict], output_dir: Path) -> None:
    """Exporta a CSV todos los picos individuales detectados (sin agrupar)."""
    rows = []
    for r in results:
        info: FileInfo = r["info"]
        peaks: PeakSet = r["peaks"]
        for f, a, adb, prom in zip(peaks.freq, peaks.amp, peaks.amp_db,
                                   peaks.prominence_db):
            rows.append({
                "rep": info.rep,
                "iso": info.iso,
                "dsb": info.dsb,
                "rpm": info.rpm,
                "condition": info.condition_id,
                "file": info.path.name,
                "freq_hz": f,
                "amp": a,
                "amp_db": adb,
                "prominence_db": prom,
            })
    df = pd.DataFrame(rows)
    path = output_dir / "tables" / "peaks_raw.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Picos individuales exportados: %s (%d picos).", path, len(df))


def aggregate_by_rpm(
    results: list[dict],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Agrupa por RPM, calcula persistencia y PSD promedio de cada velocidad.

    Returns
    -------
    tuple
        ``(tabla_global, psd_promedio)`` donde ``tabla_global`` concatena las
        tablas de grupos de todas las RPM y ``psd_promedio`` mapea cada RPM a
        ``(freq, psd_media, psd_std)``.
    """
    by_rpm: dict[float, list[dict]] = defaultdict(list)
    for r in results:
        by_rpm[r["info"].rpm].append(r)

    all_tables: list[pd.DataFrame] = []
    avg_by_rpm: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    avg_dir = output_dir / "tables" / "psd_average"
    avg_dir.mkdir(parents=True, exist_ok=True)

    for rpm in sorted(by_rpm):
        group = by_rpm[rpm]

        # PSD promedio ± desviación estándar de la velocidad.
        freq_avg, psd_mean, psd_std = average_psd(
            [PSDResult(freq=g["freq"], psd=g["psd"]) for g in group]
        )
        avg_by_rpm[rpm] = (freq_avg, psd_mean, psd_std)
        pd.DataFrame(
            {"freq_hz": freq_avg, "psd_mean": psd_mean, "psd_std": psd_std}
        ).to_csv(avg_dir / f"psd_avg_rpm{rpm:g}.csv", index=False)

        # Observaciones de picos de todas las condiciones de esta RPM.
        observations = []
        for g in group:
            info: FileInfo = g["info"]
            peaks: PeakSet = g["peaks"]
            for f, a, adb, prom in zip(peaks.freq, peaks.amp, peaks.amp_db,
                                       peaks.prominence_db):
                observations.append(PeakObservation(
                    condition_id=info.condition_id,
                    freq=float(f),
                    amp=float(a),
                    amp_db=float(adb),
                    prominence_db=float(prom),
                ))

        n_conditions = len({g["info"].condition_id for g in group})
        table = analyze_rpm_group(
            rpm=rpm,
            observations=observations,
            n_conditions=n_conditions,
            tolerance_hz=config.CLUSTER_TOLERANCE_HZ,
            avg_freq=freq_avg,
            avg_psd=psd_mean,
        )
        if not table.empty:
            all_tables.append(table)
        logger.info(
            "RPM %g: %d condiciones, %d picos -> %d grupos de frecuencia.",
            rpm, n_conditions, len(observations), len(table),
        )

    if all_tables:
        global_table = pd.concat(all_tables, ignore_index=True)
    else:
        global_table = pd.DataFrame(
            columns=["rpm", "freq_mean", "freq_std", "amp_mean", "amp_std",
                     "amp_db_mean", "n_appearances", "n_conditions",
                     "persistence", "psd_avg_at_freq"]
        )
    return global_table, avg_by_rpm


def export_tables(
    global_table: pd.DataFrame,
    persistent_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Exporta las tablas de grupos (todas y filtradas) a CSV."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    global_table.to_csv(tables_dir / "peak_groups_all.csv", index=False)

    # Tabla principal de resultados con los nombres de columna solicitados.
    renamed = persistent_table.rename(columns={
        "rpm": "RPM",
        "freq_mean": "Frecuencia_Hz",
        "freq_std": "Desviacion_estandar_Hz",
        "persistence": "Persistencia",
        "n_appearances": "Numero_apariciones",
        "n_conditions": "Numero_condiciones",
        "amp_mean": "Amplitud_promedio",
        "amp_std": "Amplitud_desviacion",
        "amp_db_mean": "Amplitud_promedio_dB",
        "psd_avg_at_freq": "Promedio_PSD",
    })
    column_order = [
        "RPM", "Frecuencia_Hz", "Persistencia", "Numero_apariciones",
        "Numero_condiciones", "Amplitud_promedio", "Amplitud_desviacion",
        "Amplitud_promedio_dB", "Desviacion_estandar_Hz", "Promedio_PSD",
    ]
    renamed = renamed[[c for c in column_order if c in renamed.columns]]
    renamed.to_csv(tables_dir / "persistent_peaks.csv", index=False)

    logger.info("Tablas exportadas en %s (grupos totales: %d, "
                "persistentes: %d).",
                tables_dir, len(global_table), len(persistent_table))


def generate_figures(
    results: list[dict],
    global_table: pd.DataFrame,
    persistent_table: pd.DataFrame,
    avg_by_rpm: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Genera todas las figuras del análisis."""
    fig_dir = output_dir / "figures"
    fmt = config.FIGURE_FORMAT
    dpi = config.FIGURE_DPI

    # Etiqueta del espectro según el modo (hay coherencia solo en CPSD).
    is_cpsd = any(r.get("coherence") is not None for r in results)
    spectrum_label = "CPSD" if is_cpsd else "PSD"

    # 1. Espectro individual de cada ensayo con sus picos.
    if config.SAVE_INDIVIDUAL_PSD_PLOTS:
        iterator = results
        if config.SHOW_PROGRESS:
            iterator = tqdm(results, desc="Figuras PSD", unit="figura")
        for r in iterator:
            info: FileInfo = r["info"]
            plotting.plot_psd_with_peaks(
                r["freq"], r["psd"], r["peaks"],
                title=info.label,
                path=fig_dir / "psd_individual" / f"{info.label}.{fmt}",
                scale=config.PSD_PLOT_SCALE,
                dpi=dpi,
                coherence=r.get("coherence"),
                coherence_threshold=(config.COHERENCE_THRESHOLD
                                     if config.CPSD_USE_COHERENCE else None),
                spectrum_label=spectrum_label,
            )

    # 2. Espectro promedio de cada velocidad con los picos persistentes.
    if config.SAVE_AVERAGE_PSD_PLOTS:
        for rpm, (freq_avg, psd_mean, psd_std) in avg_by_rpm.items():
            table_rpm = persistent_table[persistent_table["rpm"] == rpm] \
                if not persistent_table.empty else persistent_table
            plotting.plot_average_psd(
                freq_avg, psd_mean, psd_std, table_rpm, rpm,
                path=fig_dir / "psd_average" / f"psd_avg_rpm{rpm:g}.{fmt}",
                scale=config.PSD_PLOT_SCALE,
                dpi=dpi,
                spectrum_label=spectrum_label,
            )

    # 3. Figuras globales.
    plotting.plot_campbell(
        persistent_table,
        path=fig_dir / f"campbell.{fmt}",
        colormap=config.CAMPBELL_COLORMAP,
        show_harmonics=config.CAMPBELL_SHOW_HARMONIC_LINES,
        harmonic_orders=config.CAMPBELL_HARMONIC_ORDERS,
        dpi=dpi,
    )
    plotting.plot_persistence_histogram(
        global_table, path=fig_dir / f"persistence_histogram.{fmt}", dpi=dpi
    )
    plotting.plot_appearances_by_frequency(
        global_table, path=fig_dir / f"appearances_by_frequency.{fmt}", dpi=dpi
    )
    logger.info("Figuras guardadas en %s.", fig_dir)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos de línea de comandos (anulan valores de ``config.py``)."""
    parser = argparse.ArgumentParser(
        description="OMA por Peak Picking sobre un rotor Jeffcott con "
                    "cojinetes hidrodinámicos.",
    )
    parser.add_argument("--data-dir", default=None,
                        help="Carpeta raíz de la base de datos "
                             "(por defecto config.DATA_DIR).")
    parser.add_argument("--output-dir", default=None,
                        help="Carpeta de resultados "
                             "(por defecto config.OUTPUT_DIR).")
    parser.add_argument("--sensor", default=None,
                        help="Canal a analizar en modo psd "
                             "(por defecto config.SENSOR).")
    parser.add_argument("--mode", default=None, choices=["psd", "cpsd"],
                        help="Modo de análisis: 'psd' (autoespectro de un "
                             "sensor) o 'cpsd' (espectro cruzado entre las "
                             "sondas de proximidad, sin acelerómetros). "
                             "Por defecto config.ANALYSIS_MODE.")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Desactiva el procesamiento en paralelo.")
    parser.add_argument("--no-individual-plots", action="store_true",
                        help="No guardar la PSD individual de cada ensayo.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Analizar solo los primeros N archivos "
                             "(para pruebas rápidas).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el análisis completo. Devuelve el código de salida."""
    args = parse_args(argv)

    data_dir = Path(args.data_dir or config.DATA_DIR)
    output_dir = Path(args.output_dir or config.OUTPUT_DIR)
    sensor = args.sensor or config.SENSOR
    mode = args.mode or config.ANALYSIS_MODE
    parallel = config.PARALLEL_ENABLED and not args.no_parallel
    if args.no_individual_plots:
        config.SAVE_INDIVIDUAL_PSD_PLOTS = False

    setup_logging(output_dir)
    logger.info("=== OMA Peak Picking · rotor Jeffcott ===")
    if mode == "cpsd":
        logger.info("Datos: %s | Salida: %s | Modo: CPSD entre %s",
                    data_dir.resolve(), output_dir.resolve(),
                    ", ".join(config.PROXIMITY_SENSORS))
    else:
        logger.info("Datos: %s | Salida: %s | Modo: PSD | Sensor: %s",
                    data_dir.resolve(), output_dir.resolve(), sensor)

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
        logger.error("No se encontró ningún archivo de medición en %s.",
                     data_dir)
        return 1
    if args.limit:
        files = files[: args.limit]
        logger.info("Limitado a los primeros %d archivos.", len(files))

    # Índice de archivos analizados.
    index_path = output_dir / "tables" / "files_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"file": str(f.path), "rep": f.rep, "iso": f.iso, "dsb": f.dsb,
         "rpm": f.rpm, "condition": f.condition_id}
        for f in files
    ]).to_csv(index_path, index=False)

    # 2. Lectura + PSD/CPSD + peak picking (paralelizable).
    results = run_processing(files, build_params(sensor, mode), parallel)
    if not results:
        logger.error("Ningún archivo pudo procesarse; se aborta el análisis.")
        return 1

    # 3. Exportación de los picos individuales.
    export_raw_peaks(results, output_dir)

    # 4-7. Agrupamiento por RPM, clustering, persistencia y PSD promedio.
    global_table, avg_by_rpm = aggregate_by_rpm(results, output_dir)
    persistent_table = filter_persistent(
        global_table,
        min_persistence=config.MIN_PERSISTENCE,
        min_appearances=config.MIN_APPEARANCES,
    )
    logger.info("Grupos de picos: %d en total, %d superan persistencia >= "
                "%.2f y apariciones >= %d.",
                len(global_table), len(persistent_table),
                config.MIN_PERSISTENCE, config.MIN_APPEARANCES)

    # 8. Tablas CSV.
    export_tables(global_table, persistent_table, output_dir)

    # 9-10. Figuras (Campbell incluido).
    generate_figures(results, global_table, persistent_table, avg_by_rpm,
                     output_dir)

    logger.info("Análisis completado. Resultados en: %s",
                output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
