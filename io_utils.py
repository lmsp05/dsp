"""
io_utils.py
===========

Lectura de la base de datos experimental.

Este módulo se encarga de:

* Recorrer la estructura de carpetas ``repX_isoYY_dsbZ`` y localizar los
  archivos ``.txt`` de medición.
* Extraer los metadatos de cada medición (repetición, viscosidad, desbalance
  y RPM) a partir de los nombres de carpeta y archivo.
* Leer los archivos de medición ignorando los bloques de encabezado
  (``BEGIN_HEADER ... END_HEADER``, ``BEGIN_INFO ... END_INFO``) y extrayendo
  únicamente el bloque numérico ``BEGIN_DATA ... END_DATA``.
* Detectar automáticamente el delimitador de columnas, el separador decimal
  (punto o coma) y la posible columna de tiempo.
* Seleccionar la columna correspondiente al sensor configurado.

Todas las funciones incluyen manejo robusto de errores: un archivo corrupto
o incompleto se registra en el log y se omite sin detener el análisis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Marcadores de los bloques del archivo (insensibles a mayúsculas/minúsculas).
_BEGIN_DATA_RE = re.compile(r"^\s*BEGIN[_\s]?DATA\b", re.IGNORECASE)
_END_DATA_RE = re.compile(r"^\s*END[_\s]?DATA\b", re.IGNORECASE)

# Línea compuesta únicamente por números (con posibles comas decimales),
# usada como respaldo cuando el archivo no tiene marcadores BEGIN/END_DATA.
_NUMERIC_LINE_RE = re.compile(
    r"^\s*[-+]?\d[\d.,eE+\-]*(?:[\s;,\t]+[-+]?\d[\d.,eE+\-]*)*\s*$"
)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Metadatos de un archivo de medición (aún sin leer la señal)."""

    path: Path          #: Ruta completa al archivo .txt.
    rep: int            #: Número de repetición del ensayo.
    iso: int            #: Grado ISO de viscosidad del lubricante.
    dsb: int            #: Configuración de desbalance.
    rpm: float          #: Velocidad de rotación extraída del nombre.

    @property
    def condition_id(self) -> str:
        """Identificador único de la condición experimental (sin la RPM)."""
        return f"rep{self.rep}_iso{self.iso}_dsb{self.dsb}"

    @property
    def label(self) -> str:
        """Etiqueta legible del ensayo completo (condición + RPM)."""
        return f"{self.condition_id}_rpm{self.rpm:g}"


@dataclass
class MeasurementRecord:
    """Archivo de medición ya leído: metadatos + señal del sensor."""

    info: FileInfo                          #: Metadatos del archivo.
    signal: np.ndarray                      #: Señal temporal del sensor.
    channel_names: list[str] = field(default_factory=list)  #: Canales del archivo.
    sensor: str = ""                        #: Nombre del canal analizado.


# ---------------------------------------------------------------------------
# Descubrimiento de archivos
# ---------------------------------------------------------------------------

def discover_files(
    data_dir: str | Path,
    folder_pattern: str,
    rpm_pattern: str,
    file_extension: str = ".txt",
) -> list[FileInfo]:
    """Recorre la base de datos y devuelve la lista de archivos a analizar.

    Parameters
    ----------
    data_dir : str or Path
        Carpeta raíz con las carpetas de condiciones experimentales.
    folder_pattern : str
        Expresión regular con los grupos nombrados ``rep``, ``iso`` y ``dsb``.
    rpm_pattern : str
        Expresión regular con un grupo de captura para la RPM del archivo.
    file_extension : str
        Extensión de los archivos de medición.

    Returns
    -------
    list of FileInfo
        Metadatos de cada archivo encontrado, ordenados por condición y RPM.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"La carpeta de datos no existe: {data_dir.resolve()}"
        )

    folder_re = re.compile(folder_pattern, re.IGNORECASE)
    rpm_re = re.compile(rpm_pattern)

    files: list[FileInfo] = []
    skipped_folders: list[str] = []

    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        match = folder_re.search(folder.name)
        if match is None:
            skipped_folders.append(folder.name)
            continue

        rep = int(match.group("rep"))
        iso = int(match.group("iso"))
        dsb = int(match.group("dsb"))

        for file_path in sorted(folder.glob(f"*{file_extension}")):
            rpm_match = rpm_re.search(file_path.name)
            if rpm_match is None:
                logger.warning(
                    "No se pudo extraer la RPM del archivo %s; se omite.",
                    file_path,
                )
                continue
            rpm = float(rpm_match.group(1))
            files.append(FileInfo(path=file_path, rep=rep, iso=iso, dsb=dsb, rpm=rpm))

    if skipped_folders:
        logger.info(
            "Carpetas ignoradas (no coinciden con el patrón): %s",
            ", ".join(skipped_folders),
        )

    files.sort(key=lambda f: (f.rep, f.iso, f.dsb, f.rpm))
    logger.info(
        "Base de datos: %d archivos en %d condiciones, %d velocidades.",
        len(files),
        len({f.condition_id for f in files}),
        len({f.rpm for f in files}),
    )
    return files


# ---------------------------------------------------------------------------
# Lectura de un archivo de medición
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> list[str]:
    """Lee el archivo como texto probando varias codificaciones habituales."""
    for encoding in ("utf-8", "latin-1", "utf-16"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"No se pudo decodificar el archivo {path}")


def _extract_data_block(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separa el bloque de datos numéricos del resto del archivo.

    Devuelve una tupla ``(header_lines, data_lines)``. Si el archivo tiene
    los marcadores ``BEGIN_DATA``/``END_DATA`` se usan directamente; en caso
    contrario se busca la primera racha de líneas puramente numéricas.
    """
    begin_idx = end_idx = None
    for i, line in enumerate(lines):
        if begin_idx is None and _BEGIN_DATA_RE.match(line):
            begin_idx = i
        elif begin_idx is not None and _END_DATA_RE.match(line):
            end_idx = i
            break

    if begin_idx is not None:
        end_idx = end_idx if end_idx is not None else len(lines)
        header = lines[:begin_idx]
        data = [ln for ln in lines[begin_idx + 1 : end_idx] if ln.strip()]
        return header, data

    # Respaldo: primera racha larga de líneas numéricas consecutivas.
    start = None
    for i, line in enumerate(lines):
        if _NUMERIC_LINE_RE.match(line):
            if start is None:
                start = i
        elif start is not None:
            if i - start >= 10:  # racha suficientemente larga -> datos
                break
            start = None
    if start is None:
        raise ValueError("No se encontró ningún bloque de datos numéricos.")

    data = []
    for line in lines[start:]:
        if not _NUMERIC_LINE_RE.match(line):
            break
        data.append(line)
    return lines[:start], data


def _detect_delimiter_and_decimal(sample_line: str) -> tuple[str | None, str]:
    """Deduce el delimitador de columnas y el separador decimal de una línea.

    Returns
    -------
    tuple
        ``(delimitador, separador_decimal)``. Un delimitador ``None`` indica
        espacios en blanco genéricos (comportamiento por defecto de numpy).
    """
    if "\t" in sample_line:
        delimiter = "\t"
    elif ";" in sample_line:
        delimiter = ";"
    elif "," in sample_line and "." in sample_line:
        # Hay puntos decimales, por lo que la coma solo puede ser delimitador.
        delimiter = ","
    else:
        delimiter = None  # espacios

    # Si tras fijar el delimitador quedan comas dentro de los campos y no hay
    # puntos, se trata de separador decimal europeo.
    decimal = "."
    if delimiter != "," and "," in sample_line and "." not in sample_line:
        decimal = ","
    return delimiter, decimal


def _parse_data_matrix(data_lines: list[str]) -> np.ndarray:
    """Convierte las líneas del bloque de datos en una matriz float 2D.

    Se toleran líneas corruptas aisladas (se descartan con un aviso), pero si
    más del 5 % de las líneas son inválidas el archivo se considera corrupto.
    """
    delimiter, decimal = _detect_delimiter_and_decimal(data_lines[0])

    rows: list[list[float]] = []
    n_bad = 0
    n_cols: int | None = None

    for line in data_lines:
        if decimal == ",":
            line = line.replace(",", ".")
        parts = line.split(delimiter) if delimiter else line.split()
        try:
            row = [float(p) for p in parts if p.strip() != ""]
        except ValueError:
            n_bad += 1
            continue
        if not row:
            continue
        if n_cols is None:
            n_cols = len(row)
        if len(row) != n_cols:
            n_bad += 1
            continue
        rows.append(row)

    if not rows:
        raise ValueError("El bloque de datos no contiene filas válidas.")
    if n_bad > 0.05 * len(data_lines):
        raise ValueError(
            f"Demasiadas líneas corruptas: {n_bad}/{len(data_lines)}."
        )
    if n_bad:
        logger.debug("Se descartaron %d líneas corruptas.", n_bad)

    return np.asarray(rows, dtype=float)


# Token de nombre de sensor con notación por puntos (p. ej. Mach1.P1.Y).
_SENSOR_TOKEN_RE = re.compile(r"[A-Za-z]\w*(?:\.\w+)+")


def _parse_name_row(header_lines: list[str]) -> list[str] | None:
    """Extrae el mapa completo de canales de la fila ``Name`` del bloque INFO.

    El equipo de adquisición escribe una línea del tipo::

        Name<TAB>Ext. sync. 1<TAB>Input 1: Mach1.P1.Y<TAB>Input 2: Mach1.P1.X ...

    con una etiqueta por **columna de datos**, en el mismo orden. Es
    fundamental respetar ese orden: la primera columna es el tacómetro
    (``Ext. sync.``), no una sonda, de modo que ``Mach1.P1.Y`` es en realidad
    la **segunda** columna. Se devuelve la lista de nombres alineada con las
    columnas: para cada campo se extrae el token de sensor con notación por
    puntos si existe (``Mach1.P1.Y``) y, si no (tacómetro/sincronismo), se
    conserva la etiqueta limpia.

    Returns
    -------
    list of str or None
        Un nombre por columna de datos, o ``None`` si no hay fila ``Name``.
    """
    for line in header_lines:
        fields = [f.strip() for f in line.split("\t")] if "\t" in line \
            else line.split()
        if len(fields) < 3 or fields[0].strip().lower() != "name":
            continue
        cols = [f for f in fields[1:] if f != ""]
        if len(cols) < 2:
            continue
        names: list[str] = []
        for i, col in enumerate(cols):
            match = _SENSOR_TOKEN_RE.search(col)
            names.append(match.group(0) if match else (col or f"col{i + 1}"))
        logger.debug("Canales de la fila 'Name': %s", names)
        return names
    return None


def _detect_channel_names(header_lines: list[str], expected: list[str]) -> list[str]:
    """Extrae los nombres de canal del encabezado, alineados con las columnas.

    Estrategia:

    1. Fila ``Name`` del bloque INFO (formato real del equipo): da el mapa
       completo columna->canal, incluida la columna de tacómetro inicial.
    2. Respaldo: primera línea que contenga al menos dos de los nombres
       esperados (p. ej. cabeceras simplificadas o archivos sintéticos).
    3. Último recurso: los nombres por defecto de ``config.CHANNEL_NAMES``.
    """
    named = _parse_name_row(header_lines)
    if named is not None:
        return named

    for line in header_lines:
        tokens = _SENSOR_TOKEN_RE.findall(line)
        hits = [t for t in tokens if t in expected]
        if len(hits) >= 2:
            logger.debug("Canales detectados en el encabezado: %s", hits)
            return hits
    return list(expected)


def _find_time_column(matrix: np.ndarray, mode: bool | str) -> bool:
    """Decide si la primera columna del bloque de datos es la base de tiempos."""
    if mode is True:
        return True
    if mode is False:
        return False
    # Modo "auto": columna estrictamente creciente y aproximadamente uniforme.
    col = matrix[: min(len(matrix), 1000), 0]
    if len(col) < 3:
        return False
    diffs = np.diff(col)
    return bool(np.all(diffs > 0) and np.std(diffs) < 0.01 * abs(np.mean(diffs)))


def _load_matrix(
    info: FileInfo,
    channel_names: list[str],
    has_time_column: bool | str = "auto",
) -> tuple[np.ndarray, list[str], int]:
    """Lee y valida el bloque de datos de un archivo de medición.

    Returns
    -------
    tuple
        ``(matrix, channels, offset)`` donde ``matrix`` es la matriz numérica
        completa, ``channels`` los nombres de canal (detectados o por
        defecto) y ``offset`` vale 1 si la primera columna es tiempo.
    """
    lines = _read_text(info.path)
    header_lines, data_lines = _extract_data_block(lines)
    if len(data_lines) < 100:
        raise ValueError(
            f"Bloque de datos demasiado corto ({len(data_lines)} líneas)."
        )

    matrix = _parse_data_matrix(data_lines)
    channels = _detect_channel_names(header_lines, channel_names)

    offset = 0
    if matrix.shape[1] > 1 and _find_time_column(matrix, has_time_column):
        offset = 1
    return matrix, channels, offset


def _extract_channel(
    matrix: np.ndarray,
    channels: list[str],
    offset: int,
    sensor: str,
    remove_mean: bool,
) -> np.ndarray:
    """Extrae la columna de un sensor de la matriz de datos."""
    n_data_cols = matrix.shape[1] - offset
    if sensor not in channels:
        raise ValueError(
            f"El sensor '{sensor}' no está entre los canales {channels}."
        )
    col_idx = channels.index(sensor)
    if col_idx >= n_data_cols:
        raise ValueError(
            f"El archivo tiene {n_data_cols} columnas de datos, pero el "
            f"sensor '{sensor}' corresponde a la columna {col_idx + 1}."
        )
    signal = matrix[:, offset + col_idx].astype(float)
    if remove_mean:
        signal = signal - float(np.mean(signal))
    return signal


def read_measurement(
    info: FileInfo,
    sensor: str,
    channel_names: list[str],
    has_time_column: bool | str = "auto",
    remove_mean: bool = True,
) -> MeasurementRecord:
    """Lee un archivo de medición y devuelve la señal del sensor solicitado.

    Parameters
    ----------
    info : FileInfo
        Metadatos del archivo (ruta, condición, RPM).
    sensor : str
        Nombre del canal a extraer (p. ej. ``"Mach1.P1.Y"``).
    channel_names : list of str
        Orden por defecto de los canales si el encabezado no los declara.
    has_time_column : bool or "auto"
        Tratamiento de la posible columna de tiempo (ver ``config.py``).
    remove_mean : bool
        Si es True, se resta la media de la señal.

    Returns
    -------
    MeasurementRecord
        Registro con la señal del sensor.

    Raises
    ------
    ValueError
        Si el archivo está corrupto, incompleto o no contiene el sensor.
    """
    matrix, channels, offset = _load_matrix(info, channel_names, has_time_column)
    signal = _extract_channel(matrix, channels, offset, sensor, remove_mean)
    return MeasurementRecord(
        info=info,
        signal=signal,
        channel_names=channels[: matrix.shape[1] - offset],
        sensor=sensor,
    )


def read_measurement_multi(
    info: FileInfo,
    sensors: list[str],
    channel_names: list[str],
    has_time_column: bool | str = "auto",
    remove_mean: bool = True,
) -> dict[str, np.ndarray]:
    """Lee varios canales de un mismo archivo con un único parsing.

    Se usa en el modo CPSD, donde hacen falta simultáneamente todas las
    sondas de proximidad de la misma medición.

    Parameters
    ----------
    info : FileInfo
        Metadatos del archivo.
    sensors : list of str
        Canales a extraer (p. ej. las cuatro sondas de proximidad).
    channel_names, has_time_column, remove_mean
        Igual que en :func:`read_measurement`.

    Returns
    -------
    dict
        Mapa ``sensor -> señal`` con un array por canal solicitado.

    Raises
    ------
    ValueError
        Si el archivo está corrupto o falta alguno de los sensores.
    """
    matrix, channels, offset = _load_matrix(info, channel_names, has_time_column)
    return {
        sensor: _extract_channel(matrix, channels, offset, sensor, remove_mean)
        for sensor in sensors
    }
