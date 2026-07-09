"""
config.py
=========

Configuración centralizada del proyecto OMA Peak Picking.

Todos los parámetros del análisis se definen aquí para que el usuario pueda
ajustar el comportamiento del programa sin tocar el resto del código.

Los parámetros se agrupan por etapa del análisis:

1. Rutas y estructura de la base de datos.
2. Adquisición (frecuencia de muestreo, sensor, columnas).
3. Estimación espectral (Welch).
4. Detección de picos (peak picking).
5. Agrupamiento de frecuencias y persistencia.
6. Eliminación de armónicos.
7. Gráficos y salidas.
8. Ejecución (logging, paralelización).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Rutas y estructura de la base de datos
# ---------------------------------------------------------------------------

#: Carpeta del proyecto (donde está este archivo). Las rutas por defecto se
#: anclan aquí para que los resultados aparezcan siempre junto al código,
#: independientemente del directorio desde el que se ejecute ``main.py``.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Carpeta raíz que contiene las carpetas de condiciones experimentales
#: (rep1_iso32_dsb1, rep1_iso32_dsb2, ..., rep3_iso68_dsb3).
#: Puede ser una ruta absoluta (p. ej. r"C:\ensayos\rotor" o "/datos/rotor").
DATA_DIR: str = str(PROJECT_ROOT / "data")

#: Carpeta donde se guardan todos los resultados (CSV y figuras).
OUTPUT_DIR: str = str(PROJECT_ROOT / "results")

#: Patrón (expresión regular) para extraer los metadatos de la carpeta:
#: repetición, viscosidad (grado ISO) y desbalance.
#: Debe contener los grupos nombrados ``rep``, ``iso`` y ``dsb``.
FOLDER_PATTERN: str = r"rep(?P<rep>\d+)_iso(?P<iso>\d+)_dsb(?P<dsb>\d+)"

#: Patrón para extraer la velocidad de rotación (RPM) del nombre del archivo.
#: Debe contener un único grupo de captura con el valor numérico.
#: Ejemplo de archivo: ``Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt``
RPM_PATTERN: str = r"[Rr][Pp][Mm][\s_\-]*(\d+)"

#: Extensión de los archivos de medición.
FILE_EXTENSION: str = ".txt"

# ---------------------------------------------------------------------------
# 2. Adquisición y modo de análisis
# ---------------------------------------------------------------------------

#: Frecuencia de muestreo [Hz].
FS: float = 12800.0

#: Modo de análisis espectral:
#:   - "psd":  autoespectro (PSD de Welch) del sensor único ``SENSOR``.
#:   - "cpsd": espectro cruzado (CPSD) entre los pares de sensores de
#:             proximidad ``PROXIMITY_SENSORS``. Para cada archivo se
#:             promedia la magnitud de la CPSD de todos los pares en un
#:             espectro compuesto; el contenido coherente entre sondas
#:             (modos estructurales) se refuerza y el ruido propio de cada
#:             sensor se atenúa. Los sensores de aceleración (Mach1.S1)
#:             quedan fuera de este modo.
ANALYSIS_MODE: str = "psd"

#: Sensor (canal) a analizar en modo "psd". Debe coincidir con uno de
#: ``CHANNEL_NAMES`` o con un nombre detectado en el encabezado del archivo.
SENSOR: str = "Mach1.P1.Y"

#: Sensores de proximidad usados en el modo "cpsd" (sin acelerómetros).
PROXIMITY_SENSORS: list[str] = [
    "Mach1.P1.Y",
    "Mach1.P1.X",
    "Mach1.P2.Y",
    "Mach1.P2.X",
]

#: Pares de sensores para la CPSD:
#:   - "all": todas las combinaciones de ``PROXIMITY_SENSORS`` (6 pares).
#:   - lista explícita de tuplas, p. ej.
#:     [("Mach1.P1.Y", "Mach1.P2.Y"), ("Mach1.P1.X", "Mach1.P2.X")]
CPSD_PAIRS: str | list[tuple[str, str]] = "all"

#: Usar la coherencia para depurar los picos en modo "cpsd": un pico de la
#: CPSD compuesta se descarta si la coherencia media entre pares en esa
#: frecuencia es inferior al umbral (el pico no está correlacionado entre
#: sondas y difícilmente es un modo).
CPSD_USE_COHERENCE: bool = True

#: Umbral de coherencia media (0-1) para conservar un pico.
COHERENCE_THRESHOLD: float = 0.5

#: Nombres de los canales en el orden en que aparecen como columnas en los
#: archivos de datos. Se utiliza cuando el encabezado del archivo no permite
#: detectar los nombres automáticamente.
CHANNEL_NAMES: list[str] = [
    "Mach1.P1.Y",
    "Mach1.P1.X",
    "Mach1.P2.Y",
    "Mach1.P2.X",
    "Mach1.S1",
]

#: Manejo de una posible columna de tiempo como primera columna del bloque
#: de datos. Opciones:
#:   - "auto":  se detecta automáticamente (columna monótona creciente).
#:   - True:    la primera columna siempre es tiempo y se descarta.
#:   - False:   todas las columnas son canales de medición.
HAS_TIME_COLUMN: bool | str = "auto"

#: Eliminar la media de la señal antes de calcular la PSD.
REMOVE_MEAN: bool = True

# ---------------------------------------------------------------------------
# 3. Estimación espectral (método de Welch)
# ---------------------------------------------------------------------------

#: Número de muestras por segmento de Welch. A mayor valor, mejor resolución
#: en frecuencia (df = FS / NPERSEG) pero menor promediado (más varianza).
WELCH_NPERSEG: int = 16384

#: Fracción de solape entre segmentos (0.0 - 0.95). 0.5 = 50 %.
WELCH_OVERLAP: float = 0.5

#: Ventana de análisis (se pasa directamente a ``scipy.signal.welch``).
WELCH_WINDOW: str = "hann"

#: Tipo de detrending aplicado por Welch ("constant", "linear" o False).
WELCH_DETREND: str = "constant"

#: Frecuencia máxima de análisis [Hz]. La PSD se trunca a este valor.
FMAX: float = 500.0

#: Frecuencia mínima de análisis [Hz]. Útil para descartar la deriva de
#: muy baja frecuencia de los sensores de proximidad.
FMIN: float = 2.0

#: Suavizado opcional de la PSD (media móvil) antes de la detección de picos.
SMOOTHING_ENABLED: bool = False

#: Ancho de la ventana de suavizado en número de líneas espectrales (impar).
SMOOTHING_WINDOW_BINS: int = 5

# ---------------------------------------------------------------------------
# 4. Detección de picos (peak picking con scipy.signal.find_peaks)
# ---------------------------------------------------------------------------
# La detección se realiza sobre la PSD en decibelios (10*log10) porque la
# prominencia en dB es mucho más estable frente a los órdenes de magnitud
# que abarca una PSD vibratoria.

#: Modo de prominencia:
#:   - "auto":  la prominencia mínima se estima a partir del nivel de ruido
#:              de la propia PSD (mediana móvil + desviación robusta MAD).
#:   - "fixed": se usa el valor fijo ``PEAK_PROMINENCE_DB``.
PEAK_PROMINENCE_MODE: str = "auto"

#: Prominencia mínima [dB] cuando ``PEAK_PROMINENCE_MODE = "fixed"``.
PEAK_PROMINENCE_DB: float = 6.0

#: Parámetros del modo automático: prominencia = max(AUTO_MIN_DB,
#: AUTO_FACTOR * sigma_ruido), con sigma_ruido estimado de forma robusta.
PEAK_AUTO_FACTOR: float = 6.0
PEAK_AUTO_MIN_DB: float = 6.0

#: Tamaño de la ventana [Hz] de la mediana móvil usada para estimar el piso
#: de ruido en el modo automático.
PEAK_NOISE_WINDOW_HZ: float = 25.0

#: Distancia mínima entre picos [Hz].
PEAK_DISTANCE_HZ: float = 2.0

#: Ancho mínimo del pico [Hz]. ``None`` desactiva el criterio.
PEAK_WIDTH_HZ: float | None = None

#: Altura mínima absoluta del pico [dB re 1 unidad^2/Hz].
#: ``None`` desactiva el criterio (recomendado: la prominencia es más robusta).
PEAK_HEIGHT_DB: float | None = None

#: Número máximo de picos a conservar por PSD (los de mayor prominencia).
#: ``None`` conserva todos los detectados.
PEAK_MAX_PEAKS: int | None = 10

# ---------------------------------------------------------------------------
# 5. Agrupamiento de frecuencias y persistencia
# ---------------------------------------------------------------------------

#: Tolerancia [Hz] para considerar que dos picos de distintas condiciones
#: corresponden a la misma frecuencia (p. ej. 0.5 o 1.0 Hz).
CLUSTER_TOLERANCE_HZ: float = 1.0

#: Persistencia mínima (0-1) para que un grupo aparezca en el diagrama de
#: Campbell y en los resúmenes finales.
MIN_PERSISTENCE: float = 0.5

#: Número mínimo de apariciones para que un grupo sea considerado
#: (protege frente a picos espurios cuando hay pocas condiciones).
MIN_APPEARANCES: int = 3

# ---------------------------------------------------------------------------
# 6. Eliminación opcional de armónicos de la velocidad de rotación
# ---------------------------------------------------------------------------

#: Si es True, los picos situados sobre los armónicos 1X, 2X, ... NX de la
#: frecuencia de rotación se descartan (son excitación forzada por el
#: desbalance, no modos estructurales). En un rotor desbalanceado la 1X y
#: sus armónicos dominan la PSD, por lo que sin este filtro acaparan el
#: diagrama de Campbell con persistencia máxima.
REMOVE_HARMONICS: bool = True

#: Número de armónicos a eliminar (1X hasta NX).
HARMONICS_MAX_ORDER: int = 10

#: Tolerancia absoluta [Hz] alrededor de cada armónico.
HARMONICS_TOL_HZ: float = 1.5

#: Tolerancia relativa (fracción de la frecuencia del armónico). La tolerancia
#: efectiva es max(HARMONICS_TOL_HZ, HARMONICS_TOL_REL * n * f_rot).
HARMONICS_TOL_REL: float = 0.02

#: Refinar la frecuencia de rotación real a partir de la propia PSD: se busca
#: el pico más alto en una banda de ±5 % alrededor de la 1X nominal (RPM/60)
#: y los armónicos se calculan sobre esa frecuencia. Compensa la diferencia
#: entre la velocidad nominal del nombre del archivo y la velocidad real.
HARMONICS_REFINE_F1: bool = True

#: Semiancho relativo de la banda de búsqueda de la 1X real.
HARMONICS_REFINE_SEARCH_REL: float = 0.05

# ---------------------------------------------------------------------------
# 7. Gráficos y salidas
# ---------------------------------------------------------------------------

#: Guardar una figura PSD + picos por cada archivo analizado.
#: Con bases de datos grandes puede generar cientos de figuras.
SAVE_INDIVIDUAL_PSD_PLOTS: bool = True

#: Guardar la PSD promedio (media ± desviación estándar) de cada velocidad.
SAVE_AVERAGE_PSD_PLOTS: bool = True

#: Formato de las figuras ("png", "pdf", "svg").
FIGURE_FORMAT: str = "png"

#: Resolución de las figuras en puntos por pulgada.
FIGURE_DPI: int = 150

#: Escala del eje Y de las PSD ("db" o "log").
PSD_PLOT_SCALE: str = "db"

#: Dibujar en el diagrama de Campbell las líneas de armónicos 1X, 2X, ...
CAMPBELL_SHOW_HARMONIC_LINES: bool = True

#: Órdenes de armónicos a dibujar en el Campbell.
CAMPBELL_HARMONIC_ORDERS: list[int] = [1, 2, 3]

#: Mapa de colores secuencial para la persistencia (perceptualmente uniforme
#: y seguro para daltonismo).
CAMPBELL_COLORMAP: str = "viridis"

# ---------------------------------------------------------------------------
# 9. Identificación estocástica de subespacios (SSI-COV/ref)
# ---------------------------------------------------------------------------
# Parámetros del método OMA avanzado implementado en ``ssi.py`` y ejecutado
# con ``ssi_main.py`` (Peeters & De Roeck 2000 + Dreher et al. 2023). Es
# independiente del pipeline de peak picking anterior.

#: Canales de salida del modelo SSI (las 4 sondas de proximidad; el
#: acelerómetro Mach1.S1 se excluye porque mezcla unidades de aceleración con
#: los desplazamientos de las sondas).
SSI_OUTPUT_SENSORS: list[str] = [
    "Mach1.P1.Y",
    "Mach1.P1.X",
    "Mach1.P2.Y",
    "Mach1.P2.X",
]

#: Canales de referencia (subconjunto de SSI_OUTPUT_SENSORS). En el SSI
#: basado en referencias solo se usan estos canales para las columnas de la
#: matriz de covarianzas, lo que reduce el coste sin perder observabilidad si
#: las referencias están bien excitadas. Las dos sondas verticales suelen ser
#: buenas referencias en un rotor.
SSI_REF_SENSORS: list[str] = [
    "Mach1.P1.Y",
    "Mach1.P2.Y",
]

#: Factor de diezmado antes de la SSI. La frecuencia de muestreo efectiva pasa
#: a ser FS / SSI_DECIMATION. Con FS = 12800 Hz y factor 10 -> 1280 Hz
#: (Nyquist 640 Hz), suficiente para modos por debajo de SSI_FMAX y mucho más
#: barato numéricamente. Ajustar si SSI_FMAX se acerca al nuevo Nyquist.
SSI_DECIMATION: int = 10

#: Número de bloques-fila ``p`` de la matriz bloque-Toeplitz de covarianzas.
#: El orden máximo alcanzable es (nº de salidas) * p.
SSI_BLOCK_ROWS: int = 40

#: Rango de órdenes del modelo para el diagrama de estabilización (pares).
SSI_ORDER_MIN: int = 2
SSI_ORDER_MAX: int = 60
SSI_ORDER_STEP: int = 2

#: Banda de frecuencias de interés para la SSI [Hz].
SSI_FMIN: float = 2.0
SSI_FMAX: float = 500.0

#: Amortiguamiento máximo admisible para aceptar un polo como físico (0-1).
SSI_ZETA_MAX: float = 0.2

#: Tolerancias de estabilidad entre órdenes consecutivos:
#:   - SSI_DF_TOL:  desviación relativa de frecuencia (0.01 = 1 %).
#:   - SSI_DZ_TOL:  desviación relativa de amortiguamiento (0.05 = 5 %).
#:   - SSI_MAC_TOL: MAC mínimo entre formas modales de órdenes consecutivos.
SSI_DF_TOL: float = 0.01
SSI_DZ_TOL: float = 0.05
SSI_MAC_TOL: float = 0.98

#: Clustering de polos estables -> modos físicos (Dreher et al.).
#:   - SSI_CLUSTER_FREQ_WEIGHT / SSI_CLUSTER_MAC_WEIGHT: pesos de la distancia
#:     (frecuencia relativa vs 1-MAC).
#:   - SSI_CLUSTER_DISTANCE: umbral de corte del dendrograma.
#:   - SSI_MIN_CLUSTER_FRACTION: fracción mínima de órdenes que debe cubrir un
#:     grupo para considerarse un modo físico (columna del diagrama).
#:   - SSI_MIN_CLUSTER_SIZE: número mínimo absoluto de polos por grupo.
SSI_CLUSTER_FREQ_WEIGHT: float = 1.0
SSI_CLUSTER_MAC_WEIGHT: float = 1.0
SSI_CLUSTER_DISTANCE: float = 0.05
SSI_MIN_CLUSTER_FRACTION: float = 0.25
SSI_MIN_CLUSTER_SIZE: int = 3

#: Marcar (y opcionalmente excluir) los modos que coinciden con armónicos nX
#: de la velocidad de rotación. La 1X se refina desde el espectro igual que en
#: el peak picking (HARMONICS_REFINE_*).
SSI_FLAG_HARMONICS: bool = True
SSI_HARMONICS_MAX_ORDER: int = 10
SSI_HARMONICS_TOL_HZ: float = 1.5
SSI_HARMONICS_TOL_REL: float = 0.02

#: Excluir los modos marcados como armónicos de la tabla de frecuencias
#: naturales y del Campbell (recomendado en maquinaria rotativa).
SSI_EXCLUDE_HARMONICS: bool = True

#: Tolerancia [Hz] del agrupamiento de frecuencias entre condiciones de una
#: misma RPM (persistencia de los modos SSI).
SSI_CLUSTER_TOLERANCE_HZ: float = 1.0

#: Persistencia y apariciones mínimas para que un modo aparezca en el Campbell
#: SSI y en la tabla principal de frecuencias naturales.
SSI_MIN_PERSISTENCE: float = 0.5
SSI_MIN_APPEARANCES: int = 2

#: Guardar el diagrama de estabilización de cada ensayo.
SSI_SAVE_STABILIZATION_PLOTS: bool = True

# ---------------------------------------------------------------------------
# 8. Ejecución
# ---------------------------------------------------------------------------

#: Nivel de logging: "DEBUG", "INFO", "WARNING", "ERROR".
LOG_LEVEL: str = "INFO"

#: Guardar el log también en un archivo dentro de OUTPUT_DIR.
LOG_TO_FILE: bool = True

#: Procesar los archivos en paralelo con multiprocessing.
PARALLEL_ENABLED: bool = True

#: Número de procesos. ``None`` usa todos los núcleos disponibles.
PARALLEL_WORKERS: int | None = None

#: Mostrar barras de progreso con tqdm.
SHOW_PROGRESS: bool = True
