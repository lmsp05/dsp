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

# ---------------------------------------------------------------------------
# 1. Rutas y estructura de la base de datos
# ---------------------------------------------------------------------------

#: Carpeta raíz que contiene las carpetas de condiciones experimentales
#: (rep1_iso32_dsb1, rep1_iso32_dsb2, ..., rep3_iso68_dsb3).
DATA_DIR: str = "data"

#: Carpeta donde se guardan todos los resultados (CSV y figuras).
OUTPUT_DIR: str = "results"

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
# 2. Adquisición
# ---------------------------------------------------------------------------

#: Frecuencia de muestreo [Hz].
FS: float = 12800.0

#: Sensor (canal) a analizar. Debe coincidir con uno de ``CHANNEL_NAMES``
#: o con un nombre detectado automáticamente en el encabezado del archivo.
SENSOR: str = "Mach1.P1.Y"

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
PEAK_AUTO_FACTOR: float = 4.0
PEAK_AUTO_MIN_DB: float = 3.0

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
PEAK_MAX_PEAKS: int | None = 20

# ---------------------------------------------------------------------------
# 5. Agrupamiento de frecuencias y persistencia
# ---------------------------------------------------------------------------

#: Tolerancia [Hz] para considerar que dos picos de distintas condiciones
#: corresponden a la misma frecuencia (p. ej. 0.5 o 1.0 Hz).
CLUSTER_TOLERANCE_HZ: float = 1.0

#: Persistencia mínima (0-1) para que un grupo aparezca en el diagrama de
#: Campbell y en los resúmenes finales.
MIN_PERSISTENCE: float = 0.3

#: Número mínimo de apariciones para que un grupo sea considerado
#: (protege frente a picos espurios cuando hay pocas condiciones).
MIN_APPEARANCES: int = 2

# ---------------------------------------------------------------------------
# 6. Eliminación opcional de armónicos de la velocidad de rotación
# ---------------------------------------------------------------------------

#: Si es True, los picos situados sobre los armónicos 1X, 2X, ... NX de la
#: frecuencia de rotación se descartan (son excitación forzada, no modos).
REMOVE_HARMONICS: bool = False

#: Número de armónicos a eliminar (1X hasta NX).
HARMONICS_MAX_ORDER: int = 5

#: Tolerancia absoluta [Hz] alrededor de cada armónico.
HARMONICS_TOL_HZ: float = 1.0

#: Tolerancia relativa (fracción de la frecuencia del armónico). La tolerancia
#: efectiva es max(HARMONICS_TOL_HZ, HARMONICS_TOL_REL * n * f_rot).
HARMONICS_TOL_REL: float = 0.01

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
