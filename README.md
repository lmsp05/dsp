# OMA Peak Picking · Rotor Jeffcott con cojinetes hidrodinámicos

Proyecto en Python para **Operational Modal Analysis (OMA)** mediante *peak
picking* sobre una base de datos experimental de un rotor Jeffcott soportado
por cojinetes hidrodinámicos.

La idea central: en lugar de detectar picos aislados, el programa identifica
las frecuencias que **persisten entre distintas condiciones experimentales**
(repeticiones × viscosidades × desbalances) para una misma velocidad de
rotación. Un pico que sobrevive a cambios de lubricante y de desbalance es un
fuerte candidato a frecuencia natural. El resultado principal es un **diagrama
tipo Campbell** donde el tamaño y el color de cada punto codifican la
persistencia, lo que permite seguir la evolución de los modos con la velocidad.

## Estructura del proyecto

| Archivo | Contenido |
|---|---|
| `config.py` | Todos los parámetros del análisis, centralizados y documentados. |
| `io_utils.py` | Recorrido de la base de datos, extracción de metadatos (rep, iso, dsb, RPM) y lectura robusta de los archivos `.txt` (ignora los bloques `BEGIN_HEADER`/`BEGIN_INFO`, detecta delimitador, separador decimal y columna de tiempo). |
| `psd.py` | PSD por el método de Welch (ventana Hann), suavizado opcional y PSD promedio ± desviación estándar por grupo. |
| `peak_detection.py` | Peak picking con `scipy.signal.find_peaks` (prominencia, distancia, ancho, altura), estimación automática de la prominencia a partir del ruido y eliminación opcional de armónicos 1X…NX. |
| `persistence.py` | Clustering 1D de frecuencias con tolerancia configurable y cálculo de la persistencia de cada grupo. |
| `plotting.py` | Todas las figuras: PSD individuales, PSD promedio, Campbell, histogramas. |
| `main.py` | Orquestación del flujo completo (paralelizable, con logging y `tqdm`). |
| `generate_synthetic_data.py` | Genera una base de datos sintética con el formato real para validar el pipeline. |

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python ≥ 3.10.

## Estructura esperada de la base de datos

```
data/
├── rep1_iso32_dsb1/
│   ├── Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt
│   ├── Rec_stb_iso32_dsb(0+8-7)-Rpm900.txt
│   └── ...
├── rep1_iso32_dsb2/
└── ... hasta rep3_iso68_dsb3
```

* El nombre de la carpeta codifica **repetición**, **viscosidad** (grado ISO)
  y **desbalance**: `rep{N}_iso{VG}_dsb{M}`.
* El nombre de cada archivo contiene la velocidad (`...Rpm600.txt`); las RPM
  se extraen automáticamente con una expresión regular configurable.
* Cada archivo tiene bloques `BEGIN_HEADER/END_HEADER`, `BEGIN_INFO/END_INFO`
  y `BEGIN_DATA/END_DATA`; solo se lee el bloque de datos.
* Frecuencia de muestreo: **12 800 Hz** (configurable en `config.py`).
* Canales: `Mach1.P1.Y`, `Mach1.P1.X`, `Mach1.P2.Y`, `Mach1.P2.X`, `Mach1.S1`.

## Uso

```bash
# Análisis completo con la configuración de config.py
python main.py

# Cambiar carpeta de datos, sensor o carpeta de salida sin editar config.py
python main.py --data-dir /ruta/a/mis/datos --sensor Mach1.P2.X --output-dir results_p2x

# Modo CPSD: espectro cruzado entre las 4 sondas de proximidad
# (excluye el acelerómetro Mach1.S1)
python main.py --mode cpsd --output-dir results_cpsd

# Ejecución rápida de prueba (sin paralelizar, sin figuras individuales)
python main.py --no-parallel --no-individual-plots --limit 10
```

### Prueba sin datos reales

```bash
python generate_synthetic_data.py --data-dir data_synthetic
python main.py --data-dir data_synthetic --output-dir results_synthetic
```

## Modos de análisis

* **`psd`** (por defecto): autoespectro de Welch del sensor único `SENSOR`.
* **`cpsd`**: espectro cruzado (CPSD, `scipy.signal.csd`) entre los pares de
  sondas de proximidad `PROXIMITY_SENSORS` — los acelerómetros quedan fuera.
  Para cada archivo se promedia la magnitud de la CPSD de todos los pares
  (configurables con `CPSD_PAIRS`) en un espectro compuesto: el contenido
  correlacionado entre sondas (respuesta estructural) se refuerza y el ruido
  propio de cada sensor se atenúa. Además se calcula la **coherencia media**
  entre pares y los picos con coherencia inferior a `COHERENCE_THRESHOLD` se
  descartan (`CPSD_USE_COHERENCE`). Las figuras individuales incluyen un
  panel de coherencia bajo el espectro.

El modo se elige con `ANALYSIS_MODE` en `config.py` o con `--mode` en la
línea de comandos; el resto del pipeline (picos, clustering, persistencia,
Campbell) es idéntico en ambos modos.

## Flujo del análisis

1. **Descubrimiento**: se recorren las carpetas, se extraen los metadatos y se
   construye el índice de archivos (`tables/files_index.csv`).
2. **PSD**: para cada archivo se calcula la PSD por Welch (ventana Hann,
   `nperseg`, solape y `FMAX` configurables). Suavizado opcional.
3. **Peak picking**: `scipy.signal.find_peaks` sobre la PSD en dB con
   prominencia, distancia, ancho y altura configurables. En modo `auto` la
   prominencia mínima se estima del nivel de ruido (mediana móvil + MAD).
   Los armónicos 1X…NX de la rotación pueden eliminarse (`REMOVE_HARMONICS`).
4. **Agrupamiento por velocidad**: para cada RPM se reúnen los picos de todas
   las condiciones experimentales que comparten esa velocidad.
5. **Clustering de frecuencias**: picos como 41.9 / 42.1 / 42.3 / 42.4 Hz se
   agrupan si distan menos de `CLUSTER_TOLERANCE_HZ` de la media del grupo
   (robusto frente al desplazamiento de frecuencia de los cojinetes
   hidrodinámicos).
6. **Persistencia**: para cada grupo, `persistencia = condiciones donde
   aparece / condiciones totales` (p. ej. 24/27 = 0.89).
7. **PSD promedio**: media ± desviación estándar de las PSD de cada RPM.
8. **Exportación**: todas las tablas a CSV.
9. **Figuras**: PSD individuales con picos anotados, PSD promedio, histograma
   de persistencia, apariciones por frecuencia y **diagrama de Campbell**.

## Resultados generados

```
results/
├── analysis.log
├── tables/
│   ├── files_index.csv          # índice de archivos analizados
│   ├── peaks_raw.csv            # todos los picos individuales
│   ├── peak_groups_all.csv      # todos los grupos (antes del filtro)
│   ├── persistent_peaks.csv     # RESULTADO PRINCIPAL: picos persistentes
│   └── psd_average/             # PSD promedio ± std por RPM (CSV)
└── figures/
    ├── campbell.png             # RESULTADO PRINCIPAL: diagrama de Campbell
    ├── persistence_histogram.png
    ├── appearances_by_frequency.png
    ├── psd_average/             # PSD promedio por RPM
    └── psd_individual/          # PSD + picos de cada ensayo
```

`persistent_peaks.csv` contiene: `RPM`, `Frecuencia_Hz`, `Persistencia`,
`Numero_apariciones`, `Numero_condiciones`, `Amplitud_promedio`,
`Amplitud_desviacion`, `Amplitud_promedio_dB`, `Desviacion_estandar_Hz`
y `Promedio_PSD`.

## Parámetros principales (`config.py`)

| Parámetro | Significado | Valor por defecto |
|---|---|---|
| `ANALYSIS_MODE` | `"psd"` (un sensor) o `"cpsd"` (cruzado entre sondas) | `psd` |
| `SENSOR` | Canal a analizar en modo `psd` | `Mach1.P1.Y` |
| `PROXIMITY_SENSORS` | Sondas usadas en modo `cpsd` (sin acelerómetros) | P1.Y, P1.X, P2.Y, P2.X |
| `COHERENCE_THRESHOLD` | Coherencia mínima para conservar un pico (`cpsd`) | 0.5 |
| `FS` | Frecuencia de muestreo | 12 800 Hz |
| `WELCH_NPERSEG` | Muestras por segmento de Welch (df = FS/nperseg) | 16 384 (df ≈ 0.78 Hz) |
| `FMAX` / `FMIN` | Rango de análisis | 500 / 2 Hz |
| `PEAK_PROMINENCE_MODE` | `"auto"` (desde el ruido) o `"fixed"` | `auto` |
| `PEAK_DISTANCE_HZ` | Separación mínima entre picos | 2 Hz |
| `CLUSTER_TOLERANCE_HZ` | Tolerancia del agrupamiento de frecuencias | 1.0 Hz |
| `MIN_PERSISTENCE` | Persistencia mínima para el Campbell | 0.5 |
| `REMOVE_HARMONICS` | Eliminar picos 1X…NX de la rotación | `True` (hasta 10X) |
| `HARMONICS_REFINE_F1` | Estimar la 1X real desde la PSD (no la RPM nominal) | `True` |
| `PARALLEL_ENABLED` | Procesamiento multiproceso | `True` |

## Método SSI: identificación estocástica de subespacios (OMA automatizado)

Además del peak picking, el proyecto incluye un método OMA avanzado que
identifica **frecuencias naturales, amortiguamientos y formas modales** de
forma automatizada, siguiendo la bibliografía de referencia:

* **Peeters & De Roeck (2000)** — *Reference-based Stochastic Subspace
  Identification* (**SSI-COV/ref**). Para cada ensayo se estiman las
  covarianzas de salida basadas en referencias, se construye la matriz
  bloque-Toeplitz `T_{1|p}`, se descompone en valores singulares (SVD) y de
  la matriz de observabilidad se extraen las matrices de estado `A` y de
  salida `C` por invariancia de desplazamiento. Los autovalores de `A` dan
  los polos (frecuencia, amortiguamiento y forma modal). Repitiendo para un
  rango de órdenes se obtiene el **diagrama de estabilización**.
* **Dreher et al. (2023)** — *Automated OMA for Rotating Machinery Based on
  Clustering Techniques*. El diagrama de estabilización se interpreta sin
  intervención manual: los polos se etiquetan con criterios de estabilidad
  entre órdenes consecutivos (desviación de frecuencia, de amortiguamiento y
  **MAC**) y los polos estables se agrupan por **clustering jerárquico**. Cada
  grupo suficientemente poblado (columna vertical del diagrama) es un modo
  físico; los polos matemáticos/espurios quedan en grupos dispersos que se
  descartan. Los **armónicos nX** de la velocidad de giro se marcan aparte
  para no confundirlos con modos estructurales.

### Módulos SSI

| Archivo | Contenido |
|---|---|
| `ssi.py` | Núcleo SSI-COV/ref: covarianzas, bloque-Toeplitz, SVD, extracción de `A`/`C`, parámetros modales, MAC, etiquetado de estabilidad, *clustering* de polos, marcado de armónicos y agregación entre condiciones. |
| `ssi_main.py` | Orquestación del flujo SSI completo (paralelizable). |
| `plotting.plot_stabilization_diagram` | Diagrama de estabilización por ensayo. |
| `plotting.plot_ssi_campbell` | Campbell de las frecuencias naturales identificadas. |

### Uso

```bash
# Análisis SSI completo con la configuración de config.py (sección 9)
python ssi_main.py

# Otra carpeta de datos/salida, o prueba rápida
python ssi_main.py --data-dir data_synthetic --output-dir results_ssi --limit 10
python ssi_main.py --no-parallel --no-stab-plots
```

### Flujo del análisis SSI

1. **Descubrimiento**: misma estructura de carpetas `repX_isoYY_dsbZ` que el
   peak picking.
2. **Diezmado**: la señal se diezma (`SSI_DECIMATION`, por defecto ×10) de
   12 800 Hz a 1 280 Hz, suficiente para modos por debajo de `SSI_FMAX` y
   mucho más barato numéricamente.
3. **SSI-COV/ref por ensayo**: covarianzas basadas en las sondas de
   referencia (`SSI_REF_SENSORS`), Toeplitz de `SSI_BLOCK_ROWS` bloques, SVD y
   barrido de órdenes `SSI_ORDER_MIN..SSI_ORDER_MAX`.
4. **Estabilización + clustering**: se etiquetan los polos, se agrupan los
   estables en modos físicos y se marcan los armónicos (1X refinada del
   espectro).
5. **Persistencia entre condiciones**: para cada RPM se agrupan los modos de
   todas las condiciones (repeticiones × viscosidades × desbalances) por
   frecuencia y se calcula su persistencia, igual que en el peak picking.
6. **Salidas**: tablas CSV, diagramas de estabilización por ensayo y Campbell
   de las frecuencias naturales.

### Resultados generados (`results/`)

```
results/
├── ssi_analysis.log
├── tables/
│   ├── files_index.csv
│   ├── ssi_modes_raw.csv            # modos de cada ensayo (freq, ζ, orden…)
│   ├── ssi_modes_all.csv            # grupos por RPM (con persistencia)
│   └── ssi_natural_frequencies.csv  # RESULTADO PRINCIPAL: frecuencias naturales
└── figures/
    ├── ssi_campbell.png             # Campbell de frecuencias naturales
    ├── ssi_persistence_histogram.png
    └── stabilization/               # diagrama de estabilización por ensayo
```

`ssi_natural_frequencies.csv` contiene: `RPM`, `Frecuencia_Natural_Hz`,
`Amortiguamiento`, `Persistencia`, `Numero_apariciones`,
`Numero_condiciones`, `Desviacion_Frecuencia_Hz` y
`Desviacion_Amortiguamiento`.

### Parámetros principales de SSI (`config.py`, sección 9)

| Parámetro | Significado | Valor por defecto |
|---|---|---|
| `SSI_OUTPUT_SENSORS` | Salidas del modelo (4 sondas de proximidad) | P1.Y, P1.X, P2.Y, P2.X |
| `SSI_REF_SENSORS` | Sondas de referencia (subconjunto) | P1.Y, P2.Y |
| `SSI_DECIMATION` | Factor de diezmado (fs_ef = FS / factor) | 10 (→ 1 280 Hz) |
| `SSI_BLOCK_ROWS` | Bloques-fila `p` del Toeplitz | 40 |
| `SSI_ORDER_MIN/MAX/STEP` | Rango de órdenes del modelo | 2 / 60 / 2 |
| `SSI_ZETA_MAX` | Amortiguamiento máximo de un polo físico | 0.2 |
| `SSI_DF_TOL` / `SSI_DZ_TOL` / `SSI_MAC_TOL` | Tolerancias de estabilidad | 1 % / 5 % / 0.98 |
| `SSI_CLUSTER_DISTANCE` | Umbral de corte del *clustering* | 0.05 |
| `SSI_MIN_CLUSTER_FRACTION` | Fracción mínima de órdenes por modo | 0.25 |
| `SSI_FLAG_HARMONICS` / `SSI_EXCLUDE_HARMONICS` | Marcar / excluir armónicos nX | `True` / `True` |
| `SSI_MIN_PERSISTENCE` | Persistencia mínima para el Campbell | 0.5 |

### Validación con datos sintéticos

`generate_synthetic_data.py` reproduce el **formato real del equipo** (6
columnas: tacómetro + 4 sondas + acelerómetro, con la fila `Name` del bloque
INFO) y simula un sistema modal multi-salida coherente con dos modos
conocidos (44.0 Hz ζ=3 %, 83.0 Hz ζ=2 %). La SSI los recupera con precisión:

```bash
python generate_synthetic_data.py --data-dir data_synthetic --duration 30
python ssi_main.py --data-dir data_synthetic --output-dir results_ssi
# -> ssi_natural_frequencies.csv: 44.0 Hz (ζ≈3.1 %) y 83.0 Hz (ζ≈2.0 %),
#    persistencia 1.0 en las 8 condiciones de cada velocidad.
```

> **Nota sobre el formato real y el tacómetro.** Los archivos del equipo
> tienen **6 columnas**: la primera es el tacómetro (`Ext. sync.`) y las cinco
> siguientes son las sondas y el acelerómetro. `io_utils` lee la fila `Name`
> del bloque INFO y **alinea cada canal con su columna**, de modo que
> `Mach1.P1.Y` se toma de la segunda columna y no del tacómetro. Esto vale
> tanto para el peak picking como para la SSI. Para la identificación modal no
> hace falta aplicar los `Coef_A`/`Coef_B`: al eliminarse la media (separación
> DC de la sonda) y ser todas las sondas de la misma escala, ni las
> frecuencias ni el MAC dependen de ese factor.

## Extensión futura

La arquitectura separa lectura (`io_utils`), estimación espectral (`psd`),
identificación (`peak_detection` + `persistence` para peak picking; `ssi` para
subespacios) y presentación (`plotting`), de modo que otros métodos OMA (FDD,
EFDD, poly-reference LSCF) pueden añadirse como módulos nuevos que consuman las
mismas estructuras (`MeasurementRecord`, `PSDResult`) sin tocar el resto del
pipeline.
