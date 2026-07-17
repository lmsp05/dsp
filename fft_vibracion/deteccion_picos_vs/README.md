# Variantes de detección de picos (`deteccion_picos_vs`)

Cuatro variantes de `deteccion_picos.py` que **promedian** espectros a distintos
niveles antes de detectar los picos. Todas comparten el mismo núcleo
(`_nucleo_vs.py`), que a su vez reutiliza la lógica de `deteccion_picos.py`
(estimación del 1X, eliminación por hombros, `find_peaks`). Cada script-versión
solo fija: **sobre qué se promedia**, **si elimina o no** el 1X/armónicos y el
**tipo de gráfica**.

| Script | Promedia sobre | Elimina 1X/armónicos | Grafica | Grupos |
|---|---|---|---|---|
| `v1_prom_sensores.py` | sensores (por rep·iso·dsb·rpm) | **Sí** | frecuencia vs RPM | 1 por rep·iso·dsb·rpm |
| `v2_prom_sensores_velocidades.py` | sensores + velocidades (por rep·iso·dsb) | **Sí** | frecuencia vs condición | 1 por rep·iso·dsb |
| `v3_prom_sensores_velocidades_sin_elim.py` | sensores + velocidades (por rep·iso·dsb) | **No** | frecuencia vs condición | 1 por rep·iso·dsb |
| `v4_prom_global_sin_elim.py` | **todo** (sensores + condiciones + velocidades) | **No** | espectro promedio con picos | 1 (global) |

### `metricas_cascada.py` — RMS + Kurtosis + Ridge (portado del `.m`)

Método **distinto** (no promedia): apila la cascada `Z` (filas = RPM, columnas =
frecuencia) por condición y calcula tres métricas **atravesando las RPM**,
aprovechando que una natural es fija con la velocidad y el 1X/armónicos se mueven:

* **RMS(f)** = `sqrt(mean_rpm(Z²))`
* **Kurtosis(f)** = curtosis sobre las RPM (Pearson, como MATLAB)
* **Ridge(f)** = en cuántas RPM hay un pico local en esa frecuencia (persistencia
  de crestas). Es el mejor detector de naturales: aparecen en casi todas las RPM.

Portado de `x0302_nf_gph_transformations.m` (mismas fracciones de prominencia:
RMS 5 %, Kurtosis 10 %, Ridge 20 %). Genera, **por condición** (rep·iso·dsb·sensor)
y dentro de `--outdir`: `metricas_<condicion>.txt` (frecuencias por método) y
`metricas_<condicion>.png` (las 3 métricas con sus picos + la cascada con las
frecuencias marcadas). Necesita ≥2 RPM por condición y funciona igual sobre el
full spectrum complejo. También la ejecuta `comparar_deteccion_picos.py`.

---

En las variantes de promediado, el promedio se hace interpolando cada espectro a
una **grilla de frecuencia común** (las distintas RPM tienen distinto `df`); en las variantes con
eliminación, el 1X y los armónicos se quitan de **cada espectro individual**
(con su propia RPM) *antes* de promediar, de modo que las frecuencias naturales
(que no se mueven con la RPM) se refuerzan y el resto se promedia.

Cada script escribe en su `--outdir`: `picos_detectados.txt` y `picos_grafico.png`.
Aceptan los mismos filtros por niveles (`--rep`, `--iso`, …) y parámetros de
detección (`--fraccion`, `--n-armonicos`, …) que `deteccion_picos.py`.

```bash
python deteccion_picos_vs/v1_prom_sensores.py --entrada resultados_fft.txt --outdir v1
```

### `metricas_versiones.py` — las 3 métricas en las 5 configuraciones v0…v4

Aplica RMS/Kurtosis/Ridge en las **mismas 5 configuraciones** que las variantes,
seleccionadas con `--modo`. Las métricas agregan un conjunto de espectros, y en
cada modo el conjunto es lo que esa versión combina:

| `--modo` | conjunto de la métrica | se detecta por | eliminación | gráfica |
|---|---|---|---|---|
| `v0` | las RPM (cascada) | condición · sensor | **Sí** | cuadrícula 5×4 (métodos × sensores) por condición |
| `v1` | los 4 sensores | condición · velocidad | **Sí** | Campbell (freq vs RPM) |
| `v2` | sensores + velocidades | condición | **Sí** | freq vs condición |
| `v3` | sensores + velocidades | condición | **No** | freq vs condición |
| `v4` | todos los espectros | global | **No** | curva métrica vs freq |
| `v5` | las RPM (cascada) | condición · sensor | **No** | cuadrícula 5×4 (= v0 sin eliminación) |

Para v1–v4 genera, en `--outdir`: `picos_RMS.png`, `picos_Kurtosis.png`,
`picos_Ridge.png`, `picos_combinado.png` y `picos_detectados.txt`. Para v0 genera
una `v0grid_<condicion>.png` por condición (5 filas = prominencia, RMS, Kurtosis,
Ridge, combinado; 4 columnas = sensores) más el `.txt`.

```bash
python deteccion_picos_vs/metricas_versiones.py --modo v1 --entrada resultados_fft.txt --outdir mv1
```

> Nota: RMS y Ridge son los detectores robustos de naturales; la **Kurtosis**
> necesita variabilidad real entre los miembros del conjunto (con sensores muy
> correlacionados o pocas RPM sale degenerada).
>
> Los **criterios son poco severos por defecto** (fracciones de prominencia
> RMS 0.015 / Kurtosis 0.03 / Ridge 0.05 y prominencia por fila del ridge 0.04),
> para dejar pasar picos poco perceptibles. Se ajustan con `--frac-rms`,
> `--frac-kurt`, `--frac-ridge` y `--prom-ridge-fila` (súbelos para ser más
> estricto). Lo mismo aplica a `metricas_cascada.py`.

## Ejecutar todas a la vez

Desde la carpeta padre, `comparar_deteccion_picos.py` corre la original y las 4
variantes, cada una en su propia subcarpeta:

```bash
python comparar_deteccion_picos.py --entrada resultados_fft.txt --outdir comparacion
# los filtros/parámetros extra se reenvían a todas:
python comparar_deteccion_picos.py --entrada resultados_fft.txt --outdir comp --iso 32,68
```
