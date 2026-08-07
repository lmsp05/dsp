# Inventario de la base de datos · combinaciones `rep_iso_dsb_rpm`

Herramienta para auditar la base de datos experimental **antes** de analizarla:
qué ensayos están medidos, cuáles faltan, cuáles están duplicados y cuáles
tienen un archivo de tamaño sospechoso.

| Archivo | Contenido |
|---|---|
| `inventario_combinaciones.py` | Script principal: recorre la base de datos y genera los dos informes `.txt`. |
| `_generar_datos_prueba.py` | Crea una base de datos falsa (con faltantes, duplicados y pesos anómalos) para probar el script sin los datos reales. |

Solo usa la biblioteca estándar de Python (≥ 3.10): no requiere `numpy` ni
ninguna otra dependencia del proyecto.

> **Es de solo lectura sobre la base de datos.** De cada archivo lee su nombre
> y su tamaño, nada más: no renombra, no mueve ni modifica ningún archivo. Lo
> único que escribe son los dos `.txt`, y siempre dentro de la carpeta indicada
> con `--outdir`.

## Estructura de datos esperada

```
datos/
├── rep1_iso32_dsb1/
│   ├── Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt    → rep1_iso32_dsb1_rpm600
│   ├── Rec_stb_iso32_dsb(0+8-7)-Rpm1300.txt   → rep1_iso32_dsb1_rpm1300
│   └── ...
├── rep1_iso32_dsb2/
└── ... hasta rep3_iso68_dsb3
```

* La subcarpeta aporta **rep**, **iso** y **dsb**; el nombre del archivo aporta
  la **RPM** (el nombre debe terminar en `RpmXXXX`, antes de la extensión).
* La búsqueda de subcarpetas es recursiva, así que la raíz puede tener niveles
  intermedios sin que haya que aplanar nada.

### Catálogo de valores válidos

| Parámetro | Valores |
|---|---|
| `rep` | 1, 2, 3 |
| `iso` | 32, 46, 68 |
| `dsb` | 1, 2, 3 |
| `rpm` | 600, 1300, 2000, 2500, 2600, 2700, 2800, 3400, 3500, 3600, 3700, 4000, 4500, 5000, 5500 |

Son **3 × 3 × 3 × 15 = 405** combinaciones posibles. Los archivos cuyos
parámetros caen fuera de estas listas (por ejemplo una `Rpm900`) se reportan
aparte y no cuentan para la cobertura.

## Uso

```bash
# Lo mínimo: indicar la carpeta que contiene las subcarpetas repN_isoVG_dsbM
python inventario_combinaciones.py /ruta/a/mis/datos

# Equivalente con la opción larga, eligiendo la carpeta de salida
python inventario_combinaciones.py --data-dir datos --outdir inventario

# Considerar solo los .txt y ser más estricto con los outliers
python inventario_combinaciones.py datos --ext .txt --k-outlier 2.5

# Comparar el peso de cada archivo contra los de su misma RPM
python inventario_combinaciones.py datos --grupo-outliers rpm
```

### Opciones

| Opción | Descripción |
|---|---|
| `carpeta` (posicional) / `--data-dir` | Carpeta raíz con las subcarpetas `repN_isoVG_dsbM`. |
| `--outdir` | Carpeta de salida de los dos `.txt` (def. `./inventario`). |
| `--ext` | Extensión a considerar, p. ej. `.txt` (def. todas). |
| `--k-outlier` | Desviaciones robustas para marcar un peso atípico (def. 3). |
| `--grupo-outliers` | `global` (def.), `rpm` o `condicion`: con qué archivos se compara cada peso. |
| `--reps`, `--isos`, `--dsbs`, `--rpms` | Sustituyen el catálogo de valores válidos sin editar el script. |

### Prueba sin datos reales

```bash
python _generar_datos_prueba.py --outdir datos_prueba
python inventario_combinaciones.py datos_prueba --outdir inventario_prueba
```

La base de prueba incluye a propósito 6 combinaciones faltantes, 3 duplicadas,
3 archivos con peso anómalo y 1 archivo fuera de catálogo, así que sirve para
verificar que el script detecta los cuatro casos.

## Salidas

### 1. `inventario_combinaciones.txt` — informe completo

| Sección | Contenido |
|---|---|
| Resumen | Archivos, peso total, combinaciones encontradas / faltantes / repetidas, outliers y cobertura. |
| 1. Tabla de archivos | Un renglón por archivo: etiqueta, `rep`, `iso`, `dsb`, `rpm`, `cant`, peso en MB, estado y ruta. |
| 2. Combinaciones repetidas | Las que aparecen más de una vez, con su etiqueta `_cantN` y los archivos implicados. |
| 3. Combinaciones faltantes | Lista completa y resumen por condición `rep_iso_dsb`. |
| 4. Outliers por tamaño | Estadísticos de cada grupo de comparación y detalle de cada archivo atípico. |
| 5. Fuera de catálogo | Archivos con algún parámetro que no está en las listas válidas. |
| 6. Avisos | Carpetas ignoradas, carpetas vacías y archivos sin RPM en el nombre. |

La **etiqueta** de cada combinación es `repR_isoI_dsbD_rpmV` y lleva el
parámetro adicional `_cantN` cuando la combinación aparece N > 1 veces:

```
rep1_iso32_dsb1_rpm600          ← medida una sola vez
rep2_iso32_dsb1_rpm4000_cant3   ← medida tres veces
```

Esa etiqueta es un identificador que existe **solo dentro del informe**, en la
columna `ETIQUETA`; no es el nombre del archivo ni lo modifica. El nombre real
de cada archivo se muestra tal cual en la columna `ARCHIVO`:

```
ETIQUETA                       CANT  PESO_MB  ARCHIVO
rep2_iso32_dsb1_rpm4000_cant3     3    0.988  rep2_iso32_dsb1/Rec_stb_iso32_dsb(0+8-7)-Rpm4000.txt
rep2_iso32_dsb1_rpm4000_cant3     3    0.983  rep2_iso32_dsb1/Rec_stb_iso32_dsb(0+8-7)_copia1-Rpm4000.txt
rep2_iso32_dsb1_rpm4000_cant3     3    0.981  rep2_iso32_dsb1/Rec_stb_iso32_dsb(0+8-7)_copia2-Rpm4000.txt
```

### 2. `faltantes_y_outliers.txt` — solo lo accionable

Contiene únicamente las dos listas que hay que atender: las combinaciones que
**no están medidas** y las que están medidas pero con un **tamaño de archivo
atípico**.

## Criterio de outlier

Un archivo se marca como outlier de tamaño cuando

```
|peso − mediana(pesos)| > k · 1.4826 · MAD(pesos)
```

es decir, cuando se aparta más de `k` desviaciones estándar robustas de la
mediana de su grupo (`k = 3` por defecto). Se usan mediana y MAD en lugar de
media y desviación estándar porque estas últimas se contaminan con los propios
valores atípicos que se quieren detectar: un solo archivo de 4 MB entre
archivos de 1 MB infla la desviación estándar lo suficiente como para dejar de
verse a sí mismo como anómalo. El factor 1.4826 convierte la MAD en un
equivalente a la desviación estándar de una distribución normal.

Casos límite:

* **MAD = 0** (más de la mitad de los archivos pesan exactamente lo mismo) →
  se usa la IQR.
* **IQR = 0 también** → se usa una tolerancia relativa del 1 % de la mediana.
* **Menos de 4 archivos en el grupo** → no se evalúan outliers; la dispersión
  no es estimable de forma fiable y el informe lo indica como
  `sin evaluar (n<4)`.

El grupo de comparación lo elige `--grupo-outliers`: `global` compara todos los
archivos entre sí (adecuado cuando todos los registros duran lo mismo), `rpm`
compara solo los de la misma velocidad (útil si la duración del registro
depende de la RPM) y `condicion` compara los de la misma carpeta.

Los pesos se reportan en MiB (base 1024).
