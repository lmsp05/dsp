# FFT de señales de vibración de rotorkit (por lotes)

Carpeta **autónoma**: no depende del resto del repositorio. Genera el espectro
FFT (amplitud + fase) de los proximitores de todos los archivos de vibración de
una carpeta y lo guarda en un único `.txt` filtrable por condiciones, con un
conversor a `.npy` para leerlo rápido desde otros scripts.

Parte del prototipo que compartiste (`fft_confiable_rotorkit.py`), conservando lo
que estaba bien hecho (keyphasor → RPM instantánea → segmentación por
estacionariedad → ventanas de un número **entero de revoluciones** con Hann) y
añadiendo el recorrido por lotes, la selección automática de tramo y el guardado
de amplitud **y** fase.

## Archivos

| Script | Qué hace |
|---|---|
| `procesar_fft.py` | Recorre la carpeta de datos, procesa cada archivo y escribe un `.txt` tabular con el espectro de los 4 proximitores. |
| `txt_a_npy.py` | Convierte ese `.txt` en un `.npy` (array estructurado de numpy) fácil de filtrar. |
| `revision_rpms.py` | **Diagnóstico** de la RPM y de la segmentación: una figura por archivo (RPM vs tiempo, bloques, tramos aceptados/rechazados/seleccionado) + un CSV de métricas de dispersión. |
| `rpm_instantaneas.py` | Script sencillo: guarda la RPM instantánea (pulso a pulso) de cada archivo con su condición, en un `.txt`. |
| `deteccion_picos.py` | Lee los espectros de `procesar_fft.py` y detecta picos (procedimiento de `x0302_nf_gph_transformations.m`): tabla `omega1, omega2, …` + diagrama de dispersión frecuencia-vs-velocidad. |
| `inspeccionar_espectro.py` | Inspección visual de **una** condición (rep/iso/dsb/rpm/sensor): espectro original, espectro sin 1X/armónicos y picos detectados con sus métricas, en una figura de 3 paneles. |

Solo requiere `numpy` (y `scipy` no es necesario: la FFT se hace con
`numpy.fft`).

## Uso

```bash
# 1) generar el .txt con los espectros de toda la base de datos
python procesar_fft.py --data-dir /ruta/a/RAW_DATA --salida resultados_fft.txt

# 2) convertirlo a .npy
python txt_a_npy.py --entrada resultados_fft.txt --salida resultados_fft.npy
```

Opciones útiles de `procesar_fft.py` (`--help` para verlas todas):

| Opción | Def. | Significado |
|---|---|---|
| `--revs` | 15 | revoluciones enteras por bloque de FFT |
| `--solape` | 0.5 | solape entre bloques |
| `--umbral-rpm` | 1.5 | % de variación de RPM que separa tramos |
| `--min-dur` | 20 | duración mínima de un tramo utilizable [s] |
| `--fmin` / `--fmax` | 2 / 500 | rango de frecuencia guardado [Hz] |
| `--limit` | 0 | procesar solo los primeros N archivos (prueba) |

### Estructura de datos esperada

```
RAW_DATA/
├── rep1_iso32_dsb1/
│   ├── Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt
│   └── ...
├── rep1_iso32_dsb2/
└── ... hasta rep3_iso68_dsb3
```

* De la **carpeta** se extraen `rep`, `iso`, `dsb` (patrón `repN_isoVG_dsbM`).
* Del **nombre del archivo** se extrae la RPM nominal (`...Rpm600.txt` → 600).
* De cada archivo se leen solo los proximitores `P1.Y, P1.X, P2.Y, P2.X`
  (el keyphasor se usa para la RPM y el acelerómetro `S1` se ignora).

## Diagnóstico de RPM (`revision_rpms.py`)

Si la RPM promedio medida se aparta más de lo esperado de la nominal, este
script lo explica. Para cada archivo genera una figura con:

* la RPM instantánea (pulso a pulso) frente al tiempo;
* la RPM media por bloque de `bloque_s` (lo que alimenta la segmentación) y los
  **splits** donde el cambio supera el umbral, con el % anotado;
* todos los tramos candidatos: **verde** = aceptado, **rojo** = rechazado (con el
  motivo), **marco dorado** = el finalmente seleccionado, cada uno con duración,
  media, mediana y coeficiente de variación;
* el histograma de la RPM instantánea con las métricas de dispersión, resaltando
  **media vs mediana** y los outliers.

```bash
python revision_rpms.py --data-dir /ruta/a/RAW_DATA --outdir revision_rpms
```

Además escribe `revision_rpms/revision_rpms_metricas.csv` con una fila `GLOBAL`
por archivo y una fila por tramo candidato (filtrable).

**Clave para tu preocupación:** `procesar_fft.py` reporta como RPM del tramo la
**media** de la RPM instantánea, y la media es muy sensible a errores de
detección de pulsos del keyphasor, en ambos sentidos:

* **Pulsos perdidos** → intervalos dobles/triples → RPM instantáneas en
  **rpm/2, rpm/3, …** que **hunden** la media (síntoma: bandas horizontales en
  ½, ⅓, ¼ de la nominal en el gráfico de RPM). Causa típica: umbral de disparo
  demasiado alto para pulsos cortos cuya cresta muestreada varía. La detección
  actual usa un umbral **bajo** (25 % del rango, equivalente al cruce por 0 V
  del hardware de este dataset) con **histéresis**, precisamente para evitarlo.
* **Pulsos dobles** → intervalos diminutos → RPM instantáneas gigantes que
  **inflan** la media. La histéresis + la separación mínima los eliminan.

En el diagnóstico: si dentro del tramo **media ≈ mediana**, la velocidad medida
es de fiar y la diferencia con la nominal es real; si difieren (o hay outliers),
es un artefacto de detección de pulsos. Ojo también con un **escalón de RPM**
dentro del registro: la media *global* se aleja de la nominal por mezclar dos
velocidades, pero la del **tramo seleccionado** sí es correcta.

## Detección de picos (`deteccion_picos.py`)

Lee `resultados_fft.txt` (o `.npy`) y detecta los picos de **cada espectro**
replicando el procedimiento de `x0302_nf_gph_transformations.m`. En el `.m`, la
detección por espectro es el bucle de *ridge detection*:

```matlab
[~,locs] = findpeaks(row, 'MinPeakProminence', max(row)*0.1, 'MinPeakDistance', 5);
```

que aquí es `scipy.signal.find_peaks` sobre la amplitud lineal con
`prominence = max(amplitud) * fraccion` (0.1 por defecto) y `distance = 5` bins.
Los picos se ordenan por frecuencia ascendente → `omega1, omega2, …`.

**Eliminación de síncronos y armónicos (1X, 2X, … NX) — método por hombros:**
antes de detectar, se quitan del espectro las líneas de giro y sus armónicos,
para que los `omega` sean solo contenido no síncrono (frecuencias naturales):

1. Se estima el **1X real** con precisión **sub-bin**: se parte de `rpm/60`
   (usando la RPM *medida* del `.txt` si está), se toma el **pico más alto** en
   una banda `±band_1x` y se afina por interpolación parabólica. Esto lo hace
   robusto al **sesgo de la RPM medida** (la media puede caer un bin o más por
   debajo/encima del 1X real) y evita propagar error a los armónicos altos.
2. Para cada orden `n = 1…n_armonicos` se busca el **pico real más cercano** a
   `n·f1` dentro de una ventana modesta `win = máx(tol_orden·f1, 1.5·df)`. Buscar
   por orden y quedarse con el pico más cercano tolera la cuantización de bins y
   el leakage sin tragarse naturales cercanas a un orden.
3. Para cada pico síncrono se detectan sus **hombros** (se desciende a cada lado
   hasta el primer mínimo local) y se elimina el pico **completo** en ese
   intervalo, sustituyéndolo por la **línea base interpolada**. `ancho_bins` es
   solo un piso mínimo de medio ancho por si la punta es plana o ruidosa.

Ventaja frente al notch de ancho fijo: el ancho lo define el **propio pico**, así
que solo se recorta donde de verdad hay contenido síncrono. Una frecuencia
natural cercana a un orden pero separada por un valle **no** se elimina (no es el
pico cuya punta cae en el margen y queda fuera de sus hombros).

Como la prominencia es relativa al máximo, al quitar el 1X (que domina con
desbalance) el umbral baja y **afloran picos naturales débiles** que antes se
perdían. Se puede desactivar con `--conservar-armonicos`.

```bash
python deteccion_picos.py --entrada resultados_fft.txt --outdir .
# parámetros del criterio (por defecto los del .m)
python deteccion_picos.py --entrada resultados_fft.npy --fraccion 0.1 --dist-bins 5 --fmax 300
# ajustar la eliminación de armónicos
python deteccion_picos.py --entrada resultados_fft.txt --n-armonicos 12 --ancho-bins 3
# no eliminar armónicos (comportamiento anterior)
python deteccion_picos.py --entrada resultados_fft.txt --conservar-armonicos
```

**Filtrado por niveles:** con `--rep`, `--iso`, `--dsb`, `--rpm` y `--sensor` se
indican **listas de niveles** (separadas por comas); solo se procesan las
condiciones que son una combinación de esos niveles. Una propiedad sin niveles
admite cualquier valor.

```bash
# solo iso 32 y 68, rpm 600 y 1200, cualquier rep/dsb, sensor P1.Y
python deteccion_picos.py --entrada resultados_fft.txt --iso 32,68 --rpm 600,1200 --sensor P1.Y
```

El diagrama de dispersión dibuja las líneas de orden **1X…20X** como referencia.

Genera, en la carpeta de salida:

* **`picos_detectados.txt`** — tabla con una fila por archivo·sensor:
  `rep  iso  dsb  rpm  sensor  omega1  omega2  …  omegaN`
  (celdas vacías donde ese espectro tiene menos picos).
* **`picos_scatter.png`** — dispersión frecuencia (Y) vs velocidad (X), con
  **color = viscosidad ISO** y **forma = desbalance dsb**, y líneas de orden
  1X/2X/3X de referencia.
* **`picos_scatter_<sensor>.png`** — la misma dispersión, un archivo por sensor.

> Nota: como en el `.m`, la prominencia es **relativa al máximo de cada
> espectro**. Con la eliminación de armónicos activada (por defecto) el 1X ya no
> domina el máximo, así que los picos débiles se detectan mejor; aun así puedes
> bajar `--fraccion` para capturar más.
>
> Con el método por hombros, una natural solo se elimina si es *ella misma* el
> pico cuya punta cae dentro del margen de un orden (mismo bin que el armónico a
> esa RPM). Si está separada del armónico por un valle, se conserva. Aun así,
> cuando natural y armónico se solapan de verdad a una velocidad, la natural
> reaparece en las demás (no se mueve con la RPM), así que la banda horizontal
> sigue viéndose en el scatter.

## Inspección de una condición (`inspeccionar_espectro.py`)

Para revisar en detalle **una sola** medición: eliges `rep/iso/dsb/rpm/sensor` y
genera una figura de 3 paneles con el mismo pipeline que `deteccion_picos.py`:

1. **Espectro original**, con las bandas síncronas (1X…NX) sombreadas y el 1X
   estimado marcado.
2. **Espectro tras eliminar** el 1X y sus armónicos (notch + interpolación).
3. **Picos detectados** sobre el espectro limpio + las **métricas**: umbral de
   prominencia (`máx·fracción`), distancia mínima, 1X, nº de armónicos, `df`, y
   la prominencia real de cada pico dibujada como barra vertical.

Igual que `deteccion_picos.py`, filtra por **listas de niveles** (`--rep`,
`--iso`, `--dsb`, `--rpm`, `--sensor`; vacío = cualquiera) y genera **una figura
por cada combinación** que pasa el filtro.

```bash
# una sola condición
python inspeccionar_espectro.py --entrada resultados_fft.txt \
    --rep 1 --iso 46 --dsb 1 --rpm 600 --sensor P1.Y
# varios niveles -> una figura por combinación (rep{1,3} × iso{32,68} × …)
python inspeccionar_espectro.py --entrada resultados_fft.txt \
    --rep 1,3 --iso 32,68 --sensor P1.Y
```

Acepta los mismos parámetros de detección (`--fraccion`, `--n-armonicos`,
`--ancho-bins`, …), así que sirve para **calibrar** esos valores viendo el efecto
antes de lanzar `deteccion_picos.py` sobre toda la base. `--limit N` acota el
número de figuras; si ningún filtro coincide, lista las disponibles.

## Formato de salida (`.txt`)

Formato **largo/tidy**: una fila por línea espectral, con los metadatos repetidos
en cada fila para poder filtrar directamente. Columnas:

```
rep  iso  dsb  rpm_nominal  sensor  t_ini_s  t_fin_s  rpm_medida
n_bloques  revs_bloque  df_Hz  frecuencia_Hz  amplitud  fase_grados
```

Filtrar es inmediato, por ejemplo con pandas:

```python
import pandas as pd
df = pd.read_csv("resultados_fft.txt", sep="\t", comment="#")
sel = df[(df.rep == 1) & (df.iso == 32) & (df.sensor == "P1.Y")]
```

o, tras convertir a `.npy`:

```python
import numpy as np
d = np.load("resultados_fft.npy")
sel = d[(d["rep"] == 1) & (d["iso"] == 32) & (d["sensor"] == "P1.Y")]
f, amp, fase = sel["frecuencia_Hz"], sel["amplitud"], sel["fase_grados"]
```

---

## Respuestas a tus tres preguntas

### 1. ¿Cómo se promedia para obtener un único espectro del bloque?

El prototipo usa el **método de Welch**, que promedia **potencia** (`|FFT|²`) de
los distintos segmentos de un tramo, **no** la fase. Por eso Welch da un único
espectro de amplitud pero pierde la fase. Aquí se hace explícito y se conservan
las dos cosas:

* **Amplitud** → promedio **RMS de las magnitudes** de los bloques:
  `amp(f) = sqrt( mean_k |X_k(f)|² )`. Es exactamente lo que hace Welch (promedio
  de potencia), pero en unidades de amplitud. Reduce la varianza y funciona igual
  para el contenido síncrono (1X, 2X…) y para el **no** síncrono (frecuencias
  naturales, whirl de aceite) — que es lo que interesa para OMA.
* **Fase** → fase del promedio **coherente** (vectorial):
  `fase(f) = angle( mean_k X_k(f) )`. Como cada bloque arranca en un pulso del
  keyphasor, el contenido síncrono tiene fase fija y se promedia bien. El
  contenido no síncrono tiene fase aleatoria entre bloques y su fase promedio es
  ruidosa: **eso es esperable y hasta útil** (fase estable ⇒ línea de orden;
  fase que no se repite ⇒ contenido estructural).

Cada bloque son 15 revoluciones enteras con ventana **Hann** y 50 % de solape;
`nperseg` se fija a la RPM media del tramo, de modo que todos los bloques
comparten la misma grilla de frecuencia y el 1X cae prácticamente en el bin
exacto (sin *leakage*).

### 2. ¿Qué tramo estacionario conviene elegir para detectar picos luego?

`procesar_fft.py` selecciona **un** tramo por archivo con este criterio
(`seleccionar_tramo`), que es el que te recomiendo:

1. **Cerca de la RPM nominal** del nombre del archivo (≤ 5 %): así el espectro
   corresponde de verdad a la velocidad con la que está etiquetada la condición y
   no a un transitorio ni a un escalón de velocidad (recuerda el escalón real de
   ~600 → ~618 RPM que menciona el prototipo).
2. **El más largo** entre esos: más revoluciones ⇒ más bloques promediados ⇒
   menor varianza y piso de ruido más limpio. Esa reducción de varianza es
   justo lo que hace que un pico sea detectable de forma fiable.
3. **Desempate**: la RPM más estable (menor coeficiente de variación).

Idea de fondo: para *peak picking* de frecuencias naturales, un tramo largo y
estacionario a la velocidad nominal es mejor que uno corto o transitorio, aunque
"se vea" con picos, porque el promediado limpia el ruido sin ensanchar ni
desdoblar las líneas.

### 3. ¿Amplitud + fase (grados) o número complejo?

Recomendación **mixta**, que es lo que implementan los scripts:

* En el **`.txt`** (para leer con los ojos): **amplitud + fase en grados**. Es lo
  más interpretable para un ingeniero, se compara directo con la lectura de los
  instrumentos y los saltos de fase se ven a simple vista.
* En el **`.npy`** (para las máquinas): además de amplitud y fase, se guarda el
  **número complejo** (`complejo = amplitud · e^{i·fase}`). Es **sin pérdidas**:
  de él se recuperan amplitud (`abs`) y fase (`angle`) exactas, y permite
  operaciones vectoriales (promediar, restar el *runout*, comparar fases entre
  sensores) sin el problema del salto de fase en ±180°.

Así cada script aguas abajo usa la representación que le convenga sin tener que
reconvertir nada.
