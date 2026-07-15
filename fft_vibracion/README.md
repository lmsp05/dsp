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
**media** de la RPM instantánea. La media se dispara con muy pocos pulsos
espurios/perdidos del keyphasor (un pulso doble da una RPM instantánea de decenas
de miles), mientras que la **mediana** casi no se ve afectada. En este
diagnóstico:

* si dentro del tramo **media ≈ mediana**, la velocidad medida es de fiar y la
  diferencia con la nominal es real (p. ej. el rotor giraba a 596, no a 600);
* si **media ≫ mediana** (o hay outliers), el problema es de detección de pulsos:
  conviene usar la mediana o filtrar los pulsos espurios. Ojo también con el caso
  de un **escalón de RPM** dentro del registro: ahí la media *global* se aleja de
  la nominal por mezclar dos velocidades, pero la media del **tramo seleccionado**
  (una sola velocidad) sí es correcta.

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
