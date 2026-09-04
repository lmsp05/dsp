# Conversión de datos: `.txt` → `.npz`

## Comando

```bat
python "C:\Users\Owner\Documents\git\psd\dsp\conversion de datos\convertir_a_npz.py" ^
       --entrada "C:\Users\Owner\Documents\BD\All data" ^
       --salida  "D:\donde\quieras\npz"
```

El nombre de la carpeta lleva espacios, así que en Windows la ruta **debe ir
entre comillas**.

### Banderas

| Bandera | Por defecto | Qué hace |
|---|---|---|
| `--entrada` | *(obligatoria)* | Carpeta raíz que contiene las carpetas `repX_isoYY_dsbZ` |
| `--salida` | *(obligatoria)* | Carpeta donde se escriben los `.npz`; se crea si no existe |
| `--fs` | `12800` | Frecuencia de muestreo [Hz]. **Solo** se usa para generar el eje de tiempo cuando el archivo no trae uno |
| `--patron-carpeta` | `rep(?P<rep>\d+)_iso(?P<iso>\d+)_dsb(?P<dsb>\d+)` | Regex de la carpeta, con los grupos `rep`, `iso` y `dsb` |
| `--patron-rpm` | `[Rr][Pp][Mm][\s_\-]*(\d+)` | Regex con un grupo de captura para la RPM del nombre del archivo |
| `--extension` | `.txt` | Extensión de los archivos de medición |
| `--canales` | los 5 de `config.py` | Nombres de los canales en el orden de las columnas. Solo se usan si la cabecera del archivo no los declara |
| `--columna-tiempo` | `auto` | `auto`, `si` o `no`: si la primera columna del bloque de datos es la base de tiempos |
| `--comprimir` | apagado | Guarda comprimido: archivos más pequeños, carga más lenta |
| `--sobrescribir` | apagado | Rehace los `.npz` que ya existan (por defecto se omiten) |
| `--limite N` | `0` (todos) | Convierte solo los N primeros archivos |
| `--seco` | apagado | No escribe nada: solo lista lo que se convertiría |
| `--verboso` | apagado | Una línea por archivo en vez de barra de avance |

### Ejemplos

```bat
REM ver qué haría, sin escribir nada
python "%C%\convertir_a_npz.py" --entrada "%DATOS%" --salida "%NPZ%" --seco

REM probar con 3 archivos antes de lanzar el lote completo
python "%C%\convertir_a_npz.py" --entrada "%DATOS%" --salida "%NPZ%" --limite 3 --verboso

REM lote completo, comprimido
python "%C%\convertir_a_npz.py" --entrada "%DATOS%" --salida "%NPZ%" --comprimir

REM rehacer todo desde cero
python "%C%\convertir_a_npz.py" --entrada "%DATOS%" --salida "%NPZ%" --sobrescribir
```

Se puede volver a lanzar cuantas veces se quiera: los archivos ya convertidos
se omiten, así que una conversión interrumpida se reanuda sola.

---

## Qué hace

Recorre la estructura de la base de datos experimental:

```
<entrada>/
    rep1_iso32_dsb1/
        Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt
        Rec_stb_iso32_dsb(0+8-7)-Rpm1200.txt
        ...
    rep1_iso32_dsb2/
    ...
```

y escribe un `.npz` por cada `.txt`:

```
<salida>/
    rep1_iso32_dsb1_rpm600.npz
    rep1_iso32_dsb1_rpm1200.npz
    ...
```

Los `.npz` quedan **todos en una sola carpeta plana**, sin subcarpetas: la
condición experimental completa va en el nombre (`rep_iso_dsb_rpm`), así que un
archivo suelto sigue siendo identificable sin abrirlo y sin depender de dónde
esté guardado.

### Por qué convertir

Medido sobre 24 archivos sintéticos de 25 600 muestras × 5 canales:

| Formato | Tiempo de carga | Tamaño |
|---|---|---|
| `.txt` original | 1.90 s | 49.5 MB |
| `.npz` sin comprimir | 0.03 s | 29.6 MB |
| `.npz` comprimido | 0.17 s | 25.2 MB |

Un `.txt` hay que reinterpretarlo entero cada vez: detectar la codificación,
localizar el bloque `BEGIN_DATA`, deducir el delimitador y el separador
decimal, y convertir cada número desde texto. El `.npz` guarda el array binario
ya interpretado. Sobre estos datos la carga sale **~58× más rápida** sin
comprimir y **~11×** comprimida.

Usa `--comprimir` si el espacio en disco importa más que la velocidad; en caso
contrario, déjalo apagado.

---

## Cómo abrir un `.npz`

### Directamente con pandas

```python
import numpy as np, pandas as pd

d  = np.load("rep1_iso32_dsb1_rpm600.npz", allow_pickle=False)
df = pd.DataFrame(d["datos"], columns=d["columnas"])

rpm = float(d["rpm"])
fs  = float(d["fs_hz"])
```

`allow_pickle=False` funciona porque no se guarda ningún objeto de Python:
solo arrays numéricos y cadenas. Eso hace los archivos portables entre
versiones de numpy y seguros de abrir.

### Con el ayudante del módulo

```python
import sys; sys.path.insert(0, r"...\conversion de datos")
from convertir_a_npz import cargar

df, meta = cargar("rep1_iso32_dsb1_rpm600.npz")
# meta = {'rep': 1, 'iso': 32, 'dsb': 1, 'rpm': 600.0, 'fs_hz': 12800.0,
#         'tiempo_origen': 'archivo', 'n_muestras': 25600,
#         'archivo_origen': 'Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt'}
```

### Cargar toda la base en un solo DataFrame

```python
import glob, numpy as np, pandas as pd

filas = []
for f in sorted(glob.glob(r"D:\npz\*.npz")):
    with np.load(f, allow_pickle=False) as d:
        sub = pd.DataFrame(d["datos"], columns=[str(c) for c in d["columnas"]])
        sub["rep"], sub["iso"] = int(d["rep"]), int(d["iso"])
        sub["dsb"], sub["rpm"] = int(d["dsb"]), float(d["rpm"])
        filas.append(sub)
todo = pd.concat(filas, ignore_index=True)
```

> Cuidado con la memoria: son 25 600 muestras × 5 canales por archivo. Con la
> base completa esto son varios GB. Para trabajar condición por condición es
> mejor cargar solo los archivos que hagan falta.

---

## Contenido de cada `.npz`

| Clave | Tipo | Contenido |
|---|---|---|
| `datos` | `float64` (n × m) | La matriz de señales |
| `columnas` | array de texto | Nombres de las columnas de `datos`, en orden |
| `rep` | `int64` | Repetición |
| `iso` | `int64` | Grado ISO de viscosidad |
| `dsb` | `int64` | Nivel de desbalanceo |
| `rpm` | `float64` | Velocidad de giro |
| `fs_hz` | `float64` | Frecuencia de muestreo |
| `tiempo_origen` | texto | `'archivo'` o `'sintetizado'` |
| `n_muestras` | `int64` | Número de filas |
| `archivo_origen` | texto | Nombre del `.txt` de procedencia |

La **primera columna de `datos` es siempre `tiempo_s`**, seguida de un canal por
columna:

```
tiempo_s   Mach1.P1.Y   Mach1.P1.X   Mach1.P2.Y   Mach1.P2.X   Mach1.S1
0.000000     0.300181     1.391401     1.581586     1.574434   0.070305
0.000078     0.425594     1.862143     1.849765     2.250336   0.873723
```

### Sobre el eje de tiempo

- Si el `.txt` trae su propia columna de tiempo, se conserva **tal cual** y
  `tiempo_origen` vale `'archivo'`.
- Si no la trae, se genera como `n / fs` y `tiempo_origen` vale
  `'sintetizado'`.

En el segundo caso **el eje temporal es una suposición basada en `--fs`**, no
un dato medido. Comprueba `tiempo_origen` antes de usarlo para algo que dependa
de la escala temporal absoluta.

---

## Garantías y límites

**Los números no se tocan.** El script convierte de formato, nada más: no
resta la media, no filtra, no reescala ni recorta. Verificado sobre los 24
archivos de prueba: la matriz que sale del `.npz` es **idéntica bit a bit** a
la que se obtiene parseando el `.txt`.

**Un archivo ilegible no detiene el lote.** Se registra por nombre con el
motivo al final de la ejecución y la conversión continúa. Comprobado con un
archivo sin bloque numérico: 24 convertidos, 1 error reportado.

**Columnas de más.** Si un archivo trae más columnas que nombres conocidos, las
sobrantes se guardan como `canal_6`, `canal_7`… en vez de descartarse: perder
una columna al convertir sería peor que darle un nombre genérico.

**Carpetas que no cumplen el patrón se ignoran** en silencio (usa `--verboso`
para verlas). Un archivo del que no se pueda extraer la RPM se omite con un
aviso.

---

## Dependencias

`numpy` para convertir; `pandas` solo si usas `cargar()` o construyes el
DataFrame. Ambas están en el `requirements.txt` de la raíz.

El script reutiliza `io_utils.py` de la raíz del repositorio, que es donde está
resuelto el parseo de los `.txt` (codificaciones, bloques
`BEGIN_DATA`/`END_DATA`, delimitador, coma decimal, detección de canales en la
cabecera). Por eso **debe permanecer en una subcarpeta del repositorio**, junto
a `io_utils.py` y `config.py`. Si lo mueves fuera, avisa con un mensaje claro en
vez de fallar de forma confusa.
