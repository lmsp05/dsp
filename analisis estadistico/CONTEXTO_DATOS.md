# Contexto: datos en bruto

Descripción del conjunto de datos tal como sale del laboratorio, antes de
cualquier procesamiento. Qué se midió y cómo están organizados los archivos.

---

## 1. Qué se midió

Ensayo de vibración sobre un **rotor apoyado en cojinetes hidrodinámicos
lubricados**, instrumentado con **cuatro proximitores** (sondas de
desplazamiento sin contacto, tipo corriente de Foucault) y un **keyphasor**
(sensor de referencia de fase, una marca por vuelta).

Los proximitores están dispuestos en **dos planos de medida × dos direcciones
ortogonales**:

| Canal en el archivo | Identificador | Plano | Dirección |
|---|---|---|---|
| `Mach1.P1.Y` | P1Y | cojinete 1 | vertical |
| `Mach1.P1.X` | P1X | cojinete 1 | horizontal |
| `Mach1.P2.Y` | P2Y | cojinete 2 | vertical |
| `Mach1.P2.X` | P2X | cojinete 2 | horizontal |

La magnitud registrada es el **vector 1X**: la componente de la vibración
síncrona con la velocidad de giro, expresada como **amplitud** y **fase**.

> **Importante para cualquier análisis posterior.** Estos archivos **no
> contienen la forma de onda cruda**. El sistema de adquisición ya aplicó el
> filtro síncrono y exportó únicamente amplitud y fase de la componente 1X en
> función del tiempo. No es posible hacer una FFT propia, ni extraer 2X, 0.5X,
> armónicos ni contenido de banda ancha a partir de estos datos. La cabecera lo
> declara: `1X [Pk-Pk]`.

Unidades: amplitud en **micrómetros pico-pico**, fase en **grados**, tiempo en
segundos. La fase está referida al keyphasor y llega envuelta en el rango
[0, 360).

---

## 2. Condiciones ensayadas

Tres factores cruzados, todas las combinaciones presentes:

| Factor | Niveles | Valores |
|---|---|---|
| Repetición | 3 | 1, 2, 3 — réplica completa del experimento (desmontaje y remontaje) |
| Viscosidad del aceite | 3 | ISO VG 32, 46, 68 |
| Nivel de desbalanceo | 3 | 1, 2, 3 — masa de desbalanceo creciente |

**3 × 3 × 3 = 27 ensayos.** Cada ensayo barre las mismas **15 velocidades de
giro**:

```
600, 1300, 2000, 2500, 2600, 2700, 2800, 3400, 3500, 3600, 3700,
4000, 4500, 5000, 5500  rpm
```

Las velocidades **no están uniformemente espaciadas**: el muestreo se
concentra alrededor de los 2500–3700 rpm, la zona de la primera velocidad
crítica del rotor. Los 600 rpm son la velocidad más baja del barrido, muy por
debajo de la crítica.

---

## 3. Organización de los archivos

Un archivo `.xlsx` por ensayo, con el nombre codificando las condiciones:

```
rep<repetición>_<viscosidad>_<desbalanceo>_<tipo>_<fecha>.xlsx
```

Ejemplos:

```
rep1_32_1_p_0424.xlsx     repetición 1, ISO VG 32, desbalanceo 1, proximitores
rep2_46_3_p_0424.xlsx     repetición 2, ISO VG 46, desbalanceo 3, proximitores
rep3_68_2_p_0424.xlsx     repetición 3, ISO VG 68, desbalanceo 2, proximitores
```

Campos:

| Campo | Valores | Nota |
|---|---|---|
| repetición | 1, 2, 3 | |
| viscosidad | 32, 46, 68 | grado ISO VG, sin prefijo |
| desbalanceo | 1, 2, 3 | |
| tipo | `p` o `ac` | `p` = proximitores, `ac` = acelerómetros |
| fecha | p. ej. `0424` | mes y día |

El sufijo de tipo separa **dos conjuntos paralelos**: los archivos `_p_`
(27, proximitores) y los archivos `_ac_` (acelerómetros, medidos en los mismos
ensayos). Son instrumentaciones distintas y su estructura interna **no es la
misma**; lo descrito en la sección 4 corresponde a los archivos `_p_`.

---

## 4. Estructura interna de cada archivo `_p_`

### Hojas

**Una hoja por velocidad de giro**, y el nombre de la hoja **es** el valor en
rpm: `600`, `1300`, `2000`, … `5500`.

Los libros contienen además hojas auxiliares con nombres no numéricos
(`prox`, `Hoja17`, …) que no son datos de ensayo.

### Disposición dentro de una hoja

```
fila 1   canal      NVD: Prof >Bode> 1X [Pk-Pk]> Mach1.P1.Y
fila 2   magnitud   Time | Displacement | Phase
fila 3   (cabecera)
fila 4+  datos numéricos
```

- **Tres filas de cabecera**; los datos empiezan en la cuarta.
- La **fila 1** identifica el canal. El nombre del sensor va al final de la
  cadena, con la forma `Mach1.P<n>.<X|Y>`.
- La **fila 2** identifica la magnitud de la columna: `Time`, `Displacement`
  (amplitud) o `Phase`.
- Hay **una columna de tiempo** común y **dos columnas por sensor**
  (amplitud y fase), es decir 4 sensores × 2 = 8 columnas de datos.

El orden de las columnas **no es fijo entre archivos**: hay que localizarlas
por el contenido de la cabecera, no por su posición.

---

## 5. Naturaleza de cada registro

Cada hoja es un **registro temporal a velocidad constante**, no un barrido.

El rotor se lleva a la velocidad indicada, se mantiene ahí durante
**aproximadamente 3 minutos**, y durante ese tiempo se registra la evolución
del vector 1X. El resultado son muchas muestras sucesivas del mismo fasor.

Consecuencias para el análisis:

- El inicio de cada registro contiene un **transitorio**: el fasor se desplaza
  mientras se asientan las condiciones térmicas e hidrodinámicas de la película
  de aceite. La parte útil es el tramo final, ya estabilizado.
- La cantidad de muestras **varía entre hojas**: los registros no tienen todos
  la misma longitud ni el mismo número de puntos.
- La fase, al estar envuelta en [0, 360), presenta **saltos artificiales de
  360°** cuando el fasor cruza el origen del ángulo. Un salto de 359° a 1° es
  un movimiento real de 2°, no de 358°.

---

## 6. Inventario

| | |
|---|---|
| Archivos de proximitores (`_p_`) | 27 |
| Hojas de ensayo por archivo | 15 (una por velocidad) |
| Sensores por hoja | 4 |
| Columnas de datos por hoja | 9 (1 de tiempo + 4 × 2) |
| Registros temporales totales | 27 × 15 = **405** |
| Series temporales totales | 405 × 4 sensores × 2 magnitudes = **3 240** |
| Duración de cada registro | ~3 min a velocidad constante |

---

## 7. Puntos sin confirmar

- **Frecuencia de muestreo** de los registros: no está declarada en los
  archivos; solo existe la columna de tiempo, de la que puede deducirse.
- **Estructura interna de los archivos `_ac_`** (acelerómetros): distinta a la
  descrita, no documentada aquí.
- **Claro radial y geometría de los cojinetes**, temperatura del aceite y masa
  exacta de cada nivel de desbalanceo: no forman parte de estos archivos.
