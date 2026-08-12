# Guía de los resultados del paso 5 (`p5_estadistica`)

Referencia de los archivos que genera `p5_anova_split_plot.py`. Con
`--grupos-velocidad` son 51 (**25 CSV** + **26 PNG**); añadiendo
`--grupos-desbalanceo` son 82.

---

## 1. El esquema de nombres

Todo archivo pertenece a una **familia** y a una **variante**:

```
<familia>.csv                 -> las 15 velocidades juntas   (análisis global)
<familia>_Grupo_1_bajas.csv   -> solo  600, 1300, 2000 rpm
<familia>_Grupo_2_pre_1a.csv  -> solo  2500, 2600, 2700, 2800 rpm
<familia>_Grupo_3_1a_critica  -> solo  3400, 3500, 3600, 3700 rpm
<familia>_Grupo_4_altas       -> solo  4000, 4500, 5000, 5500 rpm
```

**La variante con sufijo es el mismo análisis, repetido sobre el subconjunto de
filas de ese tramo de velocidad.** Misma estructura, mismas columnas, mismo número
de filas. Solo cambian los números, porque se calculan con menos velocidades.

Existe porque la interacción `Viscosidad × Velocidad` sale significativa: el efecto
de la viscosidad **no es el mismo a todas las velocidades**, así que un único número
global lo promedia y lo esconde. Los tramos se editan en `GRUPOS_VELOCIDAD`.

> **Qué cambia y qué no entre el global y los grupos.** Los grados de libertad de
> Error(a) = 4 y Error(b) = 12 son **iguales en todas las variantes**: no dependen
> del número de velocidades. Lo que cambia es Error(c), cuyos gl son
> `a·b·(r−1)·(c−1)` = 252 con 15 velocidades, 36 con 3 y 54 con 4. Las sumas de
> cuadrados sí se recalculan dentro de cada subconjunto, así que las columnas de
> contribución son relativas **al total de ese tramo** y no se comparan en valor
> absoluto entre tramos.

> **Aviso de comparaciones múltiples.** Los análisis salen todos de los mismos
> datos y los valores p **no** están corregidos por multiplicidad. Úsalos para
> describir en qué régimen vive el efecto, no para acumular pruebas independientes.

### Variantes `_Desbalanceo_N` (con `--grupos-desbalanceo`)

```
<familia>_Desbalanceo_1 / _2 / _3
```

Ojo, estas **no** son solo un subconjunto: al fijar el desbalanceo ese factor sale
del modelo y el diseño se reduce de split-split-plot a **split-plot en bloques**
(parcela grande = viscosidad, subparcela = velocidad). Consecuencias en los
archivos:

* `anova_split_plot_Desbalanceo_N.csv` tiene **24 filas** (4 sensores × **6**
  fuentes), no 44. Las fuentes son: `Repeticion (bloque)`, `Viscosidad`,
  `Error(a) = Rep x Visc` (4 gl), `Velocidad`, `Viscosidad x Velocidad`,
  `Error(b) = residual` (84 gl). **No hay `Error(c)`.**
* `anova_ingenuo_Desbalanceo_N.csv` tiene **16 filas** (4 sensores × 4: los 3
  efectos + el residual agrupado de 90 gl).
* `componentes_varianza_Desbalanceo_N.csv` tiene **8 filas** (4 sensores × **2**
  estratos): `Parcela grande (montaje del aceite)` y
  `Subparcela (barrido de velocidad)`.
* `p5_interacciones_Desbalanceo_N.png` tiene **una sola fila** de paneles (la de
  viscosidad × velocidad): con un solo desbalanceo, la fila de abajo sería un
  punto por curva.
* `comparacion_viscosidad_*` y `posthoc_viscosidad_*` mantienen su forma (4 y 12
  filas), porque siguen hablando solo de la viscosidad.

**La viscosidad se sigue contrastando con 4 gl.** Separar por desbalanceo no añade
grados de libertad al aceite: los montajes de parcela grande siguen siendo 9.

Y una figura extra sin variantes: **`p5_viscosidad_por_desbalanceo.png`**, con la
misma estructura que `p5_viscosidad_por_grupo.png` pero comparando los tres niveles
de desbalanceo.

---

## 2. Los cinco CSV

### 2.1 `anova_split_plot*.csv` — la tabla principal

**44 filas = 4 sensores × 11 fuentes. 11 columnas.** Es el resultado central: el
ANOVA hecho bien.

Las 11 filas de cada sensor, siempre en este orden:

| # | Fuente | GL | Se contrasta contra |
|---|---|---|---|
| 1 | `Repeticion (bloque)` | 2 | Error(a) |
| 2 | **`Viscosidad`** | 2 | **Error(a)** |
| 3 | `Error(a) = Rep x Visc` | 4 | — (es un error) |
| 4 | `Desbalanceo` | 2 | Error(b) |
| 5 | `Viscosidad x Desbalanceo` | 4 | Error(b) |
| 6 | `Error(b) = Rep x Desb \| Visc` | 12 | — (es un error) |
| 7 | `Velocidad` | c−1 | Error(c) |
| 8 | `Viscosidad x Velocidad` | 2(c−1) | Error(c) |
| 9 | `Desbalanceo x Velocidad` | 2(c−1) | Error(c) |
| 10 | `Viscosidad x Desbalanceo x Velocidad` | 4(c−1) | Error(c) |
| 11 | `Error(c)` | 9·2·(c−1) | — (es un error) |

Los gl suman siempre `n_observaciones − 1` (404 en el global). Está comprobado con
una aserción en el código.

Las columnas:

| Columna | Qué es |
|---|---|
| `Sensor` | P1Y / P1X / P2Y / P2X |
| `Fuente` | el efecto o el estrato de error de esa fila |
| `SS` | suma de cuadrados: la variabilidad total atribuible a esa fuente |
| `GL` | grados de libertad de la fuente (el **numerador** de la F) |
| `Error` | qué estrato se usa de denominador. **Vacío** en las 3 filas de error |
| `MS` | cuadrado medio = `SS / GL` |
| `GL_denominador` | gl del error usado. **Vacío** en las filas de error |
| `F` | `MS_fuente / MS_error`. **Vacío** en las filas de error |
| `p` | probabilidad de ver una F así de grande si el efecto no existiera |
| `Contribucion_SS_%` | `100·SS/SS_total`. **Descriptivo**, suma 100 entre las 11 filas |
| `omega2_%` | tamaño de efecto **insesgado** |

Sobre las dos últimas, que se confunden con facilidad:

* `Contribucion_SS_%` dice **cuánta variabilidad ocupa** una fuente. No es
  significancia. `Velocidad` se lleva el 68 % simplemente porque el barrido cruza
  resonancias: es el diseño, no un hallazgo.
* `omega2_%` = `(SS − GL·MS_error) / (SS_total + MS_error)`, truncado en 0. Descuenta
  lo que esa fuente se llevaría **solo por azar**. Cuando `omega2_%` es mucho menor
  que `Contribucion_SS_%`, buena parte de esa suma de cuadrados era ruido. Está
  vacío en las filas de error.

### 2.2 `anova_ingenuo*.csv` — el análisis incorrecto, para comparar

**32 filas = 4 sensores × 8 filas. 8 columnas.** Reproduce lo que hacían
`f3_s1_v1.py` y `f3_s1_v2.py`: ignora los bloques y contrasta **todo** contra un
único residual.

Las 8 filas por sensor: los 7 efectos (sin `Repeticion`, que este modelo ni
considera) más `Residual (agrupado)`.

Columnas: `Sensor`, `Fuente`, `SS`, `GL`, `MS`, `GL_denominador`, `F`, `p`.

**Lo esencial:** `SS`, `GL` y `MS` son **idénticos** a los de `anova_split_plot`.
La descomposición no cambia. Lo único que cambia es `GL_denominador` (270 en el
global, en vez de 4 / 12 / 252) y, en consecuencia, `F` y `p`.

La fila `Residual (agrupado)` es la suma de los cuatro estratos que el modelo
ingenuo mete en el mismo saco:

```
SS(Repeticion) + SS(Error a) + SS(Error b) + SS(Error c)     gl = 2+4+12+252 = 270
```

Este archivo **no** es un resultado que reportar. Existe para poder decir con
números cuánto se inflaba la significancia antes.

### 2.3 `comparacion_viscosidad*.csv` — el resumen de una página

**4 filas, una por sensor. 10 columnas.** Es un extracto: coge la fila
`Viscosidad` de los dos ANOVAs anteriores y las pone lado a lado.

| Columna | Qué es |
|---|---|
| `Sensor` | P1Y / P1X / P2Y / P2X |
| `F_correcto`, `gl_den_correcto`, `p_correcto` | contra Error(a); `gl_den` = **4** siempre |
| `F_ingenuo`, `gl_den_ingenuo`, `p_ingenuo` | contra el residual agrupado; `gl_den` = 270 en el global |
| `veces_F_inflada` | `F_ingenuo / F_correcto`. Cuántas veces exageraba el método anterior |
| `significativa_correcto` | SI / no, al 5 %, con el error correcto ← **el que se reporta** |
| `significativa_ingenuo` | SI / no con el método antiguo |

Si buscas una sola cifra para el informe, sale de aquí.

### 2.4 `componentes_varianza*.csv` — de dónde viene el ruido

**12 filas = 4 sensores × 3 estratos. 5 columnas.** No habla de efectos, sino de
cómo se reparte la variabilidad **aleatoria**.

Las 3 filas de cada sensor:
1. `Parcela grande (montaje del aceite)`
2. `Subparcela (montaje del desbalanceo)`
3. `Sub-subparcela (barrido de velocidad)`

| Columna | Qué es |
|---|---|
| `MS` | cuadrado medio del estrato (el mismo de la tabla del ANOVA) |
| `Varianza` | componente de varianza estimado por el método de los momentos |
| `Porcentaje` | sobre la suma de los tres componentes |

Los componentes salen de:

```
σ²_parcela grande = (MS_Ea − MS_Eb) / (b·c)      truncado en 0
σ²_subparcela     = (MS_Eb − MS_Ec) / c          truncado en 0
σ²_sub-subparcela =  MS_Ec
```

**Dos avisos importantes:**

* Un `0` **no significa "cero"**, significa "no distinguible de cero": es la
  truncatura de una estimación que salió negativa (`MS_Ea < MS_Eb`), cosa habitual
  con 4 gl.
* Estos componentes se estiman con 4 y 12 gl: son **muy imprecisos**. No leas
  demasiado en las diferencias entre sensores.

Para qué sirve: si el estrato de parcela grande pesa poco, remontar el rotor para
cambiar de aceite añade poca variabilidad y el efecto de la viscosidad se puede
detectar pese a los 4 gl. Si pesara mucho, con 4 gl no demostrarías nada. **Los 4
gl son los mismos en cualquier caso** — eso es estructural del diseño.

### 2.5 `posthoc_viscosidad*.csv` — qué aceites difieren entre sí

**12 filas = 4 sensores × 3 parejas. 8 columnas.** El ANOVA dice "hay diferencia
entre los tres aceites"; esto dice **entre cuáles**.

Las 3 filas de cada sensor: `ISO 32 - ISO 46`, `ISO 32 - ISO 68`, `ISO 46 - ISO 68`.

| Columna | Qué es |
|---|---|
| `Comparacion` | la pareja de aceites |
| `Diferencia` | media del primero − media del segundo, en µm. **El signo importa**: positivo = el primero vibra más |
| `HSD_95` | diferencia mínima significativa de Tukey al 95 %. Es **la misma para las 3 parejas** de un sensor porque el diseño está balanceado |
| `IC_inf`, `IC_sup` | `Diferencia ± HSD_95` |
| `p_Tukey` | valor p corregido por comparaciones múltiples |
| `Significativa` | `SI` si `\|Diferencia\| > HSD_95`, es decir si el intervalo no cruza el cero |

Calculado con `MS_Error(a)` y sus 4 gl, y con `n = r·b·c` observaciones por media.
No con el residual global: eso volvería a inflar la significancia.

Lectura típica: que solo salga `SI` en `ISO 32 - ISO 68` significa que los extremos
se separan pero el ISO 46 intermedio no se distingue de ninguno — con 3 bloques es
un resultado muy común y perfectamente publicable, siempre que se diga así.

---

## 3. Los seis PNG

> **Todas las figuras ponen los 4 proximitores en unos mismos ejes**, no un panel
> por sensor. En las de barras, el color identifica el sensor (cojinete 1 azules,
> cojinete 2 cálidos; dirección Y oscura, X clara) y la trama rayada marca lo no
> significativo. En las de líneas, el color es la viscosidad y el estilo de línea
> el sensor, con los valores normalizados a la media de cada sensor para que
> sensores de magnitud muy distinta sean comparables.

### 3.1 `p5_contribucion*.png` — el mapa general
Cuadrícula 2×2 (un panel por sensor). 10 barras = las 11 fuentes menos
`Repeticion`. Altura = `Contribucion_SS_%`. Rojo = viscosidad, gris = estratos de
error, azul = el resto. Encima de cada barra, `*` si el efecto es significativo al
5 % **con su propio término de error**, `ns` si no.

Sirve para ver de un vistazo qué pesa y qué es real. Recuerda: altura ≠
significancia; por eso van las dos cosas por separado.

### 3.2 `p5_ingenuo_vs_correcto*.png` — cuánto cambia respetar el diseño
2×2 por sensor, barras pareadas: naranja = ANOVA ingenuo, verde = split-plot. Eje Y
= `−log10(p)` en escala **symlog** (lineal hasta 3, comprimida arriba), para que la
zona decisiva alrededor de p = 0.05 no quede aplastada por los p astronómicos del
modelo ingenuo. La línea discontinua marca p = 0.05.

Detalle que suele sorprender: para `Vel`, `Visc×Vel` y `Desb×Vel` las barras verdes
son **más altas** que las naranjas. No es un error: el residual agrupado del modelo
ingenuo estaba contaminado con la varianza de los estratos superiores, así que
perdía sensibilidad justo donde el diseño es más informativo.

### 3.3 `p5_efecto_viscosidad*.png` — el efecto principal
Fila de 4 paneles. Tres barras por panel (ISO 32 rojo, 46 azul, 68 verde) con la
media de amplitud de cada aceite. En el título, el valor p correcto.

Las barras de error son el IC al 95 % calculado como
`t(0.975, gl_Ea) · √(MS_Ea / n)`, con `gl_Ea = 4` (multiplicador t = 2.78) y
`n = r·b·c`. **Se ven pequeñas porque Error(a) es pequeño, no porque el test sea
generoso.**

Ojo con la variante global: esas medias promedian las 15 velocidades y quedan
dominadas por la resonancia. Las variantes por grupo no tienen ese problema.

### 3.4 `p5_interacciones*.png` — cómo depende el efecto de las demás condiciones
Cuadrícula 2×4:
* **Fila de arriba** — viscosidad × velocidad. Cada punto promedia
  `n_rep × n_desbalanceos` valores (9 en el global).
* **Fila de abajo** — viscosidad × desbalanceo. Cada punto promedia
  `n_rep × n_velocidades` valores (45 en el global, 9 o 12 en un grupo).

El título indica esos recuentos. Líneas paralelas = no hay interacción; líneas que
se cruzan o cambian de separación = sí la hay.

> En la variante global, la fila de abajo promedia sobre las 15 velocidades y queda
> **dominada por la resonancia** (en los datos de prueba, 3 de 15 velocidades
> aportan el 47 % del promedio). Para leer la interacción viscosidad × desbalanceo
> conviene usar las variantes por grupo.

### 3.5 `p5_estratos_varianza*.png` — el reparto del ruido
Un solo panel: una barra apilada por sensor con los tres estratos (morado = montaje
del aceite, azul = montaje del desbalanceo, gris = barrido). Es la versión gráfica
de `componentes_varianza.csv`, con sus mismos avisos.

Cuanto mayor es el morado, más limitada está la información sobre viscosidad por el
**número de montajes** y menos por el número de medidas.

### 3.6 `p5_viscosidad_por_grupo.png` — **archivo único, sin variantes**
El único que compara los cuatro tramos entre sí. Cuadrícula 2×4:
* **Arriba** — media de cada aceite en cada tramo (eje X = los 4 grupos).
* **Abajo** — evidencia del efecto en cada tramo, `−log10(p)`; verde si es
  significativo al 5 %, gris si no.

Es la lectura práctica de una interacción `Visc × Vel` significativa: dice **en qué
régimen de velocidad aparece el efecto de la viscosidad**. Si buscas una sola figura
para la conclusión, es esta.

---

## 4. Qué archivo responde a cada pregunta

| Pregunta | Archivo |
|---|---|
| ¿La viscosidad afecta a la vibración? | `comparacion_viscosidad.csv`, columnas `p_correcto` y `significativa_correcto` |
| ¿Cuánto exageraba el método anterior? | `comparacion_viscosidad.csv`, columna `veces_F_inflada` |
| ¿Entre qué aceites está la diferencia? | `posthoc_viscosidad.csv` |
| ¿De qué tamaño es el efecto? | `posthoc_viscosidad.csv` (`Diferencia`, en µm) y `anova_split_plot.csv` (`omega2_%`) |
| ¿En qué régimen de velocidad aparece? | `p5_viscosidad_por_grupo.png` y los `comparacion_viscosidad_Grupo_*.csv` |
| ¿Depende del nivel de desbalanceo? | fila `Viscosidad x Desbalanceo` de `anova_split_plot.csv` (contraste formal) y `p5_viscosidad_por_desbalanceo.png` (lectura descriptiva) |
| ¿Es repetible el banco? | `componentes_varianza.csv` / `p5_estratos_varianza.png` |
| ¿Qué domina la variabilidad? | `p5_contribucion.png` |
| ¿Por qué no puedo usar el ANOVA de antes? | `p5_ingenuo_vs_correcto.png` |

**Para un informe**, con esto basta: `comparacion_viscosidad.csv` +
`posthoc_viscosidad.csv` + `p5_viscosidad_por_grupo.png`, y
`anova_split_plot.csv` como tabla completa en un anexo.
