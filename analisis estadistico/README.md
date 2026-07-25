# Análisis estadístico de la respuesta dinámica del rotor

Pipeline completo desde los `.xlsx` de proximitores hasta el análisis de
significancia de la viscosidad, respetando el diseño de parcelas subdivididas
del ensayo.

| Script | Paso | Qué hace |
|---|---|---|
| `p1_extraer_fasores.py` | 1 | Extrae el fasor 1X (amplitud + fase) de cada ensayo y velocidad, analizando la estabilización temporal |
| `p2_graficar_fasores.py` | 2 | 12 figuras 2×2 de amplitud y fase frente a la velocidad |
| `p3_compensar_runout.py` | 3 | Resta vectorial del fasor de slow roll |
| `p4_graficar_compensados.py` | 4 | Las mismas 12 figuras con los datos ya compensados |
| `p5_anova_split_plot.py` | 5 | ANOVA de parcelas subdivididas + comparación con el análisis ingenuo |
| `ejecutar_todo.py` | — | Ejecuta los 5 pasos en orden |
| `config.py` / `comun.py` | — | Constantes y utilidades compartidas |
| `_generar_datos_prueba.py` | — | (opcional) genera `.xlsx` sintéticos para probar el pipeline |

---

## Ejecución

La carpeta tiene un espacio en el nombre, así que en Windows hay que **entrecomillar
la ruta**:

```bat
REM todo el proceso de una vez
python "C:\Users\Owner\Documents\git\psd\dsp\analisis estadistico\ejecutar_todo.py" ^
       --entrada "C:\Users\Owner\Documents\BD\All data" ^
       --salida  "C:\Users\Owner\Documents\BD\resultados_analisis" ^
       --desenvolver --grupos-velocidad
```

Paso a paso:

```bat
set A=C:\Users\Owner\Documents\git\psd\dsp\analisis estadistico
set R=C:\Users\Owner\Documents\BD\resultados_analisis

python "%A%\p1_extraer_fasores.py"      --entrada "C:\Users\Owner\Documents\BD\All data" --salida "%R%"
python "%A%\p2_graficar_fasores.py"     --salida "%R%" --desenvolver
python "%A%\p3_compensar_runout.py"     --salida "%R%" --modo viscosidad
python "%A%\p4_graficar_compensados.py" --salida "%R%" --desenvolver
python "%A%\p5_anova_split_plot.py"     --salida "%R%" --grupos-velocidad
```

Los valores por defecto de `--entrada` y `--salida` se editan en `config.py`.
Dependencias: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`.

Para probar el pipeline sin datos reales:

```bat
python "%A%\_generar_datos_prueba.py" --salida "%R%\xlsx_sinteticos"
python "%A%\ejecutar_todo.py" --entrada "%R%\xlsx_sinteticos" --salida "%R%\prueba"
```

---

## Paso 1 — Extracción de fasores

Procesa **solo** los archivos que cumplen la nomenclatura

```
rep<R>_<viscosidad>_<desbalanceo>_p_<fecha>.xlsx
```

El resto de archivos de la carpeta se ignoran (se informa cuántos). El sufijo
`_p_` selecciona proximitores; con `--tipo ac` se procesarían los acelerómetros.
Cada hoja cuyo nombre sea un número se interpreta como una velocidad en rpm; las
hojas auxiliares (`prox`, `Hoja17`, …) se descartan solas.

Las columnas se localizan **por contenido** de la cabecera (`Mach1.P1.Y` +
`Displacement`/`Phase`), no por posición fija, de modo que un cambio de orden en
la exportación no rompe nada.

### Análisis de estabilización

Cada medición dura ~3 min. Antes de promediar se busca **el tramo final más largo
que ya esté estabilizado**, es decir, en el que el fasor ya no deriva:

| Criterio | Qué mide | Tolerancia (`config.py`) |
|---|---|---|
| `deriva_amp` | \|mediana(primer cuarto) − mediana(última mitad)\| / mediana | `TOL_DERIVA_AMP` = 0.05 |
| `deriva_fase` | ángulo entre las medias circulares de ambos tramos | `TOL_DERIVA_FASE` = 5° |
| `cv_amp` | dispersión robusta de la amplitud / mediana | `TOL_CV_AMP` = 0.15 |
| `disp_fase` | desviación **circular** de la fase | `TOL_DISP_FASE` = 12° |

Se compara el *primer cuarto* contra la *última mitad* (y no mitad contra mitad)
porque un transitorio corto al principio queda diluido en la mediana de media
ventana y pasaría desapercibido.

Sobre el tramo aceptado se hace el **promedio vectorial** del fasor. El detector
se comportó así en pruebas con transitorios de magnitud conocida:

| Señal | Recorta |
|---|---|
| sin transitorio | nada |
| escalón 25 % que decae con τ = 20 s | 11 s |
| el mismo con τ = 40 s | 41 s |
| rampa del 20 % que nunca se asienta | 106 s |
| rampa del 40 % que nunca se asienta | nada aceptable → **marcado como no estable** |

**Salidas:** `p1_fasores.txt` (tabla principal) y
`p1_diagnostico_estabilizacion.txt` (una fila por ensayo × rpm × sensor, con
todas las métricas y la marca `estable`). Conviene revisar las filas con
`estable = 0`: suelen corresponder a puntos donde la amplitud colapsa y la fase
pierde sentido físico.

### Verificación del diseño

Se comprueba que estén las **27** combinaciones `rep × visc × desb` y se avisa de
las que falten, las que sobren y los duplicados. También se avisa si falta alguna
de las 15 velocidades o si hay valores que no se pudieron calcular.

---

## Pasos 2 y 4 — Figuras de fasores

12 figuras por cada juego de datos, con este reparto 2×2:

|  | columna 1 (sensor **Y**) | columna 2 (sensor **X**) |
|---|---|---|
| **fila 1** | amplitud 1X [µm] | amplitud 1X [µm] |
| **fila 2** | fase 1X [°] | fase 1X [°] |

* **6 figuras** `desbalanceo<d>_<coj>.png` — fijan desbalanceo y cojinete y
  superponen las 3 viscosidades × 3 repeticiones.
* **6 figuras** `viscosidad<v>_<coj>.png` — fijan viscosidad y cojinete y
  superponen los 3 desbalanceos × 3 repeticiones.

Convenios, constantes en **todas** las figuras:

| Codificación | Factor |
|---|---|
| **Color** | viscosidad: ISO 32 rojo · ISO 46 azul · ISO 68 verde |
| **Saturación** del color | desbalanceo: 1 el más atenuado, 3 el pleno |
| **Estilo de línea** | repetición: 1 continua, 2 discontinua, 3 punteada |

Solo línea, sin marcadores. El **eje X** reparte las 15 velocidades en 15
posiciones equiespaciadas (no a escala numérica de rpm). Los **límites del eje Y**
son comunes a las 12 figuras y salen del máximo y el mínimo globales de toda la
tabla, uno para amplitud y otro para fase.

> **`--desenvolver`** — muy recomendable. Sin él, la fila de fase muestra saltos
> verticales artificiales de 0 a 360 que ocultan la tendencia real (el desfase
> progresivo al pasar las críticas). Con él, cada curva se dibuja de forma
> continua. No está activado por defecto para que la figura muestre por defecto
> exactamente los valores de la tabla.

El paso 4 admite además `--escala-comun`, que amplía los límites para englobar
también los datos sin compensar y así poder comparar directamente las figuras del
paso 2 con las del paso 4.

---

## Paso 3 — Compensación de runout

```
z_compensado(rpm) = z_medido(rpm) − z_slow_roll
```

El vector de slow roll es el **promedio vectorial** de los fasores a 600 rpm.
`--modo` elige el agrupamiento: `viscosidad` (por defecto), `global` o
`rep_viscosidad`.

### El problema del ángulo, y por qué aquí no existe

Las fases vienen envueltas en [0, 360). Dos medidas casi idénticas pueden salir
como 359° y 1°, y su media aritmética daría 180°: exactamente lo contrario del
valor correcto. **El promedio aritmético de ángulos no es válido.**

La solución no es corregir el promedio a posteriori, sino no calcularlo nunca así:
se pasa cada medida a su fasor complejo `z = A·e^{iφ}`, se promedian parte real e
imaginaria, y del resultado se recuperan amplitud y fase. Como el ángulo nunca se
manipula como escalar, el envolvimiento deja de existir. Comprobación con nueve
medidas repartidas alrededor de 0°:

```
fases: [358.0, 359.5, 1.0, 2.5, 0.3, 357.8, 3.1, 359.0, 1.7]
  media aritmética (incorrecta) : 160.32°
  media vectorial  (correcta)   :   0.32°
  dispersión circular: 1.80°   (la desviación típica ingenua daría 177.3°)
```

Para las series en las que sí hace falta un ángulo **continuo** (dibujar curvas,
medir derivas) está `comun.desenvolver_fase`, que suma a cada muestra el múltiplo
de 360 que la deja más cerca de la anterior.

El script informa, grupo a grupo, de la discrepancia entre la media ingenua y la
vectorial, y de la concentración circular `R` (con `R < 0.9` la fase promedio es
poco representativa). Genera además `p3_slow_roll.png` con los diagramas polares
de los fasores a 600 rpm.

### ¿Tiene sentido que el slow roll dependa de la viscosidad?

Preguntas si tiene sentido que el fasor de slow roll salga distinto según el
aceite. Respuesta corta: **el runout propiamente dicho no puede depender del
aceite, pero lo que se mide a 600 rpm no es solo runout.**

* El runout **mecánico** (geometría y curvatura del eje) y el **eléctrico**
  (variaciones del material que ve la sonda de corrientes de Foucault) son
  propiedades del eje y de la sonda. El aceite no los toca. Si midieras runout
  puro, tendría que salir idéntico con los tres aceites.
* Lo que se registra a 600 rpm es runout **más** la respuesta síncrona que ya
  existe a esa velocidad. Con la primera crítica en torno a 3500 rpm,
  *r* = 600/3500 ≈ 0.17 y el factor de amplificación `r²/(1−r²)` ≈ 0.03: pequeño,
  pero no nulo.
* Además, la posición de equilibrio del muñón dentro del cojinete depende del
  número de Sommerfeld (viscosidad × velocidad / carga). Cambiar de aceite mueve
  la excentricidad y el ángulo de actitud, y con ellos los coeficientes de rigidez
  y amortiguamiento del cojinete — que a su vez modifican esa pequeña respuesta
  síncrona.

Es decir: una dependencia leve con la viscosidad a 600 rpm es **físicamente
plausible, pero no es runout: es respuesta dinámica temprana**. Y ahí está el
riesgo — si restas un slow roll distinto por viscosidad, estarás restando parte
del efecto que quieres medir.

**Qué hacer:**

1. El script cuantifica el caso con `comparar_viscosidades`: compara la separación
   *entre* las medias de las tres viscosidades con la dispersión *dentro* de cada
   viscosidad (repeticiones y desbalanceos). Mira la columna
   `razon_entre_dentro`:
   * **< 2** → la diferencia no supera al ruido de medida: no está respaldada.
     Usa `--modo global`.
   * **≫ 2** → la diferencia es medible y hay que decidir de dónde viene.
2. Si al cambiar de aceite **se desmonta y se vuelve a montar el rotor**, el
   runout mecánico sí puede cambiar de verdad — pero entonces cambia por
   *montaje*, no por *viscosidad*. En ese caso lo correcto es `--modo
   rep_viscosidad`, que asigna un slow roll a cada combinación repetición ×
   viscosidad.
3. Lo ideal sería medir el slow roll a una velocidad claramente por debajo de la
   zona de respuesta, del orden del 10–15 % de la primera crítica (≈ 350–500 rpm
   aquí). 600 rpm es ≈ 17 %: está en el límite. Con `--rpm-slow-roll` puedes usar
   otra velocidad si en algún momento registras una más baja.

---

## Paso 5 — Análisis estadístico

### Por qué el ANOVA factorial corriente no vale

El ensayo **no** es un experimento completamente aleatorizado. Es un
**split-split-plot en bloques**:

| Nivel | Factor | Niveles | Por qué |
|---|---|---|---|
| bloque | repetición | 3 | |
| parcela grande | viscosidad | 3 | cambiar el aceite es costoso: se hace pocas veces |
| subparcela | desbalanceo | 3 | se cambia la masa sin tocar el aceite |
| sub-subparcela | velocidad | 15 | barrido dentro de cada montaje |

Cada nivel de aleatorización tiene **su propio término de error**:

| Efecto | Se contrasta contra | gl del denominador |
|---|---|---|
| Viscosidad | Error(a) = Rep × Visc | **4** |
| Desbalanceo, Visc×Desb | Error(b) = Rep × Desb \| Visc | 12 |
| Velocidad y sus interacciones | Error(c) | 252 |

`f3_s1_v1.py` y `f3_s1_v2.py` contrastan **todo** contra el residual global
(270 gl). Eso trata la viscosidad como si el rotor se hubiera montado 405 veces
de forma independiente, cuando en realidad **solo hay 9 unidades de parcela
grande** (3 bloques × 3 aceites). El denominador queda demasiado pequeño y la F
se dispara.

### Cuánto importa, medido

Simulación bajo la hipótesis nula (la viscosidad **no** tiene ningún efecto), con
variabilidad de montaje realista, 600 réplicas por escenario:

| σ de parcela grande | Rechazos del test **correcto** | Rechazos del test **ingenuo** |
|---|---|---|
| 0.0 | 5.7 % | 48.3 % |
| 0.5 | 5.2 % | 78.0 % |
| 1.0 | 4.5 % | 87.7 % |
| 2.0 | 5.7 % | 91.8 % |

*(nominal esperado: 5 %)*

El test de parcelas subdivididas mantiene el 5 % nominal. El ANOVA ingenuo declara
significativa la viscosidad hasta en **9 de cada 10 casos en los que no hay
efecto alguno**. No es un matiz: es la diferencia entre un resultado y un
artefacto.

Sobre datos sintéticos con efecto real, el ANOVA ingenuo multiplicó la F de la
viscosidad por un factor de **14 a 31**, llevando el valor p de ~0.007 (real) a
~10⁻⁹⁶ (irreal).

> La descomposición de sumas de cuadrados implementada se validó contra
> `statsmodels` (`anova_lm`, tipo II): coinciden hasta ~1e-14 en todos los
> efectos, y los tres estratos de error suman exactamente el residual del modelo
> factorial. Lo que cambia no son las sumas de cuadrados, sino **contra qué se
> divide cada una**.

Ojo a un efecto secundario interesante: para los factores de sub-subparcela
(velocidad y sus interacciones) el análisis correcto da p **más** pequeños que el
ingenuo, porque el residual agrupado estaba contaminado con la varianza de los
estratos superiores.

### Salidas

En `<salida>/p5_estadistica`:

| Archivo | Contenido |
|---|---|
| `anova_split_plot.csv` | tabla completa por sensor: SS, gl, MS, término de error, F, p, contribución, ω² |
| `anova_ingenuo.csv` | el ANOVA factorial corriente, para comparar |
| `comparacion_viscosidad.csv` | F y p correctos frente a ingenuos, y el factor de inflación |
| `posthoc_viscosidad.csv` | Tukey entre aceites usando Error(a) y sus 4 gl |
| `componentes_varianza.csv` | varianza atribuible a cada estrato |
| `p5_contribucion.png` | descomposición de la variabilidad, con marca de significancia real |
| `p5_ingenuo_vs_correcto.png` | la evidencia real frente a la inflada, efecto por efecto |
| `p5_efecto_viscosidad.png` | media por aceite con IC 95 % basado en Error(a) |
| `p5_interacciones.png` | viscosidad × velocidad y viscosidad × desbalanceo |
| `p5_estratos_varianza.png` | de dónde viene la variabilidad del ensayo |

### Análisis por tramos de velocidad

Con `--grupos-velocidad` se repite **todo** el análisis dentro de cada tramo de rpm
(bajas / pre-crítica / 1ª crítica / altas), como hacía `f3_s1_v2.py` pero con los
términos de error correctos. Genera los CSV y **también las 5 figuras** de cada
tramo, con el sufijo del grupo:

```
p5_interacciones_Grupo_3_1a_critica.png
p5_efecto_viscosidad_Grupo_3_1a_critica.png
p5_contribucion_Grupo_3_1a_critica.png
...
```

Más una figura adicional, `p5_viscosidad_por_grupo.png`, que compara los cuatro
tramos entre sí: arriba la media por aceite en cada tramo, abajo la evidencia del
efecto (verde = significativo). Es la lectura que da sentido a una interacción
Visc × Vel significativa: el efecto de la viscosidad **no es el mismo a todas las
velocidades**, y esta figura dice en qué régimen aparece.

Los tramos se editan en `GRUPOS_VELOCIDAD`, al principio del script.

> Hacer el análisis por tramos corrige de paso un sesgo de las figuras globales:
> promediar la amplitud sobre las 15 velocidades queda dominado por la resonancia
> (en los datos de prueba, 3 de las 15 velocidades aportan casi la mitad del
> promedio). Dentro de un tramo, la media sí representa a ese régimen.
> Por eso el título de `p5_interacciones*.png` indica cuántos valores promedia
> cada punto en cada fila.

Por defecto analiza los datos **compensados** (`p3_fasores_compensados.txt`); con
`--entrada` se puede apuntar a `p1_fasores.txt` para analizar los sin compensar.

### Dos advertencias

* **La fase es una variable circular.** `--respuesta fase` existe, pero un ANOVA
  sobre grados solo es defendible si ninguna condición cruza el 0/360 (359° y 1°
  distan 2°, no 358°). El script lo avisa. Para un tratamiento riguroso haría
  falta estadística circular; puedo añadirla si la necesitas.
* **Con 4 gl en el denominador, la potencia para detectar el efecto de la
  viscosidad es baja.** Eso no es un defecto del análisis, es lo que el diseño
  permite: la información sobre la viscosidad la aportan 9 montajes, no 405
  medidas. Si el efecto resulta no significativo, la conclusión honesta es "este
  ensayo no tiene resolución suficiente para demostrarlo", no "no existe". Para
  ganar potencia habría que aumentar el número de **repeticiones completas**
  (bloques), que es lo que alimenta Error(a) — no más velocidades ni más
  desbalanceos.
