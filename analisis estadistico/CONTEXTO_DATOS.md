# Contexto del conjunto de datos

Documento para pegar en otra conversación y dar el contexto completo del
experimento sin tener que explicarlo desde cero.

---

## 1. Qué se midió

Ensayo de vibración sobre un **rotor montado en cojinetes hidrodinámicos
lubricados**. El objetivo es determinar si la **viscosidad del aceite** influye
en la respuesta vibratoria del rotor, y en qué zonas de velocidad lo hace.

La respuesta se mide con **proximitores** (sondas de desplazamiento sin
contacto) y se resume en el **fasor 1X**: la componente síncrona con el giro,
caracterizada por una **amplitud** y una **fase**.

---

## 2. Estructura factorial

Cuatro factores completamente cruzados y **balanceados**:

| Factor | Niveles | Valores |
|---|---|---|
| Repetición (bloque) | 3 | 1, 2, 3 |
| Viscosidad del aceite | 3 | ISO VG 32, 46, 68 |
| Nivel de desbalanceo | 3 | 1, 2, 3 (masa creciente) |
| Velocidad de giro | 15 | 600, 1300, 2000, 2500, 2600, 2700, 2800, 3400, 3500, 3600, 3700, 4000, 4500, 5000, 5500 rpm |

**3 × 3 × 3 = 27 ensayos**, cada uno barriendo las 15 velocidades →
**3 × 3 × 3 × 15 = 405 observaciones** por sensor.

Las velocidades no están uniformemente espaciadas: se concentran alrededor de
la **primera velocidad crítica** (~3400–3700 rpm). Se agrupan en cuatro tramos:

| Tramo | rpm | Significado |
|---|---|---|
| G1 | 600, 1300, 2000 | baja velocidad |
| G2 | 2500–2800 | por debajo de la 1.ª crítica |
| G3 | 3400–3700 | cerca de la 1.ª crítica |
| G4 | 4000–5500 | por encima de la 1.ª crítica |

---

## 3. Sensores

Cuatro proximitores: **dos cojinetes × dos direcciones ortogonales**.

| Nombre | Cojinete | Dirección |
|---|---|---|
| `P1Y` | 1 | vertical |
| `P1X` | 1 | horizontal |
| `P2Y` | 2 | vertical |
| `P2X` | 2 | horizontal |

Las amplitudes de los cuatro no son comparables entre sí sin más: difieren
por un factor de ~6 según la posición y la dirección.

---

## 4. Archivos de entrada

### Nomenclatura

```
rep<R>_<viscosidad>_<desbalanceo>_<p|ac>_<fecha>.xlsx

ejemplo:  rep2_46_3_p_0424.xlsx
          repetición 2, ISO VG 46, desbalanceo 3, proximitores, 24 de abril
```

`_p_` = proximitores (los que se analizan). `_ac_` = acelerómetros, presentes
pero **no procesados**. Cualquier archivo que no cumpla el patrón se ignora.

### Estructura interna de cada `.xlsx`

- **Una hoja por velocidad**, nombrada con el valor en rpm (`600`, `1300`, …).
  Las hojas auxiliares (`prox`, `Hoja17`, …) se descartan porque su nombre no
  es numérico.
- **Tres filas de cabecera**; los datos empiezan en la cuarta fila.
  - Fila 0 = canal, con la forma
    `NVD: Prof >Bode> 1X [Pk-Pk]> Mach1.P1.Y`
  - Fila 1 = magnitud: `Time`, `Displacement` o `Phase`
- Por cada sensor hay **dos columnas**: amplitud 1X pico-pico en µm y fase en
  grados, más una columna común de tiempo.
- Cada hoja es un **registro temporal de ~3 minutos** a velocidad constante:
  no es un barrido, son muchas muestras del mismo fasor mientras se estabiliza.

Las columnas se localizan **por contenido de la cabecera**, no por posición
fija, así que un cambio de orden en la exportación no rompe nada.

---

## 5. De registro temporal a un fasor por celda

Cada hoja (~3 min de muestras) se reduce a **un solo fasor 1X** por sensor:

1. Se detecta el tramo estable del registro (se descarta el transitorio
   inicial: arranque, asentamiento térmico o hidrodinámico).
2. Sobre ese tramo se hace la **media vectorial** de los fasores
   `z = A·e^{iφ}` — nunca el promedio por separado de amplitud y ángulo,
   porque la fase está envuelta en [0, 360) y promediar 359° con 1° daría 180°
   en vez de 0°.

Resultado: **405 filas × 4 sensores**, cada una con amplitud y fase.

---

## 6. La tabla de trabajo

Texto separado por tabuladores, 405 filas × 12 columnas:

```
Repetition  Viscosity  Unbalance  Speed  P1Y_amp  P1Y_phase  P1X_amp  P1X_phase  P2Y_amp  P2Y_phase  P2X_amp  P2X_phase
1           32         1          600    0.4279   178.8154   0.1272   152.6457   0.0434   144.6957   0.9317   193.9356
```

- Cuatro columnas de factor (enteros) + dos columnas por sensor.
- `*_amp` en **µm pico-pico**, `*_phase` en **grados [0, 360)**.

Existen dos versiones:

| Archivo | Contenido |
|---|---|
| `p1_phasors.txt` | fasores crudos |
| `p3_phasors_compensated.txt` | fasores con el **runout compensado** |

**Compensación de runout:** a 600 rpm el rotor gira muy por debajo de la
primera crítica, así que lo que se mide ahí es casi todo error mecánico y
eléctrico del blanco (excentricidad, marcas en el eje), no respuesta dinámica.
Ese fasor de "slow roll" se **resta vectorialmente** de todas las velocidades.
Es la tabla compensada la que se analiza por defecto.

---

## 7. Estructura estadística — esto es lo crítico

**El experimento NO está completamente aleatorizado.** Es un
**split-split-plot en bloques** (parcelas subdivididas), porque los factores
son cada vez más baratos de cambiar:

| Nivel | Factor | Por qué |
|---|---|---|
| Bloque | Repetición | réplica completa del experimento |
| Parcela grande | **Viscosidad** | cambiar el aceite es caro: hay que drenar, limpiar y rellenar |
| Subparcela | Desbalanceo | se cambia la masa dentro de un mismo aceite |
| Sub-subparcela | Velocidad | barrido dentro de un mismo montaje |

**Cada nivel de aleatorización tiene su propio término de error.** Contrastar
la viscosidad contra el residual global —lo que hace un ANOVA factorial
corriente— equivale a suponer que el rotor se montó 405 veces de forma
independiente, cuando solo hay **9 unidades de parcela grande** (3 bloques ×
3 aceites).

| Efecto | Se contrasta contra | gl del denominador |
|---|---|---|
| Viscosidad | Error(a) = Rep × Visc | **4** |
| Desbalanceo | Error(b) = Rep × Desb \| Visc | 12 |
| Visc × Desb | Error(b) | 12 |
| Velocidad | Error(c) = residual | 252 |
| Visc × Vel | Error(c) | 252 |
| Desb × Vel | Error(c) | 252 |
| Visc × Desb × Vel | Error(c) | 252 |

Los grados de libertad suman 404 = 405 − 1.

**Consecuencia práctica:** la viscosidad se juzga con **4 gl**, no con 252.
Ignorarlo infla el estadístico F en uno o dos órdenes de magnitud y produce
valores p irreales. En una simulación bajo hipótesis nula el contraste
correcto rechaza el 4.5–5.7 % de las veces (lo esperado al 5 %), mientras que
el ANOVA ingenuo rechaza entre el 48 % y el 92 %.

Con **4 gl la potencia es baja**: un efecto no significativo significa "no
resuelto con 9 montajes", no "no existe".

---

## 8. Limitaciones a tener presentes

- **Una sola observación por celda.** No hay réplicas dentro de una
  combinación (rep, visc, desb, vel); la réplica es la repetición completa.
- **La fase es una variable circular.** Un ANOVA ordinario sobre grados no es
  válido en general (359° y 1° distan 2°, no 358°). Solo es defendible si
  ninguna condición cruza el 0/360.
- **La compensación de runout usa 600 rpm**, que es ~17 % de la primera
  crítica. Ahí ya existe algo de respuesta síncrona real, y la posición de
  equilibrio del muñón depende del número de Sommerfeld —es decir, del
  aceite—. Restar un slow roll que depende de la viscosidad puede eliminar
  parte del efecto que se estudia. Existen modos alternativos de agrupación
  (global, o por repetición × viscosidad).
- **Sin corrección por multiplicidad.** El análisis global, los 4 tramos de
  velocidad y los 3 niveles de desbalanceo dan 8 contrastes de viscosidad por
  sensor. Los subanálisis son exploratorios; la conclusión principal debe
  anclarse en el análisis global.
- **Al fijar un nivel de desbalanceo** el diseño se reduce a un split-plot de
  dos estratos (bloque → viscosidad → velocidad). La viscosidad sigue
  contrastándose con 4 gl: separar por desbalanceo **no** aporta información
  sobre el aceite.

---

## 9. Constantes editables

Todo lo de las secciones 2 y 3 (niveles, velocidades, tramos, nombres de
sensores) está en `config.py` y puede cambiarse sin tocar el análisis. Lo de
las secciones 4 a 7 es estructural.
