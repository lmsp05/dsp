# Etiquetado — viscosidad según temperatura (`mu_temperatura.py`)

Calcula la viscosidad real del lubricante a la temperatura medida en cada ensayo
con la **ecuación de Walther (ASTM D341)** y añade a un **duplicado** del archivo
de entrada las columnas:

* `mu1` — viscosidad a la temperatura del cojinete 1 (`T1`)
* `mu2` — viscosidad a la temperatura del cojinete 2 (`T2`)
* `mu_prom` — promedio de `mu1` y `mu2`

La viscosidad es **cinemática** en cSt (mm²/s). Filas con `T` faltante/`NaN`
(o `iso` desconocido) quedan como `NaN`. Se preservan todas las columnas
originales (incluida la columna `orden` duplicada) y las filas cortas se
normalizan a `NaN`.

```bash
python mu_temperatura.py --entrada ensayos.txt
# salida por defecto: ensayos_mu.txt
python mu_temperatura.py --entrada ensayos.txt --salida ensayos_mu.txt --config aceites.json
```

La entrada se lee separada por tabuladores (`--sep tab`, por defecto); usa
`--sep auto` si está separada por espacios.

## La fórmula (ASTM D341 / Walther)

```
log10(log10(nu + 0.7)) = A − B · log10(T_K)
```

Con **dos** puntos (temperatura, viscosidad) de cada aceite se despejan `A` y `B`
y con ellos se calcula `nu` a cualquier temperatura. Es la relación
viscosidad–temperatura estándar para aceites minerales.

## Cómo indicar cada aceite (tu duda)

Cada grado ISO se define por **dos viscosidades de referencia**, en el diccionario
`ACEITES` del script (o en un JSON con `--config`, ver `aceites.json`):

```json
{"32": {"nu40": 32.0, "nu100": 5.4},
 "46": {"nu40": 46.0, "nu100": 6.8},
 "68": {"nu40": 68.0, "nu100": 8.7}}
```

* `nu40` = viscosidad a 40 °C = **el número del grado ISO VG** (32/46/68), ya lo sabes.
* `nu100` = viscosidad a 100 °C, que sale de la **ficha técnica** del aceite.

**Mi recomendación:** en vez de dar el "índice de viscosidad" (VI), da `nu40` y
`nu100`. Motivos:

1. El VI **se calcula a partir de** `nu40` y `nu100` (norma ASTM D2270): son el
   dato fundamental, el VI es un derivado. Dar los dos números es más directo.
2. `nu100` aparece **siempre** en la ficha del lubricante, junto a `nu40` y al VI.
3. Convertir VI → `nu100` requiere la tabla de la ASTM D2270 (paso extra y menos
   transparente). Con `nu40` (que ya lo fija el grado) más `nu100`, el
   comportamiento con la temperatura queda **exactamente** determinado.

Es decir: el único número que te falta por aceite es `nu100`. Búscalo en la ficha
de los tres aceites y ponlo en `aceites.json`. Los valores por defecto
(`5.4 / 6.8 / 8.7`) son típicos de aceite mineral con VI ≈ 100 — cámbialos por
los de tus aceites reales.

> Si de verdad solo tienes el VI y no `nu100`, dímelo y añado el cálculo
> `VI → nu100` con la tabla de la ASTM D2270.

> Nota: `mu` aquí es viscosidad **cinemática** (cSt). Si necesitas la **dinámica**
> (Pa·s) para número de Sommerfeld, etc., hay que multiplicar por la densidad
> ρ(T); puedo añadirlo si lo necesitas.

---

## Visualización de viscosidad (`graficar_viscosidad.py`)

Crea scatter plots que muestran la viscosidad promedio (`mu_prom`) en función
de la velocidad de rotación:

```bash
python graficar_viscosidad.py --entrada ensayos_mu.txt --salida ./graficos
# opciones: --figsize ancho,alto  --dpi 150 (por defecto 12,8 y 150 dpi)
```

Acepta los mismos separadores que `mu_temperatura.py` (`--sep tab` / `--sep auto`).

### Gráficos generados

1. **`viscosidad_scatter.png`** — gráfico general
   * **Color** = tipo de aceite (ISO 32/46/68)
   * **Forma de marcador** = nivel de desbalanceo (`dsb`)

2. **`viscosidad_scatter_dsb{N}.png`** — gráficos por desbalanceo (uno para cada valor de `dsb`)
   * **Color** = tipo de aceite (ISO 32/46/68)
   * **Forma de marcador** = repetición (`rep`: círculo / cuadrado / triángulo / etc.)
   * Permite comparar las diferentes repeticiones dentro de cada condición de desbalanceo
