# Statistical analysis of the rotor dynamic response

Full pipeline from the proximity-probe `.xlsx` files to the significance
analysis of oil viscosity, respecting the split-plot structure of the
experiment.

| Script | Step | What it does |
|---|---|---|
| `p1_extraer_fasores.py` | 1 | Extracts the 1X phasor (amplitude + phase) of every run and speed, after a stabilisation analysis |
| `p2_graficar_fasores.py` | 2 | Twelve 2x2 figures of amplitude and phase against speed |
| `p3_compensar_runout.py` | 3 | Vector subtraction of the slow-roll phasor |
| `p4_graficar_compensados.py` | 4 | The same twelve figures on the compensated data |
| `p5_anova_split_plot.py` | 5 | Split-plot ANOVA + comparison against the naive analysis |
| `p6_graficar_pvalores.py` | 6 | p-value figures with the significance threshold |
| `ejecutar_todo.py` | — | Runs all six steps in order |
| `config.py` / `comun.py` | — | Shared constants and helpers |
| `_generar_datos_prueba.py` | — | (optional) synthetic `.xlsx` files to exercise the pipeline |

> **[`GUIA_RESULTADOS_P5.md`](GUIA_RESULTADOS_P5.md)** — reference for every file
> step 5 produces: each CSV column by column and row by row, what each figure
> shows, and which file answers which question.

---

## Running it

The folder name contains a space, so on Windows the path must be **quoted**:

```bat
REM the whole pipeline at once
python "C:\Users\Owner\Documents\git\psd\dsp\analisis estadistico\ejecutar_todo.py" ^
       --entrada "C:\Users\Owner\Documents\BD\All data" ^
       --salida  "D:\wherever\you\want\results" ^
       --desenvolver --grupos-velocidad --grupos-desbalanceo
```

Step by step:

```bat
set A=C:\Users\Owner\Documents\git\psd\dsp\analisis estadistico
set R=D:\wherever\you\want\results

python "%A%\p1_extraer_fasores.py"      --entrada "C:\Users\Owner\Documents\BD\All data" --salida "%R%"
python "%A%\p2_graficar_fasores.py"     --salida "%R%" --desenvolver
python "%A%\p3_compensar_runout.py"     --salida "%R%" --modo viscosidad
python "%A%\p4_graficar_compensados.py" --salida "%R%" --desenvolver
python "%A%\p5_anova_split_plot.py"     --salida "%R%" --grupos-velocidad --grupos-desbalanceo
python "%A%\p6_graficar_pvalores.py"    --salida "%R%"
```

`--salida` may be any local folder; it does not have to sit inside the repo and
is created if missing. Defaults for `--entrada` and `--salida` live in
`config.py`. Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`.

To exercise the pipeline without field data:

```bat
python "%A%\_generar_datos_prueba.py" --salida "%R%\synthetic_xlsx"
python "%A%\ejecutar_todo.py" --entrada "%R%\synthetic_xlsx" --salida "%R%\test"
```

---

## Figures for a two-column paper

A figure is legible in print when its type is the right size **at the size it
will be printed**. Drawing a 16 in wide figure and shrinking it into a 3.5 in
column scales the text down by 4.6x, which is exactly why large on-screen
figures come out unreadable on paper.

`--formato` therefore fixes the FINAL width and sizes the type for it, so the
figure is placed at 100 % with no scaling:

| `--formato` | Width | Use |
|---|---|---|
| `screen` (default) | 16 in, 200 dpi | reviewing on a monitor |
| `column` | 3.5 in / 88.9 mm, 600 dpi | one column of a two-column letter page |
| `double` | 7.16 in / 181.6 mm, 600 dpi | spanning both columns |

`--paneles-por-sensor` switches the bar figures from *four probes on shared
axes* to a **2x2 grid with one panel per probe**. Which one reads better depends
on the figure:

* **Shared axes** is more compact and makes probe-to-probe differences
  immediate. Good with few, short category labels.
* **One panel per probe** keeps every label legible when there are many
  categories. This is usually the right choice at `--formato column`.

Both switches apply to steps 5 and 6:

```bat
python "%A%\p5_anova_split_plot.py" --salida "%R%" --formato double --paneles-por-sensor
python "%A%\p6_graficar_pvalores.py" --salida "%R%" --formato double --paneles-por-sensor --ymax 0.1
```

The curve figures (`p5_viscosity_effect`, `p5_interaction_speed`,
`p5_interaction_unbalance`, `p5_viscosity_by_*`) **always** use one panel per
probe, whatever the switch says: the probes differ by roughly a factor of six in
micrometres, so overlaying them on one axis would flatten the smaller one
against the axis. Their panels sit side by side with the title on top.

Colour encoding of the probes: bearing 1 in blues, bearing 2 in warm tones; the
Y direction dark and the X direction light.

### Type size, probe selection and shared scales

| Flag | Effect |
|---|---|
| `--tamano-letra 14` | base font size in points **at final size** |
| `--tamano-titulo 17` | title size, set **independently** of the base size |
| `--cojinete P1` | plot only bearing 1 (`P1Y`, `P1X`); `P2` or `todos` likewise |
| `--eje-y-comun` | every panel of a figure gets the **same Y limits** |

`--tamano-letra` is not just `font.size`: tick labels, legend, bar annotations,
line widths and marker sizes all follow it, the figure grows so the panels keep
usable room, the title wraps to the real width, the legend picks the number of
columns that fits, and **the number of Y axis divisions is reduced** — a large
font on a fixed axis height would otherwise crowd the tick labels. Set the two
sizes separately when a big title would eat the plot area:

```bat
python "%A%\p5_anova_split_plot.py" --salida "%R%" --formato double ^
    --cojinete P1 --tamano-letra 14 --tamano-titulo 17 --eje-y-comun
python "%A%\p6_graficar_pvalores.py" --salida "%R%" --formato double ^
    --cojinete P1 --tamano-letra 14 --tamano-titulo 17 --ymax 0.1
```

`--eje-y-comun` is a genuine trade-off, which is why it is a switch and not a
default: shared limits make the probes comparable by bar or curve height, while
independent limits let each panel use its full height and show its own shape.

The **bar** figures (`p5_contribution`, `p5_naive_vs_correct`, `p6_pvalues*`)
put every probe's bars **side by side in one panel**, in the aspect ratio of a
figure that spans the **full page width with no column division**, and the bar
groups are spaced so consecutive categories keep clear white space between them.
Use `--paneles-por-sensor` if you would rather have one panel per probe.

---

## Speed bands

`--grupos-velocidad` repeats the whole analysis inside each band. They are named
**G1..G4** on the axes — long names become unreadable once a figure is scaled
into a column — and the meaning is spelled out in the figure subtitle and here:

| Band | Speeds [rpm] | Meaning |
|---|---|---|
| **G1** | 600, 1300, 2000 | low speed |
| **G2** | 2500, 2600, 2700, 2800 | below the 1st critical |
| **G3** | 3400, 3500, 3600, 3700 | near the 1st critical |
| **G4** | 4000, 4500, 5000, 5500 | above the 1st critical |

Edit them in `config.GRUPOS_VELOCIDAD`. `--grupos-desbalanceo` does the same per
unbalance level, named **U1..U3**.

---

## Step 1 — Phasor extraction

Only files matching the naming convention are processed:

```
rep<R>_<viscosity>_<unbalance>_p_<date>.xlsx
```

Everything else in the folder is ignored (the count is reported). The `_p_`
suffix selects the proximity probes; `--tipo ac` would process the
accelerometers. Every sheet whose name is a number is read as a speed in rpm;
auxiliary sheets (`prox`, `Hoja17`, ...) drop out on their own.

Columns are located **by header content** (`Mach1.P1.Y` + `Displacement`/
`Phase`), not by fixed position, so a change of column order in the export does
not break anything.

### Stabilisation analysis

Each measurement lasts ~3 min. Before averaging, the script looks for **the
longest final stretch that has already settled**, i.e. where the phasor no
longer drifts:

| Criterion | What it measures | Tolerance (`config.py`) |
|---|---|---|
| `deriva_amp` | \|median(first quarter) − median(last half)\| / median | `TOL_DERIVA_AMP` = 0.05 |
| `deriva_fase` | angle between the circular means of both stretches | `TOL_DERIVA_FASE` = 5° |
| `cv_amp` | robust amplitude scatter / median | `TOL_CV_AMP` = 0.15 |
| `disp_fase` | **circular** standard deviation of the phase | `TOL_DISP_FASE` = 12° |

The *first quarter* is compared against the *last half* (rather than half
against half) because a short transient at the beginning gets diluted in the
median of half a window and would go unnoticed.

The accepted stretch is then averaged **vectorially**. Against transients of
known size the detector behaved like this:

| Signal | Trimmed |
|---|---|
| no transient | nothing |
| 25 % step decaying with τ = 20 s | 11 s |
| the same with τ = 40 s | 41 s |
| 20 % ramp that never settles | 106 s |
| 40 % ramp that never settles | nothing acceptable → **flagged as not stable** |

**Outputs:** `p1_phasors.txt` (main table) and
`p1_stabilisation_diagnostics.txt` (one row per run × rpm × probe with every
metric and the `estable` flag). Rows with `estable = 0` are worth a look: they
usually sit where the amplitude collapses and the phase loses physical meaning.

### Design check

The 27 `rep × visc × unbalance` combinations are verified, and missing, extra
and duplicated ones are reported. Missing speeds and values that could not be
computed are reported too.

---

## Steps 2 and 4 — Phasor figures

Twelve figures per dataset, laid out 2x2:

|  | column 1 (**Y** probe) | column 2 (**X** probe) |
|---|---|---|
| **row 1** | 1X amplitude [µm] | 1X amplitude [µm] |
| **row 2** | 1X phase [°] | 1X phase [°] |

* **6 figures** `unbalance<d>_<bearing>.png` — fix unbalance and bearing, overlay
  the 3 viscosities × 3 repetitions.
* **6 figures** `viscosity<v>_<bearing>.png` — fix viscosity and bearing, overlay
  the 3 unbalance levels × 3 repetitions.

Encoding, constant across **all** figures:

| Encoding | Factor |
|---|---|
| **Colour** | viscosity: ISO 32 red · ISO 46 blue · ISO 68 green |
| **Colour saturation** | unbalance: 1 the most washed out, 3 full colour |
| **Line style** | repetition: 1 solid, 2 dashed, 3 dotted |

Lines only, no markers. The **X axis** spreads the 15 speeds over 15 evenly
spaced positions (not on a numeric rpm scale). The **Y limits** are shared by
all twelve figures and come from the global maximum and minimum of the table,
one for amplitude and one for phase.

> **`--desenvolver`** — strongly recommended. Without it the phase row shows
> artificial vertical jumps from 0 to 360 that hide the real trend (the
> progressive lag through the criticals). With it every curve is drawn
> continuously. It is off by default so the figure shows exactly the values in
> the table.

Step 4 also accepts `--escala-comun`, which widens the limits to cover the
uncompensated data too, so step 2 and step 4 figures can be compared directly.

---

## Step 3 — Runout compensation

```
z_compensated(rpm) = z_measured(rpm) − z_slow_roll
```

The slow-roll vector is the **vector mean** of the phasors at 600 rpm. `--modo`
picks the grouping: `viscosidad` (default), `global` or `rep_viscosidad`.

### The angle problem, and why it does not exist here

Phases arrive wrapped into [0, 360). Two nearly identical measurements can come
out as 359° and 1°, and their arithmetic mean would be 180°: the exact opposite
of the right answer. **The arithmetic mean of angles is not valid.**

The fix is not to correct the mean afterwards, but never to compute it that way:
each measurement becomes a complex phasor `z = A·e^{iφ}`, real and imaginary
parts are averaged, and amplitude and phase are recovered from the result. Since
the angle is never handled as a scalar, wrapping stops existing. Checked on nine
measurements spread around 0°:

```
phases: [358.0, 359.5, 1.0, 2.5, 0.3, 357.8, 3.1, 359.0, 1.7]
  arithmetic mean (wrong) : 160.32°
  vector mean     (right) :   0.32°
  circular scatter: 1.80°   (the naive standard deviation would give 177.3°)
```

Where a **continuous** angle is genuinely needed (drawing curves, measuring
drift) `comun.desenvolver_fase` adds to each sample the multiple of 360 that
leaves it closest to the previous one.

The script reports, group by group, the discrepancy between the naive and the
vector mean, and the circular concentration `R` (below 0.9 the mean phase is not
representative). It also writes `p3_slow_roll.png` with the polar diagrams of
the phasors at 600 rpm.

### Does a viscosity-dependent slow roll make sense?

Short answer: **runout proper cannot depend on the oil, but what is measured at
600 rpm is not only runout.**

* Mechanical runout (shaft geometry and bow) and electrical runout (material
  variations seen by the eddy-current probe) are properties of the shaft and the
  probe. The oil does not touch them.
* What is recorded at 600 rpm is runout **plus** the synchronous response that
  already exists at that speed. With the first critical around 3500 rpm,
  *r* = 600/3500 ≈ 0.17 and the amplification factor `r²/(1−r²)` ≈ 0.03: small,
  but not zero.
* The journal equilibrium position inside the bearing also depends on the
  Sommerfeld number (viscosity × speed / load). Changing the oil moves the
  eccentricity and the attitude angle, and with them the bearing stiffness and
  damping coefficients — which in turn modify that small synchronous response.

So a mild viscosity dependence at 600 rpm is **physically plausible, but it is
not runout: it is early dynamic response**. And that is the risk — subtracting a
viscosity-dependent slow roll removes part of the very effect being measured.

**What to do:**

1. `comparar_viscosidades` quantifies it: it compares the separation *between*
   the three viscosity means against the scatter *within* each viscosity. Look
   at `razon_entre_dentro`:
   * **< 2** → the difference does not exceed measurement noise. Use
     `--modo global`.
   * **≫ 2** → the difference is measurable and its origin has to be decided.
2. If the rotor is **dismantled and re-mounted** when the oil is changed, the
   mechanical runout genuinely can change — but then it changes per *mounting*,
   not per *viscosity*. The right choice there is `--modo rep_viscosidad`.
3. Ideally the slow roll would be measured well below the response region,
   around 10–15 % of the first critical (≈ 350–500 rpm here). 600 rpm is ≈ 17 %:
   borderline. `--rpm-slow-roll` accepts another speed if a lower one is ever
   recorded.

---

## Step 5 — Statistical analysis

### Why an ordinary factorial ANOVA is wrong here

The experiment is **not** completely randomised. It is a **split-split-plot in
blocks**:

| Level | Factor | Levels | Why |
|---|---|---|---|
| block | repetition | 3 | |
| whole plot | viscosity | 3 | changing the oil is expensive, done rarely |
| subplot | unbalance | 3 | the mass changes without touching the oil |
| sub-subplot | speed | 15 | sweep within a single mounting |

Every randomisation level has **its own error term**:

| Effect | Tested against | Denominator df |
|---|---|---|
| Viscosity | Error(a) = Rep × Visc | **4** |
| Unbalance, Visc×Unb | Error(b) = Rep × Unb \| Visc | 12 |
| Speed and its interactions | Error(c) | 252 |

`f3_s1_v1.py` and `f3_s1_v2.py` test **everything** against the global residual
(270 df). That treats viscosity as if the rotor had been mounted 405
independent times, when there are only **9 whole-plot units** (3 blocks ×
3 oils). The denominator is far too small and the F ratio explodes.

### How much it matters, measured

Simulation under the null hypothesis (viscosity has **no** effect), with
realistic mounting variability, 600 replicates per scenario:

| Whole-plot σ | Rejections, **correct** test | Rejections, **naive** test |
|---|---|---|
| 0.0 | 5.7 % | 48.3 % |
| 0.5 | 5.2 % | 78.0 % |
| 1.0 | 4.5 % | 87.7 % |
| 2.0 | 5.7 % | 91.8 % |

*(nominal expectation: 5 %)*

The split-plot test holds the nominal 5 %. The naive ANOVA declares viscosity
significant in up to **9 out of 10 cases where there is no effect at all**. That
is not a nuance: it is the difference between a result and an artefact.

On synthetic data with a real effect, the naive ANOVA multiplied the viscosity F
by a factor of **14 to 31**, taking the p value from ~0.007 (real) to ~10⁻⁹⁶.

> The sum-of-squares decomposition was validated against `statsmodels`
> (`anova_lm`, type II): they agree to ~1e-14 on every effect, and the three
> error strata add up exactly to the residual of the factorial model. What
> changes is not the sums of squares but **what each one is divided by**.

Note a side effect: for the sub-subplot factors (speed and its interactions) the
correct analysis gives **smaller** p values than the naive one, because the
pooled residual was contaminated with the variance of the upper strata.

### Analysis by unbalance level

With `--grupos-desbalanceo` the analysis is repeated **for each unbalance level
separately**, suffixed `_U1/_U2/_U3`, plus the comparison figures
`p5_viscosity_by_unbalance_means.png` and
`p5_viscosity_by_unbalance_evidence.png`.

There is a change of model worth understanding: fixing the unbalance **removes
that factor from the model** and with it its error stratum. The design stops
being a split-split-plot and reduces to a **split-plot in blocks**:

| | Full design | Fixing one unbalance level |
|---|---|---|
| block | repetition | repetition |
| whole plot | viscosity | viscosity |
| subplot | unbalance | **speed** |
| sub-subplot | speed | — |
| Sources in the table | 11 | **6** |
| Error(a) | Rep × Visc, 4 df | Rep × Visc, **4 df** |
| Error(b) | Rep × Unb \| Visc, 12 df | residual, 84 df |
| Error(c) | 252 df | does not exist |

**Viscosity is still judged with 4 df.** Splitting by unbalance adds no
information about the oil, because the number of whole-plot mountings
(3 blocks × 3 oils = 9) is unchanged. What it does show is whether the oil
effect changes with the amount of unbalance — which is exactly what the
`Visc × Unb` term of the full analysis tests formally.

> Validated the same way as the full design: sums of squares match `statsmodels`
> (`y ~ C(Visc)*C(Speed)`) to ~1e-14, the df add up to n−1 = 134, and two
> identities link both implementations:
> `Σ_d SS_Visc(at d) = SS_Visc + SS_Visc×Unb` and
> `Σ_d SS_Error(a)(at d) = SS_Error(a) + SS_Rep×Visc×Unb` (12 df = 4 + 8).

### Outputs

In `<salida>/p5_statistics`:

| File | Contents |
|---|---|
| `anova_split_plot.csv` | full table per sensor: SS, df, MS, error term, F, p, contribution, ω² |
| `anova_naive.csv` | the ordinary factorial ANOVA, for comparison |
| `viscosity_comparison.csv` | correct vs naive F and p, and the inflation factor |
| `posthoc_viscosity.csv` | Tukey between oils using Error(a) and its 4 df |
| `variance_components.csv` | variance attributable to each stratum |
| `p5_contribution.png` | variability decomposition, with real significance marks |
| `p5_naive_vs_correct.png` | real evidence versus inflated evidence, effect by effect |
| `p5_viscosity_effect.png` | mean per oil with a 95 % CI based on Error(a), one panel per probe |
| `p5_interaction_speed.png` | viscosity × speed, one panel per probe |
| `p5_interaction_unbalance.png` | viscosity × unbalance, one panel per probe |
| `p5_variance_strata.png` | where the experimental variability comes from |
| `p5_viscosity_by_speed_band_means.png` | (--grupos-velocidad) mean per oil, band by band |
| `p5_viscosity_by_speed_band_evidence.png` | (--grupos-velocidad) evidence of the effect, band by band |
| `p5_viscosity_by_unbalance_means.png` | (--grupos-desbalanceo) mean per oil, level by level |
| `p5_viscosity_by_unbalance_evidence.png` | (--grupos-desbalanceo) evidence of the effect, level by level |

The magnitude and the evidence are **separate figures** rather than two rows of
one figure: they carry different units (micrometres against -log10(p)), so a
reader had to switch scales between rows to follow a single figure.

By default the **compensated** data are analysed
(`p3_phasors_compensated.txt`); `--entrada` can point at `p1_phasors.txt` to
analyse the uncompensated ones.

### Two warnings

* **Phase is a circular variable.** `--respuesta phase` exists, but an ANOVA on
  degrees is only defensible if no condition crosses 0/360 (359° and 1° are 2°
  apart, not 358°). The script warns about it. A rigorous treatment would need
  circular statistics; ask if you need it.
* **With 4 denominator df, the power to detect the viscosity effect is low.**
  That is not a flaw of the analysis, it is what the design allows: the
  information about viscosity comes from 9 mountings, not 405 measurements. If
  the effect comes out non-significant, the honest conclusion is "this
  experiment lacks the resolution to demonstrate it", not "it does not exist".
  Gaining power means more **complete repetitions** (blocks), which is what
  feeds Error(a) — not more speeds and not more unbalance levels.

---

## Step 6 — p-value figures

Same per-probe layout options as step 5, but the Y axis carries the **p value**
instead of the share of the sum of squares.

```bat
python "%A%\p6_graficar_pvalores.py" --salida "%R%"
python "%A%\p6_graficar_pvalores.py" --salida "%R%" --alfa 0.01
python "%A%\p6_graficar_pvalores.py" --salida "%R%" --ymax 0.1
```

It recomputes nothing: it reads the `anova_split_plot*.csv` files already
written by step 5 and picks up whichever variants exist.

* **Y axis** = the p value on an **ordinary linear scale**, 0 to 1. The bar
  height **is** the p value, untransformed.
* **A bar below the red line is significant.** The lower the bar, the stronger
  the evidence. The significant zone is shaded green and spelled out in the
  legend.
* **The number on each bar** is the exact p value.
* **Green** = significant · **hatched** = not.

> **Mind the direction of the scale.** `p < 0.05` means the factor **does**
> influence the response: the p value is the probability of observing these data
> *if the factor had no effect at all*, so a small p is strong evidence. This is
> the opposite of the "big bar = important" intuition. It agrees with the marks
> in `p5_contribution.png`: significant ↔ bar inside the green band.

> **`--ymax` to zoom into the threshold.** Over the full 0–1 range every
> significant bar sits on the floor: clearly below the line, but
> indistinguishable from each other. `--ymax 0.1` bounds the axis to the region
> that matters; bars that exceed it are clipped and flagged with an arrow. The
> exact number is printed on every bar either way.

Outputs in `p5_statistics`:

| File | Contents |
|---|---|
| `p6_pvalues.png` | every factor, global analysis |
| `p6_pvalues_<variant>.png` | the same per speed band and per unbalance level |
| `p6_pvalues_viscosity.png` | **the summary**: the viscosity p across every variant |
| `p6_pvalues.csv` | the same numbers as a table, with a `Significant` flag |
