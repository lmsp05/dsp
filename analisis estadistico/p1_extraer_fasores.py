"""
PASO 1 — Extraccion de fasores 1X desde los .xlsx
=================================================

Recorre una carpeta, procesa UNICAMENTE los .xlsx que cumplen la nomenclatura

    rep<R>_<viscosidad>_<desbalanceo>_p_<fecha>.xlsx      (p = proximitores)

y de cada hoja (cuyo nombre es la velocidad en rpm) extrae, para los 4 sensores
P1Y / P1X / P2Y / P2X, la amplitud 1X pico-pico [um] y la fase [grados].

Cada medicion dura ~3 min, asi que antes de promediar se ANALIZA LA
ESTABILIZACION: se busca el tramo final mas largo del registro en el que la
amplitud y la fase ya no derivan y su dispersion es baja (ver `tramo_estable`).
Sobre ese tramo se hace el promedio VECTORIAL del fasor (inmune al problema de
envolvimiento del angulo en 0/360).

Salidas en --salida:
  * p1_fasores.txt                    tabla principal (una fila por ensayo x rpm)
  * p1_diagnostico_estabilizacion.txt detalle de la estabilizacion por sensor

Comprueba ademas que esten las 27 combinaciones rep x visc x desb y avisa si no.

Uso:
    python p1_extraer_fasores.py --entrada <carpeta_xlsx> --salida <carpeta_resultados>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
import comun


# ============================================================
# LECTURA DE UNA HOJA
# ============================================================

# 'NVD: Prof >Bode> 1X [Pk-Pk]> Mach1.P1.Y'  ->  'P1Y'
_RE_SENSOR = re.compile(r"P(\d)\s*\.\s*([XY])", re.IGNORECASE)


def _mapa_columnas(cab: pd.DataFrame):
    """
    A partir de las 3 filas de cabecera devuelve (col_tiempo, {sensor: (ia, if)}).

    Fila 0 = canal ('...Mach1.P1.Y'), fila 1 = magnitud ('Time' / 'Displacement'
    / 'Phase'). Se localizan por contenido, no por posicion fija.
    """
    col_t = None
    parejas: dict[str, dict[str, int]] = {}
    for j in range(cab.shape[1]):
        canal = str(cab.iloc[0, j])
        magnitud = str(cab.iloc[1, j]).strip().lower()
        if magnitud == "time":
            col_t = j
            continue
        m = _RE_SENSOR.search(canal)
        if not m:
            continue
        sensor = f"P{m.group(1)}{m.group(2).upper()}"
        if magnitud.startswith("displacement") or magnitud.startswith("amp"):
            parejas.setdefault(sensor, {})["amp"] = j
        elif magnitud.startswith("phase") or magnitud.startswith("fase"):
            parejas.setdefault(sensor, {})["fase"] = j
    salida = {s: (d["amp"], d["fase"]) for s, d in parejas.items()
              if "amp" in d and "fase" in d}
    return col_t, salida


def leer_hoja(ruta: Path, hoja: str):
    """Devuelve (tiempo, {sensor: (amp, fase)}) de una hoja de velocidad."""
    crudo = pd.read_excel(ruta, sheet_name=hoja, header=None)
    if crudo.shape[0] < 5:
        return None, {}
    col_t, mapa = _mapa_columnas(crudo.iloc[:3])
    if col_t is None or not mapa:
        return None, {}
    cuerpo = crudo.iloc[3:].apply(pd.to_numeric, errors="coerce")
    t = cuerpo.iloc[:, col_t].to_numpy(dtype=float)
    datos = {}
    for sensor, (ia, if_) in mapa.items():
        a = cuerpo.iloc[:, ia].to_numpy(dtype=float)
        f = cuerpo.iloc[:, if_].to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(a) & np.isfinite(f)
        datos[sensor] = (t[ok], a[ok], f[ok])
    return t, datos


# ============================================================
# ANALISIS DE ESTABILIZACION
# ============================================================

def tramo_estable(t, amp, fase):
    """
    Busca el tramo final [i:] mas largo que se pueda considerar estabilizado.

    "Estabilizado" = el transitorio inicial ya paso, es decir el fasor YA NO
    DERIVA. La deriva se mide de forma robusta comparando la primera y la
    segunda mitad del tramo candidato:

      * deriva_amp  = |mediana(2a mitad) - mediana(1a mitad)| / mediana  <= TOL_DERIVA_AMP
      * deriva_fase = angulo entre las medias circulares de ambas mitades <= TOL_DERIVA_FASE

    y ademas el ruido dentro del tramo no debe ser excesivo:

      * cv_amp    = dispersion robusta de la amplitud / mediana  <= TOL_CV_AMP
      * disp_fase = desviacion CIRCULAR de la fase [grados]      <= TOL_DISP_FASE

    Todas las medidas de fase son circulares, de modo que no se ven afectadas
    por el envolvimiento 0/360 ni se disparan cuando el fasor pasa cerca del
    origen (donde el angulo carece de sentido fisico).

    Se prueban inicios cada vez mas tardios y se devuelve el PRIMERO que cumple
    (el tramo estable mas largo). Si ninguno cumple se devuelve el ultimo
    FRAC_MINIMA_ESTABLE del registro marcado como no estable.

    Devuelve (indice_inicio, estable, metricas).
    """
    n = len(amp)
    n_min = max(5, int(round(config.FRAC_MINIMA_ESTABLE * n)))
    if n < n_min + 1:
        return 0, False, _metricas(t, amp, fase, 0)

    ultimos = n - n_min                             # maximo inicio admisible
    paso = max(1, ultimos // config.N_CANDIDATOS)
    for i in range(0, ultimos + 1, paso):
        m = _metricas(t, amp, fase, i)
        if (m["cv_amp"] <= config.TOL_CV_AMP
                and m["disp_phase"] <= config.TOL_DISP_FASE
                and m["deriva_amp"] <= config.TOL_DERIVA_AMP
                and m["deriva_phase"] <= config.TOL_DERIVA_FASE):
            return i, True, m
    return ultimos, False, _metricas(t, amp, fase, ultimos)


def _metricas(t, amp, fase, i):
    """Metricas de estabilidad (robustas y circulares) del tramo [i:]."""
    ts, a, f = t[i:], amp[i:], fase[i:]
    med = max(float(np.median(a)), 1e-12)

    # Deriva = cuanto se aparta el ARRANQUE del tramo (primer cuarto) del nivel
    # ya asentado (ultima mitad). Comparar mitad contra mitad no detectaria un
    # transitorio corto al principio, porque queda diluido en la mediana.
    q, h = len(a) // 4, len(a) // 2
    if q >= 2:
        ref_a = float(np.median(a[h:]))
        deriva_amp = abs(float(np.median(a[:q])) - ref_a) / med
        m_ini, _ = comun.promedio_escalar_fase(f[:q])
        m_fin, _ = comun.promedio_escalar_fase(f[h:])
        deriva_fase = abs(comun.diferencia_angular(m_ini, m_fin))
    else:
        deriva_amp = deriva_fase = 0.0

    # Dispersion del fasor completo: |z - z_medio| relativo. Resume en un solo
    # numero cuanto se mueve la medida, sin separar amplitud de fase.
    z = comun.a_fasor(a, f)
    zm = z.mean()
    disp_vect = float(np.mean(np.abs(z - zm)) / max(abs(zm), 1e-12))

    return {
        "cv_amp": comun.dispersion_robusta(a) / med,
        "disp_phase": comun.dispersion_circular(f),
        "deriva_amp": deriva_amp,
        "deriva_phase": deriva_fase,
        "disp_vect": disp_vect,
        "t_inicio": float(ts[0]) if len(ts) else np.nan,
        "n_usadas": int(len(a)),
    }


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def procesar_archivo(info, filas, diag):
    """Procesa un .xlsx completo y acumula sus filas en `filas` y `diag`."""
    ruta = info["ruta"]
    xl = pd.ExcelFile(ruta)
    hojas = [h for h in xl.sheet_names if h.strip().isdigit()]
    if not hojas:
        print(f"  [AVISO] {ruta.name}: sin hojas de velocidad, se omite.")
        return 0

    n = 0
    for hoja in sorted(hojas, key=lambda h: int(h)):
        rpm = int(hoja.strip())
        _, datos = leer_hoja(ruta, hoja)
        if not datos:
            print(f"  [AVISO] {ruta.name} hoja '{hoja}': cabecera no reconocida.")
            continue

        fila = {
            "Repetition": info["rep"], "Viscosity": info["visc"],
            "Unbalance": info["desb"], "Speed": rpm,
        }
        for sensor in config.SENSORES:
            if sensor not in datos:
                fila[f"{sensor}_amp"] = np.nan
                fila[f"{sensor}_phase"] = np.nan
                continue
            t, a, f = datos[sensor]
            if len(a) < 3:
                fila[f"{sensor}_amp"] = np.nan
                fila[f"{sensor}_phase"] = np.nan
                continue
            i, estable, m = tramo_estable(t, a, f)
            # Promedio VECTORIAL sobre el tramo estable.
            amp_m, fase_m = comun.promedio_vectorial(a[i:], f[i:])
            fila[f"{sensor}_amp"] = amp_m
            fila[f"{sensor}_phase"] = fase_m
            diag.append({
                "Repetition": info["rep"], "Viscosity": info["visc"],
                "Unbalance": info["desb"], "Speed": rpm, "Sensor": sensor,
                "n_total": len(a), "n_usadas": m["n_usadas"],
                "frac_usada": m["n_usadas"] / len(a),
                "t_inicio_s": m["t_inicio"], "estable": int(estable),
                "cv_amp": m["cv_amp"], "disp_fase_deg": m["disp_phase"],
                "deriva_amp": m["deriva_amp"], "deriva_fase_deg": m["deriva_phase"],
                "disp_vect": m["disp_vect"],
            })
        filas.append(fila)
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paso 1: extrae los fasores 1X promedio (tramo estabilizado) "
                    "de los .xlsx de proximitores.")
    p.add_argument("--entrada", default=config.DIR_EXCEL,
                   help="Carpeta con los .xlsx")
    p.add_argument("--salida", default=config.DIR_RESULTADOS,
                   help="Carpeta donde escribir los resultados")
    p.add_argument("--tipo", default="p", choices=["p", "ac"],
                   help="'p' = proximitores (def), 'ac' = acelerometros")
    args = p.parse_args()

    entrada = Path(args.entrada).expanduser()
    salida = Path(args.salida).expanduser()
    if not entrada.is_dir():
        print(f"ERROR: no existe la carpeta de entrada {entrada}")
        return 1
    salida.mkdir(parents=True, exist_ok=True)

    validos, ignorados = comun.listar_excels(entrada, args.tipo)
    print(f"Carpeta: {entrada}")
    print(f"Archivos que cumplen la nomenclatura (_{args.tipo}_): {len(validos)}")
    if ignorados:
        print(f"Archivos ignorados: {len(ignorados)}"
              + (f"  (p.ej. {ignorados[0]})" if ignorados else ""))
    if not validos:
        print("ERROR: no hay ningun archivo que procesar.")
        return 1

    # ---- comprobacion del diseno completo (27 combinaciones) ----
    combos = {(v["rep"], v["visc"], v["desb"]) for v in validos}
    esperadas = {(r, vi, d) for r in config.REPETICIONES
                 for vi in config.VISCOSIDADES for d in config.DESBALANCEOS}
    faltan = sorted(esperadas - combos)
    sobran = sorted(combos - esperadas)
    duplicados = len(validos) - len(combos)

    print(f"\nCombinaciones rep x visc x desb encontradas: {len(combos)} "
          f"de {config.N_ARCHIVOS_ESPERADOS} esperadas")
    if faltan:
        print(f"  !! ALERTA: faltan {len(faltan)} combinaciones: "
              + ", ".join(f"rep{r}_{v}_{d}" for r, v, d in faltan))
    if sobran:
        print(f"  !! ALERTA: {len(sobran)} combinaciones fuera del diseno: "
              + ", ".join(f"rep{r}_{v}_{d}" for r, v, d in sobran))
    if duplicados > 0:
        print(f"  !! ALERTA: hay {duplicados} archivo(s) duplicado(s) para una misma combinacion")
    if not faltan and not sobran and duplicados == 0:
        print("  OK: el diseno esta completo.")

    # ---- extraccion ----
    print("\nProcesando:")
    filas, diag = [], []
    for info in validos:
        n = procesar_archivo(info, filas, diag)
        print(f"  {info['ruta'].name:40s} rep{info['rep']} ISO{info['visc']} "
              f"dsb{info['desb']} -> {n} velocidades")

    if not filas:
        print("ERROR: no se extrajo ninguna fila.")
        return 1

    df = pd.DataFrame(filas)
    df = df[comun.COLUMNAS_FACTOR + comun.columnas_fasor()]
    df = df.sort_values(comun.COLUMNAS_FACTOR).reset_index(drop=True)
    ruta_tabla = comun.escribir_tabla(df, salida / config.ARCHIVO_FASORES)

    dg = pd.DataFrame(diag)
    comun.escribir_tabla(dg, salida / config.ARCHIVO_DIAGNOSTICO)

    # ---- resumen ----
    vel_halladas = sorted(int(v) for v in df["Speed"].unique())
    print(f"\nFilas escritas: {len(df)}  (esperadas "
          f"{config.N_ARCHIVOS_ESPERADOS * len(config.VELOCIDADES)})")
    print(f"Velocidades encontradas ({len(vel_halladas)}): {vel_halladas}")
    faltan_vel = sorted(set(config.VELOCIDADES) - set(vel_halladas))
    if faltan_vel:
        print(f"  !! ALERTA: faltan velocidades del diseno: {faltan_vel}")
    n_nan = int(df[comun.columnas_fasor()].isna().sum().sum())
    if n_nan:
        print(f"  !! ALERTA: {n_nan} valores no pudieron calcularse (NaN)")

    if len(dg):
        pct = 100.0 * dg["estable"].mean()
        print(f"\nEstabilizacion: {pct:.1f} % de las series alcanzaron el criterio; "
              f"se uso en promedio el {100*dg['frac_usada'].mean():.1f} % del registro.")
        malas = dg[dg["estable"] == 0]
        if len(malas):
            print(f"  !! {len(malas)} serie(s) no estabilizaron (ver el diagnostico); "
                  "se promedio su tramo final.")

    print(f"\nTabla de fasores : {ruta_tabla}")
    print(f"Diagnostico      : {salida / config.ARCHIVO_DIAGNOSTICO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
