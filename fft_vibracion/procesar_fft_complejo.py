"""
procesar_fft_complejo.py
========================

Variante de ``procesar_fft.py`` que, en lugar de calcular el espectro de cada
proximitor por separado, calcula el ESPECTRO COMPLEJO (full spectrum) de la
senal de orbita z = x + i*y de cada COJINETE:

    C1 = P1.X + i * P1.Y      (cojinete 1)
    C2 = P2.X + i * P2.Y      (cojinete 2)

Como z es una senal COMPLEJA, su FFT NO es simetrica: las frecuencias positivas
y negativas son componentes fisicas distintas. En un rotor:

    * frecuencia POSITIVA  -> precesion DIRECTA  (forward whirl, mismo sentido
      que el giro),
    * frecuencia NEGATIVA  -> precesion RETROGRADA (backward whirl).

Asi, para cada condicion y velocidad se generan DOS espectros (C1 y C2), cada uno
con frecuencias negativas y positivas, y se pueden detectar las frecuencias
negativas (modos retrogrados), imposibles de ver en el espectro de un solo
proximitor.

El resto del procesamiento es identico a procesar_fft.py (keyphasor -> RPM ->
tramo estacionario -> bloques de N revoluciones con Hann) y se reutiliza de aquel
modulo. El archivo de salida tiene EXACTAMENTE las mismas columnas que
procesar_fft.py (la columna ``sensor`` vale ``C1``/``C2`` en vez del proximitor),
de modo que ``deteccion_picos.py`` y sus variantes lo procesan igual.

Diferencia de normalizacion: en el espectro real se duplica la amplitud (espectro
de un solo lado); aqui NO, porque cada frecuencia (positiva o negativa) es su
propia componente. La amplitud es el radio de la componente circular a esa
frecuencia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Reutiliza todo el pipeline de procesar_fft.py (mismo folder).
from procesar_fft import (
    FS, REVOLUCIONES_POR_BLOQUE, SOLAPE, ANALISIS_BLOQUE_S, UMBRAL_RPM_PCT,
    MIN_DURACION_TRAMO_S, FMIN, FMAX, RE_PROXIMITOR,
    leer_archivo, detectar_pulsos, rpm_instantanea, segmentar_estacionario,
    seleccionar_tramo, descubrir_archivos, ArchivoInfo, Tramo, escribir_espectro,
    CABECERA_COLUMNAS,
)


# ============================================================
# EMPAREJAMIENTO DE PROXIMITORES EN COJINETES (senal compleja)
# ============================================================

def emparejar_cojinetes(canales: list[str]) -> list[tuple[str, int, int]]:
    """Devuelve [(nombre, col_x, col_y), ...] para C1 (P1.X,P1.Y) y C2 (P2.X,P2.Y)."""
    pos = {}
    for i, c in enumerate(canales):
        if RE_PROXIMITOR.fullmatch(c):
            pos[c.upper()] = i
    cojinetes = []
    for b in (1, 2):
        ix = pos.get(f"P{b}.X")
        iy = pos.get(f"P{b}.Y")
        if ix is not None and iy is not None:
            cojinetes.append((f"C{b}", ix, iy))
    return cojinetes


# ============================================================
# ESPECTRO COMPLEJO (FULL SPECTRUM) PROMEDIADO EN BLOQUES
# ============================================================

def espectro_complejo(z, pulsos, fs, tramo: Tramo, revs, solape, fmin, fmax):
    """Full spectrum de la senal compleja z sobre el tramo estacionario.

    Igual estructura que espectro_amp_fase de procesar_fft (bloques de `revs`
    revoluciones que arrancan en un pulso, ventana Hann, grilla comun), pero con
    FFT COMPLETA (np.fft.fft) para conservar frecuencias negativas y sin duplicar
    amplitud. Devuelve (f, amplitud, fase_grados, n_bloques, nperseg) con f
    ordenada de negativa a positiva.
    """
    nperseg = int(round(revs * fs * 60.0 / tramo.rpm_media))
    if nperseg < 8:
        raise ValueError("nperseg demasiado pequeno; revisa RPM/revoluciones.")
    w = np.hanning(nperseg)
    cg = w.sum()
    paso = max(1, int(round(revs * (1.0 - solape))))

    i0 = int(tramo.t0 * fs)
    i1 = int(tramo.t1 * fs)
    idx_pulsos = np.where((pulsos >= i0) & (pulsos + nperseg <= i1) &
                          (pulsos + nperseg <= len(z)))[0]
    if len(idx_pulsos) == 0:
        raise ValueError("El tramo no admite ningun bloque completo.")

    Zsum = None
    Pot = None
    n = 0
    for k in range(0, len(idx_pulsos), paso):
        ini = int(pulsos[idx_pulsos[k]])
        seg = z[ini:ini + nperseg].astype(complex)
        seg = seg - seg.mean()                 # media compleja (quita el offset x e y)
        Z = np.fft.fft(seg * w) / cg           # FFT completa; sin doblar amplitud
        if Zsum is None:
            Zsum = np.zeros_like(Z)
            Pot = np.zeros(len(Z))
        Zsum += Z
        Pot += np.abs(Z) ** 2
        n += 1
    if n == 0:
        raise ValueError("No se acumulo ningun bloque.")

    f = np.fft.fftfreq(nperseg, d=1.0 / fs)
    amplitud = np.sqrt(Pot / n)                 # promedio de potencia -> amplitud (radio)
    fase = np.degrees(np.angle(Zsum / n))       # promedio coherente -> fase

    orden = np.argsort(f)                        # de negativa a positiva
    f, amplitud, fase = f[orden], amplitud[orden], fase[orden]
    m = (np.abs(f) >= fmin) & (np.abs(f) <= fmax)  # conserva ambas bandas +/-
    return f[m], amplitud[m], fase[m], n, nperseg


# ============================================================
# CABECERA
# ============================================================

def escribir_cabecera(fh, args):
    fh.write("# Full spectrum (espectro complejo z=x+iy) por cojinete\n")
    fh.write("# Generado por procesar_fft_complejo.py\n")
    fh.write("# C1 = P1.X + i*P1.Y ; C2 = P2.X + i*P2.Y  (columna 'sensor' = C1/C2)\n")
    fh.write("# frecuencia POSITIVA = precesion directa ; NEGATIVA = retrograda\n")
    fh.write("# --------------------------------------------------------------\n")
    fh.write(f"# fs_Hz\t{FS}\n")
    fh.write(f"# revoluciones_por_bloque\t{args.revs}\n")
    fh.write(f"# solape\t{args.solape}\n")
    fh.write(f"# fmin_Hz\t{args.fmin}\n")
    fh.write(f"# fmax_Hz\t{args.fmax}\n")
    fh.write("# amplitud: radio de la componente circular a esa frecuencia (um)\n")
    fh.write("# Una fila por bin. Filtrable por rep/iso/dsb/rpm_nominal/sensor(C1,C2).\n")
    fh.write("# --------------------------------------------------------------\n")
    fh.write("\t".join(CABECERA_COLUMNAS) + "\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def procesar(args) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: la carpeta de datos no existe: {data_dir}")
        return 1

    archivos = descubrir_archivos(data_dir)
    if args.limit:
        archivos = archivos[:args.limit]
    if not archivos:
        print(f"No se encontraron archivos validos en {data_dir}")
        return 1

    salida = Path(args.salida).expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    print(f"Archivos a procesar: {len(archivos)}")
    print(f"Salida (full spectrum): {salida}\n")

    n_ok = 0
    n_err = 0
    with salida.open("w", encoding="utf-8") as fh:
        escribir_cabecera(fh, args)
        for info in archivos:
            etiqueta = f"rep{info.rep}_iso{info.iso}_dsb{info.dsb}_rpm{info.rpm:g}"
            try:
                matriz, canales = leer_archivo(info.path)
            except Exception as e:
                print(f"  [ERROR lectura] {etiqueta}: {e}")
                n_err += 1
                continue

            try:
                col_sync = next(i for i, c in enumerate(canales) if c == "sync")
            except StopIteration:
                col_sync = 0
            cojinetes = emparejar_cojinetes(canales)
            if not cojinetes:
                print(f"  [aviso] sin pares X/Y de proximitores en {etiqueta}, se omite")
                n_err += 1
                continue

            try:
                pulsos = detectar_pulsos(matriz[:, col_sync])
                t_pulso, rpm = rpm_instantanea(pulsos, FS)
                tramos = segmentar_estacionario(
                    t_pulso, rpm, args.bloque_s, args.umbral_rpm, args.min_dur)
                tramo = seleccionar_tramo(tramos, info.rpm)
            except Exception as e:
                print(f"  [ERROR keyphasor] {etiqueta}: {e}")
                n_err += 1
                continue
            if tramo is None:
                print(f"  [aviso] {etiqueta}: sin tramo estacionario, se omite")
                n_err += 1
                continue

            escritos = 0
            for nombre, ix, iy in cojinetes:
                try:
                    z = matriz[:, ix].astype(float) + 1j * matriz[:, iy].astype(float)
                    f, amp, fase, nb, nperseg = espectro_complejo(
                        z, pulsos, FS, tramo, args.revs, args.solape, args.fmin, args.fmax)
                    escribir_espectro(fh, info, nombre, tramo, f, amp, fase, nb, nperseg)
                    escritos += 1
                except Exception as e:
                    print(f"  [ERROR fft {nombre}] {etiqueta}: {e}")
            if escritos:
                print(f"  OK {etiqueta}: tramo [{tramo.t0:.1f},{tramo.t1:.1f}]s "
                      f"RPM={tramo.rpm_media:.1f} | {escritos} cojinete(s)")
                n_ok += 1
            else:
                n_err += 1

    print(f"\nListo. Archivos con espectro: {n_ok} | con error/omitidos: {n_err}")
    print(f"Resultado en: {salida}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera el full spectrum (espectro complejo z=x+iy) por "
                    "cojinete de todos los archivos de una carpeta.")
    p.add_argument("--data-dir", default="datos",
                   help="Carpeta raiz con las carpetas repN_isoVG_dsbM (def: ./datos)")
    p.add_argument("--salida", default="resultados_fft_complejo.txt",
                   help="Archivo .txt de salida (def: ./resultados_fft_complejo.txt)")
    p.add_argument("--revs", type=int, default=REVOLUCIONES_POR_BLOQUE,
                   help="Revoluciones enteras por bloque de FFT")
    p.add_argument("--solape", type=float, default=SOLAPE, help="Solape entre bloques (0-0.95)")
    p.add_argument("--bloque-s", dest="bloque_s", type=float, default=ANALISIS_BLOQUE_S,
                   help="Bloque temporal para medir estacionariedad [s]")
    p.add_argument("--umbral-rpm", dest="umbral_rpm", type=float, default=UMBRAL_RPM_PCT,
                   help="%% de variacion de RPM que separa tramos")
    p.add_argument("--min-dur", dest="min_dur", type=float, default=MIN_DURACION_TRAMO_S,
                   help="Duracion minima de un tramo utilizable [s]")
    p.add_argument("--fmin", type=float, default=FMIN, help="|Frecuencia| minima guardada [Hz]")
    p.add_argument("--fmax", type=float, default=FMAX, help="|Frecuencia| maxima guardada [Hz]")
    p.add_argument("--limit", type=int, default=0, help="Procesar solo los primeros N archivos")
    return p


if __name__ == "__main__":
    args = construir_parser().parse_args()
    sys.exit(procesar(args))
