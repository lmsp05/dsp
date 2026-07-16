"""
deteccion_picos.py
==================

Lee los resultados de la FFT generados por ``procesar_fft.py`` (el .txt tabular
o su version .npy) y detecta los picos de cada espectro replicando el
procedimiento del script de MATLAB ``x0302_nf_gph_transformations.m``.

--------------------------------------------------------------------------
PROCEDIMIENTO (equivalente al de MATLAB)
--------------------------------------------------------------------------
En el .m, la deteccion de picos "por espectro" es el bucle de *ridge detection*
que se aplica a cada fila de la cascada (cada RPM):

    [~,locs] = findpeaks(row, 'MinPeakProminence', max(row)*0.1, ...
                              'MinPeakDistance', 5);

Es decir: `scipy.signal.find_peaks` sobre la AMPLITUD LINEAL del espectro, con

  * prominencia minima = `max(amplitud) * FRACCION_PROMINENCIA`  (0.1 como en el
    bucle de crestas del .m), y
  * distancia minima entre picos = `DIST_BINS` bins (5 como en el .m).

(El .m convierte la cascada de dB a lineal antes de detectar; nuestra amplitud
ya esta en unidades lineales, asi que se usa directamente.)

--------------------------------------------------------------------------
ELIMINACION DE SINCRONOS Y ARMONICOS (1X, 2X, ... NX)
--------------------------------------------------------------------------
Antes de detectar los picos se quitan del espectro las lineas SINCRONAS (el 1X
de giro) y sus ARMONICOS (2X, 3X, ...), para que la deteccion quede solo con el
contenido no sincrono (frecuencias naturales, whirl de aceite, etc.). El
procedimiento:

  1. Se estima la frecuencia de giro real f1: se parte de rpm/60 y se refina
     tomando la frecuencia de mayor amplitud en una banda +-`band_1x` alrededor
     (el 1X real puede no caer exacto en rpm/60 nominal).
  2. Se marcan como sincronas las lineas cuyo ORDEN (freq/f1) este a menos de
     `tol_orden` de un entero n = 1..`n_armonicos`.
  3. En esas bandas se sustituye la amplitud por la linea base interpolada del
     entorno no sincrono (notch), de modo que el pico sincrono desaparece sin
     dejar un hueco artificial.

Ventaja adicional: como la prominencia es relativa al maximo del espectro, al
quitar el 1X (que suele dominar con desbalance) el umbral baja y afloran picos
naturales debiles que antes quedaban por debajo.

Para cada espectro (una combinacion rep/iso/dsb/rpm/sensor) se detectan los
picos ya sin sincronos, se ordenan por frecuencia ascendente y se guardan como
omega1, omega2, ...

--------------------------------------------------------------------------
SALIDAS (en la misma carpeta que la entrada, o la indicada con --outdir)
--------------------------------------------------------------------------
  * picos_detectados.txt : tabla con columnas
        rep  iso  dsb  rpm  sensor  omega1  omega2  ...  omegaN
    (una fila por archivo y sensor; celdas vacias donde no hay mas picos).
  * picos_scatter.png     : dispersion frecuencia (Y) vs velocidad (X), con
        color = viscosidad ISO y forma = nivel de desbalance dsb.
  * picos_scatter_<sensor>.png : la misma dispersion, un archivo por sensor.

Uso:
    python deteccion_picos.py --entrada resultados_fft.txt
    python deteccion_picos.py --entrada resultados_fft.npy --outdir figuras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# Parametros del procedimiento (valores del bucle de crestas del .m)
FRACCION_PROMINENCIA = 0.1   # MinPeakProminence = max(espectro) * fraccion
DIST_BINS = 5                # MinPeakDistance en numero de bins

# Eliminacion de sincronos/armonicos
N_ARMONICOS = 10             # ordenes a eliminar: 1X ... NX
TOL_ORDEN = 0.05             # una linea es sincrona si |freq/f1 - entero| <= tol
BAND_1X = 0.10               # banda relativa para refinar el 1X real cerca de rpm/60
ANCHO_BINS_NOTCH = 2.5       # medio ancho del notch en bins (cubre el lobulo del pico)

# Codificacion visual del diagrama de dispersion
COLORES_ISO = {32: "#1f77b4", 46: "#ff7f0e", 68: "#2ca02c"}
FORMAS_DSB = {1: "o", 2: "s", 3: "^"}


# ============================================================
# LECTURA DE LOS RESULTADOS DE procesar_fft.py
# ============================================================

def cargar_espectros(path: Path) -> np.ndarray:
    """Carga el .txt tabular o el .npy y devuelve un array estructurado con,
    al menos, los campos rep, iso, dsb, rpm_nominal, sensor, frecuencia_Hz,
    amplitud."""
    if path.suffix.lower() == ".npy":
        return np.load(path)

    with path.open("r", encoding="utf-8") as fh:
        utiles = [ln.rstrip("\n") for ln in fh
                  if ln.strip() and not ln.lstrip().startswith("#")]
    if len(utiles) < 2:
        raise ValueError("El archivo no contiene filas de datos.")
    columnas = utiles[0].split("\t")
    arr = np.genfromtxt(
        utiles[1:], delimiter="\t", names=columnas, dtype=None, encoding="utf-8")
    return np.atleast_1d(arr)


# ============================================================
# DETECCION DE PICOS POR ESPECTRO (procedimiento del .m)
# ============================================================

def _f1_giro(f: np.ndarray, a: np.ndarray, rpm: float, band: float) -> float | None:
    """Frecuencia de giro real (1X): rpm/60 refinado al maximo local cercano."""
    if not rpm or rpm <= 0:
        return None
    f1 = rpm / 60.0
    m = (f >= f1 * (1 - band)) & (f <= f1 * (1 + band))
    if m.any():
        f1 = float(f[m][np.argmax(a[m])])
    return f1


def _mascara_sincrona(f: np.ndarray, f1: float | None, n_max: int,
                      tol_orden: float, ancho_bins: float) -> np.ndarray:
    """Marca las lineas que caen en el 1X y sus armonicos (1..n_max).

    El ancho del notch alrededor de cada orden n*f1 es el MAYOR entre:
      * tol_orden * f1  (tolerancia en orden), y
      * ancho_bins * df (varios bins, para cubrir el lobulo del pico y sus
        hombros y no dejar residuos que luego se detecten como picos falsos).
    """
    if not f1 or f1 <= 0:
        return np.zeros(len(f), dtype=bool)
    df = float(np.median(np.diff(f))) if len(f) > 1 else 0.0
    half = max(tol_orden * f1, ancho_bins * df)
    mask = np.zeros(len(f), dtype=bool)
    for n in range(1, n_max + 1):
        centro = n * f1
        if centro - half > f[-1]:
            break
        mask |= np.abs(f - centro) <= half
    return mask


def analizar_espectro(frecuencia: np.ndarray, amplitud: np.ndarray,
                      fraccion: float, dist_bins: int,
                      fmin: float | None, fmax: float | None,
                      rpm: float | None = None, quitar_armonicos: bool = True,
                      n_armonicos: int = N_ARMONICOS, tol_orden: float = TOL_ORDEN,
                      band_1x: float = BAND_1X, ancho_bins: float = ANCHO_BINS_NOTCH) -> dict:
    """Ejecuta el pipeline completo sobre UN espectro y devuelve todos los pasos.

    Devuelve un dict con: f, a (espectro original tras fmin/fmax), a_det (espectro
    sin sincronos), sinc (mascara de bandas sincronas), f1 (giro estimado),
    prominencia (umbral), dist_bins, picos_idx, picos_freq, picos_amp.
    """
    orden = np.argsort(frecuencia)
    f = np.asarray(frecuencia)[orden]
    a = np.asarray(amplitud)[orden]

    if fmin is not None or fmax is not None:
        m = np.ones(len(f), dtype=bool)
        if fmin is not None:
            m &= f >= fmin
        if fmax is not None:
            m &= f <= fmax
        f, a = f[m], a[m]

    vacio = {"f": f, "a": a, "a_det": a.copy(),
             "sinc": np.zeros(len(f), dtype=bool), "f1": None,
             "prominencia": 0.0, "dist_bins": dist_bins,
             "picos_idx": np.array([], dtype=int),
             "picos_freq": np.array([]), "picos_amp": np.array([])}
    if a.size == 0 or a.max() <= 0:
        return vacio

    # --- Elimina el 1X y sus armonicos del espectro (notch por interpolacion) ---
    sinc = np.zeros(len(f), dtype=bool)
    a_det = a.copy()
    f1 = None
    if quitar_armonicos:
        f1 = _f1_giro(f, a, rpm, band_1x)
        sinc = _mascara_sincrona(f, f1, n_armonicos, tol_orden, ancho_bins)
        if sinc.any() and (~sinc).any():
            a_det[sinc] = np.interp(f[sinc], f[~sinc], a[~sinc])

    # --- Deteccion de picos sobre el espectro ya sin sincronos ---
    prominencia = a_det.max() * fraccion
    picos, _ = find_peaks(a_det, prominence=prominencia, distance=max(1, dist_bins))
    # descarta cualquier pico residual que caiga en una banda sincrona
    picos = np.array([p for p in picos if not sinc[p]], dtype=int)
    return {"f": f, "a": a, "a_det": a_det, "sinc": sinc, "f1": f1,
            "prominencia": prominencia, "dist_bins": dist_bins,
            "picos_idx": picos, "picos_freq": f[picos], "picos_amp": a_det[picos]}


def detectar_picos(frecuencia: np.ndarray, amplitud: np.ndarray,
                   fraccion: float, dist_bins: int,
                   fmin: float | None, fmax: float | None,
                   rpm: float | None = None, quitar_armonicos: bool = True,
                   n_armonicos: int = N_ARMONICOS, tol_orden: float = TOL_ORDEN,
                   band_1x: float = BAND_1X, ancho_bins: float = ANCHO_BINS_NOTCH) -> np.ndarray:
    """Devuelve las frecuencias de los picos (sin sincronos), ordenadas asc."""
    return analizar_espectro(
        frecuencia, amplitud, fraccion, dist_bins, fmin, fmax, rpm,
        quitar_armonicos, n_armonicos, tol_orden, band_1x, ancho_bins)["picos_freq"]


# ============================================================
# AGRUPAMIENTO POR (rep, iso, dsb, rpm, sensor)
# ============================================================

def procesar(arr: np.ndarray, args) -> list[dict]:
    rep = arr["rep"].astype(int)
    iso = arr["iso"].astype(int)
    dsb = arr["dsb"].astype(int)
    rpm = arr["rpm_nominal"].astype(float)
    sensor = np.asarray(arr["sensor"]).astype(str)
    freq = arr["frecuencia_Hz"].astype(float)
    amp = arr["amplitud"].astype(float)
    # RPM real medida (para estimar el 1X); si no esta, se usa la nominal.
    tiene_medida = "rpm_medida" in arr.dtype.names
    rpm_medida = arr["rpm_medida"].astype(float) if tiene_medida else rpm

    claves = list(zip(rep, iso, dsb, rpm, sensor))
    vistos: dict[tuple, None] = {}
    for c in claves:
        vistos.setdefault(c, None)

    filas = []
    for clave in vistos:
        m = np.array([c == clave for c in claves], dtype=bool)
        rpm_f1 = float(np.median(rpm_medida[m])) if tiene_medida else clave[3]
        omegas = detectar_picos(
            freq[m], amp[m], args.fraccion, args.dist_bins, args.fmin, args.fmax,
            rpm=rpm_f1, quitar_armonicos=not args.conservar_armonicos,
            n_armonicos=args.n_armonicos, tol_orden=args.tol_orden,
            band_1x=args.band_1x, ancho_bins=args.ancho_bins)
        r, i_, d, v, s = clave
        filas.append({
            "rep": int(r), "iso": int(i_), "dsb": int(d), "rpm": float(v),
            "sensor": str(s), "omegas": np.sort(omegas),
        })
    filas.sort(key=lambda x: (x["rep"], x["iso"], x["dsb"], x["rpm"], x["sensor"]))
    return filas


# ============================================================
# TABLA DE RESULTADOS
# ============================================================

def escribir_tabla(filas: list[dict], salida: Path) -> int:
    n_max = max((len(f["omegas"]) for f in filas), default=0)
    cols = ["rep", "iso", "dsb", "rpm", "sensor"] + [f"omega{k+1}" for k in range(n_max)]
    lineas = ["\t".join(cols)]
    for f in filas:
        celdas = [str(f["rep"]), str(f["iso"]), str(f["dsb"]),
                  f"{f['rpm']:g}", f["sensor"]]
        celdas += [f"{w:.4f}" for w in f["omegas"]]
        celdas += [""] * (n_max - len(f["omegas"]))
        lineas.append("\t".join(celdas))
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return n_max


# ============================================================
# DIAGRAMA DE DISPERSION frecuencia vs velocidad
# ============================================================

def _dibujar_scatter(filas, ax, titulo, con_ordenes=True):
    isos = sorted({f["iso"] for f in filas})
    dsbs = sorted({f["dsb"] for f in filas})
    rpms = sorted({f["rpm"] for f in filas})

    for f in filas:
        if len(f["omegas"]) == 0:
            continue
        color = COLORES_ISO.get(f["iso"], "gray")
        marker = FORMAS_DSB.get(f["dsb"], "x")
        ax.scatter([f["rpm"]] * len(f["omegas"]), f["omegas"],
                   c=color, marker=marker, s=45, alpha=0.75,
                   edgecolors="k", linewidths=0.4)

    # lineas de orden 1X, 2X, 3X como referencia (freq = n * rpm/60)
    if con_ordenes and rpms:
        xr = np.array([min(rpms), max(rpms)], dtype=float)
        for n in (1, 2, 3):
            ax.plot(xr, n * xr / 60.0, ls="--", lw=0.8, color="gray", alpha=0.5)
            ax.text(xr[-1], n * xr[-1] / 60.0, f" {n}X", fontsize=7,
                    color="gray", va="center")

    ax.set_xlabel("Velocidad (RPM)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)

    # dos leyendas: color = ISO, forma = dsb
    leg_iso = [Line2D([0], [0], marker="o", ls="", mfc=COLORES_ISO.get(i, "gray"),
                      mec="k", ms=8, label=f"ISO {i}") for i in isos]
    leg_dsb = [Line2D([0], [0], marker=FORMAS_DSB.get(d, "x"), ls="", mfc="gray",
                      mec="k", ms=8, label=f"dsb {d}") for d in dsbs]
    l1 = ax.legend(handles=leg_iso, title="Viscosidad", loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=leg_dsb, title="Desbalance", loc="upper right", fontsize=8)


def graficar(filas: list[dict], outdir: Path):
    # 1) diagrama combinado (todos los sensores)
    fig, ax = plt.subplots(figsize=(11, 7))
    _dibujar_scatter(filas, ax, "Frecuencias detectadas vs velocidad (todos los sensores)")
    png = outdir / "picos_scatter.png"
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generados = [png]

    # 2) un diagrama por sensor
    sensores = sorted({f["sensor"] for f in filas})
    for s in sensores:
        sub = [f for f in filas if f["sensor"] == s]
        fig, ax = plt.subplots(figsize=(11, 7))
        _dibujar_scatter(sub, ax, f"Frecuencias detectadas vs velocidad — sensor {s}")
        png = outdir / f"picos_scatter_{s}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generados.append(png)
    return generados


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main(args) -> int:
    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_file():
        print(f"ERROR: no existe la entrada {entrada}")
        return 1
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else entrada.parent
    outdir.mkdir(parents=True, exist_ok=True)

    arr = cargar_espectros(entrada)
    faltan = [c for c in ("rep", "iso", "dsb", "rpm_nominal", "sensor",
                          "frecuencia_Hz", "amplitud") if c not in arr.dtype.names]
    if faltan:
        print(f"ERROR: a la entrada le faltan columnas: {faltan}")
        return 1

    filas = procesar(arr, args)
    if not filas:
        print("No se pudo formar ningun espectro.")
        return 1

    tabla = outdir / "picos_detectados.txt"
    n_max = escribir_tabla(filas, tabla)
    generados = graficar(filas, outdir)

    total_picos = sum(len(f["omegas"]) for f in filas)
    print(f"Espectros procesados: {len(filas)}")
    print(f"Picos detectados en total: {total_picos} (max por espectro: {n_max})")
    print(f"Tabla: {tabla}")
    for g in generados:
        print(f"Figura: {g}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deteccion de picos sobre los espectros de procesar_fft.py "
                    "(procedimiento de x0302_nf_gph_transformations.m) + tabla y "
                    "diagrama frecuencia-vs-velocidad.")
    p.add_argument("--entrada", default="resultados_fft.txt",
                   help="Archivo .txt o .npy generado por procesar_fft.py")
    p.add_argument("--outdir", default="",
                   help="Carpeta de salida (def: la misma de la entrada)")
    p.add_argument("--fraccion", type=float, default=FRACCION_PROMINENCIA,
                   help="MinPeakProminence = max(espectro) * fraccion (def: 0.1)")
    p.add_argument("--dist-bins", dest="dist_bins", type=int, default=DIST_BINS,
                   help="Distancia minima entre picos, en bins (def: 5)")
    p.add_argument("--fmin", type=float, default=None, help="Frecuencia minima a considerar [Hz]")
    p.add_argument("--fmax", type=float, default=None, help="Frecuencia maxima a considerar [Hz]")
    # Eliminacion de sincronos/armonicos
    p.add_argument("--conservar-armonicos", dest="conservar_armonicos", action="store_true",
                   help="NO eliminar el 1X ni sus armonicos (por defecto SI se eliminan)")
    p.add_argument("--n-armonicos", dest="n_armonicos", type=int, default=N_ARMONICOS,
                   help="Numero de ordenes a eliminar: 1X..NX (def: 10)")
    p.add_argument("--tol-orden", dest="tol_orden", type=float, default=TOL_ORDEN,
                   help="Tolerancia de orden para marcar una linea como sincrona (def: 0.05)")
    p.add_argument("--band-1x", dest="band_1x", type=float, default=BAND_1X,
                   help="Banda relativa para refinar el 1X cerca de rpm/60 (def: 0.10)")
    p.add_argument("--ancho-bins", dest="ancho_bins", type=float, default=ANCHO_BINS_NOTCH,
                   help="Medio ancho del notch de cada armonico, en bins (def: 4)")
    return p


if __name__ == "__main__":
    sys.exit(main(construir_parser().parse_args()))
