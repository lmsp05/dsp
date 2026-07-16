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
ELIMINACION DE SINCRONOS Y ARMONICOS (1X, 2X, ... NX)  [metodo por hombros]
--------------------------------------------------------------------------
Antes de detectar los picos se quitan del espectro las lineas SINCRONAS (el 1X
de giro) y sus ARMONICOS (2X, 3X, ...), para que la deteccion quede solo con el
contenido no sincrono (frecuencias naturales, whirl de aceite, etc.). El
procedimiento (eliminacion adaptativa por hombros):

  1. Se estima la frecuencia de giro real f1: se parte de rpm/60 y se toma el
     PICO mas alto en una banda +-`band_1x` alrededor. Usar el pico mas alto y
     una banda amplia hace la estimacion robusta al sesgo de la RPM medida (la
     media puede caer un bin o mas por debajo/encima del 1X real).
  2. Para cada orden n = 1..`n_armonicos` se busca el PICO REAL mas cercano a la
     posicion esperada n*f1, dentro de una ventana proporcional
     (win = max(`tol_orden`*n*f1, (piso+1)*df)). Buscar por orden y quedarse con
     el pico mas cercano tolera el error de f1 y que los armonicos no caigan
     exactos en n*f1 (anarmonicidad + cuantizacion de bins).
  3. Para cada pico sincrono se detectan sus HOMBROS: se desciende a izquierda y
     derecha hasta el primer minimo local. Se elimina el pico COMPLETO en ese
     intervalo [hombro_izq, hombro_der], sustituyendo la amplitud por la linea
     base interpolada del entorno no sincrono. (`ancho_bins` fija un piso minimo
     de medio ancho por si la punta es plana o ruidosa.)

Ventaja frente al notch de ancho fijo: el ancho lo define el propio pico, asi que
solo se elimina donde REALMENTE hay contenido sincrono. Una frecuencia natural
cercana a un orden (pero separada por un valle) NO se elimina, porque no es el
pico cuya punta cae en el margen y queda fuera de sus hombros.

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
TOL_ORDEN = 0.05             # un pico es sincrono si |punta/f1 - entero| <= tol
BAND_1X = 0.20               # banda relativa para buscar el 1X real cerca de rpm/60
ANCHO_BINS_NOTCH = 1.0       # piso minimo de medio ancho, en bins (por punta plana)

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
# FILTRADO POR NIVELES DE LAS PROPIEDADES (rep, iso, dsb, rpm, sensor)
# ============================================================

def parse_niveles(texto: str | None, tipo):
    """Convierte '1,3' -> {1, 3} con el tipo dado. Vacio/None -> None (= cualquiera)."""
    if not texto:
        return None
    vals = set()
    for tok in str(texto).replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            vals.add(tipo(tok))
    return vals or None


def filtrar_condiciones(arr: np.ndarray, reps=None, isos=None, dsbs=None,
                        rpms=None, sensores=None) -> np.ndarray:
    """Devuelve las filas cuya combinacion de niveles pasa el filtro.

    Cada argumento es un conjunto de niveles admitidos para esa propiedad, o None
    para admitir cualquier valor. El resultado son las filas que cumplen TODAS las
    restricciones a la vez (combinacion de los niveles especificados).
    """
    m = np.ones(len(arr), dtype=bool)
    if reps:
        m &= np.isin(arr["rep"].astype(int), list(reps))
    if isos:
        m &= np.isin(arr["iso"].astype(int), list(isos))
    if dsbs:
        m &= np.isin(arr["dsb"].astype(int), list(dsbs))
    if rpms:
        m &= np.isin(np.round(arr["rpm_nominal"].astype(float), 6),
                     [round(float(x), 6) for x in rpms])
    if sensores:
        m &= np.isin(np.asarray(arr["sensor"]).astype(str), list(sensores))
    return arr[m]


# ============================================================
# DETECCION DE PICOS POR ESPECTRO (procedimiento del .m)
# ============================================================

def _interp_pico(f: np.ndarray, a: np.ndarray, p: int) -> float:
    """Frecuencia del pico p con interpolacion parabolica (precision sub-bin)."""
    if 0 < p < len(a) - 1:
        y0, y1, y2 = a[p - 1], a[p], a[p + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            delta = max(-0.5, min(0.5, delta))
            return float(f[p] + delta * (f[p + 1] - f[p]))
    return float(f[p])


def _f1_giro(f: np.ndarray, a: np.ndarray, rpm: float, band: float) -> float | None:
    """Frecuencia de giro real (1X), con precision sub-bin.

    Parte de rpm/60 como prior y toma el PICO mas alto en una banda +-band
    alrededor; su posicion se afina por interpolacion parabolica. Usar el pico
    mas alto, una banda amplia y la interpolacion sub-bin hace robusta la
    estimacion al sesgo de la RPM medida y evita propagar error a los armonicos
    altos (n*f1 se mantiene preciso).
    """
    if not rpm or rpm <= 0:
        return None
    f0 = rpm / 60.0
    m = (f >= f0 * (1 - band)) & (f <= f0 * (1 + band))
    if not m.any():
        return f0
    picos, _ = find_peaks(a)
    picos_banda = [p for p in picos if m[p]]
    if picos_banda:
        p = max(picos_banda, key=lambda p: a[p])   # pico mas alto de la banda
        return _interp_pico(f, a, p)
    idx = np.where(m)[0]
    return _interp_pico(f, a, int(idx[np.argmax(a[idx])]))


def _hombros(a: np.ndarray, p: int) -> tuple[int, int]:
    """Extremos del pico en el indice p: se desciende a cada lado mientras la
    amplitud no vuelva a subir, hasta el primer minimo local (el hombro)."""
    n = len(a)
    l = p
    while l - 1 >= 0 and a[l - 1] <= a[l]:
        l -= 1
    r = p
    while r + 1 < n and a[r + 1] <= a[r]:
        r += 1
    return l, r


def _mascara_sincrona(f: np.ndarray, a: np.ndarray, f1: float | None, n_max: int,
                      tol_orden: float, ancho_bins: float):
    """Marca para eliminacion los picos SINCRONOS por el metodo de hombros.

    Para cada orden n = 1..n_max se busca el PICO REAL mas cercano a la posicion
    esperada n*f1, dentro de una ventana MODESTA y constante
        win = max(tol_orden * f1, 1.5 * df).
    Si la punta de ese pico cae dentro de la ventana, se detectan sus HOMBROS
    (primer minimo local a cada lado) y se marca TODO el intervalo del pico.
    `ancho_bins` impone un piso minimo de medio ancho por si la punta es plana.

    Como f1 se estima con precision sub-bin, n*f1 queda preciso en todos los
    ordenes, asi que basta una ventana de ~1-2 bins (cuantizacion + leakage) para
    capturar el armonico sin tragarse frecuencias naturales cercanas a un orden.

    Devuelve (mask, intervalos, picos) donde:
      * mask       : bool array de las lineas a eliminar,
      * intervalos : lista de (idx_hombro_izq, idx_punta, idx_hombro_der, orden_n),
      * picos      : indices de las puntas sincronas.
    """
    mask = np.zeros(len(f), dtype=bool)
    intervalos: list[tuple[int, int, int, int]] = []
    picos_sinc: list[int] = []
    if not f1 or f1 <= 0 or len(f) < 3:
        return mask, intervalos, picos_sinc

    df = float(np.median(np.diff(f)))
    piso = int(max(0, round(ancho_bins)))
    picos, _ = find_peaks(a)
    if len(picos) == 0:
        return mask, intervalos, picos_sinc
    fpicos = f[picos]

    usados: set[int] = set()
    for n in range(1, n_max + 1):
        centro = n * f1
        if centro - df > f[-1]:
            break
        win = max(tol_orden * f1, 1.5 * df)
        d = np.abs(fpicos - centro)
        j = int(np.argmin(d))
        if d[j] > win:
            continue
        p = int(picos[j])
        if p in usados:
            continue
        usados.add(p)
        l, r = _hombros(a, p)
        l = max(0, min(l, p - piso))
        r = min(len(f) - 1, max(r, p + piso))
        mask[l:r + 1] = True
        intervalos.append((l, p, r, n))
        picos_sinc.append(p)
    return mask, intervalos, picos_sinc


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
             "sinc": np.zeros(len(f), dtype=bool), "sinc_intervalos": [],
             "sinc_picos": [], "f1": None,
             "prominencia": 0.0, "dist_bins": dist_bins,
             "picos_idx": np.array([], dtype=int),
             "picos_freq": np.array([]), "picos_amp": np.array([])}
    if a.size == 0 or a.max() <= 0:
        return vacio

    # --- Elimina el 1X y sus armonicos por el metodo de hombros (interpolacion) ---
    sinc = np.zeros(len(f), dtype=bool)
    intervalos: list = []
    picos_sinc: list = []
    a_det = a.copy()
    f1 = None
    if quitar_armonicos:
        f1 = _f1_giro(f, a, rpm, band_1x)
        sinc, intervalos, picos_sinc = _mascara_sincrona(
            f, a, f1, n_armonicos, tol_orden, ancho_bins)
        if sinc.any() and (~sinc).any():
            a_det[sinc] = np.interp(f[sinc], f[~sinc], a[~sinc])

    # --- Deteccion de picos sobre el espectro ya sin sincronos ---
    prominencia = a_det.max() * fraccion
    picos, _ = find_peaks(a_det, prominence=prominencia, distance=max(1, dist_bins))
    # descarta cualquier pico residual que caiga en una banda sincrona
    picos = np.array([p for p in picos if not sinc[p]], dtype=int)
    return {"f": f, "a": a, "a_det": a_det, "sinc": sinc,
            "sinc_intervalos": intervalos, "sinc_picos": picos_sinc, "f1": f1,
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

    ymax_data = 0.0
    for f in filas:
        if len(f["omegas"]) == 0:
            continue
        color = COLORES_ISO.get(f["iso"], "gray")
        marker = FORMAS_DSB.get(f["dsb"], "x")
        ax.scatter([f["rpm"]] * len(f["omegas"]), f["omegas"],
                   c=color, marker=marker, s=45, alpha=0.75,
                   edgecolors="k", linewidths=0.4)
        ymax_data = max(ymax_data, float(np.max(f["omegas"])))

    # fija la escala Y segun los datos, para que las lineas de orden no la deformen
    if ymax_data > 0:
        ax.set_ylim(0, ymax_data * 1.08)
    ylim_top = ax.get_ylim()[1]

    # lineas de orden 1X..20X como referencia (freq = n * rpm/60)
    if con_ordenes and rpms:
        xr = np.array([min(rpms), max(rpms)], dtype=float)
        for n in range(1, 21):
            y = n * xr / 60.0
            ax.plot(xr, y, ls="--", lw=0.7, color="gray", alpha=0.45)
            if y[-1] <= ylim_top:   # etiqueta solo si la linea entra en la vista
                ax.text(xr[-1], y[-1], f" {n}X", fontsize=6.5,
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

    # filtro por niveles: solo las combinaciones de los niveles indicados
    arr = filtrar_condiciones(
        arr, parse_niveles(args.rep, int), parse_niveles(args.iso, int),
        parse_niveles(args.dsb, int), parse_niveles(args.rpm, float),
        parse_niveles(args.sensor, str))
    if len(arr) == 0:
        print("Ninguna condicion pasa el filtro indicado (rep/iso/dsb/rpm/sensor).")
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
    # filtros por niveles (listas separadas por comas; vacio = cualquiera)
    p.add_argument("--rep", default="", help="Niveles de repeticion, ej. 1,3 (vacio = todos)")
    p.add_argument("--iso", default="", help="Niveles de viscosidad ISO, ej. 32,68 (vacio = todos)")
    p.add_argument("--dsb", default="", help="Niveles de desbalance, ej. 1,2 (vacio = todos)")
    p.add_argument("--rpm", default="", help="RPM nominales, ej. 600,900 (vacio = todas)")
    p.add_argument("--sensor", default="", help="Sensores, ej. P1.Y,P2.X (vacio = todos)")
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
                   help="Margen de orden para que la punta de un pico sea sincrona (def: 0.05)")
    p.add_argument("--band-1x", dest="band_1x", type=float, default=BAND_1X,
                   help="Banda relativa para refinar el 1X cerca de rpm/60 (def: 0.10)")
    p.add_argument("--ancho-bins", dest="ancho_bins", type=float, default=ANCHO_BINS_NOTCH,
                   help="Piso minimo de medio ancho del recorte, en bins (def: 1)")
    return p


if __name__ == "__main__":
    sys.exit(main(construir_parser().parse_args()))
