"""
ssi.py
======

Identificación estocástica de subespacios basada en referencias
(*Reference-based Stochastic Subspace Identification*, SSI) para OMA
automatizado sobre maquinaria rotativa.

Este módulo implementa dos ideas complementarias de la bibliografía:

1. **Peeters & De Roeck (2000)** — SSI basado en covarianzas y referencias
   (*SSI-COV/ref*). En lugar de trabajar con todas las salidas, se eligen
   unos pocos canales de **referencia** (sondas con buena relación
   señal/ruido) y se construye la matriz bloque-Toeplitz de covarianzas
   ``T_{1|p}`` entre todas las salidas y las referencias. Su descomposición
   en valores singulares (SVD) proporciona la matriz de observabilidad, de la
   que se extraen las matrices de estado ``A`` y de salida ``C`` mediante la
   propiedad de invariancia por desplazamiento. Los autovalores de ``A`` dan
   los polos del sistema: frecuencia natural, amortiguamiento y forma modal.
   Repetir el proceso para un rango de órdenes del modelo produce el
   **diagrama de estabilización**.

2. **Dreher et al. (2023)** — automatización del diagrama de estabilización
   mediante *clustering*. Los polos de todos los órdenes se depuran con
   criterios físicos y de estabilidad (desviación de frecuencia, de
   amortiguamiento y MAC entre órdenes consecutivos) y los polos estables se
   agrupan por *clustering* jerárquico. Cada grupo suficientemente poblado es
   un **modo físico**; los polos matemáticos/espurios quedan en grupos
   dispersos que se descartan. Para maquinaria rotativa, los polos asociados a
   los armónicos de la velocidad de giro (1X, 2X, …) se marcan aparte.

El módulo es autocontenido (solo NumPy/SciPy) y consume las mismas
estructuras que el resto del proyecto (``MeasurementRecord`` no es necesario:
trabaja directamente con las señales multicanal leídas por ``io_utils``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import decimate
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class Pole:
    """Un polo (modo candidato) identificado en un orden concreto del modelo."""

    order: int              #: Orden del modelo en el que se identificó.
    freq: float             #: Frecuencia natural [Hz].
    zeta: float             #: Razón de amortiguamiento (0-1).
    eigval: complex         #: Autovalor continuo del sistema [rad/s].
    shape: np.ndarray       #: Forma modal compleja en los sensores de salida.
    label: str = "new"      #: Etiqueta de estabilidad respecto al orden previo.


@dataclass
class StabilizationResult:
    """Resultado completo de la estabilización de un ensayo."""

    fs: float                       #: Frecuencia de muestreo usada (tras diezmar).
    orders: list[int]               #: Órdenes del modelo evaluados.
    poles: list[Pole]               #: Todos los polos físicos de todos los órdenes.
    output_sensors: list[str]       #: Canales de salida usados.
    ref_freq: np.ndarray = field(   #: Espectro de referencia (fondo del diagrama).
        default_factory=lambda: np.empty(0))
    ref_spectrum: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def stable_poles(self) -> list[Pole]:
        """Polos etiquetados como completamente estables."""
        return [p for p in self.poles if p.label == "stable"]


@dataclass
class ModeCluster:
    """Modo físico obtenido tras agrupar los polos estables (Dreher et al.)."""

    freq_mean: float        #: Frecuencia media del grupo [Hz].
    freq_std: float         #: Desviación estándar de la frecuencia [Hz].
    zeta_mean: float        #: Amortiguamiento medio (0-1).
    zeta_std: float         #: Desviación estándar del amortiguamiento.
    n_poles: int            #: Número de polos del grupo.
    n_orders: int           #: Número de órdenes distintos representados.
    shape: np.ndarray       #: Forma modal representativa (compleja, normalizada).
    order_min: int          #: Orden mínimo en el que aparece.
    order_max: int          #: Orden máximo en el que aparece.
    is_harmonic: bool = False  #: True si coincide con un armónico de la rotación.
    harmonic_order: int = 0    #: Orden nX del armónico (0 si no es armónico).


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def mac(phi1: np.ndarray, phi2: np.ndarray) -> float:
    """Modal Assurance Criterion entre dos formas modales complejas.

    ``MAC = |phi1^H phi2|^2 / ((phi1^H phi1)(phi2^H phi2))``. Vale 1 cuando
    las formas son colineales (mismo modo) y 0 cuando son ortogonales.
    """
    a = np.asarray(phi1).ravel()
    b = np.asarray(phi2).ravel()
    num = np.abs(np.vdot(a, b)) ** 2
    den = np.real(np.vdot(a, a) * np.vdot(b, b))
    if den <= 0:
        return 0.0
    return float(num / den)


def build_output_matrix(
    signals: dict[str, np.ndarray],
    output_sensors: list[str],
) -> np.ndarray:
    """Apila las señales de salida en una matriz ``Y`` de forma ``(l, N)``.

    Parameters
    ----------
    signals : dict
        Mapa ``sensor -> señal`` (todas de la misma medición).
    output_sensors : list of str
        Canales de salida en el orden deseado.

    Returns
    -------
    np.ndarray
        Matriz ``(l, N)`` con una fila por canal de salida.
    """
    missing = [s for s in output_sensors if s not in signals]
    if missing:
        raise ValueError(f"Faltan canales de salida en la medición: {missing}")
    n = min(len(signals[s]) for s in output_sensors)
    return np.vstack([np.asarray(signals[s][:n], dtype=float) for s in output_sensors])


def decimate_matrix(
    Y: np.ndarray,
    fs: float,
    factor: int,
) -> tuple[np.ndarray, float]:
    """Diezma (con filtro antialias) todas las filas de ``Y``.

    La SSI de modos estructurales por debajo de unos cientos de Hz no necesita
    la frecuencia de muestreo original (12 800 Hz); diezmar reduce el coste
    numérico y concentra la energía en la banda de interés. Se usa
    ``scipy.signal.decimate`` (filtro IIR Chebyshev de fase cero por defecto),
    aplicado por tramos si el factor es grande.

    Parameters
    ----------
    Y : np.ndarray
        Matriz de salidas ``(l, N)``.
    fs : float
        Frecuencia de muestreo original [Hz].
    factor : int
        Factor de diezmado entero (>= 1). Si es 1 no se hace nada.

    Returns
    -------
    tuple
        ``(Y_diezmada, fs_nueva)``.
    """
    factor = int(factor)
    if factor <= 1:
        return Y, fs
    Yd = Y
    remaining = factor
    # decimate recomienda factores <= 13 por llamada; se encadena si es mayor.
    while remaining > 1:
        step = min(remaining, 10) if remaining % 10 == 0 else min(remaining, 13)
        while remaining % step != 0:
            step -= 1
        Yd = decimate(Yd, step, axis=1, ftype="iir", zero_phase=True)
        remaining //= step
    return Yd, fs / factor


# ---------------------------------------------------------------------------
# Covarianzas y matriz bloque-Toeplitz (Peeters & De Roeck, 2000)
# ---------------------------------------------------------------------------

def block_covariances(
    Y: np.ndarray,
    Yref: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Covarianzas de salida basadas en referencias ``R_i`` (i = 0..max_lag).

    ``R_i = E[y_{k+i} y_ref_k^T]`` estimada como
    ``R_i = 1/(N-i) * Y[:, i:] @ Yref[:, :N-i]^T``  ->  matriz ``(l, r)``.

    Parameters
    ----------
    Y : np.ndarray
        Salidas ``(l, N)`` (media ya eliminada).
    Yref : np.ndarray
        Referencias ``(r, N)`` (subconjunto de filas de ``Y``).
    max_lag : int
        Máximo desfase temporal a calcular.

    Returns
    -------
    np.ndarray
        Tensor ``(max_lag+1, l, r)`` con ``R_0 .. R_{max_lag}``.
    """
    l, N = Y.shape
    r = Yref.shape[0]
    R = np.empty((max_lag + 1, l, r), dtype=float)
    for i in range(max_lag + 1):
        R[i] = Y[:, i:N] @ Yref[:, : N - i].T / (N - i)
    return R


def block_toeplitz(R: np.ndarray, p: int) -> np.ndarray:
    """Construye la matriz bloque-Toeplitz ``T_{1|p}`` (dim ``lp x rp``).

    ``T[a, b] = R_{p + a - b}`` con ``a`` (fila) y ``b`` (columna) de bloque en
    ``0..p-1``; los índices recorren ``R_1`` (esquina superior derecha) hasta
    ``R_{2p-1}`` (esquina inferior izquierda), tal como en Peeters & De Roeck.

    Parameters
    ----------
    R : np.ndarray
        Covarianzas ``(>=2p, l, r)`` (se necesitan ``R_1 .. R_{2p-1}``).
    p : int
        Número de bloques-fila (semilongitud del Toeplitz).

    Returns
    -------
    np.ndarray
        Matriz ``T`` de forma ``(l*p, r*p)``.
    """
    _, l, r = R.shape
    T = np.empty((l * p, r * p), dtype=float)
    for a in range(p):
        for b in range(p):
            T[a * l:(a + 1) * l, b * r:(b + 1) * r] = R[p + a - b]
    return T


# ---------------------------------------------------------------------------
# Extracción del espacio de estados y parámetros modales
# ---------------------------------------------------------------------------

def state_space_from_svd(
    U: np.ndarray,
    s: np.ndarray,
    l: int,
    p: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrae ``(A, C)`` de la SVD de la Toeplitz para un orden dado.

    La matriz de observabilidad truncada al orden ``n`` es
    ``O_p = U_n diag(sqrt(s_n))``. De ella:

    * ``C`` = primer bloque-fila de ``O_p`` (l filas).
    * ``A`` = solución mínimos cuadrados de ``O_p^{↑} A = O_p^{↓}`` (invariancia
      por desplazamiento), con ``O_p^{↑}`` sin el último bloque-fila y
      ``O_p^{↓}`` sin el primero.

    Parameters
    ----------
    U : np.ndarray
        Matriz ``U`` de la SVD de la Toeplitz ``(l*p, k)``.
    s : np.ndarray
        Valores singulares.
    l : int
        Número de canales de salida.
    p : int
        Número de bloques-fila del Toeplitz.
    order : int
        Orden del modelo (número de valores singulares a retener).

    Returns
    -------
    tuple
        ``(A, C)`` con ``A`` de ``(order, order)`` y ``C`` de ``(l, order)``.
    """
    Un = U[:, :order]
    Op = Un * np.sqrt(s[:order])[np.newaxis, :]
    C = Op[:l, :]
    Op_up = Op[: l * (p - 1), :]
    Op_down = Op[l:, :]
    A, *_ = np.linalg.lstsq(Op_up, Op_down, rcond=None)
    return A, C


def modal_parameters(
    A: np.ndarray,
    C: np.ndarray,
    fs: float,
    order: int,
    zeta_max: float,
    fmin: float,
    fmax: float,
) -> list[Pole]:
    """Convierte ``(A, C)`` en polos físicos (frecuencia, amortiguamiento, forma).

    Se diagonaliza ``A`` (autovalores discretos ``mu``); los polos continuos
    son ``lambda = ln(mu) * fs``. Para cada polo:
    ``f_n = |lambda| / (2 pi)``  y  ``zeta = -Re(lambda) / |lambda|``.
    La forma modal en los sensores es ``phi = C psi``.

    Se conservan solo los polos con parte imaginaria positiva (un polo por par
    conjugado), frecuencia dentro de ``[fmin, fmax]`` y amortiguamiento en
    ``(0, zeta_max)``.

    Returns
    -------
    list of Pole
        Polos físicos de este orden.
    """
    mu, psi = np.linalg.eig(A)
    lam = np.log(mu.astype(complex)) * fs
    poles: list[Pole] = []
    for k in range(len(mu)):
        lk = lam[k]
        if lk.imag <= 0:  # un representante por par conjugado
            continue
        wn = abs(lk)
        if wn == 0:
            continue
        fn = wn / (2.0 * np.pi)
        zeta = -lk.real / wn
        if not (fmin <= fn <= fmax):
            continue
        if not (0.0 < zeta < zeta_max):
            continue
        shape = C @ psi[:, k]
        poles.append(Pole(order=order, freq=float(fn), zeta=float(zeta),
                          eigval=complex(lk), shape=shape))
    poles.sort(key=lambda pp: pp.freq)
    return poles


# ---------------------------------------------------------------------------
# Estabilización: barrido de órdenes y etiquetado de estabilidad (Dreher)
# ---------------------------------------------------------------------------

def label_stability(
    poles_by_order: dict[int, list[Pole]],
    orders: list[int],
    df_tol: float,
    dz_tol: float,
    mac_tol: float,
) -> None:
    """Etiqueta cada polo comparándolo con el orden inmediatamente inferior.

    Para cada polo del orden ``n`` se busca en el orden anterior el polo más
    próximo en frecuencia y se asignan las etiquetas:

    * ``"stable"`` — frecuencia, amortiguamiento y MAC dentro de tolerancia.
    * ``"stable_f_d"`` — estable en frecuencia y amortiguamiento (MAC no).
    * ``"stable_f"`` — estable solo en frecuencia.
    * ``"new"`` — sin correspondencia estable (posible polo matemático).

    Modifica ``poles_by_order`` in situ (rellena ``Pole.label``).
    """
    for oi in range(1, len(orders)):
        prev = poles_by_order[orders[oi - 1]]
        curr = poles_by_order[orders[oi]]
        if not prev:
            continue
        prev_f = np.array([pp.freq for pp in prev])
        for pole in curr:
            j = int(np.argmin(np.abs(prev_f - pole.freq)))
            ref = prev[j]
            df = abs(pole.freq - ref.freq) / max(pole.freq, 1e-12)
            dz = abs(pole.zeta - ref.zeta) / max(pole.zeta, 1e-12)
            m = mac(pole.shape, ref.shape)
            if df <= df_tol and dz <= dz_tol and m >= mac_tol:
                pole.label = "stable"
            elif df <= df_tol and dz <= dz_tol:
                pole.label = "stable_f_d"
            elif df <= df_tol:
                pole.label = "stable_f"
            else:
                pole.label = "new"


def run_stabilization(
    signals: dict[str, np.ndarray],
    output_sensors: list[str],
    ref_sensors: list[str],
    fs: float,
    *,
    decimation_factor: int = 1,
    block_rows: int = 40,
    order_min: int = 2,
    order_max: int = 60,
    order_step: int = 2,
    zeta_max: float = 0.2,
    fmin: float = 2.0,
    fmax: float = 500.0,
    df_tol: float = 0.01,
    dz_tol: float = 0.05,
    mac_tol: float = 0.98,
    remove_mean: bool = True,
) -> StabilizationResult:
    """Ejecuta SSI-COV/ref y devuelve el diagrama de estabilización completo.

    Parameters
    ----------
    signals : dict
        Mapa ``sensor -> señal`` de una medición (todas las salidas y
        referencias deben estar presentes).
    output_sensors : list of str
        Canales de salida del modelo (p. ej. las 4 sondas de proximidad).
    ref_sensors : list of str
        Subconjunto de ``output_sensors`` usado como referencias.
    fs : float
        Frecuencia de muestreo original [Hz].
    decimation_factor : int
        Factor de diezmado antes de la SSI.
    block_rows : int
        Número de bloques-fila ``p`` del Toeplitz (define el orden máximo
        alcanzable ``l*p``).
    order_min, order_max, order_step : int
        Rango de órdenes del modelo a evaluar (pares).
    zeta_max : float
        Amortiguamiento máximo admisible para un polo físico.
    fmin, fmax : float
        Banda de frecuencias de interés [Hz].
    df_tol, dz_tol, mac_tol : float
        Tolerancias de estabilidad entre órdenes consecutivos.
    remove_mean : bool
        Restar la media de cada canal antes de calcular covarianzas.

    Returns
    -------
    StabilizationResult
        Órdenes, polos etiquetados y espectro de referencia de fondo.
    """
    Y = build_output_matrix(signals, output_sensors)
    Y, fs_used = decimate_matrix(Y, fs, decimation_factor)
    if remove_mean:
        Y = Y - Y.mean(axis=1, keepdims=True)

    l = Y.shape[0]
    ref_idx = [output_sensors.index(s) for s in ref_sensors]
    Yref = Y[ref_idx, :]

    p = int(block_rows)
    if l * p < order_max:
        logger.warning(
            "order_max=%d supera l*p=%d; se limita el orden máximo.",
            order_max, l * p,
        )
        order_max = l * p

    # Toeplitz de covarianzas y su SVD (una sola vez para todos los órdenes).
    R = block_covariances(Y, Yref, max_lag=2 * p)
    T = block_toeplitz(R, p)
    U, s, _ = np.linalg.svd(T, full_matrices=False)

    orders = list(range(order_min, order_max + 1, order_step))
    poles_by_order: dict[int, list[Pole]] = {}
    for n in orders:
        if n > len(s):
            break
        A, C = state_space_from_svd(U, s, l, p, n)
        poles_by_order[n] = modal_parameters(
            A, C, fs_used, n, zeta_max, fmin, fmax
        )
    orders = [n for n in orders if n in poles_by_order]

    label_stability(poles_by_order, orders, df_tol, dz_tol, mac_tol)

    all_poles = [pole for n in orders for pole in poles_by_order[n]]

    # Espectro de referencia de fondo: media de las auto-PSD de las salidas.
    ref_freq, ref_spec = _reference_spectrum(Y, fs_used, fmin, fmax)

    logger.debug("Estabilización: %d órdenes, %d polos (%d estables).",
                 len(orders), len(all_poles),
                 sum(1 for pp in all_poles if pp.label == "stable"))
    return StabilizationResult(
        fs=fs_used, orders=orders, poles=all_poles,
        output_sensors=list(output_sensors),
        ref_freq=ref_freq, ref_spectrum=ref_spec,
    )


def _reference_spectrum(
    Y: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """PSD media de las salidas, como fondo del diagrama de estabilización."""
    from scipy.signal import welch  # import local: solo aquí se necesita

    n = Y.shape[1]
    nperseg = min(4096, 2 ** int(np.floor(np.log2(max(n, 2)))))
    freq, pxx = welch(Y, fs=fs, nperseg=nperseg, axis=1)
    spec = pxx.mean(axis=0)
    mask = (freq >= fmin) & (freq <= fmax)
    return freq[mask], spec[mask]


# ---------------------------------------------------------------------------
# Clustering de polos estables -> modos físicos (Dreher et al., 2023)
# ---------------------------------------------------------------------------

def cluster_modes(
    poles: list[Pole],
    n_orders: int,
    *,
    freq_weight: float = 1.0,
    mac_weight: float = 1.0,
    distance_threshold: float = 0.05,
    min_cluster_fraction: float = 0.2,
    min_cluster_size: int = 3,
) -> list[ModeCluster]:
    """Agrupa los polos estables en modos físicos por *clustering* jerárquico.

    La distancia entre dos polos combina la diferencia relativa de frecuencia y
    la disimilitud de forma modal (Dreher et al.):

    ``d_ij = freq_weight * |f_i - f_j| / mean(f_i, f_j)
             + mac_weight * (1 - MAC(phi_i, phi_j))``

    Se aplica *clustering* aglomerativo (enlace medio) y se cortan los grupos a
    ``distance_threshold``. Solo se conservan los grupos que aparecen en una
    fracción suficiente de órdenes (columnas verticales del diagrama de
    estabilización); los polos matemáticos, dispersos, quedan en grupos
    pequeños y se descartan.

    Parameters
    ----------
    poles : list of Pole
        Polos estables (normalmente ``StabilizationResult.stable_poles``).
    n_orders : int
        Número total de órdenes evaluados (para la fracción mínima).
    freq_weight, mac_weight : float
        Pesos de la frecuencia y del MAC en la distancia.
    distance_threshold : float
        Umbral de corte del dendrograma.
    min_cluster_fraction : float
        Fracción mínima de órdenes distintos que debe cubrir un grupo.
    min_cluster_size : int
        Número mínimo absoluto de polos en un grupo.

    Returns
    -------
    list of ModeCluster
        Modos físicos ordenados por frecuencia creciente.
    """
    if len(poles) < max(2, min_cluster_size):
        return []

    m = len(poles)
    freqs = np.array([pp.freq for pp in poles])
    # Matriz de distancias condensada.
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            fmean = 0.5 * (freqs[i] + freqs[j])
            df = abs(freqs[i] - freqs[j]) / max(fmean, 1e-12)
            dmac = 1.0 - mac(poles[i].shape, poles[j].shape)
            D[i, j] = D[j, i] = freq_weight * df + mac_weight * dmac

    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    min_orders = max(min_cluster_size, int(np.ceil(min_cluster_fraction * n_orders)))
    clusters: list[ModeCluster] = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        group = [poles[k] for k in idx]
        orders_in = {pp.order for pp in group}
        if len(orders_in) < min_orders:
            continue
        gfreqs = np.array([pp.freq for pp in group])
        gzetas = np.array([pp.zeta for pp in group])
        # Forma representativa: la del polo más próximo a la mediana de frecuencia.
        rep = group[int(np.argmin(np.abs(gfreqs - np.median(gfreqs))))]
        shape = rep.shape / np.max(np.abs(rep.shape))
        clusters.append(ModeCluster(
            freq_mean=float(gfreqs.mean()),
            freq_std=float(gfreqs.std(ddof=0)),
            zeta_mean=float(gzetas.mean()),
            zeta_std=float(gzetas.std(ddof=0)),
            n_poles=len(group),
            n_orders=len(orders_in),
            shape=shape,
            order_min=min(orders_in),
            order_max=max(orders_in),
        ))
    clusters.sort(key=lambda c: c.freq_mean)
    return clusters


def flag_harmonics(
    clusters: list[ModeCluster],
    f_rotation: float,
    max_order: int,
    tol_hz: float,
    tol_rel: float,
) -> None:
    """Marca los modos que coinciden con un armónico nX de la rotación.

    En maquinaria rotativa los armónicos de la velocidad de giro aparecen en el
    diagrama de estabilización como columnas de amortiguamiento casi nulo, muy
    parecidas a modos estructurales. Se marcan (``is_harmonic``) para poder
    excluirlos de las frecuencias naturales sin borrarlos.

    Modifica ``clusters`` in situ.
    """
    if f_rotation <= 0:
        return
    for c in clusters:
        for n in range(1, max_order + 1):
            f_harm = n * f_rotation
            tol = max(tol_hz, tol_rel * f_harm)
            if abs(c.freq_mean - f_harm) <= tol:
                c.is_harmonic = True
                c.harmonic_order = n
                break


# ---------------------------------------------------------------------------
# Tabla de modos y agregación entre condiciones
# ---------------------------------------------------------------------------

def modes_to_dataframe(
    clusters: list[ModeCluster],
    rpm: float,
    condition_id: str,
    file_name: str,
) -> pd.DataFrame:
    """Convierte los modos de un ensayo en filas de un ``DataFrame``."""
    rows = []
    for c in clusters:
        rows.append({
            "rpm": rpm,
            "condition": condition_id,
            "file": file_name,
            "freq_hz": c.freq_mean,
            "freq_std_hz": c.freq_std,
            "zeta": c.zeta_mean,
            "zeta_std": c.zeta_std,
            "n_poles": c.n_poles,
            "n_orders": c.n_orders,
            "order_min": c.order_min,
            "order_max": c.order_max,
            "is_harmonic": c.is_harmonic,
            "harmonic_order": c.harmonic_order,
        })
    return pd.DataFrame(rows)


def aggregate_modes_by_rpm(
    modes_table: pd.DataFrame,
    n_conditions: int,
    tolerance_hz: float,
    include_harmonics: bool = False,
) -> pd.DataFrame:
    """Agrupa los modos de todas las condiciones de una RPM y da su persistencia.

    Análogo a ``persistence.analyze_rpm_group`` pero sobre los modos SSI: para
    una misma velocidad se reúnen los modos identificados en todas las
    condiciones experimentales, se agrupan por proximidad en frecuencia y se
    calcula en cuántas condiciones aparece cada frecuencia natural (persistencia).

    Parameters
    ----------
    modes_table : pd.DataFrame
        Modos SSI de una RPM (varias condiciones), con columnas ``condition``,
        ``freq_hz``, ``zeta``, ``is_harmonic``.
    n_conditions : int
        Número de condiciones experimentales de esa RPM (denominador).
    tolerance_hz : float
        Tolerancia del agrupamiento de frecuencias [Hz].
    include_harmonics : bool
        Si es False, los modos marcados como armónicos se excluyen.

    Returns
    -------
    pd.DataFrame
        Una fila por frecuencia natural con ``rpm, freq_mean, freq_std,
        zeta_mean, zeta_std, n_appearances, n_conditions, persistence,
        is_harmonic``.
    """
    if modes_table.empty:
        return modes_table
    df = modes_table
    if not include_harmonics:
        df = df[~df["is_harmonic"].astype(bool)]
    if df.empty:
        return pd.DataFrame()

    rpm = float(df["rpm"].iloc[0])
    df = df.sort_values("freq_hz")
    freqs = df["freq_hz"].to_numpy()

    # Clustering 1D incremental contra la media del grupo (igual que persistence).
    groups: list[list[int]] = [[0]]
    csum = freqs[0]
    for k in range(1, len(freqs)):
        mean = csum / len(groups[-1])
        if freqs[k] - mean <= tolerance_hz:
            groups[-1].append(k)
            csum += freqs[k]
        else:
            groups.append([k])
            csum = freqs[k]

    rows = []
    idx_arr = df.index.to_numpy()
    for g in groups:
        sub = df.loc[idx_arr[g]]
        n_appear = sub["condition"].nunique()
        rows.append({
            "rpm": rpm,
            "freq_mean": float(sub["freq_hz"].mean()),
            "freq_std": float(sub["freq_hz"].std(ddof=0)),
            "zeta_mean": float(sub["zeta"].mean()),
            "zeta_std": float(sub["zeta"].std(ddof=0)),
            "n_appearances": int(n_appear),
            "n_conditions": int(n_conditions),
            "persistence": float(n_appear / n_conditions) if n_conditions else 0.0,
            "is_harmonic": bool(sub["is_harmonic"].astype(bool).any()),
        })
    out = pd.DataFrame(rows).sort_values("freq_mean").reset_index(drop=True)
    return out
