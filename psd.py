"""
psd.py
======

Estimación de la densidad espectral de potencia (PSD).

Este módulo implementa:

* El cálculo de la PSD de una señal mediante el método de Welch
  (``scipy.signal.welch``) con ventana Hann y parámetros configurables.
* El recorte del rango de frecuencias de análisis [FMIN, FMAX].
* Un suavizado opcional de la PSD mediante media móvil.
* El cálculo de la PSD promedio (media y desviación estándar) de un grupo de
  PSD, interpolando a una malla común si fuera necesario.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

logger = logging.getLogger(__name__)


@dataclass
class PSDResult:
    """PSD de una señal recortada al rango de análisis."""

    freq: np.ndarray   #: Vector de frecuencias [Hz].
    psd: np.ndarray    #: Densidad espectral de potencia [unidad^2/Hz].

    @property
    def df(self) -> float:
        """Resolución en frecuencia [Hz]."""
        return float(self.freq[1] - self.freq[0]) if len(self.freq) > 1 else 0.0


def moving_average(x: np.ndarray, window_bins: int) -> np.ndarray:
    """Suaviza un vector mediante media móvil centrada.

    Parameters
    ----------
    x : np.ndarray
        Vector a suavizar.
    window_bins : int
        Ancho de la ventana en muestras. Se fuerza a impar y como mínimo 3.

    Returns
    -------
    np.ndarray
        Vector suavizado, de la misma longitud que ``x``.
    """
    w = max(3, int(window_bins))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    # Extensión por reflexión para evitar distorsionar los bordes.
    padded = np.pad(x, w // 2, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: str | bool = "constant",
    fmin: float = 0.0,
    fmax: float | None = None,
    smoothing: bool = False,
    smoothing_window_bins: int = 5,
) -> PSDResult:
    """Calcula la PSD de una señal por el método de Welch.

    Parameters
    ----------
    signal : np.ndarray
        Señal temporal.
    fs : float
        Frecuencia de muestreo [Hz].
    nperseg : int
        Muestras por segmento. Si la señal es más corta, se reduce
        automáticamente a la mayor potencia de dos que quepa.
    overlap : float
        Fracción de solape entre segmentos (0-0.95).
    window : str
        Ventana de análisis (por defecto Hann).
    detrend : str or bool
        Detrending aplicado por Welch.
    fmin, fmax : float
        Rango de frecuencias a conservar [Hz].
    smoothing : bool
        Aplicar suavizado por media móvil a la PSD.
    smoothing_window_bins : int
        Ancho de la ventana de suavizado en líneas espectrales.

    Returns
    -------
    PSDResult
        Frecuencia y PSD en el rango solicitado.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 256:
        raise ValueError(f"Señal demasiado corta para Welch ({n} muestras).")

    nperseg_eff = int(nperseg)
    if nperseg_eff > n:
        # Mayor potencia de dos que cabe en la señal.
        nperseg_eff = 2 ** int(np.floor(np.log2(n)))
        logger.debug(
            "nperseg reducido de %d a %d (señal de %d muestras).",
            nperseg, nperseg_eff, n,
        )

    overlap = float(np.clip(overlap, 0.0, 0.95))
    noverlap = int(round(nperseg_eff * overlap))

    freq, pxx = welch(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg_eff,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
    )

    # Recorte al rango de análisis.
    fmax_eff = fmax if fmax is not None else freq[-1]
    mask = (freq >= fmin) & (freq <= fmax_eff)
    freq, pxx = freq[mask], pxx[mask]
    if len(freq) < 8:
        raise ValueError("El rango [FMIN, FMAX] deja muy pocas líneas de PSD.")

    if smoothing:
        pxx = moving_average(pxx, smoothing_window_bins)

    return PSDResult(freq=freq, psd=pxx)


def to_db(psd: np.ndarray) -> np.ndarray:
    """Convierte una PSD lineal a decibelios (10*log10) de forma segura."""
    floor = np.finfo(float).tiny
    return 10.0 * np.log10(np.maximum(np.asarray(psd, dtype=float), floor))


def average_psd(results: list[PSDResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la PSD promedio y su desviación estándar para un grupo de PSD.

    Si todas las PSD comparten la misma malla de frecuencias (caso habitual,
    ya que fs y nperseg son fijos), se promedian directamente; en caso
    contrario se interpolan a la malla de la primera PSD.

    Parameters
    ----------
    results : list of PSDResult
        PSD individuales del grupo (p. ej. todas las condiciones de una RPM).

    Returns
    -------
    tuple
        ``(freq, psd_media, psd_std)``.
    """
    if not results:
        raise ValueError("No hay PSD para promediar.")

    ref = results[0].freq
    stack = []
    for r in results:
        if len(r.freq) == len(ref) and np.allclose(r.freq, ref):
            stack.append(r.psd)
        else:
            # Mallas distintas (p. ej. una señal corta): interpolación lineal.
            stack.append(np.interp(ref, r.freq, r.psd))
    matrix = np.vstack(stack)
    return ref, matrix.mean(axis=0), matrix.std(axis=0)
