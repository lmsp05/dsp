"""
peak_detection.py
=================

Peak picking sobre la PSD.

La detección se realiza con ``scipy.signal.find_peaks`` sobre la PSD en
decibelios, ya que la prominencia expresada en dB es estable frente a los
varios órdenes de magnitud que abarca una PSD vibratoria.

Funcionalidades:

* Criterios configurables: prominencia, distancia, ancho y altura.
* Modo automático de prominencia: el umbral se estima a partir del nivel de
  ruido de la propia PSD (piso de ruido por mediana móvil + desviación
  robusta MAD de los residuos).
* Eliminación opcional de los armónicos 1X, 2X, ... NX de la frecuencia de
  rotación (excitación forzada por el desbalance, no modos estructurales).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from psd import to_db

logger = logging.getLogger(__name__)


@dataclass
class PeakSet:
    """Picos detectados en una PSD."""

    freq: np.ndarray            #: Frecuencia de cada pico [Hz].
    amp: np.ndarray             #: Amplitud de la PSD en el pico [unidad^2/Hz].
    amp_db: np.ndarray          #: Amplitud en dB.
    prominence_db: np.ndarray   #: Prominencia de cada pico [dB].
    threshold_db: float = 0.0   #: Prominencia mínima utilizada [dB].
    removed_harmonics: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )                           #: Frecuencias descartadas por ser armónicos.

    def __len__(self) -> int:
        return len(self.freq)


def estimate_noise_prominence(
    freq: np.ndarray,
    psd_db: np.ndarray,
    noise_window_hz: float,
    factor: float,
    min_db: float,
) -> float:
    """Estima la prominencia mínima a partir del nivel de ruido de la PSD.

    El piso de ruido se aproxima con una mediana móvil (robusta frente a los
    picos) y la dispersión del ruido con la desviación absoluta mediana (MAD)
    de los residuos, escalada a una desviación estándar equivalente.

    Parameters
    ----------
    freq : np.ndarray
        Vector de frecuencias [Hz].
    psd_db : np.ndarray
        PSD en decibelios.
    noise_window_hz : float
        Ancho de la ventana de la mediana móvil [Hz].
    factor : float
        Multiplicador de la desviación del ruido.
    min_db : float
        Prominencia mínima absoluta [dB].

    Returns
    -------
    float
        Prominencia mínima recomendada [dB].
    """
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0
    size = max(5, int(round(noise_window_hz / df)))
    if size % 2 == 0:
        size += 1
    noise_floor = median_filter(psd_db, size=size, mode="nearest")
    residual = psd_db - noise_floor
    # MAD escalada: sigma ~ 1.4826 * mediana(|residuo - mediana(residuo)|).
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * float(mad)
    prominence = max(min_db, factor * sigma)
    logger.debug(
        "Prominencia automática: sigma_ruido=%.2f dB -> umbral=%.2f dB",
        sigma, prominence,
    )
    return prominence


def remove_harmonic_peaks(
    peak_freqs: np.ndarray,
    rpm: float,
    max_order: int,
    tol_hz: float,
    tol_rel: float,
) -> np.ndarray:
    """Devuelve una máscara booleana que descarta los picos armónicos.

    Un pico se descarta si está a menos de ``max(tol_hz, tol_rel * n * f1)``
    de algún armónico ``n * f1`` con ``f1 = rpm / 60``.

    Parameters
    ----------
    peak_freqs : np.ndarray
        Frecuencias de los picos [Hz].
    rpm : float
        Velocidad de rotación [rpm].
    max_order : int
        Máximo orden de armónico a eliminar (1X..NX).
    tol_hz : float
        Tolerancia absoluta [Hz].
    tol_rel : float
        Tolerancia relativa (fracción de la frecuencia del armónico).

    Returns
    -------
    np.ndarray
        Máscara booleana: True = conservar el pico.
    """
    if rpm <= 0 or len(peak_freqs) == 0:
        return np.ones(len(peak_freqs), dtype=bool)

    f1 = rpm / 60.0
    keep = np.ones(len(peak_freqs), dtype=bool)
    for n in range(1, max_order + 1):
        f_harm = n * f1
        tol = max(tol_hz, tol_rel * f_harm)
        keep &= np.abs(peak_freqs - f_harm) > tol
    return keep


def detect_peaks(
    freq: np.ndarray,
    psd: np.ndarray,
    prominence_mode: str = "auto",
    prominence_db: float = 6.0,
    auto_factor: float = 4.0,
    auto_min_db: float = 3.0,
    noise_window_hz: float = 25.0,
    distance_hz: float = 2.0,
    width_hz: float | None = None,
    height_db: float | None = None,
    max_peaks: int | None = None,
    rpm: float | None = None,
    remove_harmonics: bool = False,
    harmonics_max_order: int = 5,
    harmonics_tol_hz: float = 1.0,
    harmonics_tol_rel: float = 0.01,
) -> PeakSet:
    """Detecta picos en una PSD mediante ``scipy.signal.find_peaks``.

    Parameters
    ----------
    freq, psd : np.ndarray
        PSD lineal y su vector de frecuencias.
    prominence_mode : {"auto", "fixed"}
        Cómo determinar la prominencia mínima (ver ``config.py``).
    prominence_db : float
        Prominencia mínima [dB] en modo "fixed".
    auto_factor, auto_min_db, noise_window_hz
        Parámetros del modo automático (ver ``estimate_noise_prominence``).
    distance_hz : float
        Separación mínima entre picos [Hz].
    width_hz : float or None
        Ancho mínimo del pico [Hz] (None desactiva el criterio).
    height_db : float or None
        Altura mínima absoluta [dB] (None desactiva el criterio).
    max_peaks : int or None
        Conservar solo los ``max_peaks`` picos más prominentes.
    rpm : float or None
        Velocidad de rotación del ensayo (necesaria para armónicos).
    remove_harmonics : bool
        Descarta los picos coincidentes con armónicos de la rotación.
    harmonics_max_order, harmonics_tol_hz, harmonics_tol_rel
        Parámetros de la eliminación de armónicos.

    Returns
    -------
    PeakSet
        Frecuencias, amplitudes y prominencias de los picos detectados.
    """
    psd_db = to_db(psd)
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0

    # Prominencia mínima (fija o estimada del ruido).
    if prominence_mode == "auto":
        threshold = estimate_noise_prominence(
            freq, psd_db, noise_window_hz, auto_factor, auto_min_db
        )
    else:
        threshold = float(prominence_db)

    kwargs: dict = {
        "prominence": threshold,
        "distance": max(1, int(round(distance_hz / df))),
    }
    if width_hz is not None:
        kwargs["width"] = max(1.0, width_hz / df)
    if height_db is not None:
        kwargs["height"] = height_db

    idx, props = find_peaks(psd_db, **kwargs)

    peak_freq = freq[idx]
    peak_amp = psd[idx]
    peak_amp_db = psd_db[idx]
    peak_prom = props.get("prominences", np.zeros(len(idx)))

    # Eliminación opcional de armónicos de la velocidad de rotación.
    removed = np.empty(0)
    if remove_harmonics and rpm is not None:
        keep = remove_harmonic_peaks(
            peak_freq, rpm, harmonics_max_order, harmonics_tol_hz, harmonics_tol_rel
        )
        removed = peak_freq[~keep]
        peak_freq, peak_amp = peak_freq[keep], peak_amp[keep]
        peak_amp_db, peak_prom = peak_amp_db[keep], peak_prom[keep]

    # Limitar al número máximo de picos (los más prominentes).
    if max_peaks is not None and len(peak_freq) > max_peaks:
        order = np.argsort(peak_prom)[::-1][:max_peaks]
        order = np.sort(order)  # restaurar el orden en frecuencia
        peak_freq, peak_amp = peak_freq[order], peak_amp[order]
        peak_amp_db, peak_prom = peak_amp_db[order], peak_prom[order]

    return PeakSet(
        freq=peak_freq,
        amp=peak_amp,
        amp_db=peak_amp_db,
        prominence_db=peak_prom,
        threshold_db=threshold,
        removed_harmonics=removed,
    )
