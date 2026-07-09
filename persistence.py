"""
persistence.py
==============

Agrupamiento de picos por velocidad de rotación y cálculo de persistencia.

Esta es la etapa central del análisis OMA por peak picking:

1. Para cada velocidad de rotación (RPM) se reúnen los picos detectados en
   TODAS las condiciones experimentales (todas las repeticiones, viscosidades
   y desbalances) que comparten esa RPM.
2. Los picos se agrupan por proximidad en frecuencia mediante un clustering
   1D con tolerancia configurable: picos como 41.9, 42.1, 42.3 y 42.4 Hz se
   consideran una única frecuencia si caen dentro de la tolerancia.
3. Para cada grupo se calcula la persistencia: la fracción de condiciones
   experimentales en las que el pico aparece. Un pico que aparece en 24 de
   27 condiciones tiene persistencia 24/27 = 0.89 y es un fuerte candidato a
   frecuencia natural, ya que sobrevive a cambios de viscosidad y desbalance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PeakObservation:
    """Un pico detectado en una condición experimental concreta."""

    condition_id: str   #: Identificador de la condición (rep_iso_dsb).
    freq: float         #: Frecuencia del pico [Hz].
    amp: float          #: Amplitud de la PSD en el pico [unidad^2/Hz].
    amp_db: float       #: Amplitud en dB.
    prominence_db: float  #: Prominencia del pico [dB].


def cluster_frequencies(
    observations: list[PeakObservation],
    tolerance_hz: float,
) -> list[list[PeakObservation]]:
    """Agrupa picos de distintas condiciones por proximidad en frecuencia.

    Algoritmo (clustering 1D incremental sobre las frecuencias ordenadas):
    se recorren los picos en orden creciente de frecuencia y cada pico se
    añade al grupo actual si dista menos de ``tolerance_hz`` de la media del
    grupo; en caso contrario se abre un grupo nuevo. Comparar contra la media
    del grupo (y no contra el último pico añadido) evita el encadenamiento
    progresivo que uniría picos lejanos a través de picos intermedios.

    Parameters
    ----------
    observations : list of PeakObservation
        Picos de todas las condiciones de una misma RPM.
    tolerance_hz : float
        Distancia máxima [Hz] entre un pico y la media del grupo.

    Returns
    -------
    list of list of PeakObservation
        Grupos de picos, ordenados por frecuencia creciente.
    """
    if not observations:
        return []

    obs_sorted = sorted(observations, key=lambda o: o.freq)
    clusters: list[list[PeakObservation]] = [[obs_sorted[0]]]
    cluster_sum = obs_sorted[0].freq  # suma acumulada para la media del grupo

    for obs in obs_sorted[1:]:
        mean = cluster_sum / len(clusters[-1])
        if obs.freq - mean <= tolerance_hz:
            clusters[-1].append(obs)
            cluster_sum += obs.freq
        else:
            clusters.append([obs])
            cluster_sum = obs.freq
    return clusters


def analyze_rpm_group(
    rpm: float,
    observations: list[PeakObservation],
    n_conditions: int,
    tolerance_hz: float,
    avg_freq: np.ndarray | None = None,
    avg_psd: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calcula la tabla de picos persistentes de una velocidad de rotación.

    Parameters
    ----------
    rpm : float
        Velocidad de rotación del grupo [rpm].
    observations : list of PeakObservation
        Picos de todas las condiciones experimentales con esa RPM.
    n_conditions : int
        Número total de condiciones experimentales analizadas en esa RPM
        (denominador de la persistencia).
    tolerance_hz : float
        Tolerancia del clustering de frecuencias [Hz].
    avg_freq, avg_psd : np.ndarray, optional
        PSD promedio del grupo, usada para añadir a cada pico el valor de la
        PSD promedio en su frecuencia.

    Returns
    -------
    pd.DataFrame
        Una fila por grupo de picos con las columnas:
        ``rpm, freq_mean, freq_std, amp_mean, amp_std, amp_db_mean,
        n_appearances, n_conditions, persistence, psd_avg_at_freq``.
    """
    clusters = cluster_frequencies(observations, tolerance_hz)

    rows = []
    for cluster in clusters:
        freqs = np.array([o.freq for o in cluster])
        amps = np.array([o.amp for o in cluster])
        amps_db = np.array([o.amp_db for o in cluster])

        # Una condición puede aportar más de un pico al mismo grupo (picos
        # dobles muy próximos); la persistencia cuenta condiciones únicas.
        n_appear = len({o.condition_id for o in cluster})

        f_mean = float(freqs.mean())
        row = {
            "rpm": rpm,
            "freq_mean": f_mean,
            "freq_std": float(freqs.std(ddof=0)),
            "amp_mean": float(amps.mean()),
            "amp_std": float(amps.std(ddof=0)),
            "amp_db_mean": float(amps_db.mean()),
            "n_appearances": int(n_appear),
            "n_conditions": int(n_conditions),
            "persistence": float(n_appear / n_conditions) if n_conditions else 0.0,
        }
        if avg_freq is not None and avg_psd is not None:
            row["psd_avg_at_freq"] = float(np.interp(f_mean, avg_freq, avg_psd))
        else:
            row["psd_avg_at_freq"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("freq_mean").reset_index(drop=True)
    logger.debug(
        "RPM %g: %d picos agrupados en %d frecuencias (%d condiciones).",
        rpm, len(observations), len(df), n_conditions,
    )
    return df


def filter_persistent(
    table: pd.DataFrame,
    min_persistence: float,
    min_appearances: int,
) -> pd.DataFrame:
    """Filtra la tabla de grupos según persistencia y apariciones mínimas.

    Parameters
    ----------
    table : pd.DataFrame
        Tabla producida por :func:`analyze_rpm_group` (una o varias RPM).
    min_persistence : float
        Persistencia mínima (0-1).
    min_appearances : int
        Número mínimo de condiciones en las que debe aparecer el pico.

    Returns
    -------
    pd.DataFrame
        Subconjunto de filas que cumplen ambos criterios.
    """
    if table.empty:
        return table
    mask = (table["persistence"] >= min_persistence) & (
        table["n_appearances"] >= min_appearances
    )
    return table.loc[mask].reset_index(drop=True)
