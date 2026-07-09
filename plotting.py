"""
plotting.py
===========

Generación de todas las figuras del análisis:

* PSD individual de cada ensayo con los picos detectados y su frecuencia
  anotada sobre cada pico.
* PSD promedio (media ± desviación estándar) de cada velocidad de rotación,
  con los picos persistentes marcados.
* Diagrama tipo Campbell: frecuencia frente a RPM, donde el tamaño y el color
  de cada punto crecen con la persistencia del pico.
* Histograma de persistencia.
* Número de apariciones por frecuencia.

Criterios de diseño: un solo eje por figura, rejilla discreta, un único tono
para las curvas, mapa de colores secuencial perceptualmente uniforme
(viridis) para codificar la persistencia, y etiquetas directas solo en los
puntos relevantes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla: solo se guardan archivos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peak_detection import PeakSet
from psd import to_db

logger = logging.getLogger(__name__)

# Estilo homogéneo para todas las figuras.
_LINE_COLOR = "#2f6fb2"      # azul único para las curvas PSD
_BAND_COLOR = "#2f6fb2"      # banda ±std (con transparencia)
_PEAK_COLOR = "#c2452f"      # marcador de picos
_GRID_KW = {"color": "0.85", "linewidth": 0.6}

plt.rcParams.update(
    {
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.color": _GRID_KW["color"],
        "grid.linewidth": _GRID_KW["linewidth"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Guarda y cierra una figura, creando la carpeta si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.debug("Figura guardada: %s", path)


def _psd_to_plot_scale(psd: np.ndarray, scale: str) -> np.ndarray:
    """Convierte la PSD a la escala del eje Y ('db' o 'log')."""
    return to_db(psd) if scale == "db" else psd


def plot_psd_with_peaks(
    freq: np.ndarray,
    psd: np.ndarray,
    peaks: PeakSet,
    title: str,
    path: Path,
    scale: str = "db",
    dpi: int = 150,
) -> None:
    """PSD de un ensayo con los picos detectados y su frecuencia anotada.

    Parameters
    ----------
    freq, psd : np.ndarray
        PSD lineal del ensayo y su vector de frecuencias.
    peaks : PeakSet
        Picos detectados en esa PSD.
    title : str
        Título de la figura (etiqueta del ensayo).
    path : Path
        Archivo de salida.
    scale : {"db", "log"}
        Escala del eje Y.
    dpi : int
        Resolución de la figura.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = _psd_to_plot_scale(psd, scale)
    ax.plot(freq, y, color=_LINE_COLOR, linewidth=1.0)
    if scale == "log":
        ax.set_yscale("log")

    if len(peaks):
        y_pk = _psd_to_plot_scale(peaks.amp, scale)
        ax.plot(
            peaks.freq, y_pk, "v", color=_PEAK_COLOR,
            markersize=6, linestyle="none",
        )
        # Frecuencia anotada sobre cada pico.
        y_span = np.ptp(y) if scale == "db" else None
        for fx, fy in zip(peaks.freq, y_pk):
            offset = 0.03 * y_span if y_span else fy * 1.5
            ax.annotate(
                f"{fx:.1f}",
                xy=(fx, fy),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="0.25",
                rotation=90,
            )

    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [dB re 1 u²/Hz]" if scale == "db" else "PSD [u²/Hz]")
    ax.set_title(f"{title}  ·  {len(peaks)} picos (prominencia ≥ "
                 f"{peaks.threshold_db:.1f} dB)")
    ax.set_xlim(freq[0], freq[-1])
    ax.margins(y=0.12)
    _save(fig, path, dpi)


def plot_average_psd(
    freq: np.ndarray,
    psd_mean: np.ndarray,
    psd_std: np.ndarray,
    persistent_table: pd.DataFrame,
    rpm: float,
    path: Path,
    scale: str = "db",
    dpi: int = 150,
) -> None:
    """PSD promedio de una velocidad con banda ±1σ y picos persistentes.

    Parameters
    ----------
    freq, psd_mean, psd_std : np.ndarray
        PSD promedio del grupo y su desviación estándar (escala lineal).
    persistent_table : pd.DataFrame
        Filas de :func:`persistence.analyze_rpm_group` ya filtradas por
        persistencia, para marcar los picos que sobreviven al filtro.
    rpm : float
        Velocidad de rotación del grupo.
    path : Path
        Archivo de salida.
    scale : {"db", "log"}
        Escala del eje Y.
    dpi : int
        Resolución de la figura.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_mean = _psd_to_plot_scale(psd_mean, scale)

    if scale == "db":
        # En dB la banda se construye sobre la PSD lineal y luego se convierte.
        upper = to_db(psd_mean + psd_std)
        lower = to_db(np.maximum(psd_mean - psd_std, np.finfo(float).tiny))
    else:
        upper = psd_mean + psd_std
        lower = np.maximum(psd_mean - psd_std, np.finfo(float).tiny)
        ax.set_yscale("log")

    ax.fill_between(freq, lower, upper, color=_BAND_COLOR, alpha=0.18,
                    linewidth=0, label="±1 σ entre condiciones")
    ax.plot(freq, y_mean, color=_LINE_COLOR, linewidth=1.3,
            label="PSD promedio")

    if not persistent_table.empty:
        f_pk = persistent_table["freq_mean"].to_numpy()
        y_pk = np.interp(f_pk, freq, y_mean)
        ax.plot(f_pk, y_pk, "v", color=_PEAK_COLOR, markersize=7,
                linestyle="none", label="Picos persistentes")
        for fx, fy, pers in zip(
            f_pk, y_pk, persistent_table["persistence"].to_numpy()
        ):
            ax.annotate(
                f"{fx:.1f} Hz\np={pers:.2f}",
                xy=(fx, fy),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="0.25",
            )

    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [dB re 1 u²/Hz]" if scale == "db" else "PSD [u²/Hz]")
    ax.set_title(f"PSD promedio · {rpm:g} rpm")
    ax.set_xlim(freq[0], freq[-1])
    ax.margins(y=0.15)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    _save(fig, path, dpi)


def plot_campbell(
    table: pd.DataFrame,
    path: Path,
    colormap: str = "viridis",
    show_harmonics: bool = True,
    harmonic_orders: list[int] | None = None,
    dpi: int = 150,
) -> None:
    """Diagrama tipo Campbell basado en persistencia.

    Cada punto es un pico persistente: eje X = RPM, eje Y = frecuencia.
    El tamaño y el color del punto crecen con la persistencia, de modo que
    los modos que sobreviven a muchas condiciones destacan visualmente.

    Parameters
    ----------
    table : pd.DataFrame
        Tabla global de picos persistentes (todas las RPM) con las columnas
        ``rpm``, ``freq_mean`` y ``persistence``.
    path : Path
        Archivo de salida.
    colormap : str
        Mapa de colores secuencial para la persistencia.
    show_harmonics : bool
        Dibujar las líneas de armónicos 1X, 2X, ...
    harmonic_orders : list of int
        Órdenes de los armónicos a dibujar.
    dpi : int
        Resolución de la figura.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    if table.empty:
        ax.text(0.5, 0.5, "Sin picos persistentes", ha="center", va="center",
                transform=ax.transAxes)
    else:
        rpm = table["rpm"].to_numpy()
        freq = table["freq_mean"].to_numpy()
        pers = table["persistence"].to_numpy()

        # Líneas de armónicos nX (excitación forzada) como referencia.
        if show_harmonics and harmonic_orders:
            rpm_line = np.linspace(0, rpm.max() * 1.05, 50)
            for n in harmonic_orders:
                ax.plot(rpm_line, n * rpm_line / 60.0, color="0.75",
                        linewidth=0.8, linestyle="--", zorder=1)
                ax.annotate(
                    f"{n}X",
                    xy=(rpm_line[-1], n * rpm_line[-1] / 60.0),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color="0.5",
                    va="center",
                )

        # Tamaño proporcional a la persistencia (área entre 15 y 160 pt²).
        sizes = 15.0 + 145.0 * pers
        sc = ax.scatter(
            rpm, freq, s=sizes, c=pers, cmap=colormap,
            vmin=0.0, vmax=1.0, alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Persistencia entre condiciones")

        ax.set_xlim(0, rpm.max() * 1.08)
        ax.set_ylim(0, max(freq.max() * 1.1, 1.0))

    ax.set_xlabel("Velocidad de rotación [rpm]")
    ax.set_ylabel("Frecuencia [Hz]")
    ax.set_title("Diagrama de Campbell · picos persistentes (OMA peak picking)")
    _save(fig, path, dpi)


def plot_persistence_histogram(
    table: pd.DataFrame,
    path: Path,
    dpi: int = 150,
) -> None:
    """Histograma de la persistencia de todos los grupos de picos."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if table.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                transform=ax.transAxes)
    else:
        ax.hist(
            table["persistence"].to_numpy(),
            bins=np.linspace(0, 1, 21),
            color=_LINE_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_xlabel("Persistencia")
    ax.set_ylabel("Número de grupos de picos")
    ax.set_title("Distribución de la persistencia (todas las RPM)")
    ax.set_xlim(0, 1)
    _save(fig, path, dpi)


def plot_appearances_by_frequency(
    table: pd.DataFrame,
    path: Path,
    dpi: int = 150,
) -> None:
    """Número de apariciones de cada grupo de picos frente a su frecuencia."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if table.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                transform=ax.transAxes)
    else:
        markerline, stemlines, baseline = ax.stem(
            table["freq_mean"].to_numpy(),
            table["n_appearances"].to_numpy(),
        )
        plt.setp(stemlines, color=_LINE_COLOR, linewidth=1.0)
        plt.setp(markerline, color=_LINE_COLOR, markersize=4)
        plt.setp(baseline, color="0.8", linewidth=0.8)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Número de apariciones")
    ax.set_title("Apariciones por frecuencia (todas las RPM)")
    _save(fig, path, dpi)
