"""
procesar_fft.py
===============

Procesamiento por LOTES de archivos de vibracion de rotorkit (modelo Jeffcott,
cojinetes hidrodinamicos) para generar el espectro FFT de cada sensor
proximitor y guardarlo en un unico archivo de texto filtrable por condiciones.

Este script es una version orientada a PRODUCCION del prototipo
``fft_confiable_rotorkit.py``. Conserva la parte fisicamente correcta de aquel
codigo (deteccion del keyphasor, RPM instantanea, segmentacion por
estacionariedad y ventanas de un numero entero de revoluciones con Hann) y le
anade tres cosas:

  1. Recorre TODA una carpeta de datos con la estructura
     ``repN_isoVG_dsbM/Rec_..._RpmXXX.txt`` y extrae los metadatos (rep, iso,
     dsb, rpm) de la ruta y del nombre del archivo.
  2. Selecciona automaticamente UN tramo estacionario por archivo (el mejor
     para la posterior deteccion de picos; ver `seleccionar_tramo`).
  3. Calcula, para cada uno de los 4 proximitores, el espectro promediado en
     bloques de 15 revoluciones con Hann, guardando AMPLITUD y FASE por linea
     espectral, y lo escribe en un unico .txt en formato tabular (una fila por
     bin de frecuencia), pensado para filtrarse luego por rep/iso/dsb/rpm/sensor.

--------------------------------------------------------------------------
SOBRE EL METODO DE PROMEDIADO (la duda que planteaste)
--------------------------------------------------------------------------
El prototipo usa el metodo de Welch, que promedia POTENCIA (|FFT|**2) de los
distintos bloques, NO la fase. Por eso Welch entrega un unico espectro de
amplitud por tramo pero pierde la fase. Aqui se hace explicito el promediado y
se conservan las dos cosas:

  * AMPLITUD  -> promedio RMS de las magnitudes de los bloques:
                amp(f) = sqrt( mean_k |X_k(f)|^2 ).
                Es exactamente lo que hace Welch (promedio de potencia) pero
                expresado en unidades de amplitud. Reduce la varianza y no
                depende de la fase: sirve igual para el contenido sincrono (1X,
                2X...) y para el NO sincrono (frecuencias naturales, whirl de
                aceite), que es justo lo que interesa para OMA.

  * FASE      -> fase del promedio COHERENTE (vectorial) de los bloques:
                fase(f) = angulo( mean_k X_k(f) ).
                Como cada bloque arranca en un pulso del keyphasor, el contenido
                SINCRONO tiene fase fija y se promedia de forma coherente (fase
                estable y significativa). El contenido NO sincrono tiene fase
                aleatoria entre bloques, asi que su fase promedio es ruidosa: eso
                es esperable y, de hecho, util -> una fase estable indica linea
                sincrona; una fase que no se repite indica contenido estructural.

En resumen: la AMPLITUD es la que usaras para detectar picos; la FASE es un
diagnostico que ayuda a distinguir orden (sincrono) de modo (no sincrono).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ============================================================
# CONFIGURACION POR DEFECTO (se puede sobreescribir por linea de comandos)
# ============================================================

FS = 12800.0                          # frecuencia de muestreo [Hz]
REVOLUCIONES_POR_BLOQUE = 15          # revoluciones enteras por bloque de FFT
SOLAPE = 0.5                          # solape entre bloques (0-0.95)
ANALISIS_BLOQUE_S = 5.0               # bloque temporal para medir estacionariedad
UMBRAL_RPM_PCT = 1.5                  # % de variacion de RPM que separa tramos
MIN_DURACION_TRAMO_S = 20.0           # tramos mas cortos no se usan
FMIN = 2.0                            # frecuencia minima guardada [Hz]
FMAX = 500.0                          # frecuencia maxima guardada [Hz]

# Estructura de la base de datos.
FOLDER_PATTERN = r"rep(?P<rep>\d+)_iso(?P<iso>\d+)_dsb(?P<dsb>\d+)"
RPM_PATTERN = r"[Rr][Pp][Mm][\s_\-]*(\d+)"

# Nombres de canal por defecto (si el encabezado no permite detectarlos).
# Columna 0 = keyphasor; columnas 1-4 = proximitores; columna 5 = acelerometro.
CANALES_DEFECTO = ["sync", "P1.Y", "P1.X", "P2.Y", "P2.X", "S1"]

# Solo estos canales (proximitores) se procesan; el acelerometro S1 se excluye.
RE_PROXIMITOR = re.compile(r"(P\d\.[XY])", re.IGNORECASE)
RE_SYNC = re.compile(r"sync", re.IGNORECASE)


# ============================================================
# LECTURA DEL ARCHIVO (auto-contenida, robusta)
# ============================================================

def _leer_lineas(path: Path) -> list[str]:
    for enc in ("latin-1", "utf-8", "utf-16"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"No se pudo decodificar {path}")


def _nombres_canales(lineas: list[str]) -> list[str]:
    """Extrae los nombres de canal de la linea 'Name' del encabezado INFO.

    La linea tiene la forma:
        Name<TAB>Ext. sync. 1<TAB>Input 1: Mach1.P1.Y<TAB>...
    Se devuelve una etiqueta corta por columna de datos (sync, P1.Y, ...).
    """
    for ln in lineas:
        if ln.strip().lower().startswith("name"):
            tokens = [t.strip() for t in ln.split("\t")[1:] if t.strip()]
            if not tokens:
                break
            etiquetas = []
            for tok in tokens:
                m = RE_PROXIMITOR.search(tok)
                if m:
                    etiquetas.append(m.group(1).upper().replace("X", "X").replace("Y", "Y"))
                elif RE_SYNC.search(tok):
                    etiquetas.append("sync")
                else:
                    etiquetas.append(tok.split(":")[-1].strip() or f"canal{len(etiquetas)}")
            return etiquetas
    return list(CANALES_DEFECTO)


def leer_archivo(path: Path) -> tuple[np.ndarray, list[str]]:
    """Lee un archivo de medicion y devuelve (matriz_datos, nombres_canales).

    Ignora los bloques BEGIN_HEADER/INFO y extrae solo BEGIN_DATA..END_DATA.
    """
    lineas = _leer_lineas(path)
    canales = _nombres_canales(lineas)

    ini = fin = None
    for i, ln in enumerate(lineas):
        s = ln.strip().upper()
        if ini is None and s.startswith("BEGIN_DATA"):
            ini = i
        elif ini is not None and s.startswith("END_DATA"):
            fin = i
            break
    if ini is None:
        raise ValueError(f"No se encontro BEGIN_DATA en {path}")
    fin = fin if fin is not None else len(lineas)

    datos_txt = [ln for ln in lineas[ini + 1:fin] if ln.strip()]
    if len(datos_txt) < 100:
        raise ValueError(f"Bloque de datos demasiado corto en {path}")
    matriz = np.loadtxt(datos_txt)
    if matriz.ndim == 1:
        matriz = matriz.reshape(-1, 1)
    # Ajusta la lista de canales al numero real de columnas.
    if len(canales) < matriz.shape[1]:
        canales = canales + [f"canal{k}" for k in range(len(canales), matriz.shape[1])]
    canales = canales[:matriz.shape[1]]
    return matriz, canales


# ============================================================
# KEYPHASOR: DETECCION DE PULSOS Y RPM INSTANTANEA
# ============================================================

def detectar_pulsos(sync: np.ndarray,
                    frac_disparo: float = 0.25,
                    frac_rearme: float = 0.12) -> np.ndarray:
    """Deteccion de flancos de subida del keyphasor con histeresis real.

    El umbral de DISPARO se coloca BAJO: base + frac_disparo * (max - base).
    Para el hardware tipico de este dataset (base saturada en -1 V, pulsos de
    ~3 V) equivale a cruzar por ~0 V, igual que la deteccion de referencia
    (tach < 0 -> >= 0). Un umbral alto (50-60 % del rango) PIERDE los pulsos
    cortos cuya cresta muestreada queda baja (a 12800 Hz un pulso de <0.5 ms
    tiene pocas muestras y su maximo muestreado varia pulso a pulso); cada
    pulso perdido duplica el intervalo y produce RPM instantaneas en
    rpm/2, rpm/3, ... que hunden la media del bloque.

    La HISTERESIS evita el problema contrario (duplicar pulsos por ruido):
    tras un disparo, el detector solo se rearma cuando la senal cae por
    debajo de base + frac_rearme * (max - base).
    """
    # Base = mediana (el canal pasa casi todo el tiempo en reposo);
    # pico = maximo. Robusto aunque los pulsos sean muy escasos.
    lo = float(np.median(sync))
    hi = float(np.max(sync))
    rango = hi - lo
    if rango < 1e-9:
        raise ValueError("El canal de sincronismo no varia; no hay pulsos.")
    thr_alto = lo + frac_disparo * rango
    thr_bajo = lo + frac_rearme * rango

    # Eventos de cruce (vectorizado): subidas por thr_alto y bajadas por thr_bajo.
    sube = np.where((sync[:-1] <= thr_alto) & (sync[1:] > thr_alto))[0] + 1
    baja = np.where((sync[:-1] >= thr_bajo) & (sync[1:] < thr_bajo))[0] + 1
    if len(sube) < 3:
        raise ValueError("No se detectaron suficientes pulsos de keyphasor.")

    # Histeresis: se acepta una subida solo si hubo un rearme (bajada) despues
    # de la ultima subida aceptada. Merge lineal de ambos trenes de eventos.
    pulsos = [int(sube[0])]
    j = 0
    nb = len(baja)
    for s in sube[1:]:
        while j < nb and baja[j] < pulsos[-1]:
            j += 1
        if j < nb and baja[j] < s:      # hubo rearme entre el ultimo pulso y este
            pulsos.append(int(s))
    if len(pulsos) < 3:
        raise ValueError("No se detectaron suficientes pulsos de keyphasor.")

    # Red de seguridad contra dobles residuales por ruido en el flanco: descarta
    # pulsos a menos de 0.4 x el espaciado mediano del anterior.
    p = np.asarray(pulsos, dtype=int)
    med = float(np.median(np.diff(p)))
    limpio = [int(p[0])]
    for x in p[1:]:
        if x - limpio[-1] > 0.4 * med:
            limpio.append(int(x))
    return np.asarray(limpio, dtype=int)


def rpm_instantanea(pulsos: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    t_pulso = pulsos[:-1] / fs
    rpm = 60.0 * fs / np.diff(pulsos)
    return t_pulso, rpm


# ============================================================
# SEGMENTACION POR ESTACIONARIEDAD DE RPM
# ============================================================

@dataclass
class Tramo:
    t0: float
    t1: float
    rpm_media: float
    rpm_std: float
    n_rev: int

    @property
    def dur(self) -> float:
        return self.t1 - self.t0

    @property
    def cv(self) -> float:
        """Coeficiente de variacion de la RPM en el tramo (menor = mas estable)."""
        return self.rpm_std / self.rpm_media if self.rpm_media else np.inf


def segmentar_estacionario(t_pulso, rpm, bloque_s, umbral_pct, min_dur_s) -> list[Tramo]:
    """Divide el registro en tramos donde la RPM es aproximadamente constante.

    Igual criterio que el prototipo: bloques de `bloque_s` segundos, se fusionan
    los consecutivos cuya RPM media no se aparte mas de `umbral_pct` del inicio
    del grupo, y se conservan los grupos con duracion >= `min_dur_s`.
    """
    tmax = t_pulso.max()
    bordes = np.arange(0, tmax + bloque_s, bloque_s)
    rpm_bloque = []
    for i in range(len(bordes) - 1):
        m = (t_pulso >= bordes[i]) & (t_pulso < bordes[i + 1])
        rpm_bloque.append(np.mean(rpm[m]) if m.sum() > 0 else np.nan)
    rpm_bloque = np.array(rpm_bloque)

    def construir(inicio_idx, fin_idx) -> Tramo | None:
        t0 = float(bordes[inicio_idx])
        t1 = float(min(bordes[fin_idx], tmax))
        if (t1 - t0) < min_dur_s:
            return None
        m = (t_pulso >= t0) & (t_pulso < t1)
        if m.sum() < 2:
            return None
        return Tramo(t0, t1, float(np.mean(rpm[m])), float(np.std(rpm[m])), int(m.sum()))

    tramos: list[Tramo] = []
    inicio_idx = 0
    for i in range(1, len(rpm_bloque)):
        if np.isnan(rpm_bloque[i]) or np.isnan(rpm_bloque[inicio_idx]):
            continue
        cambio = 100 * abs(rpm_bloque[i] - rpm_bloque[inicio_idx]) / rpm_bloque[inicio_idx]
        if cambio > umbral_pct:
            tr = construir(inicio_idx, i)
            if tr:
                tramos.append(tr)
            inicio_idx = i
    tr = construir(inicio_idx, len(rpm_bloque))
    if tr:
        tramos.append(tr)
    return tramos


def seleccionar_tramo(tramos: list[Tramo], rpm_nominal: float) -> Tramo | None:
    """Elige UN tramo estacionario para el analisis espectral.

    Criterio (recomendado para la posterior deteccion de picos):

      1. Preferir los tramos cuya RPM media este cerca (<= 5 %) de la RPM
         nominal del nombre del archivo: asi el espectro corresponde de verdad
         a la velocidad con la que esta etiquetada la condicion, y no a un
         transitorio ni a un escalon de velocidad.
      2. Entre esos, quedarse con el mas LARGO (mas revoluciones -> mas bloques
         promediados -> menor varianza y piso de ruido mas limpio, que es lo que
         hace que un pico sea detectable de forma fiable).
      3. Como desempate, la RPM mas estable (menor coeficiente de variacion).

    Si ningun tramo esta cerca de la nominal, se relaja el punto 1 y se toma el
    mas largo disponible.
    """
    if not tramos:
        return None

    def cerca(tr: Tramo) -> bool:
        if rpm_nominal <= 0:
            return True
        return abs(tr.rpm_media - rpm_nominal) / rpm_nominal <= 0.05

    candidatos = [t for t in tramos if cerca(t)] or tramos
    # ordena por duracion desc, luego por cv asc
    candidatos.sort(key=lambda t: (-t.dur, t.cv))
    return candidatos[0]


# ============================================================
# ESPECTRO PROMEDIADO EN BLOQUES DE N REVOLUCIONES (AMPLITUD + FASE)
# ============================================================

def espectro_amp_fase(signal, pulsos, fs, tramo: Tramo, revs, solape, fmin, fmax):
    """Espectro promediado del tramo con AMPLITUD (RMS) y FASE (coherente).

    - nperseg = muestras de `revs` revoluciones a la RPM media del tramo, de modo
      que la resolucion espectral es identica para todos los bloques (grilla de
      frecuencia comun) y el 1X cae practicamente en el bin `revs`.
    - Cada bloque arranca en un pulso del keyphasor (referencia de fase comun).
    - Ventana Hann con correccion de ganancia coherente + espectro de un solo lado.

    Devuelve (f, amplitud, fase_grados, n_bloques, nperseg).
    """
    nperseg = int(round(revs * fs * 60.0 / tramo.rpm_media))
    if nperseg < 8:
        raise ValueError("nperseg demasiado pequeno; revisa RPM/revoluciones.")
    w = np.hanning(nperseg)
    cg = w.sum()                                   # ganancia coherente de la ventana
    paso = max(1, int(round(revs * (1.0 - solape))))  # avance en revoluciones (pulsos)

    i0 = int(tramo.t0 * fs)
    i1 = int(tramo.t1 * fs)
    # pulsos dentro del tramo cuyo bloque completo cabe en el tramo
    idx_pulsos = np.where((pulsos >= i0) & (pulsos + nperseg <= i1) &
                          (pulsos + nperseg <= len(signal)))[0]
    if len(idx_pulsos) == 0:
        raise ValueError("El tramo no admite ningun bloque completo.")

    Xsum = None
    Pot = None
    n = 0
    for k in range(0, len(idx_pulsos), paso):
        ini = int(pulsos[idx_pulsos[k]])
        seg = signal[ini:ini + nperseg].astype(float)
        seg = seg - seg.mean()
        X = np.fft.rfft(seg * w) / cg              # normalizacion de amplitud
        X[1:-1] *= 2.0                             # espectro de un solo lado (no DC/Nyquist)
        if Xsum is None:
            Xsum = np.zeros_like(X)
            Pot = np.zeros(len(X))
        Xsum += X
        Pot += np.abs(X) ** 2
        n += 1
    if n == 0:
        raise ValueError("No se acumulo ningun bloque.")

    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    amplitud = np.sqrt(Pot / n)                    # promedio de potencia -> amplitud (Welch)
    fase = np.degrees(np.angle(Xsum / n))          # promedio coherente -> fase
    m = (f >= fmin) & (f <= fmax)
    return f[m], amplitud[m], fase[m], n, nperseg


# ============================================================
# RECORRIDO DE LA BASE DE DATOS
# ============================================================

@dataclass
class ArchivoInfo:
    path: Path
    rep: int
    iso: int
    dsb: int
    rpm: float


def descubrir_archivos(data_dir: Path) -> list[ArchivoInfo]:
    folder_re = re.compile(FOLDER_PATTERN, re.IGNORECASE)
    rpm_re = re.compile(RPM_PATTERN)
    encontrados: list[ArchivoInfo] = []
    for path in sorted(data_dir.rglob("*.txt")):
        mcarp = None
        for parent in path.parents:
            mcarp = folder_re.search(parent.name)
            if mcarp:
                break
        if not mcarp:
            continue
        mrpm = rpm_re.search(path.name)
        if not mrpm:
            print(f"  [aviso] sin RPM en el nombre, se omite: {path.name}")
            continue
        encontrados.append(ArchivoInfo(
            path=path,
            rep=int(mcarp.group("rep")),
            iso=int(mcarp.group("iso")),
            dsb=int(mcarp.group("dsb")),
            rpm=float(mrpm.group(1)),
        ))
    encontrados.sort(key=lambda a: (a.rep, a.iso, a.dsb, a.rpm))
    return encontrados


# ============================================================
# SALIDA: ARCHIVO DE TEXTO TABULAR FILTRABLE
# ============================================================

CABECERA_COLUMNAS = [
    "rep", "iso", "dsb", "rpm_nominal", "sensor",
    "t_ini_s", "t_fin_s", "rpm_medida", "n_bloques", "revs_bloque", "df_Hz",
    "frecuencia_Hz", "amplitud", "fase_grados",
]


def escribir_cabecera(fh, args):
    fh.write("# Espectro FFT de proximitores de rotorkit (amplitud + fase)\n")
    fh.write("# Generado por procesar_fft.py\n")
    fh.write("# --------------------------------------------------------------\n")
    fh.write(f"# fs_Hz\t{FS}\n")
    fh.write(f"# revoluciones_por_bloque\t{args.revs}\n")
    fh.write(f"# solape\t{args.solape}\n")
    fh.write(f"# umbral_rpm_pct\t{args.umbral_rpm}\n")
    fh.write(f"# min_duracion_tramo_s\t{args.min_dur}\n")
    fh.write(f"# fmin_Hz\t{args.fmin}\n")
    fh.write(f"# fmax_Hz\t{args.fmax}\n")
    fh.write("# amplitud: unidades fisicas del sensor (proximitor -> um), pico\n")
    fh.write("# fase_grados: fase del promedio coherente (referida al keyphasor)\n")
    fh.write("# Una fila por bin de frecuencia. Filtrable por rep/iso/dsb/rpm_nominal/sensor.\n")
    fh.write("# --------------------------------------------------------------\n")
    fh.write("\t".join(CABECERA_COLUMNAS) + "\n")


def escribir_espectro(fh, info: ArchivoInfo, sensor, tramo: Tramo, f, amp, fase, n_bloques, nperseg):
    df = FS / nperseg
    # revoluciones reales por bloque (nperseg se fija a la RPM media del tramo)
    revs_real = int(round(nperseg * tramo.rpm_media / (FS * 60.0)))
    base = (f"{info.rep}\t{info.iso}\t{info.dsb}\t{info.rpm:g}\t{sensor}\t"
            f"{tramo.t0:.3f}\t{tramo.t1:.3f}\t{tramo.rpm_media:.3f}\t{n_bloques}\t"
            f"{revs_real}\t{df:.5f}\t")
    lineas = (base + f"{fi:.5f}\t{ai:.6e}\t{pi:.2f}\n"
              for fi, ai, pi in zip(f, amp, fase))
    fh.writelines(lineas)


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
    print(f"Salida: {salida}\n")

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

            # localizar canal de sincronismo y proximitores
            try:
                col_sync = next(i for i, c in enumerate(canales) if c == "sync")
            except StopIteration:
                col_sync = 0
            proximitores = [(i, c) for i, c in enumerate(canales)
                            if RE_PROXIMITOR.fullmatch(c)]
            if not proximitores:
                print(f"  [aviso] sin proximitores en {etiqueta}, se omite")
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
                print(f"  [aviso] {etiqueta}: sin tramo estacionario >= {args.min_dur:g}s, se omite")
                n_err += 1
                continue

            escritos = 0
            for col, sensor in proximitores:
                try:
                    f, amp, fase, nb, nperseg = espectro_amp_fase(
                        matriz[:, col], pulsos, FS, tramo,
                        args.revs, args.solape, args.fmin, args.fmax)
                    escribir_espectro(fh, info, sensor, tramo, f, amp, fase, nb, nperseg)
                    escritos += 1
                except Exception as e:
                    print(f"  [ERROR fft {sensor}] {etiqueta}: {e}")
            if escritos:
                print(f"  OK {etiqueta}: tramo [{tramo.t0:.1f},{tramo.t1:.1f}]s "
                      f"RPM={tramo.rpm_media:.1f} (cv={tramo.cv*100:.2f}%) "
                      f"{escritos} sensores")
                n_ok += 1
            else:
                n_err += 1

    print(f"\nListo. Archivos con espectro: {n_ok} | con error/omitidos: {n_err}")
    print(f"Resultado en: {salida}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera un .txt con el espectro FFT (amplitud + fase) de "
                    "los proximitores de todos los archivos de una carpeta.")
    p.add_argument("--data-dir", default="datos",
                   help="Carpeta raiz con las carpetas repN_isoVG_dsbM (def: ./datos)")
    p.add_argument("--salida", default="resultados_fft.txt",
                   help="Archivo .txt de salida (def: ./resultados_fft.txt)")
    p.add_argument("--revs", type=int, default=REVOLUCIONES_POR_BLOQUE,
                   help="Revoluciones enteras por bloque de FFT")
    p.add_argument("--solape", type=float, default=SOLAPE, help="Solape entre bloques (0-0.95)")
    p.add_argument("--bloque-s", dest="bloque_s", type=float, default=ANALISIS_BLOQUE_S,
                   help="Bloque temporal para medir estacionariedad [s]")
    p.add_argument("--umbral-rpm", dest="umbral_rpm", type=float, default=UMBRAL_RPM_PCT,
                   help="%% de variacion de RPM que separa tramos")
    p.add_argument("--min-dur", dest="min_dur", type=float, default=MIN_DURACION_TRAMO_S,
                   help="Duracion minima de un tramo utilizable [s]")
    p.add_argument("--fmin", type=float, default=FMIN, help="Frecuencia minima guardada [Hz]")
    p.add_argument("--fmax", type=float, default=FMAX, help="Frecuencia maxima guardada [Hz]")
    p.add_argument("--limit", type=int, default=0, help="Procesar solo los primeros N archivos (prueba)")
    return p


if __name__ == "__main__":
    args = construir_parser().parse_args()
    sys.exit(procesar(args))
