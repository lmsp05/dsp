#!/usr/bin/env python3
"""
inventario_combinaciones.py
===========================

Inventario de la base de datos experimental: que combinaciones
``rep_iso_dsb_rpm`` estan medidas, cuantas veces aparece cada una, cuanto pesa
cada archivo y cuales pesos son atipicos (outliers).

--------------------------------------------------------------------------
QUE HACE
--------------------------------------------------------------------------
Recorre una carpeta raiz que contiene subcarpetas con la nomenclatura
``repN_isoVG_dsbM`` y, dentro de ellas, archivos cuyo nombre termina en
``RpmXXXX``. De cada archivo se deducen los cuatro parametros del ensayo: la
repeticion, la viscosidad y el desbalance vienen de la subcarpeta; la velocidad
viene del propio nombre del archivo.

    datos/
    |- rep1_iso32_dsb1/
    |  |- Rec_stb_iso32_dsb(0+8-7)-Rpm600.txt   -> rep1_iso32_dsb1_rpm600
    |  |- Rec_stb_iso32_dsb(0+8-7)-Rpm1300.txt  -> rep1_iso32_dsb1_rpm1300
    |- rep1_iso32_dsb2/
    |- ... hasta rep3_iso68_dsb3

Con ese inventario el script:

  * Lista todas las combinaciones ``rep_iso_dsb_rpm`` encontradas. Cuando una
    combinacion aparece mas de una vez (archivos repetidos) se le agrega la
    etiqueta adicional ``_cantN`` con el numero de veces que aparece:
    ``rep1_iso32_dsb1_rpm600_cant2``.
  * Reporta el peso en megabytes (MiB, base 1024) de cada archivo.
  * Marca los pesos outlier con un criterio robusto (mediana + MAD), que no se
    deja arrastrar por los propios valores atipicos como si lo haria la media.
  * Compara lo encontrado contra la malla completa de combinaciones posibles
    (rep x iso x dsb x rpm) y senala las que faltan, con una tabla organizada
    por condicion ``rep_iso_dsb``: cada fila dice si la condicion esta
    ``(completo)``, si no tiene ningun archivo ``(Ninguno)`` o cuantas
    velocidades le faltan, y cuales son. Un archivo outlier por peso cuenta
    como dato NO disponible, asi que su velocidad tambien aparece en esa lista.
  * Avisa de los archivos cuyos parametros quedan fuera del catalogo de valores
    validos (por ejemplo una Rpm900 que no esta en la lista esperada).

--------------------------------------------------------------------------
SALIDAS
--------------------------------------------------------------------------
1. ``inventario_combinaciones.txt`` -- informe completo: resumen, tabla de
   todos los archivos con su peso y su estado, combinaciones repetidas,
   combinaciones faltantes, outliers de tamano y valores fuera de catalogo.

2. ``faltantes_y_outliers.txt`` -- solo lo que requiere accion: las
   combinaciones que no estan medidas y las que estan medidas pero con un
   tamano de archivo atipico.

3. ``tabla_condiciones.txt`` -- la misma informacion por condicion en columnas
   separadas por TABULACION, para abrirla en Excel o leerla con pandas:
   ``condicion``, ``n_archivos``, ``n_missing``, ``n_outliers``,
   ``lst_missing``, ``lst_outliers``, ``t_missing``. Los conteos valen 0 cuando
   no hay nada que contar y las listas quedan vacias.

El script es de SOLO LECTURA sobre la base de datos: de cada archivo lee su
nombre y su tamano, y nada mas. No renombra, no mueve ni modifica ningun
archivo. Las etiquetas (incluido el sufijo ``_cantN``) existen unicamente como
texto dentro de los informes; los nombres reales de los archivos aparecen sin
cambios en la columna ARCHIVO. Lo unico que se escribe son los dos .txt, y
siempre dentro de la carpeta indicada con ``--outdir``.

--------------------------------------------------------------------------
CRITERIO DE OUTLIER
--------------------------------------------------------------------------
Los outliers se separan en BAJOS (pesan menos de lo normal) y ALTOS (pesan
mas). Solo los BAJOS cuentan como dato faltante: un registro mas corto de lo
habitual ha perdido informacion, mientras que uno mas largo la conserva y
simplemente sobra. Los ALTOS se reportan igualmente para poder revisarlos.

Se marca un archivo como outlier de tamano cuando

    |peso - mediana(pesos)| > k * 1.4826 * MAD(pesos)

es decir, cuando se aparta mas de ``k`` desviaciones estandar robustas de la
mediana del grupo (``k = 3`` por defecto). El factor 1.4826 convierte la MAD en
un equivalente a la desviacion estandar de una distribucion normal.

Si la MAD es cero (mas de la mitad de los archivos pesan exactamente lo mismo)
se cae a la IQR, y si esta tambien es cero se usa una tolerancia relativa del
1 % respecto de la mediana. Con menos de 4 archivos en el grupo no se evaluan
outliers: la dispersion no es estimable de forma fiable.

El grupo de comparacion se elige con ``--grupo-outliers``:

  * ``global``   -> todos los archivos entre si (por defecto).
  * ``rpm``      -> archivos de la misma velocidad entre si; util cuando la
                    duracion del registro depende de la RPM.
  * ``condicion``-> archivos de la misma carpeta ``repN_isoVG_dsbM`` entre si.

--------------------------------------------------------------------------
USO
--------------------------------------------------------------------------
    python inventario_combinaciones.py /ruta/a/mis/datos
    python inventario_combinaciones.py --data-dir datos --outdir inventario
    python inventario_combinaciones.py datos --ext .txt --k-outlier 2.5
    python inventario_combinaciones.py datos --grupo-outliers rpm
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# ============================================================
# CATALOGO DE VALORES VALIDOS
# ============================================================

#: Repeticiones del ensayo.
REPS_VALIDAS: tuple[int, ...] = (1, 2, 3)

#: Grados ISO de viscosidad del lubricante.
ISOS_VALIDOS: tuple[int, ...] = (32, 46, 68)

#: Configuraciones de desbalance.
DSBS_VALIDOS: tuple[int, ...] = (1, 2, 3)

#: Velocidades de rotacion nominales [rpm].
RPMS_VALIDAS: tuple[int, ...] = (
    600, 1300, 2000, 2500, 2600, 2700, 2800,
    3400, 3500, 3600, 3700, 4000, 4500, 5000, 5500,
)

#: Nomenclatura de las subcarpetas: repN_isoVG_dsbM.
PATRON_CARPETA = re.compile(r"rep(?P<rep>\d+)_iso(?P<iso>\d+)_dsb(?P<dsb>\d+)", re.IGNORECASE)

#: Velocidad al final del nombre del archivo: ...Rpm600
PATRON_RPM = re.compile(r"[Rr][Pp][Mm][\s_\-]*(\d+)\s*$")

#: Megabyte usado en el informe (MiB, base 1024).
BYTES_POR_MB = 1024 * 1024


# ============================================================
# ESTRUCTURAS DE DATOS
# ============================================================

@dataclass
class Archivo:
    """Un archivo de medicion con sus parametros y su peso."""

    ruta: Path
    rep: int
    iso: int
    dsb: int
    rpm: int
    bytes: int
    outlier: bool = False       #: Peso atipico dentro de su grupo de comparacion.
    grupo: str = ""             #: Grupo con el que se comparo el peso.
    desviacion: float = 0.0     #: Peso menos la mediana del grupo [MB].

    @property
    def mb(self) -> float:
        """Peso del archivo en megabytes."""
        return self.bytes / BYTES_POR_MB

    @property
    def outlier_bajo(self) -> bool:
        """True si el archivo es outlier por pesar MENOS de lo normal.

        Solo estos cuentan como dato no disponible: un registro mas corto de lo
        habitual ha perdido informacion, mientras que uno mas largo la conserva.
        """
        return self.outlier and self.desviacion < 0

    @property
    def tipo_outlier(self) -> str:
        """Etiqueta del tipo de outlier: BAJO, ALTO o cadena vacia."""
        if not self.outlier:
            return ""
        return "BAJO" if self.desviacion < 0 else "ALTO"

    @property
    def combo(self) -> tuple[int, int, int, int]:
        """Combinacion (rep, iso, dsb, rpm) a la que pertenece el archivo."""
        return (self.rep, self.iso, self.dsb, self.rpm)

    @property
    def valido(self) -> bool:
        """True si los cuatro parametros estan en el catalogo de valores validos."""
        return combo_valido(self.combo)


@dataclass
class Combinacion:
    """Todos los archivos que comparten la misma combinacion rep_iso_dsb_rpm."""

    rep: int
    iso: int
    dsb: int
    rpm: int
    archivos: list[Archivo] = field(default_factory=list)

    @property
    def combo(self) -> tuple[int, int, int, int]:
        """Combinacion (rep, iso, dsb, rpm) que identifica al grupo."""
        return (self.rep, self.iso, self.dsb, self.rpm)

    @property
    def cant(self) -> int:
        """Numero de veces que aparece la combinacion."""
        return len(self.archivos)

    @property
    def etiqueta(self) -> str:
        """Etiqueta de la combinacion; agrega ``_cantN`` si aparece repetida."""
        base = etiqueta_combo((self.rep, self.iso, self.dsb, self.rpm))
        return f"{base}_cant{self.cant}" if self.cant > 1 else base

    @property
    def tiene_outlier(self) -> bool:
        """True si alguno de sus archivos tiene un peso atipico."""
        return any(a.outlier for a in self.archivos)


def etiqueta_combo(combo: tuple[int, int, int, int]) -> str:
    """Etiqueta canonica ``repR_isoI_dsbD_rpmV`` de una combinacion."""
    rep, iso, dsb, rpm = combo
    return f"rep{rep}_iso{iso}_dsb{dsb}_rpm{rpm}"


def combo_valido(combo: tuple[int, int, int, int]) -> bool:
    """True si los cuatro parametros pertenecen al catalogo de valores validos."""
    rep, iso, dsb, rpm = combo
    return (rep in REPS_VALIDAS and iso in ISOS_VALIDOS
            and dsb in DSBS_VALIDOS and rpm in RPMS_VALIDAS)


def malla_completa() -> list[tuple[int, int, int, int]]:
    """Todas las combinaciones posibles del catalogo, en orden."""
    return [(rep, iso, dsb, rpm)
            for rep in REPS_VALIDAS
            for iso in ISOS_VALIDOS
            for dsb in DSBS_VALIDOS
            for rpm in RPMS_VALIDAS]


# ============================================================
# RECORRIDO DE LA BASE DE DATOS
# ============================================================

def descubrir_archivos(data_dir: Path, ext: str = "") -> tuple[list[Archivo], list[str]]:
    """Recorre la carpeta raiz y devuelve los archivos con sus parametros.

    Busca de forma recursiva todas las subcarpetas cuyo nombre siga la
    nomenclatura ``repN_isoVG_dsbM`` y, dentro de cada una, los archivos cuyo
    nombre (sin extension) termine en ``RpmXXXX``.

    Parameters
    ----------
    data_dir : Path
        Carpeta raiz que contiene las subcarpetas de condiciones.
    ext : str
        Si se indica (por ejemplo ``.txt``), solo se consideran archivos con esa
        extension. Vacio = cualquier extension.

    Returns
    -------
    (archivos, avisos)
        Lista de archivos encontrados y lista de avisos sobre carpetas o
        archivos que no se pudieron interpretar.
    """
    archivos: list[Archivo] = []
    avisos: list[str] = []
    ext = ext.lower()

    carpetas = sorted(p for p in data_dir.rglob("*")
                      if p.is_dir() and PATRON_CARPETA.search(p.name))
    if PATRON_CARPETA.search(data_dir.name):
        carpetas.insert(0, data_dir)

    if not carpetas:
        avisos.append(f"No se encontro ninguna subcarpeta con la nomenclatura "
                      f"repN_isoVG_dsbM dentro de {data_dir}")
        return archivos, avisos

    # Subcarpetas que no siguen la nomenclatura y tampoco contienen ninguna que
    # si la siga: suelen ser erratas en el nombre (dbs en vez de dsb, etc.).
    for hija in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not any(c == hija or hija in c.parents for c in carpetas):
            avisos.append(f"Carpeta ignorada, no sigue la nomenclatura "
                          f"repN_isoVG_dsbM: {hija.name}")

    for carpeta in carpetas:
        m = PATRON_CARPETA.search(carpeta.name)
        rep, iso, dsb = int(m.group("rep")), int(m.group("iso")), int(m.group("dsb"))

        candidatos = sorted(p for p in carpeta.iterdir() if p.is_file())
        if not candidatos:
            avisos.append(f"Carpeta vacia: {carpeta.name}")
            continue

        n_carpeta = 0
        for ruta in candidatos:
            if ext and ruta.suffix.lower() != ext:
                continue
            m_rpm = PATRON_RPM.search(ruta.stem)
            if m_rpm is None:
                avisos.append(f"Sin RPM al final del nombre, se omite: "
                              f"{carpeta.name}/{ruta.name}")
                continue
            archivos.append(Archivo(
                ruta=ruta, rep=rep, iso=iso, dsb=dsb,
                rpm=int(m_rpm.group(1)), bytes=ruta.stat().st_size,
            ))
            n_carpeta += 1

        if n_carpeta == 0:
            avisos.append(f"Sin archivos utilizables: {carpeta.name}")

    archivos.sort(key=lambda a: (a.rep, a.iso, a.dsb, a.rpm, a.ruta.name))
    return archivos, avisos


def agrupar_combinaciones(archivos: list[Archivo]) -> dict[tuple[int, int, int, int], Combinacion]:
    """Agrupa los archivos por combinacion rep_iso_dsb_rpm."""
    combos: dict[tuple[int, int, int, int], Combinacion] = {}
    for a in archivos:
        c = combos.get(a.combo)
        if c is None:
            c = Combinacion(rep=a.rep, iso=a.iso, dsb=a.dsb, rpm=a.rpm)
            combos[a.combo] = c
        c.archivos.append(a)
    return combos


# ============================================================
# DETECCION DE OUTLIERS DE TAMANO
# ============================================================

def _cuartiles(valores: list[float]) -> tuple[float, float]:
    """Primer y tercer cuartil (metodo de la mediana de las mitades)."""
    v = sorted(valores)
    n = len(v)
    mitad = n // 2
    q1 = median(v[:mitad])
    q3 = median(v[mitad + 1:] if n % 2 else v[mitad:])
    return q1, q3


def _umbral_robusto(valores: list[float], k: float) -> tuple[float, float, str]:
    """Centro, umbral de desviacion y metodo usado para marcar outliers.

    Un valor es outlier cuando ``|valor - centro| > umbral``.
    """
    med = median(valores)
    mad = median([abs(v - med) for v in valores])
    if mad > 0:
        return med, k * 1.4826 * mad, "MAD"

    q1, q3 = _cuartiles(valores)
    iqr = q3 - q1
    if iqr > 0:
        return med, k * iqr / 1.349, "IQR"

    # Todos (o casi todos) los archivos pesan lo mismo: cualquier desviacion
    # relativa mayor al 1 % de la mediana ya es un peso anomalo.
    return med, max(abs(med) * 0.01, 1e-9), "tolerancia 1%"


def _fmt_mb(mb: float) -> str:
    """Peso compacto: ``20`` en vez de ``20.00``, pero ``0.15`` sigue completo.

    Un archivo de unos pocos kB se mostraria como ``0``, asi que por debajo de
    0.01 MB se usan cifras significativas en lugar de decimales fijos.
    """
    if 0 < mb < 0.01:
        return f"{mb:.3g}"
    return f"{mb:.2f}".rstrip("0").rstrip(".")


@dataclass
class EstadoCondicion:
    """Estado de una condicion ``repN_isoVG_dsbM`` frente a todas las velocidades.

    Una condicion esta completa cuando tiene un archivo utilizable para cada una
    de las velocidades del catalogo. Una velocidad se considera NO disponible en
    dos casos: cuando no hay ningun archivo (``mis``) y cuando los archivos que
    hay son todos outlier por peso BAJO (``out``), porque un registro mas corto
    de lo normal ha perdido informacion. Un archivo mas pesado de lo normal no
    resta: la informacion sigue ahi, solo sobra. Si una velocidad esta repetida
    y al menos una copia es utilizable, la velocidad cuenta como disponible.
    """

    rep: int
    iso: int
    dsb: int
    n_archivos: int = 0                             #: Archivos con RPM del catalogo.
    mis: list[int] = field(default_factory=list)    #: RPM sin ningun archivo.
    out: list[int] = field(default_factory=list)    #: RPM sin ninguna copia utilizable.
    pesos_out: dict[int, float] = field(default_factory=dict)  #: RPM -> peso perdido [MB].
    altos: list[int] = field(default_factory=list)  #: RPM con algun archivo de peso alto.

    @property
    def etiqueta(self) -> str:
        """Etiqueta de la condicion, sin la velocidad."""
        return f"rep{self.rep}_iso{self.iso}_dsb{self.dsb}"

    @property
    def n_faltantes(self) -> int:
        """Velocidades no disponibles: sin archivo mas outlier."""
        return len(self.mis) + len(self.out)

    @property
    def sin_datos(self) -> bool:
        """True si la condicion no tiene ningun archivo del catalogo."""
        return self.n_archivos == 0

    @property
    def completa(self) -> bool:
        """True si estan todas las velocidades y ninguna es outlier."""
        return self.n_faltantes == 0

    @property
    def texto_estado(self) -> str:
        """Celda de la columna FALTANTES: (completo), (Ninguno) o el numero."""
        if self.sin_datos:
            return "(Ninguno)"
        if self.completa:
            return "(completo)"
        return str(self.n_faltantes)

    @property
    def texto_velocidades(self) -> str:
        """Celda con las velocidades no disponibles: ``mis[...] out[...]``."""
        partes = []
        if self.mis:
            partes.append("mis[" + ", ".join(str(v) for v in sorted(self.mis)) + "]")
        if self.out:
            partes.append("out[" + ", ".join(str(v) for v in sorted(self.out)) + "]")
        return " ".join(partes)

    @property
    def lst_missing(self) -> str:
        """Velocidades sin ningun archivo, separadas por coma."""
        return ", ".join(str(v) for v in sorted(self.mis))

    @property
    def lst_outliers(self) -> str:
        """Velocidades perdidas por peso bajo, con el peso del archivo al lado.

        Formato ``600 (0.15MB), 1300 (0.39MB)``. Cuando la velocidad tiene varias
        copias se muestra la mas pequena, que es la que marca la perdida.
        """
        return ", ".join(f"{v} ({_fmt_mb(self.pesos_out[v])}MB)"
                         for v in sorted(self.out))


def estado_por_condicion(
    combos: dict[tuple[int, int, int, int], Combinacion],
) -> list[EstadoCondicion]:
    """Evalua las 27 condiciones rep_iso_dsb del catalogo contra sus velocidades.

    Se recorren todas las condiciones posibles, no solo las que tienen carpeta,
    para que una condicion sin ningun archivo aparezca igualmente en la tabla.
    """
    estados: list[EstadoCondicion] = []
    for rep in REPS_VALIDAS:
        for iso in ISOS_VALIDOS:
            for dsb in DSBS_VALIDOS:
                e = EstadoCondicion(rep=rep, iso=iso, dsb=dsb)
                for rpm in RPMS_VALIDAS:
                    c = combos.get((rep, iso, dsb, rpm))
                    if c is None:
                        e.mis.append(rpm)
                        continue
                    e.n_archivos += c.cant
                    if any(a.tipo_outlier == "ALTO" for a in c.archivos):
                        e.altos.append(rpm)
                    # Solo se pierde la velocidad si NINGUNA copia es utilizable.
                    if all(a.outlier_bajo for a in c.archivos):
                        e.out.append(rpm)
                        e.pesos_out[rpm] = min(a.mb for a in c.archivos)
                estados.append(e)
    return estados


def tabla_condiciones(estados: list[EstadoCondicion]) -> list[str]:
    """Tabla principal de faltantes: una fila por condicion rep_iso_dsb."""
    return tabla(
        ["CONDICION", "N_ARCHIVOS", "FALTANTES", "VELOCIDADES_FALTANTES"],
        [[e.etiqueta, str(e.n_archivos), e.texto_estado, e.texto_velocidades]
         for e in estados],
    )


#: Columnas del archivo tabulado, en orden.
COLUMNAS_TSV = ("condicion", "n_archivos", "n_missing", "n_outliers",
                "lst_missing", "lst_outliers", "t_missing")


def tabla_tsv(estados: list[EstadoCondicion]) -> str:
    """Tabla separada por tabulaciones, una fila por condicion rep_iso_dsb.

    Pensada para abrirse directamente en Excel o leerse con pandas. Los conteos
    valen 0 cuando no hay nada que contar y las listas quedan vacias.
    """
    lineas = ["\t".join(COLUMNAS_TSV)]
    for e in estados:
        lineas.append("\t".join([
            e.etiqueta,
            str(e.n_archivos),
            str(len(e.mis)),
            str(len(e.out)),
            e.lst_missing,
            e.lst_outliers,
            str(e.n_faltantes),
        ]))
    return "\n".join(lineas) + "\n"


def filas_no_disponibles(estados: list[EstadoCondicion]) -> list[list[str]]:
    """Listado plano de combinaciones no disponibles, por condicion y velocidad."""
    filas: list[list[str]] = []
    for e in estados:
        if e.completa:
            continue
        motivos = {rpm: "SIN ARCHIVO" for rpm in e.mis}
        motivos.update({rpm: "OUTLIER PESO BAJO" for rpm in e.out})
        for rpm in sorted(motivos):
            filas.append([f"{e.etiqueta}_rpm{rpm}", motivos[rpm]])
    return filas


def leyenda_condiciones() -> list[str]:
    """Explicacion de como leer la tabla de condiciones."""
    return [
        f"Una fila por cada una de las {len(REPS_VALIDAS) * len(ISOS_VALIDOS) * len(DSBS_VALIDOS)} "
        f"condiciones rep_iso_dsb posibles.",
        "",
        "Columna FALTANTES:",
        f"  (completo)  estan las {len(RPMS_VALIDAS)} velocidades y ninguna es outlier",
        "  (Ninguno)   la condicion no tiene ningun archivo",
        "  N           numero de velocidades no disponibles (sin archivo + outlier)",
        "",
        "Columna VELOCIDADES_FALTANTES (de menor a mayor):",
        "  mis[...]    velocidades sin ningun archivo",
        "  out[...]    velocidades cuyos archivos son todos outlier por peso BAJO;",
        "              el registro existe pero es mas corto de lo normal, asi que",
        "              ha perdido informacion y no es utilizable",
        "",
        "Un archivo mas PESADO de lo normal no resta: la informacion sigue ahi.",
        "Se reporta en la seccion de outliers, pero no cuenta como faltante.",
        "",
    ]


def _clave_natural(texto: str) -> list:
    """Clave de orden natural: rpm600 antes que rpm1300 (y no al reves)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", texto)]


def marcar_outliers(archivos: list[Archivo], k: float, modo: str) -> list[dict]:
    """Marca en sitio los archivos con peso atipico y devuelve el resumen por grupo.

    Parameters
    ----------
    archivos : list of Archivo
        Archivos a evaluar (se modifican los campos ``outlier`` y ``grupo``).
    k : float
        Numero de desviaciones robustas a partir del cual un peso es atipico.
    modo : {'global', 'rpm', 'condicion'}
        Criterio para formar los grupos de comparacion.

    Returns
    -------
    list of dict
        Un registro por grupo con su centro, umbral, metodo y numero de outliers.
    """
    grupos: dict[str, list[Archivo]] = {}
    for a in archivos:
        if modo == "rpm":
            clave = f"rpm{a.rpm}"
        elif modo == "condicion":
            clave = f"rep{a.rep}_iso{a.iso}_dsb{a.dsb}"
        else:
            clave = "global"
        a.grupo = clave
        grupos.setdefault(clave, []).append(a)

    resumen: list[dict] = []
    for clave, lote in sorted(grupos.items(), key=lambda kv: _clave_natural(kv[0])):
        pesos = [a.mb for a in lote]
        if len(lote) < 4:
            # Muestra demasiado pequena para estimar dispersion de forma fiable.
            resumen.append({
                "grupo": clave, "n": len(lote), "mediana": median(pesos),
                "umbral": float("nan"), "metodo": "sin evaluar (n<4)", "n_outliers": 0,
                "n_bajos": 0, "min": min(pesos), "max": max(pesos),
            })
            continue

        centro, umbral, metodo = _umbral_robusto(pesos, k)
        n_out = n_bajos = 0
        for a in lote:
            a.desviacion = a.mb - centro
            a.outlier = abs(a.desviacion) > umbral
            n_out += int(a.outlier)
            n_bajos += int(a.outlier_bajo)
        resumen.append({
            "grupo": clave, "n": len(lote), "mediana": centro,
            "umbral": umbral, "metodo": metodo, "n_outliers": n_out,
            "n_bajos": n_bajos, "min": min(pesos), "max": max(pesos),
        })
    return resumen


# ============================================================
# FORMATO DE TABLAS
# ============================================================

def tabla(encabezados: list[str], filas: list[list[str]]) -> list[str]:
    """Devuelve las lineas de una tabla de texto con columnas alineadas."""
    if not filas:
        return ["(sin registros)"]
    anchos = [len(h) for h in encabezados]
    for fila in filas:
        for i, celda in enumerate(fila):
            anchos[i] = max(anchos[i], len(celda))

    # Una columna se alinea a la derecha solo si todas sus celdas son numeros.
    derecha = [i > 0 and all(_es_num(f[i]) for f in filas)
               for i in range(len(encabezados))]

    def fmt(fila: list[str]) -> str:
        celdas = [c.rjust(anchos[i]) if derecha[i] else c.ljust(anchos[i])
                  for i, c in enumerate(fila)]
        return "  ".join(celdas).rstrip()

    sep = "  ".join("-" * a for a in anchos)
    return [fmt(encabezados), sep] + [fmt(f) for f in filas]


def _es_num(texto: str) -> bool:
    """True si la celda es un numero (para alinearla a la derecha)."""
    try:
        float(texto)
        return True
    except ValueError:
        return False


def titulo(texto: str, nivel: int = 1) -> list[str]:
    """Encabezado de seccion del informe."""
    car = "=" if nivel == 1 else "-"
    return ["", car * 78, texto, car * 78, ""]


# ============================================================
# INFORMES
# ============================================================

def informe_completo(
    data_dir: Path,
    archivos: list[Archivo],
    combos: dict[tuple[int, int, int, int], Combinacion],
    faltantes: list[tuple[int, int, int, int]],
    estados: list[EstadoCondicion],
    fuera_catalogo: list[Archivo],
    resumen_grupos: list[dict],
    avisos: list[str],
    args: argparse.Namespace,
) -> str:
    """Construye el texto del informe completo."""
    lineas: list[str] = []
    esperadas = malla_completa()
    encontradas_validas = [c for c in combos.values() if combo_valido(c.combo)]
    repetidas = [c for c in combos.values() if c.cant > 1]
    outliers = [a for a in archivos if a.outlier]
    total_mb = sum(a.mb for a in archivos)
    n_solo_outlier = sum(len(e.out) for e in estados)
    n_no_disponibles = sum(e.n_faltantes for e in estados)
    completas = [e for e in estados if e.completa]
    sin_datos = [e for e in estados if e.sin_datos]

    lineas += titulo("INVENTARIO DE COMBINACIONES rep_iso_dsb_rpm")
    lineas += [
        f"Carpeta analizada : {data_dir.resolve()}",
        f"Extension filtrada: {args.ext or '(todas)'}",
        f"Criterio outlier  : k={args.k_outlier:g} desviaciones robustas, "
        f"grupo='{args.grupo_outliers}'",
        "",
        "Catalogo de valores validos:",
        f"  rep : {', '.join(str(v) for v in REPS_VALIDAS)}",
        f"  iso : {', '.join(str(v) for v in ISOS_VALIDOS)}",
        f"  dsb : {', '.join(str(v) for v in DSBS_VALIDOS)}",
        f"  rpm : {', '.join(str(v) for v in RPMS_VALIDAS)}",
        "",
        "RESUMEN",
        f"  Archivos encontrados             : {len(archivos)}",
        f"  Peso total                       : {total_mb:.2f} MB",
        f"  Condiciones rep_iso_dsb          : {len(estados)} "
        f"(completas: {len(completas)}, sin ningun archivo: {len(sin_datos)})",
        f"  Combinaciones posibles (catalogo): {len(esperadas)}",
        f"  Combinaciones con archivo        : {len(encontradas_validas)}",
        f"  Combinaciones sin archivo        : {len(faltantes)}",
        f"  Combinaciones perdidas por peso  : {n_solo_outlier} "
        f"(sin ninguna copia utilizable)",
        f"  Combinaciones NO disponibles     : {n_no_disponibles} "
        f"(sin archivo + peso bajo)",
        f"  Combinaciones repetidas (cant>1) : {len(repetidas)}",
        f"  Archivos con peso outlier        : {len(outliers)} "
        f"(bajos: {sum(1 for a in outliers if a.outlier_bajo)}, "
        f"altos: {sum(1 for a in outliers if not a.outlier_bajo)})",
        f"  Archivos fuera de catalogo       : {len(fuera_catalogo)}",
        f"  Cobertura utilizable             : "
        f"{100 * (len(esperadas) - n_no_disponibles) / len(esperadas):.1f} %",
    ]

    # ---- Tabla principal: un renglon por archivo -------------------------
    lineas += titulo("1. TABLA DE ARCHIVOS ENCONTRADOS", nivel=2)
    lineas += [
        "ETIQUETA lleva el sufijo _cantN cuando la combinacion aparece N>1 veces.",
        "PESO_MB en megabytes (MiB, base 1024). ESTADO: OK / OUTLIER-TAMANO.",
        "",
    ]
    filas = []
    for combo in sorted(combos):
        c = combos[combo]
        for a in c.archivos:
            filas.append([
                c.etiqueta, str(a.rep), str(a.iso), str(a.dsb), str(a.rpm),
                str(c.cant), f"{a.mb:.3f}",
                "OUTLIER-TAMANO" if a.outlier else "OK",
                "" if a.valido else "FUERA-CATALOGO",
                str(a.ruta.relative_to(data_dir)),
            ])
    lineas += tabla(
        ["ETIQUETA", "REP", "ISO", "DSB", "RPM", "CANT", "PESO_MB",
         "ESTADO", "CATALOGO", "ARCHIVO"],
        filas,
    )

    # ---- Combinaciones repetidas ----------------------------------------
    lineas += titulo("2. COMBINACIONES REPETIDAS (parametro adicional _cantN)", nivel=2)
    if repetidas:
        filas = [[c.etiqueta, str(c.cant),
                  f"{sum(a.mb for a in c.archivos):.3f}",
                  " | ".join(a.ruta.name for a in c.archivos)]
                 for c in sorted(repetidas, key=lambda x: (-x.cant, x.rep, x.iso, x.dsb, x.rpm))]
        lineas += tabla(["ETIQUETA", "CANT", "PESO_TOTAL_MB", "ARCHIVOS"], filas)
    else:
        lineas += ["No hay combinaciones repetidas: cada una aparece una sola vez."]

    # ---- Estado por condicion (tabla principal de faltantes) -------------
    lineas += titulo("3. ESTADO POR CONDICION rep_iso_dsb", nivel=2)
    lineas += leyenda_condiciones()
    lineas += tabla_condiciones(estados)

    incompletas = [e for e in estados if not e.completa]
    if incompletas:
        lineas += ["", f"Condiciones completas: {len(estados) - len(incompletas)} "
                       f"de {len(estados)}.", "",
                   "Detalle de las combinaciones no disponibles:"]
        lineas += tabla(["ETIQUETA", "MOTIVO"], filas_no_disponibles(estados))
    else:
        lineas += ["", "Todas las condiciones estan completas: no falta ninguna "
                       "velocidad y ningun archivo es outlier."]

    # ---- Outliers de tamano ---------------------------------------------
    lineas += titulo("4. OUTLIERS POR TAMANO DE ARCHIVO", nivel=2)
    lineas += [
        "TIPO BAJO: pesa menos de lo normal, el registro perdio informacion y la",
        "velocidad se cuenta como faltante. TIPO ALTO: pesa mas de lo normal, se",
        "reporta pero NO cuenta como faltante.",
        "",
        "Estadisticos por grupo de comparacion:",
        "",
    ]
    lineas += tabla(
        ["GRUPO", "N", "MEDIANA_MB", "MIN_MB", "MAX_MB", "DESV_MAX_MB", "METODO",
         "N_OUTLIERS", "N_BAJOS"],
        [[g["grupo"], str(g["n"]), f"{g['mediana']:.3f}", f"{g['min']:.3f}",
          f"{g['max']:.3f}",
          "-" if g["umbral"] != g["umbral"] else f"{g['umbral']:.3f}",
          g["metodo"], str(g["n_outliers"]), str(g["n_bajos"])]
         for g in resumen_grupos],
    )
    lineas += [""]
    if outliers:
        centro = {g["grupo"]: g["mediana"] for g in resumen_grupos}
        lineas += tabla(
            ["ETIQUETA", "TIPO", "PESO_MB", "MEDIANA_GRUPO_MB", "DESVIACION_MB",
             "GRUPO", "ARCHIVO"],
            [[combos[a.combo].etiqueta, a.tipo_outlier, f"{a.mb:.3f}",
              f"{centro[a.grupo]:.3f}", f"{a.desviacion:+.3f}", a.grupo,
              str(a.ruta.relative_to(data_dir))]
             for a in sorted(outliers, key=lambda x: -abs(x.desviacion))],
        )
    else:
        lineas += ["No se detectaron pesos atipicos."]

    # ---- Fuera de catalogo ----------------------------------------------
    lineas += titulo("5. ARCHIVOS CON PARAMETROS FUERA DEL CATALOGO", nivel=2)
    if fuera_catalogo:
        lineas += [
            "Estos archivos existen pero alguno de sus parametros no esta en la",
            "lista de valores validos; no cuentan para la cobertura.",
            "",
        ]
        lineas += tabla(
            ["ETIQUETA", "REP", "ISO", "DSB", "RPM", "PESO_MB", "ARCHIVO"],
            [[etiqueta_combo(a.combo), str(a.rep), str(a.iso), str(a.dsb),
              str(a.rpm), f"{a.mb:.3f}", str(a.ruta.relative_to(data_dir))]
             for a in fuera_catalogo],
        )
    else:
        lineas += ["Todos los archivos tienen parametros dentro del catalogo."]

    # ---- Avisos ----------------------------------------------------------
    lineas += titulo("6. AVISOS DEL RECORRIDO", nivel=2)
    lineas += avisos if avisos else ["Sin avisos."]

    return "\n".join(lineas) + "\n"


def informe_acciones(
    data_dir: Path,
    combos: dict[tuple[int, int, int, int], Combinacion],
    faltantes: list[tuple[int, int, int, int]],
    estados: list[EstadoCondicion],
    outliers: list[Archivo],
    resumen_grupos: list[dict],
) -> str:
    """Construye el texto del informe reducido: solo faltantes y outliers."""
    centro = {g["grupo"]: g["mediana"] for g in resumen_grupos}
    lineas: list[str] = []
    incompletas = [e for e in estados if not e.completa]
    n_no_disponibles = sum(e.n_faltantes for e in estados)

    lineas += titulo("COMBINACIONES FALTANTES Y OUTLIERS DE TAMANO")
    lineas += [
        f"Carpeta analizada: {data_dir.resolve()}",
        "",
        "Este archivo contiene solo lo que hay que atender:",
        f"  * {len(faltantes)} combinacion{'es' if len(faltantes) != 1 else ''} "
        f"sin ningun archivo.",
        f"  * {sum(len(e.out) for e in estados)} combinaciones perdidas por peso "
        f"bajo (sin ninguna copia utilizable).",
        f"  * {n_no_disponibles} combinaciones NO disponibles en total, sobre "
        f"{len(malla_completa())} posibles.",
        f"  * {len(incompletas)} de {len(estados)} condiciones rep_iso_dsb "
        f"incompletas.",
    ]

    lineas += titulo("A. ESTADO POR CONDICION rep_iso_dsb", nivel=2)
    lineas += leyenda_condiciones()
    lineas += tabla_condiciones(estados)

    lineas += titulo("B. LISTADO PLANO DE COMBINACIONES NO DISPONIBLES", nivel=2)
    if incompletas:
        lineas += tabla(["ETIQUETA", "MOTIVO"], filas_no_disponibles(estados))
    else:
        lineas += ["(ninguna: la malla esta completa y sin outliers)"]

    lineas += titulo("C. DETALLE DE LOS ARCHIVOS OUTLIER", nivel=2)
    lineas += [
        "Solo los de TIPO BAJO cuentan como dato faltante; los de TIPO ALTO se",
        "reportan porque conviene revisarlos, pero no restan.",
        "Un archivo BAJO tampoco resta si su combinacion esta repetida y alguna",
        "copia tiene un peso normal: la velocidad sigue disponible.",
        "",
    ]
    if outliers:
        lineas += tabla(
            ["ETIQUETA", "TIPO", "PESO_MB", "MEDIANA_GRUPO_MB", "DESVIACION_MB",
             "ARCHIVO"],
            [[combos[a.combo].etiqueta, a.tipo_outlier, f"{a.mb:.3f}",
              f"{centro[a.grupo]:.3f}", f"{a.desviacion:+.3f}",
              str(a.ruta.relative_to(data_dir))]
             for a in sorted(outliers, key=lambda x: -abs(x.desviacion))],
        )
    else:
        lineas += ["(ninguno: todos los pesos son consistentes)"]

    return "\n".join(lineas) + "\n"


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def procesar(args: argparse.Namespace) -> int:
    data_dir = Path(args.carpeta or args.data_dir).expanduser()
    if not data_dir.is_dir():
        print(f"ERROR: la carpeta no existe: {data_dir.resolve()}", file=sys.stderr)
        return 1

    _aplicar_catalogo(args)

    print(f"Recorriendo {data_dir.resolve()} ...")
    archivos, avisos = descubrir_archivos(data_dir, args.ext)
    if not archivos:
        print("ERROR: no se encontro ningun archivo con nomenclatura "
              "repN_isoVG_dsbM/...RpmXXXX", file=sys.stderr)
        for aviso in avisos:
            print(f"  - {aviso}", file=sys.stderr)
        return 1

    resumen_grupos = marcar_outliers(archivos, args.k_outlier, args.grupo_outliers)
    combos = agrupar_combinaciones(archivos)

    encontradas = {c for c in combos if combo_valido(c)}
    faltantes = [c for c in malla_completa() if c not in encontradas]
    fuera_catalogo = [a for a in archivos if not a.valido]
    outliers = [a for a in archivos if a.outlier]
    estados = estado_por_condicion(combos)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    ruta_completo = outdir / "inventario_combinaciones.txt"
    ruta_acciones = outdir / "faltantes_y_outliers.txt"
    ruta_tsv = outdir / "tabla_condiciones.txt"

    ruta_completo.write_text(
        informe_completo(data_dir, archivos, combos, faltantes, estados,
                         fuera_catalogo, resumen_grupos, avisos, args),
        encoding="utf-8",
    )
    ruta_acciones.write_text(
        informe_acciones(data_dir, combos, faltantes, estados, outliers,
                         resumen_grupos),
        encoding="utf-8",
    )
    ruta_tsv.write_text(tabla_tsv(estados), encoding="utf-8")

    repetidas = sum(1 for c in combos.values() if c.cant > 1)
    total_esperadas = len(malla_completa())
    n_no_disp = sum(e.n_faltantes for e in estados)
    completas = sum(1 for e in estados if e.completa)
    sin_datos = sum(1 for e in estados if e.sin_datos)
    print(f"  Archivos encontrados      : {len(archivos)}")
    print(f"  Condiciones completas     : {completas} de {len(estados)}"
          + (f" ({sin_datos} sin ningun archivo)" if sin_datos else ""))
    print(f"  Combinaciones con archivo : {len(encontradas)} de {total_esperadas}")
    print(f"  Combinaciones repetidas   : {repetidas}")
    print(f"  Combinaciones sin archivo : {len(faltantes)}")
    print(f"  Perdidas por peso bajo    : {sum(len(e.out) for e in estados)}")
    print(f"  Combinaciones NO disponib.: {n_no_disp} "
          f"(cobertura utilizable {100 * (total_esperadas - n_no_disp) / total_esperadas:.1f} %)")
    print(f"  Archivos con peso outlier : {len(outliers)} "
          f"(bajos: {sum(1 for a in outliers if a.outlier_bajo)}, "
          f"altos: {sum(1 for a in outliers if not a.outlier_bajo)})")
    if fuera_catalogo:
        print(f"  Archivos fuera de catalogo: {len(fuera_catalogo)}")
    if avisos:
        print(f"  Avisos                    : {len(avisos)} (ver informe)")
    print(f"\nInforme completo    : {ruta_completo}")
    print(f"Faltantes + outliers: {ruta_acciones}")
    print(f"Tabla tabulada      : {ruta_tsv}")
    return 0


def _aplicar_catalogo(args: argparse.Namespace) -> None:
    """Sustituye el catalogo por defecto si se paso por linea de comandos."""
    global REPS_VALIDAS, ISOS_VALIDOS, DSBS_VALIDOS, RPMS_VALIDAS
    if args.reps:
        REPS_VALIDAS = tuple(args.reps)
    if args.isos:
        ISOS_VALIDOS = tuple(args.isos)
    if args.dsbs:
        DSBS_VALIDOS = tuple(args.dsbs)
    if args.rpms:
        RPMS_VALIDAS = tuple(args.rpms)


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inventario de combinaciones rep_iso_dsb_rpm: que hay, "
                    "cuantas veces, cuanto pesa, que falta y que pesa raro.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplo: python inventario_combinaciones.py /ruta/a/datos --outdir inventario",
    )
    p.add_argument("carpeta", nargs="?", default=None,
                   help="Carpeta raiz con las subcarpetas repN_isoVG_dsbM")
    p.add_argument("--data-dir", default="datos",
                   help="Igual que el argumento posicional (def: ./datos)")
    p.add_argument("--outdir", default="inventario",
                   help="Carpeta de salida para los dos .txt (def: ./inventario)")
    p.add_argument("--ext", default="",
                   help="Extension a considerar, p.ej. .txt (def: todas)")
    p.add_argument("--k-outlier", dest="k_outlier", type=float, default=3.0,
                   help="Desviaciones robustas para marcar un peso atipico (def: 3)")
    p.add_argument("--grupo-outliers", dest="grupo_outliers", default="global",
                   choices=("global", "rpm", "condicion"),
                   help="Grupo con el que se compara el peso de cada archivo (def: global)")
    p.add_argument("--reps", type=int, nargs="+", default=None,
                   help="Sustituye la lista de repeticiones validas")
    p.add_argument("--isos", type=int, nargs="+", default=None,
                   help="Sustituye la lista de grados ISO validos")
    p.add_argument("--dsbs", type=int, nargs="+", default=None,
                   help="Sustituye la lista de desbalances validos")
    p.add_argument("--rpms", type=int, nargs="+", default=None,
                   help="Sustituye la lista de RPM validas")
    return p


if __name__ == "__main__":
    sys.exit(procesar(construir_parser().parse_args()))
