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
    (rep x iso x dsb x rpm) y senala las que faltan.
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

--------------------------------------------------------------------------
CRITERIO DE OUTLIER
--------------------------------------------------------------------------
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

    @property
    def mb(self) -> float:
        """Peso del archivo en megabytes."""
        return self.bytes / BYTES_POR_MB

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
                "min": min(pesos), "max": max(pesos),
            })
            continue

        centro, umbral, metodo = _umbral_robusto(pesos, k)
        n_out = 0
        for a in lote:
            a.outlier = abs(a.mb - centro) > umbral
            n_out += int(a.outlier)
        resumen.append({
            "grupo": clave, "n": len(lote), "mediana": centro,
            "umbral": umbral, "metodo": metodo, "n_outliers": n_out,
            "min": min(pesos), "max": max(pesos),
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
        f"  Combinaciones posibles (catalogo): {len(esperadas)}",
        f"  Combinaciones encontradas        : {len(encontradas_validas)}",
        f"  Combinaciones faltantes          : {len(faltantes)}",
        f"  Combinaciones repetidas (cant>1) : {len(repetidas)}",
        f"  Archivos con peso outlier        : {len(outliers)}",
        f"  Archivos fuera de catalogo       : {len(fuera_catalogo)}",
        f"  Cobertura                        : "
        f"{100 * len(encontradas_validas) / len(esperadas):.1f} %",
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

    # ---- Combinaciones faltantes ----------------------------------------
    lineas += titulo("3. COMBINACIONES FALTANTES (no estan en la base de datos)", nivel=2)
    if faltantes:
        lineas += [f"Faltan {len(faltantes)} de {len(esperadas)} combinaciones posibles.", ""]
        lineas += tabla(
            ["ETIQUETA", "REP", "ISO", "DSB", "RPM"],
            [[etiqueta_combo(c), str(c[0]), str(c[1]), str(c[2]), str(c[3])]
             for c in faltantes],
        )
        lineas += ["", "Resumen de faltantes por condicion rep_iso_dsb:"]
        por_cond: dict[tuple[int, int, int], list[int]] = {}
        for rep, iso, dsb, rpm in faltantes:
            por_cond.setdefault((rep, iso, dsb), []).append(rpm)
        lineas += tabla(
            ["CONDICION", "N_FALTANTES", "RPM_FALTANTES"],
            [[f"rep{r}_iso{i}_dsb{d}", str(len(rpms)),
              ", ".join(str(v) for v in sorted(rpms))]
             for (r, i, d), rpms in sorted(por_cond.items())],
        )
    else:
        lineas += ["No falta ninguna combinacion: la malla esta completa."]

    # ---- Outliers de tamano ---------------------------------------------
    lineas += titulo("4. OUTLIERS POR TAMANO DE ARCHIVO", nivel=2)
    lineas += ["Estadisticos por grupo de comparacion:", ""]
    lineas += tabla(
        ["GRUPO", "N", "MEDIANA_MB", "MIN_MB", "MAX_MB", "DESV_MAX_MB", "METODO", "N_OUTLIERS"],
        [[g["grupo"], str(g["n"]), f"{g['mediana']:.3f}", f"{g['min']:.3f}",
          f"{g['max']:.3f}",
          "-" if g["umbral"] != g["umbral"] else f"{g['umbral']:.3f}",
          g["metodo"], str(g["n_outliers"])]
         for g in resumen_grupos],
    )
    lineas += [""]
    if outliers:
        centro = {g["grupo"]: g["mediana"] for g in resumen_grupos}
        lineas += tabla(
            ["ETIQUETA", "PESO_MB", "MEDIANA_GRUPO_MB", "DESVIACION_MB", "GRUPO", "ARCHIVO"],
            [[combos[a.combo].etiqueta, f"{a.mb:.3f}", f"{centro[a.grupo]:.3f}",
              f"{a.mb - centro[a.grupo]:+.3f}", a.grupo,
              str(a.ruta.relative_to(data_dir))]
             for a in sorted(outliers, key=lambda x: -abs(x.mb - centro[x.grupo]))],
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
    outliers: list[Archivo],
    resumen_grupos: list[dict],
) -> str:
    """Construye el texto del informe reducido: solo faltantes y outliers."""
    centro = {g["grupo"]: g["mediana"] for g in resumen_grupos}
    lineas: list[str] = []

    lineas += titulo("COMBINACIONES FALTANTES Y OUTLIERS DE TAMANO")
    lineas += [
        f"Carpeta analizada: {data_dir.resolve()}",
        "",
        "Este archivo contiene solo las combinaciones que hay que atender:",
        f"  * {len(faltantes)} combinacion{'es' if len(faltantes) != 1 else ''} "
        f"que NO {'estan' if len(faltantes) != 1 else 'esta'} medida"
        f"{'s' if len(faltantes) != 1 else ''}.",
        f"  * {len(outliers)} archivo{'s' if len(outliers) != 1 else ''} "
        f"medido{'s' if len(outliers) != 1 else ''} con un peso atipico.",
    ]

    lineas += titulo("A. COMBINACIONES QUE NO ESTAN", nivel=2)
    if faltantes:
        lineas += [etiqueta_combo(c) for c in faltantes]
    else:
        lineas += ["(ninguna: la malla esta completa)"]

    lineas += titulo("B. COMBINACIONES CON PESO OUTLIER", nivel=2)
    if outliers:
        lineas += tabla(
            ["ETIQUETA", "PESO_MB", "MEDIANA_GRUPO_MB", "DESVIACION_MB", "ARCHIVO"],
            [[combos[a.combo].etiqueta, f"{a.mb:.3f}", f"{centro[a.grupo]:.3f}",
              f"{a.mb - centro[a.grupo]:+.3f}", str(a.ruta.relative_to(data_dir))]
             for a in sorted(outliers, key=lambda x: -abs(x.mb - centro[x.grupo]))],
        )
    else:
        lineas += ["(ninguna: todos los pesos son consistentes)"]

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

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    ruta_completo = outdir / "inventario_combinaciones.txt"
    ruta_acciones = outdir / "faltantes_y_outliers.txt"

    ruta_completo.write_text(
        informe_completo(data_dir, archivos, combos, faltantes,
                         fuera_catalogo, resumen_grupos, avisos, args),
        encoding="utf-8",
    )
    ruta_acciones.write_text(
        informe_acciones(data_dir, combos, faltantes, outliers, resumen_grupos),
        encoding="utf-8",
    )

    repetidas = sum(1 for c in combos.values() if c.cant > 1)
    total_esperadas = len(malla_completa())
    print(f"  Archivos encontrados      : {len(archivos)}")
    print(f"  Combinaciones encontradas : {len(encontradas)} de {total_esperadas} "
          f"({100 * len(encontradas) / total_esperadas:.1f} %)")
    print(f"  Combinaciones repetidas   : {repetidas}")
    print(f"  Combinaciones faltantes   : {len(faltantes)}")
    print(f"  Archivos con peso outlier : {len(outliers)}")
    if fuera_catalogo:
        print(f"  Archivos fuera de catalogo: {len(fuera_catalogo)}")
    if avisos:
        print(f"  Avisos                    : {len(avisos)} (ver informe)")
    print(f"\nInforme completo    : {ruta_completo}")
    print(f"Faltantes + outliers: {ruta_acciones}")
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
