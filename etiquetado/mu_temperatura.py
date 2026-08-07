"""
mu_temperatura.py
=================

Calcula la viscosidad real del lubricante en funcion de la temperatura medida en
cada ensayo, usando la ecuacion de Walther (norma ASTM D341), y anade al archivo
de entrada las columnas mu1, mu2 (viscosidad a la temperatura del cojinete 1 y 2)
y mu_prom (promedio de ambas).

--------------------------------------------------------------------------
LA FORMULA (ASTM D341 / Walther)
--------------------------------------------------------------------------
    log10(log10(Z)) = A - B * log10(T)          con  Z = nu + 0.7 + f(nu)

donde nu es la viscosidad cinematica [cSt = mm^2/s], T la temperatura ABSOLUTA
[K] y f(nu) una correccion despreciable por encima de ~2 cSt. Con DOS puntos
(temperatura, viscosidad) conocidos de un aceite se despejan A y B, y con ellos
se calcula nu a cualquier temperatura.

--------------------------------------------------------------------------
COMO SE INDICA CADA ACEITE (ver ACEITES abajo)
--------------------------------------------------------------------------
Cada grado ISO (32/46/68) se define por sus DOS viscosidades de referencia:
  * nu40  : viscosidad cinematica a 40 C  (= el numero del grado ISO VG)
  * nu100 : viscosidad cinematica a 100 C (de la ficha tecnica del aceite)
El "indice de viscosidad" (VI) se deriva justamente de esos dos numeros, asi que
dar nu40 y nu100 es lo mas directo y exacto (ver la nota final del README).
Tambien puedes pasar un archivo JSON con --config para no editar el codigo.

Uso:
    python mu_temperatura.py --entrada ensayos.txt
    python mu_temperatura.py --entrada ensayos.txt --salida ensayos_mu.txt --config aceites.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ============================================================
# DEFINICION DE LOS ACEITES  (EDITA ESTOS VALORES CON TU FICHA TECNICA)
# nu40  = viscosidad a 40 C  [cSt]  (normalmente = el grado ISO VG)
# nu100 = viscosidad a 100 C [cSt]  (de la ficha del lubricante)
# Los valores por defecto son tipicos de aceite mineral con VI ~ 100.
# ============================================================

ACEITES = {
    32: {"nu40": 32.0, "nu100": 5.4},
    46: {"nu40": 46.0, "nu100": 6.8},
    68: {"nu40": 68.0, "nu100": 8.7},
}

TEMP_REF_BAJA = 40.0    # C, temperatura del primer punto de referencia
TEMP_REF_ALTA = 100.0   # C, temperatura del segundo punto de referencia


# ============================================================
# ECUACION DE WALTHER (ASTM D341)
# ============================================================

def _Z(nu: float) -> float:
    """Z = nu + 0.7 + correccion (ASTM D341). La correccion solo pesa si nu < ~2."""
    return nu + 0.7 + math.exp(-1.47 - 1.84 * nu - 0.51 * nu * nu)


def _nu_desde_Z(Z: float) -> float:
    """Invierte Z -> nu resolviendo la correccion de forma iterativa."""
    nu = Z - 0.7
    for _ in range(60):
        nu_new = Z - 0.7 - math.exp(-1.47 - 1.84 * nu - 0.51 * nu * nu)
        if abs(nu_new - nu) < 1e-12:
            nu = nu_new
            break
        nu = nu_new
    return nu


def ajustar_walther(t1_c: float, nu1: float, t2_c: float, nu2: float):
    """Devuelve (A, B) de log10(log10(Z)) = A - B*log10(T_K) a partir de 2 puntos."""
    x1 = math.log10(t1_c + 273.15)
    x2 = math.log10(t2_c + 273.15)
    w1 = math.log10(math.log10(_Z(nu1)))
    w2 = math.log10(math.log10(_Z(nu2)))
    B = (w1 - w2) / (x2 - x1)
    A = w1 + B * x1
    return A, B


def nu_a_temperatura(A: float, B: float, t_c: float) -> float:
    """Viscosidad cinematica [cSt] a la temperatura t_c [C] segun (A, B)."""
    w = A - B * math.log10(t_c + 273.15)
    Z = 10.0 ** (10.0 ** w)
    return _nu_desde_Z(Z)


def coeficientes_por_aceite(aceites: dict) -> dict:
    """Precalcula (A, B) de Walther para cada grado ISO."""
    coef = {}
    for iso, d in aceites.items():
        A, B = ajustar_walther(TEMP_REF_BAJA, d["nu40"], TEMP_REF_ALTA, d["nu100"])
        coef[int(iso)] = (A, B)
    return coef


# ============================================================
# LECTURA / ESCRITURA DEL ARCHIVO (preserva el formato original)
# ============================================================

def _split(linea: str, sep: str | None):
    return linea.split(sep) if sep else linea.split()


def _es_nan(txt: str) -> bool:
    return txt is None or txt.strip() == "" or txt.strip().lower() in ("nan", "na", "-")


def _a_float(txt: str):
    if _es_nan(txt):
        return None
    try:
        return float(txt.strip().replace(",", "."))
    except ValueError:
        return None


def procesar(args) -> int:
    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_file():
        print(f"ERROR: no existe la entrada {entrada}")
        return 1

    aceites = ACEITES
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        aceites = {int(k): v for k, v in cfg.items()}
    coef = coeficientes_por_aceite(aceites)

    sep = None if args.sep == "auto" else ("\t" if args.sep == "tab" else args.sep)

    lineas = entrada.read_text(encoding="utf-8", errors="ignore").splitlines()
    # primera linea no vacia = cabecera
    h = next((i for i, ln in enumerate(lineas) if ln.strip()), None)
    if h is None:
        print("ERROR: archivo vacio.")
        return 1
    cabecera = _split(lineas[h], sep)
    ncol = len(cabecera)
    low = [c.strip().lower() for c in cabecera]

    def col(nombre):
        return low.index(nombre) if nombre in low else None

    i_iso, i_t1, i_t2 = col("iso"), col("t1"), col("t2")
    faltan = [n for n, i in (("iso", i_iso), ("T1", i_t1), ("T2", i_t2)) if i is None]
    if faltan:
        print(f"ERROR: la cabecera no tiene las columnas {faltan}. Cabecera: {cabecera}")
        return 1

    salida = (Path(args.salida).expanduser().resolve() if args.salida
              else entrada.with_name(entrada.stem + "_mu" + entrada.suffix))

    n_ok = n_nan = 0
    out = ["\t".join(cabecera + ["mu1", "mu2", "mu_prom"])]
    for ln in lineas[h + 1:]:
        if not ln.strip():
            continue
        campos = _split(ln, sep)
        campos = (campos + ["NaN"] * (ncol - len(campos)))[:ncol]  # normaliza filas cortas

        iso_v = _a_float(campos[i_iso])
        t1 = _a_float(campos[i_t1])
        t2 = _a_float(campos[i_t2])

        def mu(T):
            if T is None or iso_v is None or int(iso_v) not in coef:
                return None
            A, B = coef[int(iso_v)]
            return nu_a_temperatura(A, B, T)

        mu1, mu2 = mu(t1), mu(t2)
        if mu1 is not None and mu2 is not None:
            mu_prom = 0.5 * (mu1 + mu2)
        elif mu1 is not None:
            mu_prom = mu1
        elif mu2 is not None:
            mu_prom = mu2
        else:
            mu_prom = None

        def fmt(x):
            return f"{x:.4f}" if x is not None else "NaN"

        out.append("\t".join(campos + [fmt(mu1), fmt(mu2), fmt(mu_prom)]))
        if mu_prom is not None:
            n_ok += 1
        else:
            n_nan += 1

    salida.write_text("\n".join(out) + "\n", encoding="utf-8")

    print("Aceites (grado ISO -> nu40, nu100 [cSt] -> Walther A,B):")
    for iso in sorted(coef):
        A, B = coef[iso]
        d = aceites[iso]
        print(f"  ISO {iso}: nu40={d['nu40']:.1f}  nu100={d['nu100']:.1f}  "
              f"| A={A:.4f} B={B:.4f} | mu(25C)={nu_a_temperatura(A, B, 25):.1f} "
              f"mu(40C)={nu_a_temperatura(A, B, 40):.1f} mu(60C)={nu_a_temperatura(A, B, 60):.1f} cSt")
    print(f"\nFilas con viscosidad calculada: {n_ok} | sin datos (NaN): {n_nan}")
    print(f"mu = viscosidad cinematica [cSt = mm^2/s]. Salida: {salida}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Anade mu1, mu2 y mu_prom (viscosidad ASTM D341 segun T1, T2) "
                    "a un duplicado del archivo de ensayos.")
    p.add_argument("--entrada", required=True, help="Archivo de ensayos (tabular)")
    p.add_argument("--salida", default="", help="Archivo de salida (def: <entrada>_mu)")
    p.add_argument("--config", default="", help="JSON opcional con los aceites (grado -> {nu40,nu100})")
    p.add_argument("--sep", default="tab",
                   help="Separador: 'tab' (def), 'auto' (espacios) o un caracter")
    return p


if __name__ == "__main__":
    sys.exit(procesar(construir_parser().parse_args()))
