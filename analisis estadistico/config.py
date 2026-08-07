"""
config.py — parametros comunes del modulo de analisis estadistico.
==================================================================

Edita aqui las rutas por defecto y las constantes del ensayo. Todos los scripts
aceptan ademas argumentos por linea de comandos que tienen prioridad sobre estos
valores.
"""

from __future__ import annotations

# ============================================================
# RUTAS POR DEFECTO  (se pueden sobreescribir con --entrada / --salida)
# ============================================================

# Carpeta donde estan los .xlsx originales (rep1_32_1_p_0424.xlsx, ...)
DIR_EXCEL = r"C:\Users\Owner\Documents\BD\All data"

# Carpeta donde se escriben todos los resultados de este modulo
DIR_RESULTADOS = r"C:\Users\Owner\Documents\BD\resultados_analisis"


# ============================================================
# DISENO EXPERIMENTAL
# ============================================================

REPETICIONES = [1, 2, 3]        # bloques
VISCOSIDADES = [32, 46, 68]     # parcela principal  (grado ISO VG)
DESBALANCEOS = [1, 2, 3]        # subparcela
VELOCIDADES = [600, 1300, 2000, 2500, 2600, 2700, 2800,
               3400, 3500, 3600, 3700, 4000, 4500, 5000, 5500]  # sub-subparcela

# Numero de archivos .xlsx esperados = rep x visc x desb
N_ARCHIVOS_ESPERADOS = len(REPETICIONES) * len(VISCOSIDADES) * len(DESBALANCEOS)  # 27

# Velocidad usada como "slow roll" para la compensacion de runout (paso 3)
RPM_SLOW_ROLL = 600


# ============================================================
# SENSORES
# ============================================================

SENSORES = ["P1Y", "P1X", "P2Y", "P2X"]
COJINETES = ["P1", "P2"]
DIRECCIONES = ["Y", "X"]        # orden de columnas en las figuras 2x2


# ============================================================
# DETECCION DE ESTABILIZACION (paso 1)
# ============================================================
# Cada medicion dura ~3 min. Se busca el tramo final mas largo en el que la
# amplitud y la fase ya estan estabilizadas (sin deriva y con poca dispersion).

FRAC_MINIMA_ESTABLE = 0.30   # el tramo estable debe cubrir >= 30 % del registro

# Criterio principal: que el fasor YA NO DERIVE (el transitorio termino).
TOL_DERIVA_AMP = 0.05        # cambio maximo de amplitud entre mitades (fraccion)
TOL_DERIVA_FASE = 5.0        # cambio maximo de fase entre mitades [grados]

# Criterio secundario: que el ruido dentro del tramo no sea excesivo. A alta
# velocidad la dispersion real crece, por eso las tolerancias son holgadas.
TOL_CV_AMP = 0.15            # coef. de variacion robusto maximo de la amplitud
TOL_DISP_FASE = 12.0         # desviacion CIRCULAR maxima de la fase [grados]

N_CANDIDATOS = 120           # posiciones de inicio evaluadas por sensor


# ============================================================
# GRAFICAS (pasos 2 y 4)
# ============================================================
# Color = viscosidad ; saturacion/oscuridad = desbalanceo ; estilo = repeticion.

COLOR_BASE_VISCOSIDAD = {
    32: (0.85, 0.08, 0.08),   # rojo
    46: (0.10, 0.25, 0.85),   # azul
    68: (0.02, 0.52, 0.14),   # verde
}

# Mezcla hacia blanco: desbalanceo 1 = mas atenuado, 3 = color pleno.
ATENUACION_DESBALANCEO = {1: 0.58, 2: 0.30, 3: 0.0}

ESTILO_REPETICION = {1: "-", 2: "--", 3: ":"}

FIGSIZE = (15, 9)
DPI = 200

NOMBRE_DIR_SIN_COMP = "fasores sin compensacion runout"
NOMBRE_DIR_CON_COMP = "fasores con compensacion runout"


# ============================================================
# NOMBRES DE ARCHIVOS INTERMEDIOS
# ============================================================

ARCHIVO_FASORES = "p1_fasores.txt"
ARCHIVO_DIAGNOSTICO = "p1_diagnostico_estabilizacion.txt"
ARCHIVO_FASORES_COMP = "p3_fasores_compensados.txt"
ARCHIVO_SLOW_ROLL = "p3_vectores_slow_roll.txt"
DIR_ESTADISTICA = "p5_estadistica"
