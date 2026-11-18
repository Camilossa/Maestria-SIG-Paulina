# =====================================================================
# CONFIGURACIÓN
# Carpetas, años y parámetros. Es el único archivo que hay que editar.
# =====================================================================

import os
from pathlib import Path

# ---------------- Carpetas ----------------

# Copia local de la carpeta "APLICACIONES SIG" del Drive, dentro de este proyecto.
# Termina en ".nosync" para que iCloud no intente subir los ~50 GB de datos.
CARPETA_DATOS = Path(__file__).parent / "datos.nosync"

# ---------------- Años ----------------

ANIOS = list(range(2017, 2026))   # 2017, 2018, ..., 2025
ANIO_ENTRENAMIENTO = 2021         # año de los puntos de entrenamiento (hay ESA WorldCover 2020 y 2021)
ANIO_IDEAM_ETIQUETA = 2019        # mapa del IDEAM más cercano a ese año (el IDEAM llega hasta 2019)

# ---------------- Paso 2: puntos de entrenamiento ----------------

PUNTOS_POR_CLASE = 1500           # puntos de bosque y de no bosque
VENTANA_PIXEL_PURO = 5            # el punto y sus 5 x 5 vecinos deben ser de la misma clase en ESA
USAR_DYNAMIC_WORLD = False        # Dynamic World se consulta en Google Earth Engine
                                  # (en False hasta tener permiso en el proyecto de abajo)
PROYECTO_EARTH_ENGINE = "tesis-503417"
# Etiqueta final de cada punto:
#   "consenso" = solo se usan los puntos en los que ESA, IDEAM y Dynamic World están de acuerdo
#   "esa"      = se usa la etiqueta de ESA en todos los puntos
ETIQUETA_FINAL = "consenso"
SEMILLA = 42                      # para que los sorteos al azar salgan siempre iguales

# Clases de ESA WorldCover
ESA_BOSQUE = 10
ESA_NO_BOSQUE = [20, 30, 40, 50, 60, 90, 100]   # matorral, pasto, cultivo, construido, suelo, humedal, musgo
ESA_AGUA = 80

# ---------------- Paso 3: modelos ----------------

TAMANO_BLOQUE_KM = 10             # cuadros de 10 km para separar entrenamiento y prueba
PROPORCION_PRUEBA = 0.3           # 30 % de los cuadros se guardan para la prueba
MODELO_PARA_MAPAS = "mejor"       # "mejor" (mayor kappa), "RandomForest", "XGBoost" o "SVM"

# ---------------- Paso 4: mapas y máscara de agua ----------------

TESELAS_POR_ANIO = 81             # un año solo se mapea si están todas sus teselas
FILAS_POR_LECTURA = 256           # cada tesela se lee por pedazos para no llenar la memoria
ANIO_ESA_AGUA = 2021              # el agua de ESA WorldCover de este año se enmascara
# Ríos dobles y embalses (polígonos). No se incluye drenaje_sencillo: son 10 892 quebradas
# dibujadas como líneas que casi siempre pasan bajo el bosque.
CAPAS_HIDROGRAFIA = ["DRENAJE_DOBLE/drenaje_doble.shp", "EMBALSE/embalse.shp"]

# ---------------- Paso 5: deforestación ----------------

HECTAREAS_POR_PIXEL = 0.0099      # píxel de AlphaEarth en el Huila: ≈ 10 m x 9.9 m
TAMANO_CELDA_CALOR = 100          # el mapa de calor suma la pérdida en cuadros de 100 x 100 píxeles (≈ 1 km)

# ---------------- Paso 6: IDEAM ----------------

# Bosque / No bosque del IDEAM: 1 = bosque, 2 = no bosque, 3 = sin información
IDEAM_BOSQUE = {2017: "SBQ_SMBYC_BQNBQ_V7_2017.tif", 2019: "SBQ_SMBYC_BQNBQ_V8_2019.tif"}
# Cambio 2017-2018 del IDEAM: 2 = deforestación
IDEAM_CAMBIO_2017_2018 = "DCCB_SMBYC_CBBQ_V5_2017_2018.tif"
PIXELES_PARA_COMPARAR = 200_000   # píxeles al azar para comparar mapas

# ---------------- Rutas que se derivan de lo anterior (no hace falta tocarlas) ----------------

# Para la prueba automática (carpeta pruebas/): usa otra carpeta y no usa Earth Engine
if "CARPETA_DATOS" in os.environ:
    CARPETA_DATOS = Path(os.environ["CARPETA_DATOS"])
    USAR_DYNAMIC_WORLD = False
    TESELAS_POR_ANIO = 4

DATOS = CARPETA_DATOS / "DATOS"
CARPETA_EMBEDDINGS = DATOS / "embeddings"             # una subcarpeta por año, 81 teselas .tif
CARPETA_ESA = DATOS / "clases_esa"                    # esa_wc_huila_2020.tif, esa_wc_huila_2021.tif
CARPETA_IDEAM = DATOS / "INVENTARIO FORESTAL"
RESULTADOS = CARPETA_DATOS / "resultados_deforestacion"
CARPETA_MAPAS = RESULTADOS / "mapas_bosque"           # bosque_2017.tif ... bosque_2025.tif
