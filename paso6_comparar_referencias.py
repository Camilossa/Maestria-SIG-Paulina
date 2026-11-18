# =====================================================================
# PASO 6 - Comparar los mapas con las referencias de IDEAM y ESA
#     uv run python paso6_comparar_ideam.py
#
# A) Bosque / no bosque: nuestro mapa de un año contra el IDEAM (2017, 2019)
#    y contra ESA WorldCover (2020, 2021), con PIXELES_PARA_COMPARAR píxeles al
#    azar. Mide la concordancia con las dos definiciones de bosque.
#    (El IDEAM es una referencia independiente; ESA 2021 no, porque de ahí
#    salieron los puntos de entrenamiento.)
# B) Deforestación 2017-2018: lo que marca el IDEAM contra lo que marcamos.
#
# Cada mapa de referencia se lleva a la cuadrícula de nuestros mapas (reproject).
# El IDEAM es de 30 m y nuestros mapas de 10 m: en los bordes del bosque
# siempre habrá algo de desacuerdo.
#
# Resultado: comparacion_referencias.csv
# =====================================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

import configuracion as cfg
from utilidades import metricas


def en_nuestra_cuadricula(ruta, perfil):
    """Lee un mapa de referencia y lo devuelve en la cuadrícula de nuestros mapas."""
    valores = np.zeros((perfil["height"], perfil["width"]), dtype=np.uint8)
    with rasterio.open(ruta) as src:
        reproject(rasterio.band(src, 1), valores, dst_transform=perfil["transform"], dst_crs=perfil["crs"],
                  dst_nodata=0, resampling=Resampling.nearest)
    return valores


# Referencias: (fuente, año, archivo, valores que son bosque, valores que son no bosque)
referencias = [("ESA", a, cfg.CARPETA_ESA / f"esa_wc_huila_{a}.tif", [cfg.ESA_BOSQUE], cfg.ESA_NO_BOSQUE)
               for a in (2020, 2021)]
for a, archivo in cfg.IDEAM_BOSQUE.items():
    encontrado = list(cfg.CARPETA_IDEAM.rglob(archivo))
    if encontrado:
        referencias.append(("IDEAM", a, encontrado[0], [1], [2]))

sorteo = np.random.default_rng(cfg.SEMILLA)
resultados = []

# ---- A) Bosque / no bosque ----
for fuente, anio, ruta, valores_bosque, valores_no_bosque in referencias:
    ruta_mapa = cfg.CARPETA_MAPAS / f"bosque_{anio}.tif"
    if not ruta_mapa.exists() or not ruta.exists():
        print(fuente, anio, ": falta el mapa o la referencia")
        continue
    with rasterio.open(ruta_mapa) as src:
        nuestro = src.read(1).ravel()
        referencia = en_nuestra_cuadricula(ruta, src.profile).ravel()

    # Píxeles al azar donde los dos mapas dicen bosque o no bosque
    pos = sorteo.integers(0, nuestro.size, 10 * cfg.PIXELES_PARA_COMPARAR)
    pos = pos[(nuestro[pos] <= 1) & np.isin(referencia[pos], valores_bosque + valores_no_bosque)]
    pos = pos[:cfg.PIXELES_PARA_COMPARAR]

    real = np.isin(referencia[pos], valores_bosque).astype(int)
    resultados.append({"comparacion": f"{fuente} {anio}", "pixeles": len(pos), **metricas(real, nuestro[pos])})
    print(f"{fuente} {anio}: exactitud {resultados[-1]['exactitud']:.3f}, kappa {resultados[-1]['kappa']:.3f}")

# ---- B) Deforestación 2017-2018 del IDEAM ----
ruta_ideam = list(cfg.CARPETA_IDEAM.rglob(cfg.IDEAM_CAMBIO_2017_2018))
ruta_perdida = cfg.RESULTADOS / "anio_perdida.tif"
if ruta_ideam and ruta_perdida.exists():
    with rasterio.open(ruta_perdida) as src:
        perdida = src.read(1)
        ideam = en_nuestra_cuadricula(ruta_ideam[0], src.profile)
    comun = (perdida != 65535) & np.isin(ideam, [1, 2, 4, 5])   # ambos con dato (3 = sin información)
    del_ideam = comun & (ideam == 2)                            # 2 = deforestación en el IDEAM
    nuestra = comun & (perdida == 2018)
    ambos = (del_ideam & nuestra).sum()
    resultados.append({"comparacion": "Deforestación IDEAM 2017-2018",
                       "pixeles": comun.sum(),
                       "deforestados_ideam": del_ideam.sum(),
                       "deforestados_nuestro": nuestra.sum(),
                       "en_ambos": ambos})
    print(f"Deforestación 2017-2018: IDEAM {del_ideam.sum()} píxeles, nosotros {nuestra.sum()}, en ambos {ambos}")

pd.DataFrame(resultados).round(4).to_csv(cfg.RESULTADOS / "comparacion_referencias.csv", index=False)
