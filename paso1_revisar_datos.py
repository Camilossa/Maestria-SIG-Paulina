# =====================================================================
# PASO 1 - Revisar qué datos están y cuáles faltan
#     uv run python paso1_revisar_datos.py
# =====================================================================

import configuracion as cfg
from utilidades import teselas_del_anio


def revisar(nombre, existe):
    print("  OK    " if existe else "  FALTA ", nombre)


print("Carpeta de datos:", cfg.CARPETA_DATOS)

print("\nESA WorldCover")
for anio in sorted({cfg.ANIO_ENTRENAMIENTO, cfg.ANIO_ESA_AGUA}):
    revisar(f"esa_wc_huila_{anio}.tif", (cfg.CARPETA_ESA / f"esa_wc_huila_{anio}.tif").exists())

print("\nEmbeddings de AlphaEarth (81 teselas por año)")
for anio in cfg.ANIOS:
    n = len(teselas_del_anio(anio))
    mapa_hecho = (cfg.CARPETA_MAPAS / f"bosque_{anio}.tif").exists()
    revisar(f"{anio}: {n} teselas" + ("  (el mapa ya está hecho)" if mapa_hecho else ""), n > 0 or mapa_hecho)

print("\nHidrografía")
for capa in cfg.CAPAS_HIDROGRAFIA:
    revisar(capa, (cfg.DATOS / capa).exists())

print("\nIDEAM")
for archivo in [*cfg.IDEAM_BOSQUE.values(), cfg.IDEAM_CAMBIO_2017_2018]:
    revisar(archivo, any(cfg.CARPETA_IDEAM.rglob(archivo)))
