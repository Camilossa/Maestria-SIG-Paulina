# =====================================================================
# PASO 4 - Mapa de bosque / no bosque de cada año (2017-2025)
#     uv run python paso4_mapas_bosque.py
#
# El modelo del paso 3 clasifica todos los píxeles del Huila, año por año.
# Luego se marca el agua: ríos y embalses (capas de hidrografía) y el agua de
# ESA WorldCover.
#
# Resultado: mapas_bosque/bosque_AAAA.tif
#   0 = no bosque, 1 = bosque, 2 = agua, 255 = sin dato (fuera del Huila)
#
# Se salta los años que ya tienen mapa y los que no tienen embeddings descargados.
# =====================================================================

import warnings

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

import configuracion as cfg
from utilidades import teselas_del_anio

# La capa de ríos tiene un polígono con huecos y la librería que la lee avisa.
# El aviso no afecta el resultado, así que se oculta.
warnings.filterwarnings("ignore", module="pyogrio")


def mascara_agua(perfil):
    """True donde hay agua, en la cuadrícula del mapa."""
    forma = (perfil["height"], perfil["width"])

    # Agua de ESA WorldCover, llevada a la cuadrícula del mapa
    esa = np.zeros(forma, dtype=np.uint8)
    with rasterio.open(cfg.CARPETA_ESA / f"esa_wc_huila_{cfg.ANIO_ESA_AGUA}.tif") as src:
        reproject(rasterio.band(src, 1), esa, dst_transform=perfil["transform"], dst_crs=perfil["crs"],
                  resampling=Resampling.nearest)
    agua = esa == cfg.ESA_AGUA

    # Ríos y embalses: los polígonos se "pintan" sobre la cuadrícula (rasterize)
    for capa in cfg.CAPAS_HIDROGRAFIA:
        poligonos = gpd.read_file(cfg.DATOS / capa).to_crs(perfil["crs"]).geometry
        agua |= rasterize(poligonos, out_shape=forma, transform=perfil["transform"], all_touched=True) == 1
    return agua


# ---- Modelo: el de mayor kappa del paso 3, o el que diga la configuración ----
nombre = cfg.MODELO_PARA_MAPAS
if nombre == "mejor":
    nombre = pd.read_csv(cfg.RESULTADOS / "comparacion_modelos.csv")["modelo"].iloc[0]
modelo = joblib.load(cfg.RESULTADOS / f"modelo_{nombre}.joblib")
print("Modelo:", nombre)
cfg.CARPETA_MAPAS.mkdir(parents=True, exist_ok=True)

for anio in cfg.ANIOS:
    salida = cfg.CARPETA_MAPAS / f"bosque_{anio}.tif"
    teselas = teselas_del_anio(anio)
    if salida.exists():
        print(anio, "ya tiene mapa")
        continue
    if not teselas:
        print(anio, "no tiene embeddings descargados")
        continue
    if len(teselas) < cfg.TESELAS_POR_ANIO:
        # Pasa si la descarga no ha terminado: el mapa quedaría con huecos
        print(anio, f"tiene {len(teselas)} de {cfg.TESELAS_POR_ANIO} teselas: falta terminar la descarga")
        continue
    print(anio, "-", len(teselas), "teselas")

    # ---- Tamaño del mapa completo: el rectángulo que cubre todas las teselas ----
    with rasterio.open(teselas[0]) as src:
        pixel, crs = src.res[0], src.crs
    bordes = [rasterio.open(t).bounds for t in teselas]
    oeste, norte = min(b.left for b in bordes), max(b.top for b in bordes)
    ancho = round((max(b.right for b in bordes) - oeste) / pixel)
    alto = round((norte - min(b.bottom for b in bordes)) / pixel)
    perfil = {"driver": "GTiff", "width": ancho, "height": alto, "count": 1, "dtype": "uint8",
              "crs": crs, "transform": rasterio.transform.from_origin(oeste, norte, pixel, pixel),
              "nodata": 255, "compress": "lzw", "tiled": True}

    # ---- Clasificar cada tesela por franjas de filas ----
    mapa = np.full((alto, ancho), 255, dtype=np.uint8)
    for tesela in teselas:
        with rasterio.open(tesela) as src:
            fila0 = round((norte - src.bounds.top) / pixel)     # dónde va la tesela en el mapa
            col0 = round((src.bounds.left - oeste) / pixel)
            for fila in range(0, src.height, cfg.FILAS_POR_LECTURA):
                n = min(cfg.FILAS_POR_LECTURA, src.height - fila)
                bandas = src.read(window=Window(0, fila, src.width, n))     # (64, n, ancho)
                pixeles = bandas.reshape(64, -1).T.astype(np.float32)      # una fila por píxel
                con_dato = ~np.isnan(pixeles).any(axis=1)                  # fuera del Huila es NaN
                clase = np.full(len(pixeles), 255, dtype=np.uint8)
                if con_dato.any():
                    clase[con_dato] = modelo.predict(pixeles[con_dato])
                mapa[fila0 + fila:fila0 + fila + n, col0:col0 + src.width] = clase.reshape(n, src.width)

    # ---- Máscara de agua y guardar ----
    mapa[mascara_agua(perfil) & (mapa != 255)] = 2
    with rasterio.open(salida, "w", **perfil) as destino:
        destino.write(mapa, 1)
    print("   guardado", salida.name)
