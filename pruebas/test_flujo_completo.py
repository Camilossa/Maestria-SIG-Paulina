# Prueba automática: crea un "Huila" de mentira (bosque, potreros, dos ríos y un
# parche que se tala en 2019) y corre los 6 pasos. No usa Earth Engine.
#     uv run pytest

import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterio.transform import from_origin
from shapely.geometry import box

PROYECTO = Path(__file__).resolve().parents[1]
ANIOS = [2017, 2018, 2019, 2020, 2021]
OESTE, NORTE = -75.6, 2.9
PIXEL = 0.0009    # ≈ 100 m, para que en 40 km quepan varios bloques de 10 km
LADO = 400
rng = np.random.default_rng(0)


def paisaje(anio, filas, columnas):
    """1 = bosque, 0 = no bosque, 2 = agua. En 2019 se tala un parche."""
    bosque = (filas // 40 + columnas // 40) % 2 == 0
    if anio >= 2019:
        bosque &= ~((filas >= 160) & (filas < 200) & (columnas >= 80) & (columnas < 120))
    clase = bosque.astype(np.uint8)
    clase[(columnas >= 200) & (columnas < 203)] = 2
    return clase


def escribir(ruta, datos, transform, crs="EPSG:4326", nodata=None):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = datos if datos.ndim == 3 else datos[None]
    with rasterio.open(ruta, "w", driver="GTiff", width=datos.shape[2], height=datos.shape[1],
                       count=datos.shape[0], dtype=datos.dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(datos)


def crear_datos(raiz):
    datos = raiz / "DATOS"
    filas, columnas = np.mgrid[0:LADO, 0:LADO]
    fuera = (filas - LADO / 2) ** 2 + (columnas - LADO / 2) ** 2 > (LADO * 0.48) ** 2
    huella = np.stack([rng.normal(-0.3, 0.05, 64), rng.normal(0.3, 0.05, 64), rng.normal(0, 0.05, 64)])

    # Embeddings: 4 teselas de 200 x 200 por año, NaN fuera del "Huila"
    for anio in ANIOS:
        cubo = huella[paisaje(anio, filas, columnas)].transpose(2, 0, 1) + rng.normal(0, 0.08, (64, LADO, LADO))
        cubo[:, fuera] = np.nan
        for i in (0, 200):
            for j in (0, 200):
                escribir(datos / "embeddings" / str(anio) / f"emb_{i}_{j}.tif", cubo[:, i:i + 200, j:j + 200],
                         from_origin(OESTE + j * PIXEL, NORTE - i * PIXEL, PIXEL, PIXEL))

    # ESA WorldCover con otra resolución: 10 bosque, 30 pasto, 80 agua, 0 fuera
    p = PIXEL * 0.93
    f, c = (np.mgrid[0:int(LADO / 0.93), 0:int(LADO / 0.93)] * 0.93).astype(int)
    for anio in (2020, 2021):
        clase = paisaje(anio, f, c)
        esa = np.select([clase == 1, clase == 2], [10, 80], 30).astype(np.uint8)
        esa[fuera[f, c]] = 0
        escribir(datos / "clases_esa" / f"esa_wc_huila_{anio}.tif", esa, from_origin(OESTE, NORTE, p, p), nodata=0)

    # Hidrografía: otro río (polígono) y un embalse, en el sistema de coordenadas del IGAC
    for capa, x in [("DRENAJE_DOBLE/drenaje_doble.shp", 100), ("EMBALSE/embalse.shp", 300)]:
        poligono = box(OESTE + x * PIXEL, NORTE - 380 * PIXEL, OESTE + (x + 4) * PIXEL, NORTE - 20 * PIXEL)
        (datos / capa).parent.mkdir(parents=True)
        gpd.GeoDataFrame(geometry=[poligono], crs="EPSG:4326").to_crs("EPSG:9377").to_file(datos / capa)

    # IDEAM: mismo paisaje en EPSG:3857 con píxeles de 300 m
    m = 300.0
    x0, y0 = rasterio.warp.transform("EPSG:4326", "EPSG:3857", [OESTE], [NORTE])
    n = int(LADO * PIXEL * 111_320 / m)
    lon = OESTE + (np.arange(n) + 0.5) * m / 111_320
    lat = NORTE - (np.arange(n) + 0.5) * m / 111_320
    F, C = np.meshgrid(((NORTE - lat) / PIXEL).astype(int), ((lon - OESTE) / PIXEL).astype(int), indexing="ij")
    ideam = datos / "INVENTARIO FORESTAL"
    for anio, nombre in [(2017, "SBQ_SMBYC_BQNBQ_V7_2017.tif"), (2019, "SBQ_SMBYC_BQNBQ_V8_2019.tif")]:
        escribir(ideam / str(anio) / nombre, np.where(paisaje(anio, F, C) == 1, 1, 2).astype(np.uint8),
                 from_origin(x0[0], y0[0], m, m), crs="EPSG:3857", nodata=0)
    antes, despues = paisaje(2017, F, C) == 1, paisaje(2018, F, C) == 1
    escribir(ideam / "2018" / "DCCB_SMBYC_CBBQ_V5_2017_2018.tif",
             np.select([antes & despues, antes & ~despues], [1, 2], 5).astype(np.uint8),
             from_origin(x0[0], y0[0], m, m), crs="EPSG:3857", nodata=0)


def correr(paso, raiz):
    r = subprocess.run([sys.executable, paso], cwd=PROYECTO, capture_output=True, text=True,
                       env={**os.environ, "CARPETA_DATOS": str(raiz)})
    print(r.stdout)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_flujo_completo(tmp_path):
    crear_datos(tmp_path)
    res = tmp_path / "resultados_deforestacion"

    assert "FALTA  2022" in correr("paso1_revisar_datos.py", tmp_path)

    correr("paso2_muestras.py", tmp_path)
    muestras = pd.read_csv(res / "muestras.csv")
    assert muestras["bosque"].value_counts().min() > 1000
    assert muestras["bloque"].nunique() >= 10

    correr("paso3_entrenar_modelos.py", tmp_path)
    tabla = pd.read_csv(res / "comparacion_modelos.csv")
    assert (tabla["kappa"] > 0.9).all() and (tabla["auc"] > 0.9).all()
    assert (pd.read_csv(res / "sobreajuste.csv")["brecha_kappa"] < 0.05).all()
    assert len(pd.read_csv(res / "particiones_repetidas.csv")) == 3
    assert (res / "puntos_dificiles.csv").exists()

    correr("paso4_mapas_bosque.py", tmp_path)
    with rasterio.open(res / "mapas_bosque" / "bosque_2021.tif") as src:
        assert set(np.unique(src.read(1))) == {0, 1, 2, 255}   # no bosque, bosque, agua, sin dato

    correr("paso5_deforestacion.py", tmp_path)
    tabla = pd.read_csv(res / "deforestacion_por_anio.csv").set_index("anio")
    # el parche de 40 x 40 píxeles (1600 x 0.0099 ha) se tala en 2019; los demás años casi nada
    assert tabla.loc[2019, "deforestacion_ha"] > 12
    assert tabla["deforestacion_ha"].drop(2019).max() < 0.1 * tabla.loc[2019, "deforestacion_ha"]
    assert not tabla.loc[2021, "confirmada"]

    correr("paso6_comparar_referencias.py", tmp_path)
    comparacion = pd.read_csv(res / "comparacion_referencias.csv")
    assert (comparacion["exactitud"].dropna() > 0.9).all()

    assert "ya tiene mapa" in correr("paso4_mapas_bosque.py", tmp_path)
