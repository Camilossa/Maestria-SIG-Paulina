# =====================================================================
# PASO 2 - Puntos de entrenamiento y validación
#     uv run python paso2_muestras.py
#
# 1. Sortea puntos "puros" en ESA WorldCover: puntos cuyos 5 x 5 vecinos son
#    de la misma clase (se evitan bordes y mezclas).
# 2. A cada punto le pone tres etiquetas (1 = bosque, 0 = no bosque):
#    ESA WorldCover, IDEAM y Dynamic World.
# 3. Lee los 64 valores de AlphaEarth en cada punto.
# 4. Le asigna a cada punto el cuadro de 10 km en el que cae (bloque espacial).
# 5. Con ETIQUETA_FINAL = "consenso" se quedan solo los puntos en los que las
#    fuentes están de acuerdo. Los demás se guardan aparte.
#
# Resultados: resultados_deforestacion/muestras.csv y puntos_desacuerdo.csv
# =====================================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

import configuracion as cfg
from utilidades import BANDAS, embeddings_en_puntos

anio = cfg.ANIO_ENTRENAMIENTO
sorteo = np.random.default_rng(cfg.SEMILLA)


def etiqueta_ideam(lon, lat):
    """Bosque / no bosque del IDEAM en cada punto (NaN = sin información)."""
    ruta = next(cfg.CARPETA_IDEAM.rglob(cfg.IDEAM_BOSQUE[cfg.ANIO_IDEAM_ETIQUETA]))
    with rasterio.open(ruta) as src:
        x, y = transform("EPSG:4326", src.crs, lon, lat)   # el IDEAM usa otro sistema de coordenadas
        valor = np.array([v[0] for v in src.sample(zip(x, y))])
    return np.select([valor == 1, valor == 2], [1, 0], np.nan)


def etiqueta_dynamic_world(lon, lat):
    """Bosque / no bosque de Dynamic World (Google Earth Engine) en cada punto."""
    import ee
    ee.Initialize(project=cfg.PROYECTO_EARTH_ENGINE)
    puntos = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(float(x), float(y)), {"id": i})
                                   for i, (x, y) in enumerate(zip(lon, lat))])
    # Clase más frecuente del año en cada píxel. En Dynamic World la clase 1 es "árboles".
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(f"{anio}-01-01", f"{anio + 1}-01-01")
          .select("label").mode())
    respuesta = dw.sampleRegions(collection=puntos, scale=10, geometries=False).getInfo()
    etiqueta = np.full(len(lon), np.nan)
    for punto in respuesta["features"]:
        etiqueta[punto["properties"]["id"]] = punto["properties"]["label"] == 1
    return etiqueta


# ---- 1. Puntos puros en ESA WorldCover ----
with rasterio.open(cfg.CARPETA_ESA / f"esa_wc_huila_{anio}.tif") as src:
    clases = src.read(1)
    transformacion = src.transform

# Se sortean 2 millones de píxeles al azar (lejos del borde del mapa)
r = cfg.VENTANA_PIXEL_PURO // 2
filas = sorteo.integers(r, clases.shape[0] - r, 2_000_000)
columnas = sorteo.integers(r, clases.shape[1] - r, 2_000_000)
clase = clases[filas, columnas]

# Puro = todos los vecinos de la ventana tienen la misma clase que el centro
puro = np.ones(len(filas), dtype=bool)
for df in range(-r, r + 1):
    for dc in range(-r, r + 1):
        puro &= clases[filas + df, columnas + dc] == clase

bosque = np.flatnonzero(puro & (clase == cfg.ESA_BOSQUE))
no_bosque = np.flatnonzero(puro & np.isin(clase, cfg.ESA_NO_BOSQUE))
print("Píxeles puros: bosque", len(bosque), "- no bosque", len(no_bosque))
elegidos = np.concatenate([sorteo.choice(bosque, cfg.PUNTOS_POR_CLASE, replace=False),
                           sorteo.choice(no_bosque, cfg.PUNTOS_POR_CLASE, replace=False)])

lon, lat = rasterio.transform.xy(transformacion, filas[elegidos], columnas[elegidos])
puntos = pd.DataFrame({"lon": lon, "lat": lat, "clase_esa": clase[elegidos]})

# ---- 2. Etiquetas de las tres fuentes ----
puntos["esa"] = (puntos["clase_esa"] == cfg.ESA_BOSQUE).astype(int)
puntos["ideam"] = etiqueta_ideam(puntos["lon"], puntos["lat"])
fuentes = ["esa", "ideam"]
if cfg.USAR_DYNAMIC_WORLD:
    puntos["dw"] = etiqueta_dynamic_world(puntos["lon"], puntos["lat"])
    fuentes.append("dw")

print("\nConcordancia con ESA (puntos con dato):")
for fuente in fuentes[1:]:
    con_dato = puntos[fuente].notna()
    acuerdo = (puntos.loc[con_dato, fuente] == puntos.loc[con_dato, "esa"]).mean()
    print(f"  {fuente}: {acuerdo:.1%} de {con_dato.sum()} puntos")

# ---- 3. Embeddings de AlphaEarth ----
puntos[BANDAS] = embeddings_en_puntos(anio, puntos["lon"].to_numpy(), puntos["lat"].to_numpy())
puntos = puntos.dropna(subset=BANDAS)   # fuera de las teselas no hay embedding

# ---- 4. Bloque espacial: número del cuadro de 10 km en x y en y (1 grado ≈ 111 km) ----
tamano = cfg.TAMANO_BLOQUE_KM / 111.32
puntos["bloque"] = ((puntos["lon"] // tamano).astype(int).astype(str) + "_"
                    + (puntos["lat"] // tamano).astype(int).astype(str))

# ---- 5. Etiqueta final ----
puntos["bosque"] = puntos["esa"]
desacuerdo = np.zeros(len(puntos), dtype=bool)
if cfg.ETIQUETA_FINAL == "consenso":
    # Se descarta un punto si alguna fuente con dato dice otra cosa que ESA
    for fuente in fuentes[1:]:
        desacuerdo |= puntos[fuente].notna() & (puntos[fuente] != puntos["esa"])
    print("Puntos descartados por desacuerdo:", desacuerdo.sum())

# Los descartados se guardan aparte: el paso 3 los usa como prueba difícil
cfg.RESULTADOS.mkdir(parents=True, exist_ok=True)
puntos[desacuerdo].to_csv(cfg.RESULTADOS / "puntos_desacuerdo.csv", index=False)
puntos = puntos[~desacuerdo]
puntos.to_csv(cfg.RESULTADOS / "muestras.csv", index=False)
print(f"\nPuntos: {len(puntos)} (bosque {puntos['bosque'].sum()}, no bosque {(puntos['bosque'] == 0).sum()})"
      f" en {puntos['bloque'].nunique()} bloques")
