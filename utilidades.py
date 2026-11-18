# Funciones que usan varios pasos.

import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score

import configuracion as cfg

# Los embeddings de AlphaEarth tienen 64 bandas: A00, A01, ..., A63
BANDAS = [f"A{i:02d}" for i in range(64)]


def teselas_del_anio(anio):
    """Archivos .tif de embeddings de un año (lista vacía si no se han descargado)."""
    return sorted((cfg.CARPETA_EMBEDDINGS / str(anio)).glob("*.tif"))


def embeddings_en_puntos(anio, lon, lat):
    """Los 64 valores de AlphaEarth en cada punto. NaN si el punto no cae en ninguna tesela."""
    valores = np.full((len(lon), 64), np.nan)
    for tesela in teselas_del_anio(anio):
        with rasterio.open(tesela) as src:
            b = src.bounds
            dentro = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
            if dentro.any():
                valores[dentro] = list(src.sample(zip(lon[dentro], lat[dentro])))
    return valores


def metricas(real, predicho, probabilidad=None):
    """
    Métricas de una clasificación bosque (1) / no bosque (0).
    vp = verdaderos positivos (bosque bien clasificado), vn = verdaderos negativos,
    fp = falsos positivos (no bosque clasificado como bosque), fn = falsos negativos.
    """
    vn, fp, fn, vp = confusion_matrix(real, predicho, labels=[0, 1]).ravel()
    resultado = {
        "exactitud": accuracy_score(real, predicho),
        "kappa": cohen_kappa_score(real, predicho),
        "precision": vp / (vp + fp),        # de lo que se llamó bosque, cuánto lo era
        "sensibilidad": vp / (vp + fn),     # del bosque real, cuánto se encontró
        "especificidad": vn / (vn + fp),    # del no bosque real, cuánto se encontró
    }
    resultado["f1"] = 2 * resultado["precision"] * resultado["sensibilidad"] / (
        resultado["precision"] + resultado["sensibilidad"])
    if probabilidad is not None:
        resultado["auc"] = roc_auc_score(real, probabilidad)
    resultado.update({"vp": vp, "fp": fp, "fn": fn, "vn": vn})
    return resultado
