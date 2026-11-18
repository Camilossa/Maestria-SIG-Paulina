# =====================================================================
# PASO 3 - Entrenar y comparar Random Forest, SVM y XGBoost
#     uv run python paso3_entrenar_modelos.py
#
# Los bloques de 10 km se reparten al azar: 70 % para entrenar y 30 % para
# probar. Se separa por bloques (y no punto por punto) porque dos puntos
# vecinos se parecen mucho: si uno queda en entrenamiento y el otro en prueba,
# el modelo parece mejor de lo que es.
#
# Cada modelo se entrena con los bloques de entrenamiento y se evalúa con los
# de prueba: matriz de confusión, exactitud, kappa, precisión, sensibilidad,
# especificidad, F1 y AUC.
#
# Revisión del sobreajuste (que el modelo se aprenda los puntos de memoria):
#   A. Brecha: las mismas métricas con los puntos de entrenamiento. Si el
#      modelo es mucho mejor con lo que ya vio que con lo nuevo, hay sobreajuste.
#   B. 10 particiones: se repite el reparto 70/30 con 10 sorteos distintos.
#      Si los resultados cambian mucho de un sorteo a otro, no son confiables.
#   C. Puntos difíciles: los puntos en los que ESA e IDEAM no estaban de acuerdo
#      (paso 2). El modelo nunca los vio y son los casos dudosos (bordes,
#      rastrojos, árboles dispersos).
#
# Resultados: comparacion_modelos.csv, sobreajuste.csv, particiones_repetidas.csv,
#             puntos_dificiles.csv, matrices_confusion.png, curvas_roc.png,
#             modelo_RandomForest.joblib, modelo_SVM.joblib, modelo_XGBoost.joblib
# =====================================================================

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import configuracion as cfg
from utilidades import BANDAS, metricas

muestras = pd.read_csv(cfg.RESULTADOS / "muestras.csv")


def separar(semilla):
    """Reparte los bloques al azar: devuelve los puntos de entrenamiento y de prueba."""
    sorteo = np.random.default_rng(semilla)
    bloques = muestras["bloque"].unique()
    bloques_prueba = sorteo.choice(bloques, round(len(bloques) * cfg.PROPORCION_PRUEBA), replace=False)
    prueba = muestras["bloque"].isin(bloques_prueba)
    return (muestras.loc[~prueba, BANDAS].to_numpy(), muestras.loc[~prueba, "bosque"].to_numpy(),
            muestras.loc[prueba, BANDAS].to_numpy(), muestras.loc[prueba, "bosque"].to_numpy())


X_entrena, y_entrena, X_prueba, y_prueba = separar(cfg.SEMILLA)
print(f"Entrenamiento: {len(y_entrena)} puntos - Prueba: {len(y_prueba)} puntos")

# ---- Los tres modelos ----
# El SVM necesita las 64 variables en la misma escala (StandardScaler). Como el SVM
# no da probabilidades (las usa la curva ROC), se "calibra" con CalibratedClassifierCV.
modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=cfg.SEMILLA),
    "SVM": CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC(C=10)), ensemble=False),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=cfg.SEMILLA),
}

# ---- Entrenar, evaluar y guardar ----
resultados = []
brechas = []
figura_matrices, ejes = plt.subplots(1, 3, figsize=(13, 4))
figura_roc, eje_roc = plt.subplots(figsize=(5, 5))

for eje, (nombre, modelo) in zip(ejes, modelos.items()):
    modelo.fit(X_entrena, y_entrena)
    predicho = modelo.predict(X_prueba)
    probabilidad = modelo.predict_proba(X_prueba)[:, 1]   # probabilidad de bosque
    resultados.append({"modelo": nombre, **metricas(y_prueba, predicho, probabilidad)})
    joblib.dump(modelo, cfg.RESULTADOS / f"modelo_{nombre}.joblib")

    # A. Las mismas métricas con los puntos de entrenamiento, para medir la brecha
    entrena = metricas(y_entrena, modelo.predict(X_entrena), modelo.predict_proba(X_entrena)[:, 1])
    fila = {"modelo": nombre}
    for m in ["auc", "kappa", "f1", "exactitud"]:
        fila[f"{m}_entrenamiento"] = entrena[m]
        fila[f"{m}_prueba"] = resultados[-1][m]
        fila[f"brecha_{m}"] = entrena[m] - resultados[-1][m]
    brechas.append(fila)

    ConfusionMatrixDisplay.from_predictions(y_prueba, predicho, display_labels=["No bosque", "Bosque"],
                                            ax=eje, colorbar=False, cmap="Greens")
    eje.set_title(nombre)
    RocCurveDisplay.from_predictions(y_prueba, probabilidad, name=nombre, ax=eje_roc)

figura_matrices.savefig(cfg.RESULTADOS / "matrices_confusion.png", dpi=150, bbox_inches="tight")
figura_roc.savefig(cfg.RESULTADOS / "curvas_roc.png", dpi=150, bbox_inches="tight")

tabla = pd.DataFrame(resultados).sort_values("kappa", ascending=False)
tabla.round(4).to_csv(cfg.RESULTADOS / "comparacion_modelos.csv", index=False)
print("\nResultados con los bloques de prueba:")
print(tabla.round(3).to_string(index=False))
print("\nMejor modelo (mayor kappa):", tabla.iloc[0]["modelo"])

# ---- A. Brecha = entrenamiento - prueba ----
# Guía: menos de 0.02 bien; de 0.02 a 0.05 leve; más de 0.05 sobreajuste.
sobreajuste = pd.DataFrame(brechas)
sobreajuste.round(4).to_csv(cfg.RESULTADOS / "sobreajuste.csv", index=False)
print("\nA. Brecha entre entrenamiento y prueba:")
print(sobreajuste[["modelo", "auc_entrenamiento", "auc_prueba", "brecha_auc",
                   "kappa_entrenamiento", "kappa_prueba", "brecha_kappa"]].round(4).to_string(index=False))

# ---- B. Repetir el reparto de bloques con 10 sorteos distintos ----
repeticiones = []
for semilla in range(cfg.SEMILLA, cfg.SEMILLA + 10):
    Xe, ye, Xp, yp = separar(semilla)
    for nombre, modelo in modelos.items():
        copia = clone(modelo).fit(Xe, ye)   # clone = el mismo modelo, sin entrenar
        m = metricas(yp, copia.predict(Xp), copia.predict_proba(Xp)[:, 1])
        repeticiones.append({"modelo": nombre, "semilla": semilla, "auc": m["auc"], "kappa": m["kappa"]})
repeticiones = pd.DataFrame(repeticiones)
resumen = repeticiones.groupby("modelo")[["auc", "kappa"]].agg(["mean", "std", "min"])
resumen.columns = ["auc_media", "auc_desviacion", "auc_minimo", "kappa_media", "kappa_desviacion", "kappa_minimo"]
resumen.round(4).to_csv(cfg.RESULTADOS / "particiones_repetidas.csv")
print("\nB. 10 repartos distintos de bloques (media, desviación y mínimo):")
print(resumen.round(4).to_string())

# ---- C. Puntos difíciles: donde ESA e IDEAM no estaban de acuerdo ----
ruta = cfg.RESULTADOS / "puntos_desacuerdo.csv"
dificiles = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()
if len(dificiles) > 0:
    filas = []
    for nombre, modelo in modelos.items():
        predicho = modelo.predict(dificiles[BANDAS].to_numpy())
        filas.append({"modelo": nombre, "puntos": len(dificiles),
                      "dice_bosque_pct": 100 * predicho.mean(),
                      "coincide_con_esa_pct": 100 * (predicho == dificiles["esa"]).mean(),
                      "coincide_con_ideam_pct": 100 * (predicho == dificiles["ideam"]).mean()})
    dificiles_tabla = pd.DataFrame(filas)
    dificiles_tabla.round(2).to_csv(cfg.RESULTADOS / "puntos_dificiles.csv", index=False)
    print(f"\nC. Puntos difíciles ({len(dificiles)}; ESA dice bosque en {dificiles['esa'].sum()},"
          f" IDEAM en {int(dificiles['ideam'].sum())}):")
    print(dificiles_tabla.round(1).to_string(index=False))
