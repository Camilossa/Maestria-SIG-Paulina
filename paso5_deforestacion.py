# =====================================================================
# PASO 5 - Deforestación año a año y consistencia temporal
#     uv run python paso5_deforestacion.py
#
# Compara los mapas de bosque de años seguidos, píxel por píxel:
#   cambio bruto    = bosque en t-1 y no bosque en t
#   cambio espurio  = el cambio bruto se "deshace": vuelve a ser bosque en t+1
#                     (casi siempre es un error del modelo, no una tala)
#   deforestación   = cambio bruto que sigue sin bosque en t+1
# La deforestación del último año no se puede confirmar (no hay año siguiente).
# Si un píxel se deforesta dos veces, se cuenta solo la primera.
# Agua y píxeles sin dato se excluyen.
#
# Resultados:
#   deforestacion_por_anio.csv  hectáreas de bosque, cambios brutos, espurios y deforestación
#   anio_perdida.tif            año en que cada píxel perdió el bosque (0 = no lo perdió)
#   grafica_deforestacion.png   bosque y deforestación en el tiempo
#   mapa_calor.png              hectáreas deforestadas por km²
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import configuracion as cfg

anios = [a for a in cfg.ANIOS if (cfg.CARPETA_MAPAS / f"bosque_{a}.tif").exists()]
print("Años con mapa:", anios)
if len(anios) < 2:
    raise SystemExit("Se necesitan al menos 2 mapas de bosque (paso 4).")
if anios[-1] - anios[0] + 1 != len(anios):
    print("OJO: faltan años intermedios; la pérdida de esos años se suma al siguiente año con mapa.")

mapas = [rasterio.open(cfg.CARPETA_MAPAS / f"bosque_{a}.tif") for a in anios]
alto, ancho = mapas[0].height, mapas[0].width
perfil = mapas[0].profile
perfil.update(dtype="uint16", nodata=65535)   # uint16 porque los años no caben en 0-255

# Contadores de píxeles por año
bosque_px = {a: 0 for a in anios}
bruto_px = {a: 0 for a in anios}
espurio_px = {a: 0 for a in anios}
deforestado_px = {a: 0 for a in anios}

celda = cfg.TAMANO_CELDA_CALOR
calor = np.zeros((alto // celda + 1, ancho // celda + 1))   # hectáreas deforestadas por cuadro

with rasterio.open(cfg.RESULTADOS / "anio_perdida.tif", "w", **perfil) as salida:
    # El mapa se recorre por franjas de filas para no llenar la memoria
    for fila in range(0, alto, celda * 10):
        n = min(celda * 10, alto - fila)
        franja = Window(0, fila, ancho, n)
        pila = np.stack([m.read(1, window=franja) for m in mapas])   # pila[i] = mapa del año anios[i]
        valido = (pila <= 1).all(axis=0)     # bosque o no bosque todos los años (sin agua ni vacíos)
        bosque = (pila == 1) & valido

        anio_perdida = np.zeros((n, ancho), dtype=np.uint16)
        for i, a in enumerate(anios):
            bosque_px[a] += bosque[i].sum()
            if i == 0:
                continue
            bruto = bosque[i - 1] & ~bosque[i] & valido
            if i + 1 < len(anios):
                espurio = bruto & bosque[i + 1]
                confirmado = bruto & ~bosque[i + 1]
            else:
                espurio = np.zeros_like(bruto)
                confirmado = bruto
            nuevo = confirmado & (anio_perdida == 0)
            anio_perdida[nuevo] = a
            bruto_px[a] += bruto.sum()
            espurio_px[a] += espurio.sum()
            deforestado_px[a] += nuevo.sum()

        # Sumar la pérdida en cuadros de celda x celda píxeles para el mapa de calor
        filas_p, columnas_p = np.nonzero(anio_perdida)
        np.add.at(calor, ((fila + filas_p) // celda, columnas_p // celda), cfg.HECTAREAS_POR_PIXEL)

        anio_perdida[~valido] = 65535
        salida.write(anio_perdida, 1, window=franja)

bordes, pixel = mapas[0].bounds, mapas[0].res[0]
for m in mapas:
    m.close()

# ---- Tabla ----
ha = cfg.HECTAREAS_POR_PIXEL
tabla = pd.DataFrame({
    "anio": anios,
    "bosque_ha": [bosque_px[a] * ha for a in anios],
    "cambio_bruto_ha": [bruto_px[a] * ha for a in anios],
    "cambio_espurio_ha": [espurio_px[a] * ha for a in anios],
    "deforestacion_ha": [deforestado_px[a] * ha for a in anios],
})
tabla["espurio_pct"] = 100 * tabla["cambio_espurio_ha"] / tabla["cambio_bruto_ha"]
tabla["tasa_deforestacion_pct"] = 100 * tabla["deforestacion_ha"] / tabla["bosque_ha"].shift(1)
tabla["confirmada"] = tabla["anio"] != anios[-1]
tabla.loc[0, ["cambio_bruto_ha", "cambio_espurio_ha", "deforestacion_ha", "espurio_pct"]] = np.nan
tabla.round(2).to_csv(cfg.RESULTADOS / "deforestacion_por_anio.csv", index=False)
print(tabla.round(1).to_string(index=False))
print("\nDeforestación total:", round(tabla["deforestacion_ha"].sum()), "ha")

# ---- Gráfica: bosque y deforestación en el tiempo ----
figura, (eje1, eje2) = plt.subplots(1, 2, figsize=(12, 4))
eje1.plot(tabla["anio"], tabla["bosque_ha"] / 1000, marker="o", color="darkgreen")
eje1.set_title("Bosque")
eje1.set_ylabel("miles de hectáreas")
eje2.bar(tabla["anio"] - 0.2, tabla["deforestacion_ha"], width=0.4, color="firebrick", label="deforestación")
eje2.bar(tabla["anio"] + 0.2, tabla["cambio_espurio_ha"], width=0.4, color="gray", label="cambio espurio")
eje2.set_title("Cambios por año (el último sin confirmar)")
eje2.set_ylabel("hectáreas")
eje2.legend()
figura.savefig(cfg.RESULTADOS / "grafica_deforestacion.png", dpi=150, bbox_inches="tight")

# ---- Mapa de calor ----
figura, eje = plt.subplots(figsize=(7, 8))
lado = celda * pixel   # tamaño de un cuadro del mapa de calor, en grados
imagen = eje.imshow(np.where(calor > 0, calor, np.nan), cmap="YlOrRd",
                    extent=[bordes.left, bordes.left + calor.shape[1] * lado,
                            bordes.top - calor.shape[0] * lado, bordes.top])
figura.colorbar(imagen, ax=eje, label="hectáreas deforestadas por km²", shrink=0.7)
eje.set_title(f"Deforestación {anios[0]}-{anios[-1]}")
figura.savefig(cfg.RESULTADOS / "mapa_calor.png", dpi=150, bbox_inches="tight")
