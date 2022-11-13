import os

import geopandas as geop
import numpy as np
from tqdm import tqdm

THRESHOLD = 4000
percentage = 100

dir = r'input'
rfile = 'potentialareas_400m_forest.shp'

gdf = geop.read_file(os.path.join(dir, rfile))
cols = gdf.columns

gdf = gdf.rename(columns={'distance': 'distanz_umspannwerk', '_mean': 'energieleistungsdichte'})
cols = gdf.columns
gdf_optimization = gdf[['distanz_umspannwerk', 'energieleistungsdichte', 'geometry']]
area = gdf_optimization["geometry"].area.to_numpy()
gdf_np = gdf_optimization.to_numpy()
gdf_np = np.insert(gdf_np, 3, area, axis=1)
# gdf_np[:,3] = area
# geometry objekte haben einfach eine distance methode
gdf_np = gdf_np[gdf_np[:, 3] > THRESHOLD]

gdf_np = gdf_np[:int(gdf_np.shape[0] * (percentage / 100)), :]

# distance_matrix = np.genfromtxt(f"{THRESHOLD}ThresholdCSV.csv", delimiter=",")
# print("Distance Matrix Loaded")


print("Kalkuliere Matrix")
geometrys = gdf_np[:, 2]
distance_matrix = np.zeros((geometrys.shape[0], geometrys.shape[0]))
for i in tqdm(range(geometrys.shape[0])):
    for j in range(i):
        d = geometrys[i].distance(geometrys[j])
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
with open(f"{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "wb") as f:
    np.save(f, distance_matrix)
