import os
import pickle
from itertools import combinations

import geopandas as geop
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
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

try:
    with open(f"{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "rb") as f:
        distance_matrix = np.load(f)
    print("Vorkalkulierte Distanz Matrix gefunden und geladen")
except FileNotFoundError:
    print("Keine Vorkalkulierte Distanz Matrix gefunden")
    geometrys = gdf_np[:, 2]
    distance_matrix = np.zeros((geometrys.shape[0], geometrys.shape[0]))
    for i in tqdm(range(geometrys.shape[0])):
        for j in range(i):
            d = geometrys[i].distance(geometrys[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
    with open(f"{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "wb") as f:
        np.save(f, distance_matrix)


# Die distanzmatrix enthält jetzt alle relevanten distanz informationen

class WindEnergySiteSelectionProblem(Problem):

    def __init__(self):
        # super().__init__(n_var=gdf_optimization.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        super().__init__(n_var=len(gdf_np), n_obj=2, n_ieq_constr=0, xl=0.0,
                         xu=1.0)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def _evaluate(self, x, out, *args, **kwargs):
        DISTANCE_THRESHOLD = 4000

        def correct(x1, y1, y2):
            # Statisch immer das erste element auf False setzen
            # Todo: Durch was sinnvolles ersetzen
            x[x1, y1] = False

        # def regenerate(true_indices):
        #     for val in range(len(x)):
        #         indices = true_indices[true_indices[:, 0] == val]
        #         combs = combinations(indices[:, 1], 2)
        #         for item in combs:
        #             if distance_matrix[item[0], item[1]] < DISTANCE_THRESHOLD:
        #                 correct(val, item[0], item[1])

        # true_indices = np.asarray(list(zip(*np.where(x))))
        # regenerate(true_indices)

        def regenerate(val):
            true_indices = np.asarray(list(zip(*np.where(x[val, :]))))
            combs = combinations(true_indices[:], 2)
            combs_np = np.asarray(list(combs))
            combs_reshaped = combs_np.reshape((combs_np.shape[0], 2))
            for item in combs_reshaped:
                if distance_matrix[item[0], item[1]] < DISTANCE_THRESHOLD:
                    correct(val, item[0], item[1])
                    regenerate(val)
                    break

        for val in tqdm(range(len(x))):
            regenerate(val)

        # true_indices2 = np.asarray(list(zip(*np.where(x[0, :]))))
        # combs2 = combinations(true_indices2[:], 2)
        # np_combs2 = np.asarray(combs2_list)
        # np2 = np_combs2.reshape((30807325, 2))

        # objective function values are supposed to be written into out["F"]
        # example:
        # Array mit 100 elementen was amit jeder zeile verknüpft ist

        abstand_such = np.where(x, gdf_np[:, 0], 0)
        abstand_summe = np.sum(abstand_such, axis=1)
        # for 2 objectives: out["F"] = np.column_stack([f1, f2])

        energie_such = np.where(x, gdf_np[:, 1], 0)
        energie_summe = np.sum(energie_such, axis=1)
        energie_summe_corrected = energie_summe * -1

        out["F"] = np.column_stack([abstand_summe, energie_summe_corrected])

        ## Threshold Berechnung

        # out["G"] = constraints_np


algorithm = NSGA2(pop_size=100,
                  sampling=BinaryRandomSampling(),
                  crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(),
                  eliminate_duplicates=True)

problem = WindEnergySiteSelectionProblem()

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True,
               save_history=True)

with open("result2.pkl", "wb") as out:
    pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

fitness_vals = []

for iteration in res.history:
    x = []
    y = []
    for item in iteration.off:
        x.append(item.F[0])
        y.append(item.F[1])
    fitness_vals.append([x, y])
np_fitness = np.asarray(fitness_vals)

# Manual Scotter
plot_val = [1, 10, 30, 50]
for i in plot_val:
    plt.scatter(np_fitness[i, 0], np_fitness[i, 1])
plt.scatter(res.F[:, 0], res.F[:, 1])
plt.show()

# Konvergenz plot von Pymoo
# n_evals = np.array([e.evaluator.n_eval for e in res.history])
# opt = np.array([e.opt[0].F for e in res.history])
#
# plt.title("Convergence")
# plt.plot(n_evals, opt, "--")
# plt.yscale("log")
# plt.show()

# Pymoo scatter
# Scatter().add(res.F).show()
