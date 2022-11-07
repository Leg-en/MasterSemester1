import os
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

THRESHOLD = 40000

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

# for item in np.nditer(gdf_np[:,2], ["refs_ok"]):
#    val = item.item()
#    distances = np.empty()
#    print(item.item().area)
# print()


# geometrys = gdf_np[:, 2]
# distance_matrix = np.zeros((geometrys.shape[0], geometrys.shape[0]))
# grid1, grid2 = np.meshgrid(geometrys, geometrys, indexing="ij")
# for i in tqdm(range(geometrys.shape[0])):
#    # print(str((i/geometrys.shape[0]*100)) + "& Fortschritt")
#    for j in range(geometrys.shape[0]):
#        distance_matrix[i, j] = grid1[i, j].distance(grid2[i, j])

distance_matrix = np.genfromtxt(f"{THRESHOLD}ThresholdCSV.csv", delimiter=",")
print("Distance Matrix Loaded")


# Die distanzmatrix enthält jetzt alle relevanten distanz informationen

class WindEnergySiteSelectionProblem(Problem):

    def __init__(self):
        # super().__init__(n_var=gdf_optimization.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        super().__init__(n_var=len(gdf_np), n_obj=2, n_ieq_constr=1, xl=0.0,
                         xu=1.0)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def _evaluate(self, x, out, *args, **kwargs):
        # Todo: Mindestabtand 4 KM implementieren, Konvergenz Darstellen

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

        # Suche alle distanzen die positiv sind

        DISTANCE_THRESHOLD = 4000
        distanz = []

        # print("Starting Constraint Check...")

        # def permutations(item):
        #    d = distance_matrix[item[0], item[1]]

        for idx, i in enumerate(x):
            # temp = []
            # for idx2, item in enumerate(i):
            #    if item:
            #        temp.append(idx2)
            temp = np.where(i == True)
            temp = list(temp)[0].tolist()
            permut = list(combinations(temp, 2))
            iter_val = 1
            for val in permut:
                d = distance_matrix[val[0], val[1]]
                if DISTANCE_THRESHOLD > d:
                    iter_val = -1
                    break
            distanz.append(iter_val)
        distanz = np.asarray(distanz)

        out["G"] = distanz


algorithm = NSGA2(pop_size=100,
                  sampling=BinaryRandomSampling(),
                  crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(),
                  eliminate_duplicates=True)

problem = WindEnergySiteSelectionProblem()

res = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=True,
               save_history=True)

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
plot_val = 40
plt.scatter(np_fitness[plot_val, 0], np_fitness[plot_val, 1])
plt.scatter(res.F[:, 0], res.F[:, 1])
plt.show()

# Pymoo scatter
# Scatter().add(res.F).show()
