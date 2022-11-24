import multiprocessing
import os
import pickle
from itertools import combinations

import geopandas as geop
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from tqdm import tqdm

mode = "Win"

THRESHOLD = 4000
percentage = 100

if mode == "Win":
    dir = r'input'
    rfile = 'potentialareas_400m_forest.shp'
elif mode == "WSL":
    dir = r'/home/emily/workspace/input'
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
    if mode == "Win":
        with open(f"{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "rb") as f:
            distance_matrix = np.load(f)
        print("Vorkalkulierte Distanz Matrix gefunden und geladen")
    elif mode == "WSL":
        with open(f"/home/emily/workspace/{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "rb") as f:
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
    if mode == "Win":
        with open(f"{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "wb") as f:
            np.save(f, distance_matrix)
    elif mode == "WSL":
        with open(f"/home/emily/workspace/{THRESHOLD}_AREA_{percentage}_PERC_DIST_MAT_NEW.npy", "wb") as f:
            np.save(f, distance_matrix)


# Die distanzmatrix enthält jetzt alle relevanten distanz informationen

class WindEnergySiteSelectionProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        # super().__init__(n_var=gdf_optimization.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        super().__init__(n_var=len(gdf_np), n_obj=2, n_ieq_constr=1, xl=0.0,
                         xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def _evaluate(self, x, out, *args, **kwargs):
        DISTANCE_THRESHOLD = 4000

        def correct(y1, y2):
            # Statisch immer das erste element auf False setzen
            # Todo: Durch was sinnvolles ersetzen
            x[y1] = False
            return y1

        # Die automatische korrektur dauert einfach zu lange...
        # indices = np.where(x)
        # combs = combinations(indices[0], 2)
        # corrected = np.zeros((0))
        # for item in combs:
        #     if distance_matrix[item[0], item[1]] < DISTANCE_THRESHOLD and item[0] not in corrected and \
        #             item[1] not in corrected:
        #         c_val = correct(item[0], item[1])
        #         corrected = np.append(corrected, c_val)

        indices = np.where(x)
        combs = combinations(indices[0], 2)
        constraints_np = 1
        for item in combs:
            if distance_matrix[item[0], item[1]] < DISTANCE_THRESHOLD:
                constraints_np = -1
                break

        abstand_such = np.where(x, gdf_np[:, 0], 0)
        abstand_summe = np.sum(abstand_such)
        # for 2 objectives: out["F"] = np.column_stack([f1, f2])

        energie_such = np.where(x, gdf_np[:, 1], 0)
        energie_summe = np.sum(energie_such)
        energie_summe_corrected = energie_summe * -1

        out["F"] = np.column_stack([abstand_summe, energie_summe_corrected])

        out["G"] = np.asarray([constraints_np])


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.off = {}

    def notify(self, algorithm):
        self.off[algorithm.n_gen] = algorithm.off

def main():
    algorithm = NSGA2(pop_size=100,
                      sampling=BinaryRandomSampling(),
                      crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True)

    n_proccess = 10
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    problem = WindEnergySiteSelectionProblem(elementwise_runner=runner)
    # problem = WindEnergySiteSelectionProblem()
    callback = MyCallback()
    res = minimize(problem,
                   algorithm,
                   callback=callback,
                   termination=('n_gen', 100),
                   seed=1,
                   verbose=True)

    with open("result.pkl", "wb") as out:
        pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

    with open("callback.pkl", "wb") as out:
        pickle.dump(callback, out, pickle.HIGHEST_PROTOCOL)
    # fitness_vals = []
    #
    # for iteration in res.history:
    #     x = []
    #     y = []
    #     for item in iteration.off:
    #         x.append(item.F[0])
    #         y.append(item.F[1])
    #     fitness_vals.append([x, y])
    # np_fitness = np.asarray(fitness_vals)
    #
    # # Manual Scotter
    # plot_val = [1, 10, 30, 50]
    # for i in plot_val:
    #     plt.scatter(np_fitness[i, 0], np_fitness[i, 1])
    # plt.scatter(res.F[:, 0], res.F[:, 1])
    # plt.show()

    # Konvergenz plot von Pymoo
    # n_evals = np.array([e.evaluator.n_eval for e in res.history])
    # opt = np.array([e.opt[0].F for e in res.history])
    #
    # plt.title("Convergence")
    # plt.plot(n_evals, opt, "--")
    # plt.yscale("log")
    # plt.show()

    # Pymoo scatter
    Scatter().add(res.F).show()


if __name__ == "__main__":
    main()
