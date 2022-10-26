from matplotlib import cm
import numpy as np
from solution import Solution
from problem import Point
import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# plot function for the 'mountain'
def plot_mountain(fig, X, Y):
    X, Y = np.meshgrid(X, Y)
    Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=10,
                    cmap=cm.Oranges, linewidth=0.08, alpha=0.4, antialiased=True)
    return ax


# plot function for the position of the hiker
def plot_position(ax, p, z_offset=2, color='b', size=40):
    # set an offset for point to make sure it is visible
    ax.scatter(p.x, p.y, p.z + z_offset, marker='o', s=size, c=color, edgecolors='black', linewidths=0.2)
    return ax


def get_random_state(seed):
    return np.random.RandomState(seed)


def get_height(point):
    return point.z


def select_best_from_parents_and_children(parent_generation, child_generation):
    combined_population = np.hstack((parent_generation, child_generation))
    objective_values_parents = [p.fitness for p in parent_generation]
    objective_values_children = [c.fitness for c in child_generation]
    all_objective_values = np.array(objective_values_parents + objective_values_children)
    # sort by the objective values and get the ids of the better half
    better_half = np.array(
        heapq.nlargest(parent_generation.shape[0], enumerate(all_objective_values), key=lambda x: x[1]),
        dtype=np.dtype(np.int32))
    survival_ids = better_half[:, 0]
    return combined_population[survival_ids]


def tournament_selection(population, minimization, random_state, selection_pressure=0.1):
    arena_size = int(len(population) * selection_pressure)
    arena = random_state.choice(population, size=arena_size)
    best = arena[0]
    for elem in arena:
        if minimization:
            if elem.fitness < best.fitness:
                best = elem
        else:
            if elem.fitness > best.fitness:
                best = elem
    return best


def geometrical_crossover(solution_A, solution_B, random_state):
    # hint: return Solution(Point(x_child_A, y_child_A)), Solution(Point(x_child_B, y_child_B))
    percentage = random_state.uniform()
    inverse_percentage = 1 - percentage
    a = [solution_A.representation.x, solution_A.representation.y]
    b = [solution_B.representation.x, solution_B.representation.y]
    dist = [a[0] - b[0], a[1] - b[1]]
    sol_a = Solution(Point(solution_B.representation.x + (dist[0] * percentage),
                           solution_B.representation.y + (dist[1] * percentage)))
    sol_b = Solution(Point(solution_B.representation.x + (dist[0] * inverse_percentage),
                           solution_B.representation.y + (dist[1] * inverse_percentage)))
    # plt.scatter([solution_A.representation.x, solution_B.representation.x],
    #             [solution_A.representation.y, solution_B.representation.y], color="red")
    # plt.scatter([sol_a.representation.x],
    #             [sol_a.representation.y], color="blue")
    # plt.scatter([sol_b.representation.x],
    #             [sol_b.representation.y], color="green")
    # plt.show()
    return sol_a, sol_b


def ball_mutation(solution, random_state, max_step_size=0.1):
    # hint: return Solution(Point(x,y))
    # Step size ist der mögliche radius in dem der neue punkt ausgehend von dem alten liegen darf. Berechnung mit random uniform distrubtion+
    return Solution(Point(solution.representation.x + random_state.uniform(-max_step_size, max_step_size),
                          solution.representation.y + random_state.uniform(-max_step_size, max_step_size)))
