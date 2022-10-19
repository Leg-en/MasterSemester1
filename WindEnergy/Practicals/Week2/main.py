import numpy as np
from problem import ContinuousProblem, Point
from solution import Solution
from search_algorithms import GeneticAlgorithm
from utility_functions import plot_mountain, plot_position, get_height, get_random_state, \
    tournament_selection, ball_mutation, geometrical_crossover, select_best_from_parents_and_children
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # create the problem instance
    mountain_boundary = [[0, 3], [0, 3]]
    hiking_problem = ContinuousProblem(search_space=mountain_boundary, objective_function=get_height)

    # setup random state
    seed = 3
    random_state = get_random_state(seed)

    p_start = Point(1, 0)
    start_solution = Solution(p_start)
    hiking_problem.evaluate(start_solution)

    ga = GeneticAlgorithm(problem_instance=hiking_problem,
                          random_state=random_state,
                          initial_solution=start_solution,
                          population_size=40,
                          selection=tournament_selection,
                          crossover=geometrical_crossover,
                          crossover_probability=0.8,
                          mutation=ball_mutation,
                          mutation_probability=0.1,
                          survival=select_best_from_parents_and_children
                          )
    max_stepsize_for_ballmutation = 0.1
    ga_solutions = ga.search(n_iterations=20, max_stepsize=max_stepsize_for_ballmutation)

    # set up an empty plot
    fig = plt.figure()
    # produce the mountain plot
    X = np.linspace(0, 3, 100)
    Y = np.linspace(0, 3, 100)
    ax = plot_mountain(fig, X, Y)
    for i in range(len(ga_solutions)):
        ax = plot_position(ax, ga_solutions[i].representation, color='black', size=5)
    # show the plots
    plt.show()
