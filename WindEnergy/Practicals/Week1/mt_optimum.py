import enum
from multiprocessing import Pool
import numpy as np
import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # get the corresponding height from the rastrigin function
        self.z = (self.x ** 2 - 10 * np.cos(2 * np.pi * self.x)) + (self.y ** 2 - 10 * np.cos(2 * np.pi * self.y)) + 20


class SearchAlgorithm:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance

    def initialize(self):
        pass

    def search(self, n_iterations, report=False):
        pass

    def get_best(self, candidate_a, candidate_b):
        if self.problem_instance.minimization:
            if candidate_a.fitness >= candidate_b.fitness:
                return candidate_b
            else:
                return candidate_a
        else:
            if candidate_a.fitness <= candidate_b.fitness:
                return candidate_b
            else:
                return candidate_a

    # Here, we define reporting functions. They are not a requirement, but convenient.  
    def verbose_reporter(self):
        print("Best solution found:")
        self.best_solution.verbose()

    def verbose_reporter_inner(self, solution, iteration):
        print("> > > Current best solution at iteration %d:" % iteration)
        solution.verbose()


def random_neighbor_position(random_state, current_position, max_stepsize):
    def positive_or_negative():
        if random_state.random() < 0.5:
            return 1
        else:
            return -1

    current_x = current_position.x
    current_y = current_position.y
    random_neighbor_x = current_x + (positive_or_negative() * (random_state.random() * max_stepsize))
    random_neighbor_y = current_y + (positive_or_negative() * (random_state.random() * max_stepsize))
    return Point(random_neighbor_x, random_neighbor_y)


class RandomSearch(SearchAlgorithm):
    def __init__(self, problem_instance, random_state, initial_solution,
                 neighborhood_function=random_neighbor_position,
                 max_stepsize=0.01):
        SearchAlgorithm.__init__(self, problem_instance)
        self.random_state = random_state
        self.best_solution = initial_solution
        self.neighborhood_function = neighborhood_function
        self.max_stepsize = max_stepsize
        self.problem_instance.evaluate(self.best_solution)

    def search(self, n_iterations, report=False):
        all_solutions = []
        i = self.best_solution

        for iteration in range(n_iterations):
            j = self.generate_new_position(i.representation)
            i = self.get_best(i, j)
            all_solutions.append(i)

            if report:
                self.verbose_reporter_inner(i, iteration)

        self.best_solution = i

        return all_solutions

    def generate_new_position(self, current_position):
        random_new_position = self.neighborhood_function(self.random_state, current_position=current_position,
                                                         max_stepsize=self.max_stepsize)
        random_new_solution = Solution(random_new_position)
        self.problem_instance.evaluate(random_new_solution)
        return random_new_solution


class Annealing(RandomSearch):
    def __init__(self, problem_instance, random_state, initial_solution,
                 neighborhood_function=random_neighbor_position,
                 max_stepsize=0.01, initial_temperature=1, alpha=0.01):
        RandomSearch.__init__(self, problem_instance, random_state, initial_solution,
                              neighborhood_function, max_stepsize)
        self.initial_temperature = initial_temperature
        self.alpha = alpha

    def search(self, n_iterations, report=False):
        # this is the equivalent to the temperature parameter of simulated annealing
        all_solutions = []
        current_temperature = self.initial_temperature

        i = self.best_solution

        for iteration in range(n_iterations):
            j = self.generate_new_position(i.representation)
            i = self.get_p_best(i, j, current_temperature)
            all_solutions.append(i)

            if report:
                self.verbose_reporter_inner(i, iteration)

            if current_temperature >= 0.01:
                # Change this to something non Lienar
                # current_temperature -= self.alpha
                # current_temperature = self.temp_reduction(mode="linear", temperature=current_temperature, alpha=self.alpha)
                current_temperature = self.temp_reduction(mode="linear", temperature=current_temperature,
                                                          alpha=self.alpha)

        self.best_solution = i

        return all_solutions

    # Possible Solution for the first task?
    def temp_reduction(self, mode, temperature, alpha):
        if mode == "linear":
            return temperature - alpha
        if mode == "non-linear":
            return temperature - pow(2, alpha)

    def get_p_best(self, candidate_a, candidate_b, actual_c):
        p = 0
        if candidate_b.valid:
            if self.problem_instance.minimization:
                # if the new solution is better we accept it
                if candidate_b.fitness <= candidate_a.fitness:
                    solution = candidate_b
                # if the new solution is worse we accept it with a certain probability
                elif self.random_state.uniform(0, 1) < (
                        math.exp((candidate_b.fitness - candidate_a.fitness)) / actual_c):
                    solution = candidate_b
                else:
                    solution = candidate_a
            else:
                # if the new solution is better we accept it
                if candidate_b.fitness >= candidate_a.fitness:
                    solution = candidate_b
                # if the new solution is worse we accept it with a certain probability
                elif self.random_state.uniform(0, 1) < (
                        math.exp((candidate_b.fitness - candidate_a.fitness)) / actual_c):
                    solution = candidate_b
                else:
                    solution = candidate_a
        else:
            solution = candidate_a

        return solution

    def generate_new_position(self, current_position):
        random_new_position = self.neighborhood_function(self.random_state, current_position=current_position,
                                                         max_stepsize=self.max_stepsize)
        random_new_solution = Solution(random_new_position)
        self.problem_instance.evaluate(random_new_solution)
        return random_new_solution


def get_random_state(seed):
    return np.random.RandomState(seed)


class Problem:
    def __init__(self, search_space, objective_function, minimization):
        self.search_space = search_space
        self.objective_function = objective_function
        self.minimization = minimization

    def validate(self, solution):
        pass

    def evaluate(self, solution):
        pass


class ContinuousProblem(Problem):

    def __init__(self, search_space, objective_function, minimization=False):
        Problem.__init__(self, search_space, objective_function, minimization)

    def evaluate(self, solution):
        point = solution.representation

        # The validation process determines whether a solution is a feasible solution to the problem. 
        # For this specific hiking problem, we want to ensure that the position always lies within the mountain area.
        solution.valid = self.validate(point)

        if solution.valid:
            solution.fitness = self.objective_function(point)
        else:
            if self.minimization:
                solution.fitness = np.iinfo(np.int32).max
            else:
                solution.fitness = 0

    def validate(self, point):
        validity = True
        # check whether the x position is within the defined mountain region
        if point.x < self.search_space[0][0] or point.x > self.search_space[0][1]:
            validity = False
        # check whether the y position is within the defined mountain region
        if point.y < self.search_space[1][0] or point.y > self.search_space[1][1]:
            validity = False
        return validity


def get_height(point):
    return point.z


class Solution:
    _id = 0

    def __init__(self, representation):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation

    def verbose(self, print_representation=True):
        print("Solution ID: %d\nFitness: %.2f\nIs admissible?\tR: %s" %
              (self._solution_id, self.fitness, self.valid))


def getAnnealingOptim(step_size):
    intern_best = -1
    step_size_best = 0
    alpha_best = 0
    init_temp_best = 0
    p_start = Point(x=1, y=0)
    mountain_boundary = [[0, 3], [0, 3]]
    random_state = get_random_state(20)
    start_solution = Solution(p_start)
    complex_hiking_problem = ContinuousProblem(search_space=mountain_boundary, objective_function=get_height)
    for alpha in np.linspace(0.1, 1, 101):
        for init_temp in range(1, 100):
            sa = Annealing(problem_instance=complex_hiking_problem, max_stepsize=step_size, random_state=random_state,
                           initial_solution=start_solution, initial_temperature=init_temp, alpha=alpha)
            simulated_annealing_solutions = sa.search(n_iterations=2000, report=False)
            if sa.best_solution.fitness > intern_best:
                intern_best = sa.best_solution.fitness
                step_size_best = step_size
                alpha_best = alpha
                init_temp_best = init_temp
    return [intern_best, step_size_best, alpha_best, init_temp_best]


def start():
    p_start = Point(x=1, y=0)
    mountain_boundary = [[0, 1], [0, 1]]
    hiking_problem = ContinuousProblem(search_space=mountain_boundary, objective_function=get_height)
    random_state = get_random_state(20)
    start_solution = Solution(p_start)
    hiking_problem.evaluate(start_solution)
    sa = Annealing(problem_instance=hiking_problem, random_state=random_state,
                   initial_solution=start_solution, initial_temperature=50)

    simulated_annealing_solutions = sa.search(n_iterations=400, report=False)
    sa.verbose_reporter()


def start_mt():
    with Pool(8) as p:
        result = p.map(getAnnealingOptim, np.linspace(0, 10, 100))
        best = [0]
        print(result)
        for val in result:
            if val[0] > best[0]:
                best = val
        print(best)
        npar = np.asarray(result)
        np.savetxt("results5.csv", npar, delimiter=",")


if __name__ == "__main__":
    start_mt()
