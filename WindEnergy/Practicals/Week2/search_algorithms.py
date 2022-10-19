from solution import Solution
from problem import Point
import math
from copy import deepcopy
import numpy as np


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
        if random_state.normal() < 0.5:
            return 1
        else:
            return -1

    current_x = current_position.x
    current_y = current_position.y
    random_neighbor_x = current_x + (positive_or_negative() * (random_state.uniform() * max_stepsize))
    random_neighbor_y = current_y + (positive_or_negative() * (random_state.uniform() * max_stepsize))
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
                current_temperature -= self.alpha

        self.best_solution = i

        return all_solutions

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


class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, initial_solution, population_size,
                 selection, crossover, crossover_probability, mutation, mutation_probability, survival):
        RandomSearch.__init__(self, problem_instance, random_state, initial_solution)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.crossover_probability = crossover_probability
        self.mutation = mutation
        self.mutation_probability = mutation_probability
        self.survival = survival
        self.initialize()

    def initialize(self):
        self.population = self.generate_random_valid_chromosomes()
        self.elite = self.get_elite()

    def search(self, n_iterations, max_stepsize, report=False):
        elite_solutions = []

        for iteration in range(n_iterations):
            offsprings = []

            copy_parent_population = deepcopy(self.population)

            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self.random_state) for _ in
                    range(2)]

                if self.random_state.uniform() < self.crossover_probability:
                    off1, off2 = self.crossover(p1, p2, self.random_state)

                if self.random_state.uniform() < self.mutation_probability:
                    off1 = self.mutation(off1, self.random_state, max_step_size=max_stepsize)
                    off2 = self.mutation(off2, self.random_state, max_step_size=max_stepsize)

                if not hasattr(off1, 'fitness'):
                    self.problem_instance.evaluate(off1)

                if not hasattr(off2, 'fitness'):
                    self.problem_instance.evaluate(off2)

                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            # for convenience turn list into numpy array
            offsprings = np.array(offsprings)

            self.population = self.survival(copy_parent_population, offsprings)

            self.elite = self.get_elite()
            elite_solutions.append(self.elite)

            if report:
                self.verbose_reporter_inner(self.elite, iteration)

        return elite_solutions

    def crossover(self, p1, p2):
        off1, off2 = self.crossover(p1, p2, self.random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def mutation(self, chromosome):
        mutant = self.mutation(chromosome, self.random_state)
        mutant = Solution(mutant)
        return mutant

    def get_elite(self):
        objective_values = []
        for solution in self.population:
            objective_values.append(solution.fitness)
        objective_values = np.array(objective_values)
        if self.problem_instance.minimization:
            elite_solution = self.population[
                np.unravel_index(np.argmin(objective_values, axis=None), objective_values.shape)]
        else:
            elite_solution = self.population[
                np.unravel_index(np.argmax(objective_values, axis=None), objective_values.shape)]
        return elite_solution

    def recombine(selfself):
        pass

    def phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings) - np.std(fitness_parents)

    def generate_random_valid_solution(self):
        x_position = self.random_state.uniform(low=self.problem_instance.search_space[0][0],
                                               high=self.problem_instance.search_space[0][1],
                                               size=None)
        y_position = self.random_state.uniform(low=self.problem_instance.search_space[1][0],
                                               high=self.problem_instance.search_space[1][1],
                                               size=None)

        random_new_solution = Solution(Point(x_position, y_position))
        self.problem_instance.evaluate(random_new_solution)
        return random_new_solution

    def generate_random_valid_chromosomes(self):
        chromosomes = np.array([self.generate_random_valid_solution() for _ in range(self.population_size)])
        return chromosomes
