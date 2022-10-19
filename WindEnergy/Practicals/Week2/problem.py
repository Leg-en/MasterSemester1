import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # get the corresponding height from the rastrigin function
        self.z = (self.x ** 2 - 10 * np.cos(2 * np.pi * self.x)) + (self.y ** 2 - 10 * np.cos(2 * np.pi * self.y)) + 20


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
