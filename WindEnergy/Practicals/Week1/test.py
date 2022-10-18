from multiprocessing import Pool
import numpy as np
import math

class RandomSearch(SearchAlgorithm):
  def __init__(self, problem_instance, random_state, initial_solution, 
               neighborhood_function = random_neighbor_position, 
               max_stepsize = 0.01):
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

    self.best_solution=i

    return all_solutions

  def generate_new_position(self, current_position):
    random_new_position = self.neighborhood_function(self.random_state, current_position = current_position, 
                                                     max_stepsize = self.max_stepsize)
    random_new_solution = Solution(random_new_position)
    self.problem_instance.evaluate(random_new_solution)
    return random_new_solution

class Annealing(RandomSearch):
  def __init__(self, problem_instance, random_state, initial_solution,
               neighborhood_function = random_neighbor_position, 
               max_stepsize = 0.01, initial_temperature = 1, alpha = 0.01):
    RandomSearch.__init__(self, problem_instance, random_state,initial_solution,
                          neighborhood_function, max_stepsize)
    self.initial_temperature = initial_temperature
    self.alpha = alpha
    
  def search(self, n_iterations, report=False):
    #this is the equivalent to the temperature parameter of simulated annealing
    all_solutions = []
    current_temperature=self.initial_temperature

    i = self.best_solution
    

    for iteration in range(n_iterations):
      j = self.generate_new_position(i.representation)
      i = self.get_p_best(i, j, current_temperature)
      all_solutions.append(i)

      if report:
        self.verbose_reporter_inner(i, iteration)
      
      if current_temperature>=0.01:
          #Change this to something non Lienar
        #current_temperature -= self.alpha
        #current_temperature = self.temp_reduction(mode="linear", temperature=current_temperature, alpha=self.alpha)
        current_temperature = self.temp_reduction(mode="non-linear", temperature=current_temperature, alpha=self.alpha)

    self.best_solution=i

    return all_solutions
  #Possible Solution for the first task?
  def temp_reduction(self, mode, temperature, alpha):
      if mode == "linear":
          return temperature - alpha
      if mode == "non-linear":
          return  temperature - pow(2, alpha)

  def get_p_best(self, candidate_a, candidate_b, actual_c):
    p=0
    if candidate_b.valid:
      if self.problem_instance.minimization:
        # if the new solution is better we accept it
        if candidate_b.fitness <= candidate_a.fitness:
          solution = candidate_b
        #if the new solution is worse we accept it with a certain probability
        elif self.random_state.uniform(0, 1) < (math.exp((candidate_b.fitness - candidate_a.fitness)) / actual_c):
          solution = candidate_b
        else:
          solution = candidate_a
      else:
          # if the new solution is better we accept it
        if candidate_b.fitness >= candidate_a.fitness:
          solution = candidate_b
        #if the new solution is worse we accept it with a certain probability
        elif self.random_state.uniform(0, 1) < (math.exp((candidate_b.fitness - candidate_a.fitness)) / actual_c):
          solution = candidate_b
        else:
          solution = candidate_a
    else:
      solution = candidate_a
    
    return solution
      
  def generate_new_position(self, current_position):
    random_new_position = self.neighborhood_function(self.random_state, current_position = current_position, 
                                                     max_stepsize = self.max_stepsize)
    random_new_solution = Solution(random_new_position)
    self.problem_instance.evaluate(random_new_solution)
    return random_new_solution


def getAnnealingOptim(step_size):
    intern_best = -1
    for alpha in np.linspace(0.1,1, 11):
        for init_temp in range(1,11):
            sa = Annealing(problem_instance=complex_hiking_problem, max_stepsize=step_size, random_state=random_state,
            initial_solution=start_solution, initial_temperature=init_temp, alpha=alpha)
            simulated_annealing_solutions = sa.search(n_iterations=2000, report=False)
            if sa.best_solution.fitness > intern_best:
                intern_best = sa.best_solution.fitness
    return intern_best

if __name__ == "__main__":
    with Pool(2) as p:
            print(p.map(test, [1,2]))