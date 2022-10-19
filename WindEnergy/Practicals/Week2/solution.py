class Solution:
    _id = 0

    def __init__(self, representation):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation

    def verbose(self, print_representation=True):
        print("Solution ID: %d\nFitness: %.2f\nIs admissible?\tR: %s" %
              (self._solution_id, self.fitness, self.valid))
