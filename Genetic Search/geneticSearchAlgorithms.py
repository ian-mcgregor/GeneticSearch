from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
import random 
import math 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams
import time
#############################
class GASolver: 
    def __init__(self, params, lst_of_identifiers, n):
        # Parameters for GA: see geneticAlgParams
        # Also includes test data for regression and checking validity
        self.params = params
        # The population size 
        self.N = n
        # Store the actual population (you can use other data structures if you wish)
        self.pop = []
        # A list of identifiers for the expressions
        self.identifiers = lst_of_identifiers
        # Maintain statistics on best fitness in each generation
        self.population_stats = []
        # Store best solution so far across all generations
        self.best_solution_so_far = None
        # Store the best fitness so far across all generations
        self.best_fitness_so_far = -float('inf')
        # Store k value for simplicity
        self.k = int(self.params.elitism_fraction * self.N)
        # List to store elites in each generation
        self.elites = []
        self.iterNum = 0
    
    
    
    #############################
    # TODO #1:
    # Generate Initial Population
    def generate_initial_pop(self):
        while(len(self.pop) < self.N):
            expr = generate_random_expr(self.params.depth, self.identifiers, self.params)
            if(is_viable_expr(expr, self.identifiers, self.params)):
                fit = compute_fitness(expr, self.identifiers, self.params)
                self.pop.append((expr, fit))
    
    #############################
    # TODO #2:
    # Mutation // Crossover
    def mutate(self):
        # Empty list to store N - k mutations
        mutations = []
        # Empty list  to compute weights
        weightList = []
        # Compute weight for probability of being randomly selected
        for sample in self.pop:
            weightList.append(math.exp(sample[1]/self.params.temperature))
        # While # of mutations < N - k
        while(len(mutations) < (self.N - self.k - 1)):
            # Generate e1 and e2
            e1, e2 = random.choices(self.pop, weights = weightList, k=2)
            # Generate cross
            e1_cross, e2_cross = random_subtree_crossover(e1[0], e2[0], copy = True)
            # Generate Mutations
            e1_mutation = random_expression_mutation(e1_cross, self.identifiers, self.params, copy = True)
            e2_mutation = random_expression_mutation(e2_cross, self.identifiers, self.params, copy = True)
            
            # If mutations are viable, append them to mutations list
            if(is_viable_expr(e1_mutation, self.identifiers, self.params)):
                mutations.append((e1_mutation, compute_fitness(e1_mutation, self.identifiers, self.params)))
            if(is_viable_expr(e2_mutation, self.identifiers, self.params)):
                mutations.append((e2_mutation, compute_fitness(e2_mutation, self.identifiers, self.params)))
        # Return list of mutations
        return mutations
    
    #############################
    # Compute Fitness Helper
    def take_second(self, x):
        return x[1]
    
    #############################
    # TODO #3:
    # Elitism
    def elitism(self):
        self.pop = sorted(self.pop, key = self.take_second, reverse = True)
        # Append top k elites to self.elites
        self.elites = self.pop[:self.k]

    #############################
    # TODO #3:
    # Next Generation Population
    def next_gen(self, mutations):
        
        # Make a list that concatenates elites with mutations from this iteration
        nextGen = self.elites + mutations
        # Update self.pop
        self.pop = nextGen
        # Reorder population to observe best so far
        self.pop = sorted(self.pop, key = self.take_second, reverse = True)
        bestThisGen = self.pop[0][1]
        if(self.iterNum <= 1):
            self.iterNum = self.iterNum + 1
            print('Initial Fitness:', self.best_fitness_so_far)
        # If best this generation is more fit than the best from previous generations
        if(bestThisGen > self.best_fitness_so_far):
            # Update best so far and print notification
            self.best_fitness_so_far = bestThisGen
            self.best_solution_so_far = self.pop[0][0]
        # Append best fitness for population statistics metrics
        self.population_stats.append(self.best_fitness_so_far)
    
    #############################
    # Pretty Print Runtime
    def printTime(self, runtime):
        runtime %= (24 * 3600)
        runtime %= 3600
        minutes = runtime // 60
        seconds = runtime % 60
        print('Runtime (m:s): {0}:{1}'.format(int(minutes), int(seconds)))

    #############################
    # GA Driver
    def run_ga_iterations(self, n_iter=1000):
        start = time.time()
        # Initialize best fitness to -inf so that we catch the first generation that is greater 
        self.best_fitness_so_far = -math.inf
        # Empty list to store mutations
        mutations = []
        # Generate Initial Population
        self.generate_initial_pop()

        for i in range(n_iter):
            # Mutate & Crossover
            mutations = self.mutate()
            # Elitism
            self.elitism()
            # NextGen
            self.next_gen(mutations)
        finish = time.time()
        runtime = finish - start
        self.printTime(runtime)
## Function: curve_fit_using_genetic_algorithms
# Run curvefitting using given parameters and return best result, best fitness and population statistics.
# DO NOT MODIFY
def curve_fit_using_genetic_algorithm(params, lst_of_identifiers, pop_size, num_iters):
    solver = GASolver(params, lst_of_identifiers, pop_size)
    solver.run_ga_iterations(num_iters)
    return (solver.best_solution_so_far, solver.best_fitness_so_far, solver.population_stats)