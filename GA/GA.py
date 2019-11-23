import numpy as np
from random import seed, randint as random, random as randU


mutation_prob   = 0.05
crossover_prob  = 0.7
min_weight = -10
max_weight = 10
MAX_Opt = 1000000
nb_population = 8
class GA:

    def __init__(self, mlp):
        W1, W2 = mlp.getAllParams()
        self.mlp = mlp
        self.W1Shape = np.shape(W1)
        self.W2Shape = np.shape(W2)
        self.chromosome_len = self.W1Shape[0] * self.W1Shape[1] * self.W2Shape[0] 
        self.init_params()
    
    def init_params(self):
        self.population = self.get_random_population_of(nb_population)
        
    def matrix_to_chromosome(self, W1, W2):
        W1 = np.matrix(W1).A1
        W2 = np.matrix(W2).A1
        return np.concatenate((W1, W2))
    
    def chromosome_to_matrix(self, chromosome):
        W1_len = self.W1Shape[0] * self.W1Shape[1]
        chromosome_W1 = chromosome[:W1_len]
        chromosome_W2 = chromosome[W1_len:]
        W1 = chromosome_W1.reshape(self.W1Shape)
        W2 = chromosome_W2.reshape(self.W2Shape)
        return (W1, W2)

    def prob_mutation(self):
        return mutation_prob > randU()

    def prob_crossover(self):
        return crossover_prob > randU()


    def get_random_pos_chromosome(self):
        return random(0, self.chromosome_len-1)

    def get_random_weight(self):
        return randU()*10

    def get_random_chromosome(self):
        W1, W2 = self.mlp.getRandomConfig()
        return self.matrix_to_chromosome(W1, W2)

    def get_random_population_of(self, n):
        population = []
        for i in range (0, n):
            population.append(self.get_random_chromosome())
        return population

    def calcul_fitness(self, chromosome):
        W1, W2 = self.chromosome_to_matrix(chromosome)
        self.mlp.setAllParams(W1, W2)
        self.mlp.forward_propagation()
        return MAX_Opt - self.mlp.calc_error() 


    def calcul_fitness_table(self, population):
        fitnesses = []
        for chromosome in population:
            fitnesses.append(self.calcul_fitness(chromosome))
        return fitnesses

    def calcul_max_fitness_table(self, population_fitnesses):
        return max(population_fitnesses)

    def get_best_chromosome(self, population):
        population_fitnesses = self.calcul_fitness_table(population)
        best_fit = self.calcul_max_fitness_table(population_fitnesses)
        best_chromosome = population[population_fitnesses.index(best_fit)]
        return (best_chromosome, best_fit)

    def crossover_one(self, parents):
        p1, p2 = parents
        if self.prob_crossover():    
            i = self.get_random_pos_chromosome()
            c1 = np.concatenate((p1[:i], p2[i:]))
            c2 = np.concatenate((p2[:i], p1[i:]))
            return (c1, c2)
        else:
            return (p1, p2)

    def crossover(self, population_parents):
        newPopulation = []
        for parents in population_parents:
            children = self.crossover_one(parents)
            newPopulation.append(children[0])
            newPopulation.append(children[1])
        return newPopulation

    def mutation_one(self, c):
        if self.prob_mutation():
            i = self.get_random_pos_chromosome()
            w = self.get_random_weight()
            while(w == c[i]):
                w = self.get_random_weight()
            c[i] = w
        return c

    def mutation(self, population):
        for i in range(0, len(population)):
            population[i] = self.mutation_one(population[i])
        return population

    def selection_one(self, population, population_fitnesses):
        spot = 0
        select_index = randU()
        for i in range(0, len(population_fitnesses)):
            if select_index > spot:
                spot += population_fitnesses[i]
            else:
                select_index = i-1
                return population[select_index]
        return population[-1]
        
    def selection(self, population):
        newPopulation = []
        population_fitnesses = self.calcul_fitness_table(population)
        for i in range(0, len(population)):
            newPopulation.append(self.selection_one(population, population_fitnesses))
        self.population = newPopulation
        return newPopulation

    def select_parents(self, population):
        i= 0
        parents = []
        while(i < len(population)):
            parents.append((population[i], population[i+1]))
            i +=2
        return parents


    # i=0
    # best_guess_ever = []
    # best_fit_ever = 0
    # while(1):
    #     i +=1
    #     population = selection(population)
    #     parents = select_parents(population)
    #     children = crossover(parents)
    #     newPopulation = mutation(children)
    #     population = newPopulation
    #     population_fitnesses = calcul_fitness_table(population)
    #     max_fitness = calcul_max_fitness_table(population_fitnesses)
    #     if max_fitness > 0.9:
    #         break
    #     best_guess, best_fit = get_best_guess(population)
    #     if(best_fit_ever<best_fit):
    #         best_guess_ever, best_fit_ever = best_guess, best_fit
    #     print('[%5d]: where the best current guess is "%s" with fitness: %2.1f'%(i, best_guess, best_fit*100), 
    #     '%', ' while best guess ever is "%s" with fitness: %2.1f'%(best_guess_ever, best_fit_ever*100), '%')

    # best_guess, best_fit = get_best_guess(population)
    # print('[%5d]: where the best guess is "%s" with fitness : %2.1f'%(i, best_guess, best_fit*100), "%")


# for i in range (0, 30000):
#     population = selection(population)
#     parents = select_parents(population)
#     children = crossover(parents)
#     newPopulation = mutation(children)
#     population = newPopulation
#     population_fitnesses = calcul_fitness_table(population)
#     max_fitness = calcul_max_fitness_table(population_fitnesses)
#     if max_fitness > 0.9:
#         break
# best_guess, best_fit = get_best_guess(population)
# print('finished at %d where the best guess is "%s" with fitness : %d'%(i, best_guess, best_fit*100), "%")


# for i in range (0, len(population)):
#     fitness = calcul_fitness(population[i]) * 100
#     print(population[i], ' : %2.4f'% fitness)
