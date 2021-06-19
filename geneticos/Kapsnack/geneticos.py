from random import random

class Individual:
    def __init__ (self, chromosome_size) -> None:
        self.chromosome_size = chromosome_size
        self.fitness = 0
        self.chromosome =  [1 if random() > 0.5 else 0 for i in range(chromosome_size)]


class Population:
    def __init__ (self, population_size, chromosome_size = 10, ) -> None:
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.population_fitness = 0
        self.individuals = [Individual(chromosome_size) for i in range(population_size)]

    
    def get_fittest(self, offset = 0):
        return sorted(self.individuals , key = lambda individual: individual.fitness, reverse=True)[offset]


class GeneticAlgoritim:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_count, fittnes_calc = None) -> None:
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.crossover_rate  = crossover_rate
        self.elitism_count   = elitism_count
        self.fittnes_calc    = fittnes_calc
    
    def init_population(self, chromosome_size):

        return Population(self.population_size, chromosome_size)

    def calculate_fitness(self, individual):
        
        if self.fittnes_calc == None:
            individual.fitness =  sum(individual.chromosome)/individual.chromosome_size
        else:
            individual.fitness = self.fittnes_calc(individual)
            
        return individual.fitness

    def evaluate_population(self, population):
        population_fitness = 0
        for i in range(population.population_size):
            population_fitness += self.calculate_fitness(population.individuals[i])
        population.population_fitness = population_fitness

    def select_parent(self, population):

        roullete_whell_position = random()* population.population_size
        spin_wheel = 0
        for i in range(population.population_size):
            spin_wheel += population.get_fittest(i).fitness
            if spin_wheel >= roullete_whell_position:
                return population.get_fittest(i)
        
        return population.get_fittest(-1)

    def crossover_population(self, population):
        new_population  = Population(population.population_size, population.chromosome_size)
        for i in range(population.population_size):
            parent1 = population.get_fittest(i)
            if self.crossover_rate > random() and i > self.elitism_count:
                parent2 = self.select_parent(population)
                offspring = Individual(population.chromosome_size)
                cut_index = int(random() * parent1.chromosome_size)

                for j in range(parent1.chromosome_size):
                    if j <= cut_index:
                        offspring.chromosome[j] = parent1.chromosome[j]
                    else:
                        offspring.chromosome[j] = parent2.chromosome[j]
                
                new_population.individuals[i] = offspring

            else:
                
                new_population.individuals[i] = parent1 

        return new_population

    def mutate_population(self, population):
        new_population = Population(population.population_size, population.chromosome_size)

        for i in range(population.population_size):
            individual = population.individuals[i]
            
            if i > self.elitism_count:
                for j in range(individual.chromosome_size):
                    if self.mutation_rate > random():
                        if individual.chromosome[j] == 1:
                            individual.chromosome[j] == 0
                        else:
                            individual.chromosome[j] = 1

            new_population.individuals[i] = individual

        return new_population

    



if __name__ == '__main__':

    generation = 0
    ga = GeneticAlgoritim(100, 0.05, 0.90, 2)
    population = ga.init_population(15)
    ga.evaluate_population(population)

    while generation < 100:
        print(f"Best solution: {population.get_fittest().chromosome}")
        population = ga.crossover_population(population)
        population = ga.mutate_population(population)
        ga.evaluate_population(population)
        generation += 1

    print(f"found solution in {generation} generations")
    print(f"Best solution Is: {population.get_fittest().chromosome}")



