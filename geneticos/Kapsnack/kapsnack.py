import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from genetic.geneticos import GeneticAlgoritim
from genetic.geneticos import Individual

from io import StringIO
import pandas as pd

item_values = open("./P08-cwp.txt", "r").read()
capacity_start= item_values.find("CAPACITY = ") + len("CAPACITY = ")
capacity_end  = item_values.find("\n", capacity_start)
capacity = int(item_values[capacity_start:capacity_end])

values_to_dataset = item_values[item_values.find("ID"):]

data = StringIO(values_to_dataset)
df = pd.read_csv(data, sep=",")



def calculate_profit(individual):
    value  = 0
    weight = 0 
    for i in range(individual.chromosome_size):
        value  += individual.chromosome[i] * df["PROFIT"].values[i]
        weight += individual.chromosome[i] * df["WEIGHT"].values[i]

    return (value, weight)


def fit_func(individual):
    
    value, weight = calculate_profit(individual)

    if weight <  capacity:
        return value
    else:
        return 0


if __name__ == '__main__':


    generation = 0
    ga = GeneticAlgoritim(50, 0.05, 0.90, 2, fittnes_calc= fit_func)
    population = ga.init_population(len(df))
    ga.evaluate_population(population)

    while generation < 100:
        #print(f"Best solution: {population.get_fittest().chromosome}")
        population = ga.crossover_population(population)
        population = ga.mutate_population(population)
        ga.evaluate_population(population)
        generation += 1

    value, weight = calculate_profit(population.get_fittest())

   
    print(f"found solution in {generation} generations")
    print(f"Best solution Is:")
    print(f"Profit: {str(value)}")
    print(f"Weight: {str(weight)}")
    print(f"Capacity: {str(capacity)}")



