from geneticos import GeneticAlgoritim
from geneticos import Individual

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


def item_print(individual):

    output_list = ""
    print(f"Best solution Is: {individual.chromosome}")
    output_list += str(individual.chromosome)
    output_list += "\n\n"
    for i in range(individual.chromosome_size):
        if individual.chromosome[i]:
            item   = individual.chromosome[i] * df["ID"].values[i]
            value  = individual.chromosome[i] * df["PROFIT"].values[i]
            weight = individual.chromosome[i] * df["WEIGHT"].values[i]
            output_list += f"ID: {item}, VALUE: {value}, WEIGHT: {weight}\n"

    value, weight = calculate_profit(population.get_fittest())

    output_list += (f"\n\n\nTotal Profit: {str(value)}\n")
    output_list += (f"Total Weight: {str(weight)}\n")
    output_list += (f"Capacity: {str(capacity)}\n")

    print(output_list)

    open("./out_list.txt", "w").write(output_list)
        
    



def fit_func(individual):
    
    value, weight = calculate_profit(individual)

    if weight <  capacity:
        return value
    else:
        return 0


if __name__ == '__main__':


    generation = 0
    ga = GeneticAlgoritim(50, 0.01, 0.97, 3, fittnes_calc= fit_func)
    population = ga.init_population(len(df))
    ga.evaluate_population(population)

    while generation < 100:
        #print(f"Best solution: {population.get_fittest().chromosome}")
        population = ga.crossover_population(population)
        population = ga.mutate_population(population)
        ga.evaluate_population(population)
        generation += 1


   
    print(f"found solution in {generation} generations")
    item_print(population.get_fittest())



