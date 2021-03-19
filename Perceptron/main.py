import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
import os

def removeComma(St):
    St = St.replace(",",".")
    St = float(St)
    return St
    

#dataset = pd.read_csv('datasets/dataset-treinamento.csv')
dataset = pd.read_csv('datasets/dataset-treinamento.csv',delimiter=";")
dataset = dataset.applymap(removeComma)
print(dataset)
X = dataset.iloc[:,0:-1].values
d = dataset.iloc[:,-1:].values



p = Perceptron(X, d, learning_rate=0.01)

p.train()


#evaluateDataSet = X
evaluateDataSet = pd.read_csv('datasets/dataset-teste.csv',delimiter=";")
evaluateDataSet = evaluateDataSet.applymap(removeComma)

results = []


for _x in evaluateDataSet.values:
    results.append(p.evaluate(_x))
    print(f"Evaluating {_x} =>"+str(results[-1]))

evaluateDataSet["Result"] = results
evaluateDataSet["Pesos Iniciais"] = list(p.initW) + [0 for i in range(len(results)-len(p.initW))]
evaluateDataSet["Pesos Finais"] = list(p.W) + [0 for i in range(len(results)-len(p.W))]
evaluateDataSet["Theta Inicial"] = [float(p.initTheta)] + [0 for i in range(len(results)-1)]
evaluateDataSet["Theta Final"] = [float(p.theta)] + [0 for i in range(len(results)-1)]
evaluateDataSet["Epocas"] = [float(p.epochs)] + [0 for i in range(len(results)-1)]



i = 1
csvPath = f"./results/Resultado{i}.csv"

while os.path.exists(csvPath):
    i+=1
    csvPath = f"./results/Resultado{i}.csv"

evaluateDataSet.to_csv(csvPath)

    
    


