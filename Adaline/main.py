import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaline import Adaline
from activation_function import SignFunction
import os

def removeComma(St):
    St = St.replace(",",".")
    St = float(St)
    return St
    

#dataset = pd.read_csv('datasets/dataset-treinamento.csv')
dataset = pd.read_csv('datasets/dataset-treinamento.csv',delimiter=",")
#dataset = dataset.applymap(removeComma)
print(dataset)
X = dataset.iloc[:,0:-1].values
d = dataset.iloc[:,-1:].values



a = Adaline(X, d, learning_rate=0.01,precision=1e-3,activation_function=SignFunction)

a.train()


#evaluateDataSet = X
evaluateDataSet = pd.read_csv('datasets/dataset-teste.csv',delimiter=",")
#evaluateDataSet = evaluateDataSet.applymap(removeComma)

results = []


for _x in evaluateDataSet.values:
    results.append(a.evaluate(_x))
    print(f"Evaluating {_x} =>"+str(results[-1]))

evaluateDataSet["Result"] = results
evaluateDataSet["Pesos Iniciais"] = list(a.initW) + [0 for i in range(len(results)-len(a.initW))]
evaluateDataSet["Pesos Finais"] = list(a.W) + [0 for i in range(len(results)-len(a.W))]
evaluateDataSet["Theta Inicial"] = [float(a.initTheta)] + [0 for i in range(len(results)-1)]
evaluateDataSet["Theta Final"] = [float(a.theta)] + [0 for i in range(len(results)-1)]
evaluateDataSet["Epocas"] =[float(a.epochs)] + [0 for i in range(len(results)-1)]
plt.plot(range(len(a.eqms)), a.eqms, "b-")
i = 1
pltPath = f"./plot{i}.png"
while os.path.exists(pltPath):
    i+=1
    pltPath = f"./plot{i}.png"


plt.savefig(pltPath)

i = 1
csvPath = f"./Resultado{i}.csv"

while os.path.exists(csvPath):
    i+=1
    csvPath = f"./Resultado{i}.csv"

evaluateDataSet.to_csv(csvPath)

    
    


