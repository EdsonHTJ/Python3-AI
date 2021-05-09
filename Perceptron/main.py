import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import MLPerceptron
import os


def post_process(arr):
    array_sum = sum(arr)
    
    for i in range(len(arr)):
        s = 100*(arr[i]/array_sum)
        arr[i] = 1 if s >= 50 else 0
        
    return arr
        

train_dataset = pd.read_csv("datasets/MLP_Treino.csv")
    
train_inputs = train_dataset[["x1","x2","x3","x4"]].values
train_outputs = train_dataset[["d1","d2","d3"]].values


test_dataset = pd.read_csv("datasets/MLP_Teste.csv")


test_inputs  = test_dataset[["x1","x2","x3","x4"]].values
test_putputs = test_dataset[["d1","d2","d3"]].values



plt.title("Eqm/Epochs")

#funções sigmóide ja definidas por default
MLP = MLPerceptron(train_inputs, train_outputs, [15,3],learning_rate=0.1,precision=1e-6, post_processing=post_process)


full_result = test_dataset
full_result = full_result.iloc[:, :-1]

    
MLP.train()


responses=[]
for j in test_inputs:
    responses.append(MLP.evaluate(j))

test_result = pd.DataFrame(data=responses,columns=['T1','T2','T3'])
full_result = pd.concat([full_result,test_result],axis=1)



Acertos_T1 = 0
Acertos_T2 = 0
Acertos_T3 = 0
for i in range(len(full_result)):
    if ((full_result['T1']).values[i] == (full_result['d1']).values[i]):
        Acertos_T1 +=1
    if ((full_result['T2']).values[i] == (full_result['d2']).values[i]):
        Acertos_T2 +=1
    if ((full_result['T3']).values[i] == (full_result['d3']).values[i]):
        Acertos_T3 +=1

Acertos_T1 = Acertos_T1*100/len(full_result)
Acertos_T2 = Acertos_T2*100/len(full_result)
Acertos_T3 = Acertos_T3*100/len(full_result)

full_result["Acertos_T1"] = [Acertos_T1] + [0 for i in range(len(full_result)-1)]
full_result["Acertos_T2"] = [Acertos_T2] + [0 for i in range(len(full_result)-1)]
full_result["Acertos_T3"] = [Acertos_T3] + [0 for i in range(len(full_result)-1)]


plt.plot(range(len(MLP.eqms)), MLP.eqms, "b-")

i = 1
pltPath = f"./results/plot{i}.png"
while os.path.exists(pltPath):
    i+=1
    pltPath = f"./results/plot{i}.png"


plt.savefig(pltPath)


i = 1
csvPath = f"./results/Resultado{i}.csv"

while os.path.exists(csvPath):
    i+=1
    csvPath = f"./results/Resultado{i}.csv"


full_result.to_csv(csvPath)


print(full_result)