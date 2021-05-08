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


test_dataset = pd.read_csv("datasets/MLP_Treino.csv")


test_inputs  = test_dataset[["x1","x2","x3","x4"]].values
test_putputs = test_dataset[["d1","d2","d3"]].values



plt.title("Eqm/Epochs")

MLP = MLPerceptron(train_inputs, train_outputs, [15,3],learning_rate=0.1,precision=1e-6, post_processing=post_process)


full_result = test_dataset
full_result = full_result.iloc[:, :-1]

    
MLP.train()


responses=[]
for j in test_inputs:
    responses.append(MLP.evaluate(j))

test_result = pd.DataFrame(data=responses,columns=['T1','T2','T3'])
full_result = pd.concat([full_result,test_result],axis=1)


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