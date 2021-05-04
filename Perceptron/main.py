import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
import os


def post_process(arr):
    array_sum = sum(arr)
    
    for i in range(len(arr)):
        s = 100*(arr[i]/array_sum)
        arr[i] = 1 if s>50 else 0
        
    return arr
        

train_dataset = pd.read_csv("datasets/MLP_Treino.csv")
    
train_inputs = train_dataset[["x1","x2","x3","x4"]].values
train_outputs = train_dataset[["d1","d2","d3"]].values


test_dataset = pd.read_csv("datasets/MLP_Treino.csv")


test_inputs  = test_dataset[["x1","x2","x3","x4"]].values
test_putputs = test_dataset[["d1","d2","d3"]].values



plt.title("Eqm/Epochs")

MLP = Perceptron(train_inputs, train_outputs, [15,3],learning_rate=0.1,precision=1e-6, post_processing=post_process)


full_result = test_dataset
for i in range(2):
    
    MLP.train()
    #plt.subplot(3,1,i+1)
    plt.plot(MLP.eqms)

    respostas=[]
    for j in test_inputs:
        respostas.append(MLP.evaluate(j))

    test_result = pd.DataFrame(data=respostas,columns=[f'T{i+1},1',f'T{i+1},2',f'T{i+1},3'])
    full_result = pd.concat([full_result,test_result],axis=1)
        
    plt.show()


print(full_result)