import numpy as np

from activation_function import BinaryStep
from activation_function import SignFunction

class Perceptron:
    
    def __init__(self, input_values, output_values, learning_rate = 1e-3, activation_function = SignFunction ):
        self.input_values = input_values
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(len(input_values[0]))
        self.initW = self.W
        self.theta = np.random.rand(1)[0]
        self.initTheta = self.theta
        self.epochs = 0
        
        
        if self.activation_function == SignFunction:
            
            for i in range(len(self.output_values)):
                
                if self.output_values[i] != 1 :
                    self.output_values[i]=-1
                    
        elif self.activation_function == BinaryStep:
            
            for i in range(len(self.output_values)):
                
                if self.output_values[i] != 1 :
                    self.output_values[i]=0
            
            
        
    def train(self):
        error = True
        print("Pesos Iniciais: ")
        print("Theta:"+str(self.theta))
        print("W:"+str(self.W))
        
        while error:
                self.epochs +=1
              #  print(f"Epoca {self.epochs}")
                error = False
                for x, d in zip(self.input_values, self.output_values):
                    u = np.dot(np.transpose(x), self.W) - self.theta
                    y = self.activation_function.g(u)
                    
                    if y != d:
                        
                        #print(f"Valor encontrado: {y} , esperado: {d}")
                        #print(f"Erro: {d-y}")
                        #print(f"W: {self.W}")
                        #print(f"Theta: {self.theta}")
                        #print(f"--Ajuste--")
                        self.theta = self.theta + self.learning_rate * (d - y) * -1
                        self.W = self.W + self.learning_rate * (d - y) * x
                        error =True
                        #print(f"W: {self.W}")
                        #print(f"Theta: {self.theta}")
                        
                        break
                    else:
                        ...
                        #print(f"Valor encontrado {y}, esperado {d}")
                        #print("OK")
                        
                        
                #print("======================================")
                
        print("Pesos Finais:")
        print("Theta:"+str(self.theta))
        print("W:"+str(self.W))
        print("Ultima Epoca:"+str(self.epochs))
            
            
    def evaluate(self,input_value):
            u = np.dot(np.transpose(input_value), self.W) - self.theta
            return self.activation_function.g(u)