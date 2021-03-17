import numpy as np


from activation_function import BinaryStep
from activation_function import SignFunction

class Adaline:
    
    def __init__(self, input_values, output_values, precision= 1e-3,learning_rate = 1e-3, activation_function = BinaryStep):
        self.input_values = input_values
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.precision = precision
        self.activation_function = activation_function
        self.W = np.random.rand(len(input_values[0]))
        self.initW = self.W
        self.theta = np.random.rand(1)[0]
        self.initTheta= self.theta
        self.epochs = 0
        
        if self.activation_function == SignFunction:
            
            for i in range(len(self.output_values)):
                
                if self.output_values[i] == 0 :
                    self.output_values[i]=-1
            
        print(self.output_values)
        
    def eqm(self):
        eqm = 0
        for x, d in zip(self.input_values, self.output_values):
            u = np.dot(np.transpose(x), self.W) - self.theta
            eqm += (d-u)**2
        
        return eqm / len(self.output_values)
        
        
        
            
    def train(self):
    
        print("Pesos Iniciais: ")
        print("Theta:"+str(self.theta))
        print("W:"+str(self.W))
        last_eqm = 0
        actual_eqm = 0
        while True:
            self.epochs +=1
            
            last_eqm = self.eqm()
            for x, d in zip(self.input_values, self.output_values):
                u = np.dot(np.transpose(x), self.W) - self.theta
                    
                   
                self.theta = self.theta + self.learning_rate * (d - u) * -1
                self.W = self.W + self.learning_rate * (d - u) * x
                  
            
            actual_eqm = self.eqm()
            
            if abs(actual_eqm-last_eqm)<self.precision:
                break
                
        
        print(f"Epoca Final: {self.epochs}")
        print("Pesos Finais:")
        print("Theta:"+str(self.theta))
        print("W:"+str(self.W))
        
        
            
            
    def evaluate(self,input_value):
            u = np.dot(np.transpose(input_value), self.W) - self.theta
            return self.activation_function.g(u)