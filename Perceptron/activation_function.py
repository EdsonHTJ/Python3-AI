from abc import ABC, abstractmethod
import math
import numpy as np


class Function(ABC):
    
    @staticmethod
    @abstractmethod
    def g(u):
        ...
             
class BinaryStep(Function):
    
    def g(u):
        return 1 if u >= 0 else 0
    
class SignFunction(Function):
        
    def g(u):
        return 1 if u>= 0 else -1
    

def TanH(u):
    return (1-math.e**(-2*u))/(1+math.e**(-2*u))


def TanHDerivative(u):
    return 1-TanH(u)


def sigmoid(u):
    return 1/(1+np.exp(-u))

def sigmoid_derivative(u):
    return 1 - sigmoid(u)