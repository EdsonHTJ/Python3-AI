from abc import ABC, abstractmethod

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