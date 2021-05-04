import numpy as np

from activation_function import sigmoid
from activation_function import sigmoid_derivative

class Perceptron:
    
    def __init__(self,input_values,output_values,layers,learning_rate=1e-2,
                 precision=1e-6,activation_function=sigmoid,derivative_function=sigmoid_derivative,post_processing=None):


       ones_column = np.ones((len(input_values), 1)) * -1
       self.input_values = np.append(ones_column, input_values, axis=1)
       
       self.output_values = output_values
       self.learning_rate = learning_rate
       self.precision = precision
       self.activation_function = activation_function
       self.derivative_function = derivative_function
       self.post_processing = post_processing

       self.I = []
       self.Y = []
       self.W = []

       n_input = self.input_values.shape[1]

       for i in range(len(layers)):
            self.W.append(np.random.rand(layers[i], n_input))
            self.I.append(np.zeros(layers[i]))
            self.Y.append(np.zeros(layers[i]))
            n_input = layers[i] + 1
    
      
       self.epochs = 0
       self.eqms = []


    def evaluate(self,x):


        y = self.full_propagation(np.append(-1,x))
        
        
        if(self.post_processing != None):
            y = self.post_processing(y)


        return y

               
    def train(self):

        error = True
        eqm_actual = self.eqm()
        
        while error:
            error = False
            eqm_previous = eqm_actual
            for x, d in zip(self.input_values,self.output_values):

                self.Y[-1] = self.full_propagation(x)
                self.back_full(x,d)
                
            eqm_actual = self.eqm()
            self.eqms.append(eqm_actual)
            self.epochs+=1
            if abs(eqm_actual-eqm_previous)>self.precision:
                error=True

        print(self.epochs)
        return self.eqms 


    def full_propagation(self,x):
        Y=x.copy()
        
        for i,w in enumerate(self.W):
                
            self.I[i] = np.dot(w, Y)
            Y    = self.activation_function(self.I[i])
            
            if i <len(self.W)-1:
                Y = np.append(-1,Y)
            self.Y[i]=Y
            
        return self.Y[i]
        

    def back_layer(self,w,w_old,y,d,u,x,out):
        if not out:
            y= y[1:]
            delta=sum(d*w_old)[1:]*self.derivative_function(u)
            delta=delta.reshape(len(delta),1)
        else:
            delta=((d-y)*self.derivative_function(u))
            delta=delta.reshape(len(delta),1)
            
        w += self.learning_rate*delta*x

        return w,delta


    def back_full(self,X,d):
        out=True
        w_old= 0
        for i,w in reversed(list(enumerate(self.W))):
            x=self.Y[i-1]
            if i == 0:
                x=X
            w,d = self.back_layer(w,w_old,self.Y[i],d,self.I[i],x,out)
            out=False
            w_old = self.W[i]



    def eqm(self):
        eq = 0
        
        for x, d in zip(self.input_values, self.output_values):
            Y = self.full_propagation(x)
            eq += 0.5 * sum((d - Y) ** 2)
            
        return eq/len(self.output_values)