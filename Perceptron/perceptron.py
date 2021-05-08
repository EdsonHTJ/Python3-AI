import numpy as np

from activation_function import sigmoid
from activation_function import sigmoid_derivative


class MLPerceptron:

    def __init__(self, input_values, output_values, layers, learning_rate=1e-2,
                 precision=1e-6, activation_function=sigmoid, derivative_function=sigmoid_derivative, post_processing=None):

        minus_col = np.ones((len(input_values), 1)) * -1
        # Coloca a coluna de (-1) no inicio dos valores de Input
        self.input_values = np.append(minus_col, input_values, axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.precision = precision
        self.post_processing = post_processing

        self.G = activation_function
        self.dG = derivative_function

        self.I = []
        self.Y = []
        self.W = []

        self.W.append(np.random.rand(layers[0], len(self.input_values[0])))
        self.I.append(np.zeros(layers[0]))
        self.Y.append(np.zeros(layers[0]))

        for i in range(1, len(layers)):
            self.W.append(np.random.rand(layers[i], layers[i-1] + 1))
            self.I.append(np.zeros(layers[i]))
            self.Y.append(np.zeros(layers[i]))

        self.epochs = 0
        self.eqms = []

    def evaluate(self, input_eval):

        output = self.full_propagation(np.append(-1, input_eval))

        if(self.post_processing != None):
            output = self.post_processing(output)
        return output

    def train(self):

        error = True
        eqm_actual = self.eqm()

        while error:

            error = False
            eqm_previous = eqm_actual
            
            for x, d in zip(self.input_values, self.output_values):
                
                #realiza a predicao do valor de saida
                self.Y[-1] = self.full_propagation(x)
                
                #realiza o ajuste na rede
                self.back_propagation(x, d)

            eqm_actual = self.eqm()
            self.eqms.append(eqm_actual)
            self.epochs += 1

            # check de precisao
            if abs(eqm_actual-eqm_previous) > self.precision:
                error = True

        return self.eqms

    def full_propagation(self, x):

        Y = np.array(x)

        for i, w in enumerate(self.W):

            # Calcula In como sendo Wn * Yn-1
            self.I[i] = np.dot(w, Y)

            # Atualiza o Yn obtendo G(I)
            Y = self.G(self.I[i])

            if i < len(self.W)-1:
                # adiciona Y0 = -1
                Y = np.append(-1, Y)

            self.Y[i] = Y

        return self.Y[i]

    def back_propagation(self, x_input, d):

        cnt = 0

        for i in reversed(range(len(self.W))):

            if i == 0:
                x = x_input
            else:
                x = self.Y[i-1]

            y_aux = self.Y[i]

            if i == len(self.W) - 1:
                delta = ((d-y_aux)*self.dG(self.I[i]))
            else:
                delta = sum(d*self.W[i+1])[1:]*self.dG(self.I[i])

            # realiza a transposta do delta
            delta = np.array([delta]).T

            # ajusta os pesos
            self.W[i] += self.learning_rate*delta*x

            # faz com que dn = delta(n+1)
            d = delta
            cnt += 1

    def eqm(self):
        
        eq = 0

        for x, d in zip(self.input_values, self.output_values):

            Y = self.full_propagation(x)
            eq += 0.5 * sum((d - Y) ** 2)

        eqm = eq/len(self.output_values)

        return eqm
