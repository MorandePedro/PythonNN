import numpy as np
import pandas as pd
import IPython

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def previsao(input, pesos, bias):
    camada_1 = np.dot(input,pesos)
    camada_2 = sigmoid(camada_1)
    # IPython.embed()
    return camada_2

### CALCULO DOS DOTS CAMADA 1 ###

# # DOT input e peso 1
# first_index = input_vector[0] * peso_1[0]
# second_index = input_vector[1] * peso_1[1]
# dot_peso1 = first_index + second_index

# # DOT peso 1 usando numpy
# dot_peso1 = np.dot(input_vector,peso_1)

# # DOT peso 1 usando numpy
# dot_peso2 = np.dot(input_vector,peso_2)


### PRIMEIRA PREVISAO USANDO FUNCAO DE SIGMOIDE COM A SAÍDA DA CAMADA 1 ###

input_vector = np.array([1.534, 3.465])
peso_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

# predict = previsao(input_vector,peso_1,bias)
# print(f"A previsão foi: {predict}")


### FORCANDO O ERRO DA PREVISAO PARA CALCULAR O ERRO (MSE FUNCTION) ###

input_vector = np.array([2, 1.5]) 
alvo = 0

predict = previsao(input_vector, peso_1,bias)
print(f"A previsão foi: {predict}") # Errou, resultado foi maior q 0.5, ou seja, output = 1

# Calculando erro MSE
mse = np.square(predict - alvo)

print(f"Previsão {predict}, Erro {mse}")
