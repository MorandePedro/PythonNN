import numpy as np
import pandas as pd
import IPython
import os
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def previsao(input, pesos, bias):
    camada_1 = np.dot(input,pesos)
    camada_2 = sigmoid(camada_1)
    # IPython.embed()
    return camada_2

def img2GrayScaleMatrix(img_file):
    img = Image.open(img_file)
    img_array = np.asarray(img)
    gray_array = []

    for row in img_array:
        gray_row = []
        for pixel in row:
            R = pixel[0] / 3
            G = pixel[1] / 3
            B = pixel[2] / 3
            if (R+G+B) == 0:
                gray_scale = 1.0
            else:
                gray_scale = 1/(R+G+B)

            if gray_scale < 0.4:
                gray_row.append(0)
            else:
                gray_row.append(gray_scale)
        gray_array.append(gray_row)
    return gray_array

folder1 = 'C:\\Users\\Pedro Morande\\Documents\\PythonNN\\1'
folder2 = 'C:\\Users\\Pedro Morande\\Documents\\PythonNN\\2'
folder3 = 'C:\\Users\\Pedro Morande\\Documents\\PythonNN\\3'
folder4 = 'C:\\Users\\Pedro Morande\\Documents\\PythonNN\\4'
folder5 = 'C:\\Users\\Pedro Morande\\Documents\\PythonNN\\5'

folders = [folder1,folder2,folder3,folder4,folder5]

training_matrix_dict = {}
for names in folders:
    for file in os.listdir(names):
        number = names.split('\\')[len(names.split('\\'))-1]
        training_matrix_dict[f'{number}-{file}'] = [img2GrayScaleMatrix(f'{names}\\{file}'),int(number)]


##########
pesos = np.random.random([20,20])



IPython.embed()