'''Implementation of ADAptive LInear NEuron (ADALINE)'''

######################## Sección de Imports ######################

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


######################## Clase Adaline  #########################


class Adaline:


    def __init__(self, training_csv = None, validation_csv = None, test_csv = None, tolerance = 10**-7, learning_rate = 0.0001, max_iterations = 50):
        '''Método constructor'''

        # Si se han introducido datos para entrenar el modelo, se separan en matriz de ejemplos y vector de salida
        if training_csv and validation_csv:
            self.training_set, self.training_tags = self.splitData(training_csv)
            self.validation_set, self.validation_tags = self.splitData(validation_csv)

        #Si se ha recibido un conjunto de datos para testar el modelo, se separan en ejemplos y salida
        if test_csv:
            self.test_set, self.test_tags = self.splitData(test_csv)

        # Se inicialican las variables necesarias: learning rate, iteraciones máximas y criterio de parada
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance  # Cuando la diferencia entre el mse de un ciclo y el siguiente sea menor, se para
        
        #Se inicializa aleatoriamente un vector de pesos, valores [0, 1)
        self.weights = self.initWeights(len(self.training_set[0]))


    def loadModel(self, trained_model):
        '''Método para cargar los datos (pesos) de un modelo entrenado anteriormente'''

        # Se abre el modelo entrenado cargando su ruta
        with open(trained_model, 'r') as file:
            model = json.load(file)

        # Se cargan los pesos en el vector de pesos
        self.weights[:-1] = np.array(model['Weights'])
        self.weights[-1] = np.array(model['Bias']) # El vector de pesos tiene una poiscón extra para la bias


    def saveModel(self):
        '''Método para salvar el modelo en un JSON llamado adaline_trained_model.json'''

        # Se da formato JSON a los pesos y la bias
        model = {'Weights': self.weights[:-1], 'Bias': self.weights[-1]}

        # Se elimina un modelo previo en caso de haberlo
        if os.path.exists('./../output/adaline_trained_model.json'):
            os.remove('./../output/adaline_trained_model.json')

        # Se guarda en el JSON el modelo
        with open('./../output/adaline_trained_model.json', 'a') as file:
            json.dump(model, file, indent = 4)


    def splitData(self, path):
        '''Método que de un CSV con filas de longitud n, devuelve una matriz de datos de ejemplos y el vector de salidas '''

        # Se cargan los datos de un csv en formato pandas Data Frame
        df = self.loadFromCSV(path)

        # De la columna 0 a la n-1 son atributos, la columna n es la salida
        return np.array(df.iloc[:, :-1]), np.array(df.iloc[:, -1:])


    @staticmethod
    def loadFromCSV(path):
        '''Método para cargar un csv a un pandas DataFrame'''

        # Se carga el csv y se devuelve en el formato deseado
        return pd.read_csv(path, sep = ',', header = None)


    def initWeights(self, columns):
        '''Método para inicializar un vector de pesos de la misma longitud que los atributos de entrada'''
        
        # Posiciones 0 a n son pesos, posición n+1 es la bias (o umbral)  
        return np.random.rand((columns + 1), 1)


    """def calcOutput(self, examples, weights):
        '''Método para calcular la predecida por la neurona'''

        # ∑(wi * xi) + ø
        return np.dot(examples, weights[:-1]) + weights[-1]"""


    def calcSingleOutput(self, example):

        return (example * self.weights[:-1]).sum() + self.weights[-1]


    def calcMse(self, output, tags):
        '''Método que devuelve el error medio cuadrático'''

        return (((tags - output)**2).sum())/len(tags)


    def calcMseOne(self, output, tags):
        return (((tags - output)**2).sum())/2


    def printError(self, info):
        '''Método para imprimir por stdout una tabla con la prograsión de los errores'''

        # Strings de formato para la tabla
        sep = ' -----------------------------------------------------------------------------'
        head = '|{:^25}|{:^25}|{:^25}|'.format('CICLE', 'MEAN TRAINING ERROR', 'MEAN VALIDATION ERROR')

        # Se imprime la cabecera
        print(sep)
        print(head)
        print(sep)

        # Para cada iteración se imprime el contenido
        for index, row in enumerate(info):
            print('|{:^25}|{:^25}|{:^25}|'.format(index + 1, row[0], row[1]))
            print(sep)


    def saveError(self, info):
        '''Método guardar por stdout una tabla con la prograsión de los errores'''

        # Strings de formato para la tabla
        sep = ' -----------------------------------------------------------------------------'
        head = '|{:^25}|{:^25}|{:^25}|'.format('CICLE', 'MEAN TRAINING ERROR', 'MEAN VALIDATION ERROR')

        # Si ya existe un fichero con este contenido, se elimina
        if (os.path.exists('error-progression.txt')):
            os.remove('error-progression.txt')

        # Se escribe la cabecera
        with open('error-progression.txt', 'a') as file:
            file.write(sep + '\n')
            file.write(head + '\n')
            file.write(sep + '\n')

            # Se escribe cada una de las líneas
            for index, row in enumerate(info):
                file.write('|{:^25}|{:^25}|{:^25}|'.format(index + 1, row[0], row[1]) + '\n')
                file.write(sep + '\n')
 

    def trainModel(self):
        '''Método para entrenar el modelo'''

        # Buffer de información para guardar la progresión del error
        info = []

        # Mientras no se hayan alcanzado las iteraciones máximas (máx de ciclos)
        for i in range(self.max_iterations):
            output_training = []
            output_validation = []

            for line in range(len(self.training_set)):
                # (1) Se calcula la salida con el conjunto de entrenamiento
                output = self.calcSingleOutput(self.training_set[line])
                output_training.append(output)

                # (2) Ajuste de pesos y umbral  iterando por cada patron (line)
                self.adjustWeights(output, self.training_tags[line], self.training_set[line])
            
            # Fin de ciclo i

            # (3) Se evalua el modelo con el conjunto de validación, y se guarda el error del entrenamiento.
            for l in self.validation_set:
                vout = self.calcSingleOutput(l)
                output_validation.append(vout)

            mse_validation = self.calcMse(output_validation, self.validation_tags)
            mse_training = self.calcMse(output_training, self.training_tags)

            # Criterio de parada, si el error de validación empieza a crecer o decrece con una tolerancia menor a la definida
            # (4) se da por concluido el entrenamiento y se devuelve la progresión del error
            if len(info):
                if (info[-1][-1] < mse_validation) or ((info[-1][-1] >= mse_validation) and abs(info[-1][-1] - mse_validation) < self.tolerance):
                    return info

            # Se añade información sobre los errores al buffer
            info.append((mse_training, mse_validation))

        # Si se alcanzan las iteraciones/ciclos máximas se devuelve la info de los errores.
        return info


    def adjustWeights(self, output, tags, examples):
        #examples --> patrones / matriz de ( columnas - 1 (salida)  X 1 )
        weights_variation = self.learning_rate * examples.reshape(len(examples), 1) * (tags - output)
        bias_variation = self.learning_rate * (tags - output)

        self.weights[:-1] += weights_variation
        self.weights[-1] += bias_variation


    def testModel(self):
        '''Método para testar el modelo'''

        output_test = []

        for l in self.test_set:
            vout = self.calcSingleOutput(l)
            output_test.append(vout)

            
        #Se prueba el modelo con los datos de test
        #output_test = self.calcOutput(self.test_set, self.weights)
        mse_test = self.calcMse(output_test, self.test_tags)

        print(len(output_test))

        #Se devuelve el error cuadrático medio y la salida de la predicción
        print('LEN : output_test', len(output_test))
        return mse_test, output_test


    def plotMse(self, training_info):
        '''Introducing the vector obtained from training, it plots the evolution over cycles of MSE'''

        mse_hist = []
        mae_hist = []

        for i in training_info:
            mae_hist.append(i[0])
            mse_hist.append(i[1])

        plt.plot(mse_hist, label = 'Validation MSE')
        plt.plot(mae_hist, label = 'Training MSE')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squeared Error')
        plt.legend()
        plt.show()



if __name__ == '__main__':

    # Preparación  y aprendizaje de la red
    ada = Adaline('./../data/training_set.csv', './../data/validation_set.csv', './../data/test_set.csv')
    info = ada.trainModel()

    # Calcular el error sobre el conjunto de test una vez finalizado el aprendizaje.
    res, output_test = ada.testModel()
    
    #Impresion en la terminal  de datos 
    ada.printError(info) 
    print('\nITERACIONES OPTIMAS: ', len(info))
    print(f'\nMSE TEST: {res}\n\n')

    ada.plotMse(info)
