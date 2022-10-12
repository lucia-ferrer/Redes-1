'''Implementacion de Adaline'''


#------------- Importación de librerías -------------#
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

#----------------------------------------------------#


class Adaline:
    '''Clase Adaline'''

    def __init__(self, training_csv = None, validation_csv = None, test_csv = None, tolerance = 10**-6, learning_rate = 0.00001, max_iterations = 500000):
        '''Método constructor'''

        # Si se pasan como parámetros el los conjuntos de entrenamiento y validacion
        if training_csv and validation_csv:
            # Se hace una division en [0, n-1] columnas de atributos, y columna n como valor de salida
            self.training_set, self.training_tags = self.splitData(training_csv)
            self.validation_set, self.validation_tags = self.splitData(validation_csv)

        # Si hay un conjunto de test se procede de la misma manera
        if test_csv:
            self.test_set, self.test_tags = self.splitData(test_csv)

        # Se inicializan los hiperparámetros de la red
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Se inicializan de manera aleatoria los pesos
        self.weights = self.initWeights(len(self.training_set[0]))


    def loadModel(self, trained_model):
        '''Funcion para cargar un modelo ya guardado en ejecuciones anteriores'''

        # Se abre el fichero en formato JSON
        with open(trained_model, 'r') as file:
            model = json.load(file)

        # Se cargan los pesos y la bias del modelo en la instancia
        self.weights[:-1] = np.array(model['Weights'])
        self.weights[-1] = np.array(model['Bias'])


    def saveModel(self, name = 'adaline_trained_model.json'):
        '''Funcion para salvar el modelo con el nombre deseado'''

        # Se escribe el modelo en formato JSON
        model = {'Weights': self.weights[:-1].tolist(), 'Bias': self.weights[-1].tolist()}

        # Si existe ya un modelo con ese nombre se borra
        if os.path.exists(f'./../output/{name}'):
            os.remove(f'./../output/{name}')

        # Se guarda el nuevo modelo en la carpeta output
        with open(f'./../output/{name}', 'a') as file:
            json.dump(model, file, indent = 4)


    def splitData(self, path):
        '''Metodo para dividir un fichero n-1 columnas de atributos y 1 columna de resultado'''

        # Se carga el DataFrame a partir de la ruta especificada
        df = self.loadFromCSV(path)

        # Se devuelve el conjunto subdividido en numpy arrays
        return np.array(df.iloc[:, :-1]), np.array(df.iloc[:, -1:])


    @staticmethod
    def loadFromCSV(path: str):
        '''Metodo para cargar un csv en un pd DataFrame'''

        return pd.read_csv(path, sep = ',', header = None)


    def initWeights(self, columns):
        '''Metodo para inicializar los pesos aleatoriamente'''
        
        return np.random.rand((columns + 1), 1)


    def calcOutput(self, examples, weights):
        '''Metodo para calcular el vector de salida predecido por la neurona'''

        # Sumatorio de (wi * xi) + ø
        return np.dot(examples, weights[:-1]) + weights[-1] 


    def calcMse(self, output, tags):
        '''Metodo para calcular el MSE'''

        return (((tags - output)**2).sum())/len(tags)


    def printError(self, info):
        '''Metodo para imprimir en pantalla la progresion del error'''

        sep = ' -----------------------------------------------------------------------------'
        head = '|{:^25}|{:^25}|{:^25}|'.format('CICLE', 'MEAN TRAINING ERROR', 'MEAN VALIDATION ERROR')

        print(sep)
        print(head)
        print(sep)

        for index, row in enumerate(info):
            print('|{:^25}|{:^25}|{:^25}|'.format(index + 1, row[0], row[1]))
            print(sep)


    def saveError(self, info):
        '''Metodo para escribir en un fichero la prograsion del error'''

        sep = ' -----------------------------------------------------------------------------'
        head = '|{:^25}|{:^25}|{:^25}|'.format('CICLE', 'MEAN TRAINING ERROR', 'MEAN VALIDATION ERROR')

        # Si ya existe el fichero se elimina
        if (os.path.exists('./../output/error-progression.txt')):
            os.remove('./../output/error-progression.txt')

        with open('./../output/error-progression.txt', 'a') as file:
            file.write(sep + '\n')
            file.write(head + '\n')
            file.write(sep + '\n')

            for index, row in enumerate(info):
                file.write('|{:^25}|{:^25}|{:^25}|'.format(index + 1, row[0], row[1]) + '\n')
                file.write(sep + '\n')


    def trainModel(self):
        '''Metodo para entrenar el modelo'''

        # Se inicializa un buffer para guardar la progresion MSE
        info = []

        # Se itera max_iterations veces como maximo
        for i in range(self.max_iterations):

            # Se calcula el output y el MSE de entrenamiento
            output_training = self.calcOutput(self.training_set, self.weights)
            mse_training = self.calcMse(output_training, self.training_tags)

            # Se calcula el output y el MSE de validacion
            output_validation = self.calcOutput(self.validation_set, self.weights)
            mse_validation = self.calcMse(output_validation, self.validation_tags)

            # Si se lleva mas de una ejecucion
            if len(info):

                # Y se cumple el criterio de parada
                if (info[-1][-1] < mse_validation) or ((info[-1][-1] >= mse_validation) and abs(info[-1][-1] - mse_validation) < self.tolerance):

                    # Se devuelve el buffer y el modelo queda guardado
                    return info

            # Se ajustan los pesos en funcion de la salida de entrenamiento
            self.adjustWeights(output_training, self.training_tags, self.training_set)

            # Se añade al buffer la progresion del error
            info.append((mse_training, mse_validation))

        # Se devuelve el buffer y el modelo queda guardado
        return info


    def adjustWeights(self, output, tags, examples):
        '''Metodo para ajustar los pesos del modelo'''

        # Se valculan las variaciones en el peso y en el bias
        weights_variation = self.learning_rate * (examples.T @ (tags - output))
        bias_variation = self.learning_rate * (tags - output).sum()

        # Se actualizan los pesos
        self.weights[:-1] += weights_variation
        self.weights[-1] += bias_variation


    def testModel(self):
        '''Metodo para probar la eficacia del modelo con el conjunto de test'''
        
        # Se calcula la salida de la red y el error cometid
        output_test = self.calcOutput(self.test_set, self.weights)
        mse_test = self.calcMse(output_test, self.test_tags)

        # Se devuelven ambos valores
        return mse_test, output_test


    def plotMse(self, training_info):
        '''Metodo para crear una grafica a partir de los errores cometidos'''

        mse_hist = []
        mse_v_hist = []

        for i in training_info:
            mse_v_hist.append(i[0])
            mse_hist.append(i[1])

        plt.plot(mse_hist, label = 'Validation MSE')
        plt.plot(mse_v_hist, label = 'Training MSE')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squeared Error')
        plt.legend()
        plt.show()
