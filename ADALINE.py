'''Implementation of ADAptive LInear NEuron (ADALINE)'''


import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt


class Adaline:


    def __init__(self, training_csv = None, validation_csv = None, test_csv = None, tolerance = 0.0001, learning_rate = 0.001, max_iterations = 1000):
        '''Constructor method'''

        if training_csv and validation_csv:
            self.training_set, self.training_tags = self.splitData(training_csv)
            self.validation_set, self.validation_tags = self.splitData(validation_csv)

        if test_csv:
            self.test_set, self.test_tags = self.splitData(test_csv)

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self.weights = self.initWeights(len(self.training_set[0]))


    def loadModel(self, trained_model):
        '''Load model from JSON saved in other executions'''

        with open(trained_model, 'r') as file:
            model = json.load(file)

        self.weights[:-1] = np.array(model['Weights'])
        self.weights[-1] = np.array(model['Bias'])


    def saveModel(self):
        '''Save model into path'''

        model = {'Weights': self.weights[:-1], 'Bias': self.weights[-1]}

        if os.path.exists('adaline_trained_model.json'):
            os.remove('adaline_trained_model.json')

        with open('adaline_trained_model.json', 'a') as file:
            json.dump(model, file, indent = 4)


    def splitData(self, path):
        '''Method splitting data into n-1 attribute cols nparray and 1 example col'''

        df = self.loadFromCSV(path)

        return np.array(df.iloc[:, :-1]), np.array(df.iloc[:, -1:])


    @staticmethod
    def loadFromCSV(path: str):
        '''Loads data from CSV into a pandas DataFrame'''

        return pd.read_csv(path, sep = ',', header = None)


    def initWeights(self, columns):
        '''Method to iniatilze randomly'''
        
        return np.random.rand((columns + 1), 1)


    def calcOutput(self, examples, weights):
        '''Method for calculating the output of the neuron'''

        # sumatory of (wi * xi) + ø
        return np.dot(examples, weights[:-1]) + weights[-1] 


    def calcMse(self, output, tags):
        '''Method for computing MSE'''

        return (((tags - output)**2).sum())/len(tags)


    def printError(self, info):
        '''Method to print formatted table '''

        sep = ' -----------------------------------------------------------------------------'
        head = '|{:^25}|{:^25}|{:^25}|'.format('CICLE', 'MEAN TRAINING ERROR', 'MEAN VALIDATION ERROR')

        print(sep)
        print(head)
        print(sep)

        for index, row in enumerate(info):
            print('|{:^25}|{:^25}|{:^25}|'.format(index + 1, row[0], row[1]))
            print(sep)


    def saveError(self, info):
        '''Method for storing the progression of MSE and MAE over cycles'''

        sep = ' -----------------------------------------------------------------------------'
        head = '|{:^25}|{:^25}|{:^25}|'.format('CICLE', 'MEAN TRAINING ERROR', 'MEAN VALIDATION ERROR')

        if (os.path.exisits('error-progression.txt')):
            os.remove('error-progression.txt')

        with open('error-progression.txt', 'a') as file:
            file.write(sep + '\n')
            file.write(head + '\n')
            file.write(sep + '\n')

            for index, row in enumerate(info):
                file.write('|{:^25}|{:^25}|{:^25}|'.format(index + 1, row[0], row[1]) + '\n')
                file.write(sep + '\n')



    def trainModel(self):
        '''Method for training the model'''

        info = []

        for i in range(self.max_iterations):

            output_training = self.calcOutput(self.training_set, self.weights)
            mse_training = self.calcMse(output_training, self.training_tags)

            output_validation = self.calcOutput(self.validation_set, self.weights)
            mse_validation = self.calcMse(output_validation, self.validation_tags)

            if len(info):
                #if abs(info[-1][-1] - mse_validation) < self.tolerance:
                if (info[-1][-1] < mse_validation) or ((info[-1][-1] >= mse_validation) and abs(info[-1][-1] - mse_validation) < self.tolerance):
                    return info

            self.adjustWeights(output_validation, self.validation_tags, self.validation_set)

            info.append((mse_training, mse_validation))

        return info


    def adjustWeights(self, output, tags, examples):
        '''Method for adjusting the weights'''

        weights_variation = self.learning_rate * (examples.T @ (tags - output))
        bias_variation = self.learning_rate * (tags - output).sum()

        self.weights[:-1] += weights_variation
        self.weights[-1] += bias_variation


    def testModel(self):
        '''Method for testing the model with the test-set'''
        
        output_test = self.calcOutput(self.test_set, self.weights)
        mse_test = self.calcMse(output_test, self.test_tags)

        return mse_test, output_test


    def plotMse(self, training_info):
        '''Introducing the vector obtained from training, it plots the evolution over cycles of MSE'''

        mse_hist = []
        mae_hist = []

        for i in training_info:
            mae_hist.append(i[0])
            mse_hist.append(i[1])

        plt.plot(mse_hist, color = 'blue')
        plt.plot(mae_hist, color = 'red')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squeared Error')
        plt.show()



if __name__ == '__main__':

    ada = Adaline('training_set.csv', 'validation_set.csv', 'test_set.csv')
    info = ada.trainModel()
    res, output_test = ada.testModel()
    ada.printError(info)
    print('\nITERACIONES OPTIMAS: ', len(info))
    print(f'\nMSE TEST: {res}\n\n')

    ada.plotMse(info)

