import os
import numpy as np
import pandas as pd

from ADALINE import Adaline


def unnormalize(list_outputs, dat_origen = './../data/compactiv.csv', csv_test = './../data/test_set.csv'):
    """ Método paraa desnormalizar una lista de datos (list_outputs), teniendo en cuenta los 
    datos iniciales dat_origen: compactive.dat/artificialData.csv 
    datos exclusivos del test : csv_test """
        
    if '.csv' in dat_origen: df = pd.read_csv(dat_origen, header = None)
    elif '.dat' in dat_origen: df = pd.read_table(dat_origen, sep = ',', skiprows = 26, engine = 'python', header = None)

    df_test = pd.read_csv(csv_test, header = None)

    #conseguir los maximos y minimos usados en la normalización
    max_out, min_out = df.iloc[:, -1].max(), df.iloc[:, -1].min()

    out = {"obtenido" : [], "deseado" : []}
    
    #desnormalizar segun 
    for y in list_outputs[:-1]:
        val = y*(max_out - min_out) + min_out
        out["obtenido"].append(val[0])
    
    #desnormalizar valores del test
    array_test = np.array(df_test.iloc[:, -1])
    for y in array_test[:-1]:
        val = y*(max_out - min_out) + min_out
        out["deseado"].append(val)

    outputs = pd.DataFrame(out, columns = ['obtenido', 'deseado'])

    #borrar primero archivo si existe
    if (os.path.exists('./../output/output_comparison.csv')):
        os.remove('./../output/output_comparison.csv')

    outputs.to_csv('./../output/output_comparison.csv', index = None)
    
    saveOutput(out)


def saveOutput(output):
    '''Method for storing the progression of MSE and MAE over cycles'''

    sep = ' -----------------------------------------------------'
    head = '|{:^25}|{:^25}|'.format('OBTENIDO', 'DESEADO')

    if (os.path.exists('./../output/output_comparison.txt')):
        os.remove('./../output/output_comparison.txt')

    with open('./../output/output_comparison.txt', 'a') as file:
        file.write(sep + '\n')
        file.write(head + '\n')
        file.write(sep + '\n')

        for index in range(len(output['deseado'])):
            file.write('|{:^25}|{:^25}|'.format(output['obtenido'][index], output['deseado'][index]) + '\n')
            file.write(sep + '\n')



if __name__ == '__main__':

    ada = Adaline('./../data/training_set.csv', './../data/validation_set.csv', './../data/test_set.csv')
    info = ada.trainModel()
    res, output_test = ada.testModel()
    ada.saveError(info)
    ada.saveModel()

    unnormalize(output_test)

    ada.printError(info)
    print('\nITERACIONES OPTIMAS: ', len(info))
    print(f'\nMSE TEST: {res}\n\n')

    ada.plotMse(info)
    
    # Descomentar esta sección de codigo y comentar la anterior si
    # se quiere cargar un modelo ya entrenado. Introducir en loadModel
    # la ruta del modelo que se quiere cargar.

    '''
    ada = Adaline('./../data/training_set.csv', './../data/validation_set.csv', './../data/test_set.csv')
    info = ada.loadModel('./../output/adaline_trained_model_1159.json')
    res, output_test = ada.testModel()
    ada.saveError(info)
    ada.saveModel()

    unnormalize(output_test)

    ada.printError(info)
    print('\nITERACIONES OPTIMAS: ', len(info))
    print(f'\nMSE TEST: {res}\n\n')

    ada.plotMse(info)'''
