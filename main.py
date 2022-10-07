from ADALINE import Adaline
from prueba import unnormalize
import os


if __name__ == '__main__':

    # Preparación  y aprendizaje de la red
    """if not os.exists('training_set.csv') or not os.exists('validation_set.csv') or not os.exists('test_set.csv'):
        normalize(df)"""
    
    ada = Adaline('training_set.csv', 'validation_set.csv', 'test_set.csv')
    info = ada.trainModel()

    # Calcular el error sobre el conjunto de test una vez finalizado el aprendizaje.
    res, output_test = ada.testModel()
    
    # Guardar en fichero las salidas de la red para todas las instancias de test.
    unnormalize(output_test)

    # Guardar en fichero el modelo, es decir, los pesos y el umbral (bias) de la red una vez finalizado el aprendizaje.
    # ada.saveModel()
    
    #Impresion en la terminal  de datos 
    ada.printError(info) 
    print('\nITERACIONES OPTIMAS: ', len(info))
    print(f'\nMSE TEST: {res}\n\n')

    ada.plotMse(info)

