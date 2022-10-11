import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def absolute_maximum_scale(series):
    return series / series.abs().max()

def normalize(dat_origen = 'compactiv.dat'):

    if '.csv' in dat_origen: df = pd.read_csv(dat_origen)
    elif '.dat' in dat_origen: df = pd.read_table(dat_origen, sep = ',', skiprows = 26, engine = 'python', header = None)

    df_normalized = df.copy()

    #NORMALIZE
    for col in df.columns:
        df_normalized[column] = (df_normalized[column] - df_normalized[column].min()) / (df_normalized[column].max() - df_normalized[column].min())

    #SPLIT
    set_train, set_temp = train_test_split(np.array(df_normalized), test_size = 0.30, random_state = 42, shuffle = True)
    set_validation, set_test = train_test_split(set_temp, test_size = 0.50, random_state = 42, shuffle = True)

    #SAVE
    np.savetxt("training_artificial_2.csv", set_train, delimiter = ",")
    np.savetxt("validation_artificial_2.csv", set_validation, delimiter = ",")
    np.savetxt("test_artificial_2.csv", set_test, delimiter = ",")


def unnormalize(list_outputs, dat_origen = 'compactiv.dat', csv_test = 'test_set.csv'):
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
    for y in list_outputs[:1]:
        val = y*(max_out - min_out) + min_out
        out["obtenido"].append(val)
    
    #desnormalizar valores del test
    array_test = np.array(df_test.iloc[:, -1])
    for y in array_test[:1]:
        val = y*(max_out - min_out) + min_out
        out["deseado"].append(val)

    outputs = pd.DataFrame(out, columns=['obtenido', 'deseado'])

    #borrar primero archivo si existe
    if (os.path.exists('outputs_test.csv')):
        os.remove('outputs_test.csv')
    
    outputs.to_csv('outputs_test.csv', index = None)
