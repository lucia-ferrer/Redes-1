import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

#df = pd.read_table('compactiv.dat', sep = ',', skiprows = 26, engine = 'python', header = None)
df = pd.read_csv('artificialData.csv')

def absolute_maximum_scale(series):
    return series / series.abs().max()

def normalize(df_original):
    df_normalized = df_original.copy()
    for col in df_original.columns:
        df_normalized[column] = (df_normalized[column] - df_normalized[column].min()) / (df_normalized[column].max() - df_normalized[column].min())

    set_train, set_temp = train_test_split(np.array(df), test_size = 0.30, random_state = 42, shuffle = True)
    set_validation, set_test = train_test_split(set_temp, test_size = 0.50, random_state = 42, shuffle = True)

    np.savetxt("training_artificial.csv", set_train, delimiter = ",")
    np.savetxt("validation_artificial.csv", set_validation, delimiter = ",")
    np.savetxt("test_artificial.csv", set_test, delimiter = ",")

def unnormalize(list_outputs):
    max_out, min_out = df.iloc[:, -1].max(), df.iloc[:, -1].min()

    out = {"obtenido" : []}

    for y in list_outputs:
        val = y*(max_out - min_out) + min_out
        out["obtenido"].append(val)
    
    outputs = pd.DataFrame(out, columns=['obtenido'])
    #unir ambas columnas bajo los headers especificados
    outputs = pd.concat([outputs, df.iloc[:, -1]], axis=1)
    outputs.columns = ["obtenido", "deseado"]

    #borrar primero archivo si existe
    if (os.path.exists('outputs_test.csv')):
        os.remove('outputs_test.csv')
    outputs.to_csv('outputs_test.csv', index=False)

original_normalizado = (df.iloc[:, -1] - df.iloc[:, -1].min()) / (df.iloc[:, -1].max() - df.iloc[:, -1].min())
unnormalize(original_normalizado)