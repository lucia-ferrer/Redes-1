import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#df = pd.read_table('compactiv.dat', sep = ',', skiprows = 26, engine = 'python', header = None)
df = pd.read_csv('artificialData.csv')

df_normalized = df.copy()

def absolute_maximum_scale(series):
    return series / series.abs().max()

for col in df.columns:
    df_normalized[column] = (df_normalized[column] - df_normalized[column].min()) / (df_normalized[column].max() - df_normalized[column].min())

set_train, set_temp = train_test_split(np.array(df), test_size = 0.30, random_state = 42, shuffle = True)
set_validation, set_test = train_test_split(set_temp, test_size = 0.50, random_state = 42, shuffle = True)

np.savetxt("training_artificial.csv", set_train, delimiter = ",")
np.savetxt("validation_artificial.csv", set_validation, delimiter = ",")
np.savetxt("test_artificial.csv", set_test, delimiter = ",")

