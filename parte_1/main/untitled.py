import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


training_info = pd.read_csv('./../output/output_comparison.csv', sep = ',')

obtenido = np.array(training_info.iloc[:, :-1])
deseado = np.array(training_info.iloc[:, -1:])


mse_hist = []
mae_hist = []


for i in range(len(obtenido)):
    mae_hist.append(obtenido[i])
    mse_hist.append(deseado[i])

    if i > 200:
        break

plt.plot(mse_hist, label = 'Deseado')
plt.plot(mae_hist, label = 'Obtenido')
plt.xlabel('Instancias')
plt.ylabel('Valor')
plt.legend()
plt.show()