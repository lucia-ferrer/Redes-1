import pandas as pd

df = pd.read_excel("datos_EXCEL_NOR.xlsx")

p_train = 0.80 # Porcentaje de train.

train = df[:int((len(df))*p_train)] 
test = df[int((len(df))*p_train):]


print(type(test))