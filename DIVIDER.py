import pandas as pd

df = pd.read_excel("datos_EXCEL_NOR.xlsx")

p_train = 0.80 # Porcentaje de train.

train = df[:int(len(df) * 0.70)] 
validation = df[int((len(df))*0.70):int(len(df) * 0.85)]
test = df[int(len(df) * 0.85):]

trainWriter = pd.ExcelWriter('train.xlsx', engine = 'xlsxwriter')
train.to_excel(trainWriter, sheet_name = 'welcome', index = False)
trainWriter.save()

validationWriter = pd.ExcelWriter('validation.xlsx', engine = 'xlsxwriter')
validation.to_excel(validationWriter, sheet_name = 'welcome', index = False)
validationWriter.save()

testWriter = pd.ExcelWriter('test.xlsx', engine = 'xlsxwriter')
test.to_excel(testWriter, sheet_name = 'welcome', index = False)
testWriter.save()

