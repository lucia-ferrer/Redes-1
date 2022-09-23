import pandas as pd

'''
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
'''

df = pd.read_excel("datos_EXCEL_NOR.xlsx")

set_train, set_temp = train_test_split(np.array(df), test_size = 0.30, random_state = 42, shuffle = True)
set_validation, set_test = train_test_split(set_temp, test_size = 0.50, random_state = 42, shuffle = True)

np.savetxt("training_artificial.csv", set_train, delimiter = ",")
np.savetxt("validation_artificial.csv", set_validation, delimiter = ",")
np.savetxt("test_artificial.csv", set_test, delimiter = ",")