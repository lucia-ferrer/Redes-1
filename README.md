# README

## PARTE 1

La carpeta parte 1 incluye todo el código fuente y de la implementación del ADALINE. El árbol de directorios es el siguiente:

.
├── data
│   ├── compactiv.dat
│   ├── test_set.csv
│   ├── training_set.csv
│   └── validation_set.csv
├── main
│   ├── ADALINE2.py
│   ├── ADALINE.py
│   ├── main.py
│   ├── plot_output.py
|   └── preprocesamiento.py
└── output
    └── mejor_output
        ├── adaline_trained_model.json
        ├── error-progression.txt
        ├── output_comparison.csv
        └── output_comparison.txt

En la carpeta data se encuentran todos los datos que emplea Adaline, si se mueven de ese directorio el programa no funcionará.

En el directorio main se encuentran las dos implementaciones de Adaline, ADALINE.py y ADALINE2.py, que se corresponden con
el modelo 1 y modelo 2 explicados en la memoria de la práctica. El 1 corresponde al mejor modelo obtenido, que actualiza todos
los pesos tras presentar todos los ejemplos de entrenamiento, mientras que el 2 los actualiza uno a uno cada vez que se introduce un
ejemplo.

main.py es el fichero que se ejecuta para entrenar el modelo ADALINE.py, todas las salidas de el proceso se guardan en output: la 
progresión del error, la predicción obtenida frente a la deseada y el modelo en formato json. Se puede modificar el main para cargar
un modelo ya entrenarlo y solo probarlo. El main actual entrena y prueba. ADALINE2.py tiene su propio main en el propio fichero
por si se quisiera probar (python3 ADALINE2.py).

El resto son scripts empleados para el preprocesamiento y plotting de datos.

El subdirectorio mejor_output dentro de output contiene la salida del mejor modelo. Predicción frente a deseado en csv y txt formateado,
el json con los valores de los pesos y bias y la progresión del error en la prueba que muestra el documento. Si se quisiera cargar el json
del modelo habría que moverlo al directorio padre output.

Cualquier ejecución del main sobreescribirá las salidas del modelo presentes en /output, por lo que si se quieren hacer pruebas conservando
las salidas, es necesario moverlas a otro directorio (../output/mejor_output/ por ejemplo) o cambiarles los nombres a los ficheros.