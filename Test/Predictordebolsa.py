import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Función para graficar los resultados de la predicción vs el valor real de las acciones.
def graficar_predicciones(fechas, real, prediccion):
    plt.figure(figsize=(10, 6)) 
    
    plt.plot(fechas, real, color='red', label='Valor real de la acción')
    plt.plot(fechas, prediccion, color='blue', label='Predicción de la acción')
    
    minimo = min(np.min(real), np.min(prediccion)) # Calcula el valor mínimo entre los datos reales y las predicciones
    maximo = max(np.max(real), np.max(prediccion)) # Homologo para el valor máximo
    plt.ylim(minimo * 0.95, maximo * 1.05)
    
    plt.title('Predicción del valor de las acciones')
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la acción')
    plt.legend()
    
    plt.savefig('grafica_acciones.png', dpi=300, bbox_inches='tight')
    print("¡Gráfico guardado exitosamente como 'grafica_acciones.png'!")


# Lectura de los datos y preparación del dataset
dataset = pd.read_csv('datos_aapl.csv', index_col='Fecha', parse_dates=['Fecha'], dayfirst=True)
dataset = dataset.sort_index()
dataset.head()


# Sets de entrenamiento y validación 
# La LSTM se entrenará con datos de 2024 hacia atrás. La validación se hará con datos de 2025 en adelante.
# En ambos casos sólo se usará el valor más alto de la acción para cada día


#set_entrenamiento = dataset[:'2024'].iloc[:,2:3]
set_entrenamiento = dataset.loc[:'2024', ['Máximo']]
#set_validacion = dataset['2025':].iloc[:,2:3]
set_validacion = dataset['2025':].loc[:, ['Máximo']]


#set_entrenamiento['Máximo'].plot(legend=True)
#set_validacion['Máximo'].plot(legend=True)
#plt.legend(['Entrenamiento (2006-2024)', 'Validación (2025)'])
#plt.show()

# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción).
time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Red LSTM
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 128

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=100,batch_size=64)


# Validación (predicción del valor de las acciones)
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Como usamos un time_step de 60, los primeros 60 días no tienen predicción.
# Extraemos las fechas correspondientes recortando los primeros 60 días:
fechas_prediccion = set_validacion.index[time_step:]

# Alineamos los valores reales quitando también los primeros 60 días:
valores_reales = set_validacion.values[time_step:]

# Graficamos pasándole las fechas recuperadas
graficar_predicciones(fechas_prediccion, valores_reales, prediccion)