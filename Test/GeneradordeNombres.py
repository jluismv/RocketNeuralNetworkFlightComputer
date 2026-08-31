import os
import numpy as np
import random as rn
np.random.seed(5)
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
from keras.layers import Input, Dense, SimpleRNN
from keras.layers import Input, Dense, GRU, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
 
# 1. LECTURA DEL SET DE DATOS
# ===========================================================
nombres = open('nombres_dinosaurios.txt','r').read()
nombres = nombres.lower()

# Crear diccionario (listado de caracteres que no se repiten)
alfabeto = list(set(nombres))
tam_datos, tam_alfabeto = len(nombres), len(alfabeto)
print("En total hay %d caracteres, y el diccionario tiene un tamaño de %d caracteres." % (tam_datos, tam_alfabeto))

# Conversión de caracteres a índices y viceversa
car_a_ind = { car:ind for ind,car in enumerate(sorted(alfabeto))}
ind_a_car = { ind:car for ind,car in enumerate(sorted(alfabeto))}
#print(car_a_ind)
#print(ind_a_car)

# 2. MODELO (Corregido para Matrices 3D)
# ===========================================================

n_a = 128    # Numero de unidades en la capa oculta
entrada  = Input(shape=(None, tam_alfabeto))        # Entrada de la red neuronal 
HState = Input(shape=(n_a,))                        # Estado oculto inicial

# Usamos GRU con return_sequences=True
celda_recurrente = GRU(n_a, activation='tanh', return_sequences=True) #Usamos una red GRU con activación tanh y return_sequences=True para obtener la salida en cada paso de tiempo
capa_salida = TimeDistributed(Dense(tam_alfabeto, activation='softmax')) #Capa de salida con activación softmax

hs = celda_recurrente(entrada, initial_state=HState) #Celda recurrente con entrada y estado oculto inicial
salida = capa_salida(hs) #Capa de salida aplicada a la salida de la celda recurrente
modelo = Model([entrada, HState], salida) #Modelo final con entrada y estado oculto inicial como entradas y salida como salida  

# Compilación del modelo con optimizador Adam y función de pérdida categorical_crossentropy
#opt = SGD(learning_rate=0.001)
opt = Adam(learning_rate=0.001)
modelo.compile(optimizer=opt, loss='categorical_crossentropy')


# 3. EJEMPLOS DE ENTRENAMIENTO
# ===========================================================
# Leer y limpiar ejemplos
with open("nombres_dinosaurios.txt") as f:
    ejemplos = f.readlines()
ejemplos = [x.lower().strip() for x in ejemplos] #Los convertimos a minúsculas y eliminamos espacios en blanco al inicio y al final. 
#print("Ejemplos de entrenamiento (primeros 5):", ejemplos[:5])

X_lista = []
Y_lista = []

# Procesar todos los nombres del archivo
for ejemplo in ejemplos: 
    X_num = [None] + [car_a_ind[c] for c in ejemplo] #Convertimos cada carácter del nombre a su índice correspondiente en el diccionario de caracteres. 
    Y_num = X_num[1:] + [car_a_ind['\n']] #Desplazamos la secuencia hacia la derecha y agregamos un carácter de nueva línea al final para indicar el final del nombre.
    # Convertir a One-Hot encoding individual
    x_onehot = to_categorical(X_num[1:], tam_alfabeto)
    # Rellenar con una fila de ceros al inicio para simular el estado nulo [None]
    x_final = np.vstack([np.zeros((1, tam_alfabeto)), x_onehot])
    
    y_final = to_categorical(Y_num, tam_alfabeto)
    
    X_lista.append(x_final)
    Y_lista.append(y_final)

# Aplicar Padding para que todas las secuencias tengan la misma longitud fija
X_padded = pad_sequences(X_lista, padding='post', dtype='float32')
Y_padded = pad_sequences(Y_lista, padding='post', dtype='float32')
# Crear estados iniciales (ceros) para toda la matriz de entrenamiento
HState_padded = np.zeros((X_padded.shape[0], n_a))

# 4. ENTRENAMIENTO DIRECTO EN MATRIZ
# ===========================================================
BATCH_SIZE = 80 #Numero de ejemplos que se procesan antes de actualizar los pesos del modelo

#Bucle de entrenamiento del modelo con los datos procesados y rellenados.
print("Iniciando entrenamiento...")
historia = modelo.fit(
    x=[X_padded, HState_padded],
    y=Y_padded, 
    batch_size=BATCH_SIZE, 
    epochs=50,  # Numero de ciclos de iteracion para entrenar el modelo
    verbose=1
)

# 5. GENERACIÓN DE NOMBRES (INFERENCIA PASO A PASO)
# ===========================================================
def generar_lista_nombres_fiel(modelo, cantidad=30):
    # Extraer las capas individuales del modelo entrenado
    capa_rnn = modelo.layers[2]   # La capa GRU o SimpleRNN
    capa_dense = modelo.layers[3]  # La capa TimeDistributed(Dense)
    
    nombres_generados = []
    fin_linea = '\n'
    
    for _ in range(cantidad):
        # Carácter inicial: vector de ceros (None)
        x_paso = np.zeros((1, 1, tam_alfabeto))
        # Estado oculto inicial: vector de ceros
        a_paso = np.zeros((1, n_a))
        
        nombre = ""
        caracter = ""
        contador = 0
        
        while caracter != fin_linea and contador < 25:
            # 1. Ejecutar la celda RNN manualmente para UN SOLO paso de tiempo
            # Esto nos da el nuevo estado oculto actualizado
            a_paso = capa_rnn.cell(x_paso[:, 0, :], states=[a_paso])[0]
            a_paso = np.array(a_paso) 
            
            # 2. Pasar el estado oculto por la capa densa de salida
            prediccion_densa = capa_dense.layer(a_paso)
            probabilidades = prediccion_densa[0].numpy() #Arroja un vector de probabilidades para cada carácter en el alfabeto
            
            # 3. Muestrear el siguiente carácter
            idx = np.random.choice(list(range(tam_alfabeto)), p=probabilidades.ravel()) #Seleccion aleatoria del índice del siguiente carácter basado en las probabilidades predichas.
            caracter = ind_a_car[idx]
            
            if caracter != fin_linea:
                nombre += caracter
            
            # 4. Preparar la matriz de entrada para el siguiente paso de tiempo
            x_paso = to_categorical(idx, tam_alfabeto).reshape(1, 1, tam_alfabeto)
            
            contador += 1
            
        if len(nombre) > 2:
            nombres_generados.append(nombre)
        
    return nombres_generados

# ===========================================================
# EJECUCIÓN REVISADA
# ===========================================================
print("\n--- Generando nombres con inferencia de estado oculto ---")
lista_dinos = generar_lista_nombres_fiel(modelo, cantidad=15)

for dino in lista_dinos:
    print(f"{dino.capitalize()}")