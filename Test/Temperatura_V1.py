import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

# 1. Definir los datos
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# 2. Crear el modelo con la sintaxis actualizada (Keras 3)
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),      # Capa de entrada explícita
    tf.keras.layers.Dense(units=1)   # Capa densa
])

# 3. Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), 
    loss='mean_squared_error'
)

# 4. Entrenar el modelo
print("Entrenando el modelo... (esto puede tardar unos minutos la primera vez)")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("¡Modelo entrenado!")

# 5. Graficar la pérdida
plt.xlabel('Epochs')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'])
plt.show()

# 6. Hacer una predicción
print("Predicción de temperatura en Fahrenheit:")
dato = np.array([[125.0]]) # Se pasa como arreglo 2D
resultado = modelo.predict(dato)
print(f"100 grados Celsius son aproximadamente: {resultado[0][0]:.2f} Fahrenheit")