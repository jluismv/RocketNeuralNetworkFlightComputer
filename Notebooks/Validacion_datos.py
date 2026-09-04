import pandas as pd
import numpy as np
from pathlib import Path

# Ajusta a tu ruta de archivos originales
PROCESSED_PATH = Path("/Users/darioromero/Documents/IAC 2026/REPO/RocketNeuralNetworkFlightComputer/Data/Processed Data/20Hz")
archivos = list(PROCESSED_PATH.glob("*.csv"))

print(f"Iniciando auditoría de {len(archivos)} vuelos...\n")

archivos_sospechosos = []

for archivo in archivos:
    df = pd.read_csv(archivo)
    df.columns = df.columns.str.strip()
    df = df.sort_values("time").reset_index(drop=True)
    
    alertas = []
    
    # 1. Búsqueda de Valores Nulos (NaN)
    if df[['time', 'baro_altitude', 'accl_x', 'accl_y', 'accl_z']].isnull().values.any():
        alertas.append("Contiene valores NaN o nulos.")
        
    # 2. Estabilidad de la frecuencia de muestreo (20Hz = 0.05s)
    dt = df['time'].diff().dropna()
    dt_max = dt.max()
    if dt_max > 0.1: # Tolerancia de 100ms
        alertas.append(f"Salto de tiempo detectado (Max dt: {dt_max:.3f}s).")
        
    # 3. Saturación del Acelerómetro (Clipping)
    # Ajusta el valor 150 según el límite en m/s^2 de tu IMU específica
    if df['accl_x'].max() > 150 or df['accl_y'].max() > 150 or df['accl_z'].max() > 150:
        alertas.append("Posible saturación del acelerómetro (>150 m/s²).")
        
    # 4. Lógica de Apogeo vs Tiempo
    idx_apogeo = df['baro_altitude'].idxmax()
    tiempo_apogeo = df.loc[idx_apogeo, 'time']
    
    # Reconstruir At para encontrar el despegue
    if 'At' not in df.columns:
        df['At'] = np.sqrt(df['accl_x']**2 + df['accl_y']**2 + df['accl_z']**2)
        
    despegues = df[df['At'] > 15].index
    if len(despegues) == 0:
        alertas.append("No se detectó un despegue claro (>15 m/s²).")
    else:
        tiempo_despegue = df.loc[despegues[0], 'time']
        if tiempo_apogeo <= tiempo_despegue + 1.0:
            alertas.append("El apogeo ocurrió antes o durante el quemado del motor.")
            
    # Si hay alertas, guardamos el reporte
    if alertas:
        archivos_sospechosos.append({
            "archivo": archivo.name,
            "alertas": alertas
        })

# Imprimir el reporte final
if not archivos_sospechosos:
    print("✅ Todos los archivos pasaron la auditoría física y estructural.")
else:
    print(f"⚠️ Se encontraron anomalías en {len(archivos_sospechosos)} archivos:\n")
    for reporte in archivos_sospechosos:
        print(f"Archivo: {reporte['archivo']}")
        for alerta in reporte['alertas']:
            print(f"  - {alerta}")
        print("-" * 40)