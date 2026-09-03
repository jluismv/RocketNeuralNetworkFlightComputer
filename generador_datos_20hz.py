import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.signal import resample_poly

# --- CONFIGURACIÓN ---
NUM_VUELOS_SINTETICOS = 10
FS_TARGET = 20

# Ajusta estas rutas según tu entorno de trabajo
DIRECTORIO_ENTRADA = Path("/Users/darioromero/Documents/IAC 2026/REPO/RocketNeuralNetworkFlightComputer/Data/Raw Data") 
DIRECTORIO_SALIDA = Path("/Users/darioromero/Documents/IAC 2026/REPO/RocketNeuralNetworkFlightComputer/Data/Processed Data/Augmented")

def isa_atmosphere(h):
    """Cálculo vectorizado de la Atmósfera Estándar Internacional (Troposfera)."""
    T0 = 288.15       # K
    P0 = 101325.0     # Pa
    L = 0.0065        # K/m
    R = 287.05        # J/(kg*K)
    g = 9.80665       # m/s^2
    gamma = 1.4

    # Clipping para evitar altitudes negativas por ruido en barómetro
    h = np.clip(np.asarray(h, dtype=float), 0, 11000)

    T = T0 - L * h
    P = P0 * (T / T0) ** (g / (R * L))
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    
    return T, P, rho, a

def normalizar_a_20hz(df):
    """
    Verifica la frecuencia de muestreo y remuestrea a 20 Hz (con filtro anti-aliasing)
    si el vuelo original viene a ~50 Hz. Retorna intacto si ya es ~20 Hz.
    """
    dt_mean = df["time"].diff().mean()
    if pd.isna(dt_mean) or dt_mean == 0:
        return df
        
    freq_hz = 1 / dt_mean
    
    if 48 <= freq_hz <= 52:
        n_original = len(df)
        n_target = int(np.ceil(n_original * FS_TARGET / 50))
        
        df_20 = pd.DataFrame()
        df_20["time"] = np.arange(n_target) / FS_TARGET
        
        columnas_senal = [col for col in df.columns if col not in ["time", "flight_id"]]
        for col in columnas_senal:
            resampled = resample_poly(df[col].to_numpy(), up=2, down=5)
            df_20[col] = resampled[:n_target]
            
        if "flight_id" in df.columns:
            df_20["flight_id"] = df["flight_id"].iloc[0]
            
        return df_20
    else:
        return df

def enriquecer_vuelo(df):
    """
    Calcula la aceleración total (At) y las variables ISA. 
    Se ejecuta DESPUÉS de aplicar el ruido para mantener coherencia física.
    """
    df_nuevo = df.copy()
    
    if all(col in df_nuevo.columns for col in ['accl_x', 'accl_y', 'accl_z']):
        df_nuevo['At'] = np.sqrt(df_nuevo['accl_x']**2 + df_nuevo['accl_y']**2 + df_nuevo['accl_z']**2)
        
    if 'baro_altitude' in df_nuevo.columns:
        T, P, rho, a = isa_atmosphere(df_nuevo['baro_altitude'].values)
        df_nuevo['T'] = T
        df_nuevo['P'] = P
        df_nuevo['rho'] = rho
        df_nuevo['a'] = a
        
    return df_nuevo

def generar_vuelo_sintetico(df_base):
    """Aplica Data Augmentation sobre las variables estandarizadas crudas."""
    df_sintetico = df_base.copy()
    
    # 1. Escalado aleatorio (Scaling)
    factor_escala = np.random.uniform(0.97, 1.03)
    columnas_escala = ['accl_x', 'accl_y', 'accl_z', 'baro_altitude']
    for col in columnas_escala:
        if col in df_sintetico.columns:
            df_sintetico[col] = df_sintetico[col] * factor_escala

    # 2. Inyección de Ruido Gaussiano (Jittering)
    # 0.5 m/s^2 de vibración aleatoria y 0.2 m de error barométrico
    ruido_aceleracion = np.random.normal(0, 0.5, len(df_sintetico))
    ruido_altitud = np.random.normal(0, 0.2, len(df_sintetico)) 
    
    for col in ['accl_x', 'accl_y', 'accl_z']:
        if col in df_sintetico.columns:
            df_sintetico[col] += ruido_aceleracion
            
    if 'baro_altitude' in df_sintetico.columns:
        df_sintetico['baro_altitude'] += ruido_altitud
    
    # 3. Recalcular las variables dependientes con las nuevas medidas perturbadas
    return enriquecer_vuelo(df_sintetico)

# --- EJECUCIÓN PRINCIPAL ---
DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)

# Busca todos los CSV que coincidan con el esquema estandarizado
archivos = list(DIRECTORIO_ENTRADA.glob("*_standardized*.csv"))

# Fallback por si la carpeta de entrada no existe durante las pruebas locales
if not archivos and Path("1_standardized_20Hz.csv").exists():
    archivos = [Path("1_standardized_20Hz.csv")]

print(f"Archivos localizados: {len(archivos)}. Iniciando procesamiento...")

for archivo in archivos:
    print(f"\nProcesando lote para: {archivo.name}")
    df_original = pd.read_csv(archivo)
    
    # Limpieza básica
    df_original = df_original.dropna(subset=['time', 'baro_altitude'])
    df_original = df_original[df_original['time'] >= 0]
    
    # Normalización dinámica a 20Hz
    df_20hz = normalizar_a_20hz(df_original)
    
    # Guardado del archivo base enriquecido
    df_base_enriquecido = enriquecer_vuelo(df_20hz)
    nombre_base = f"{archivo.stem}_base_enriquecido.csv"
    df_base_enriquecido.to_csv(DIRECTORIO_SALIDA / nombre_base, index=False)
    
    # Generación y exportación de sintéticos
    for i in range(1, NUM_VUELOS_SINTETICOS + 1):
        df_sintetico = generar_vuelo_sintetico(df_20hz)
        nombre_sintetico = f"{archivo.stem}_sintetico_{i:03d}.csv"
        df_sintetico.to_csv(DIRECTORIO_SALIDA / nombre_sintetico, index=False)
        
print(f"\n¡Procesamiento masivo finalizado!")