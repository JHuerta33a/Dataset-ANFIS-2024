!pip install pykrige
!pip install geopandas
!pip install contextily

import pandas as pd
import numpy as np
import os
import cProfile
import pstats
import io
import time
import random
import warnings
import seaborn as sns
import geopandas as gpd
import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.stats import chi2, zscore
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from shapely.geometry import Point
from statsmodels.tsa.statespace.sarimax import SARIMAX

"""# ***Ubicaciones de Estaciones***"""

def leer_hojas_csv(ruta_archivo):
    # Leer cada hoja en un DataFrame separado
    df_CAP = pd.read_excel(ruta_archivo, sheet_name='CAP')
    df_EPG = pd.read_excel(ruta_archivo, sheet_name='EPG')
    df_FEO = pd.read_excel(ruta_archivo, sheet_name='FEO')

    return df_CAP, df_EPG, df_FEO

ruta_archivo = 'BD_SMCAQ_15ABR2025-31AGO2025.xlsx'
df_CAP, df_EPG, df_FEO = leer_hojas_csv(ruta_archivo)

df_ubicaciones = pd.DataFrame(columns=['Estacion', 'Latitud', 'Longitud'])

df_ubicaciones = pd.concat([df_ubicaciones, pd.DataFrame([{'Estacion': "EPG", 'Latitud': 20.627625, 'Longitud': -100.40876666666668}])], ignore_index=True)
df_ubicaciones = pd.concat([df_ubicaciones, pd.DataFrame([{'Estacion': "FEO",'Latitud': 20.635125, 'Longitud': -100.45924722222223}])], ignore_index=True)

# Imprimir el DataFrame
df_ubicaciones

"""# ***Conversión de ppb a ppm***"""

df_CAP = df_CAP.rename(columns={'O3': 'O3 ppb', 'SO2': 'SO2 ppb', 'NO2': 'NO2 ppb', 'CO': 'CO ppb'})
df_EPG = df_EPG.rename(columns={'O3': 'O3 ppb', 'SO2': 'SO2 ppb', 'NO2': 'NO2 ppb', 'CO': 'CO ppb'})
df_FEO = df_FEO.rename(columns={'O3': 'O3 ppb', 'SO2': 'SO2 ppb', 'NO2': 'NO2 ppb', 'CO': 'CO ppb'})

# Convert ppb to ppm (1 ppb = 10^-3 ppm)
# Ensure columns are numeric before multiplication
df_CAP['O3'] = (pd.to_numeric(df_CAP['O3 ppb'], errors='coerce') * 0.001).astype(float)
df_CAP['SO2'] = (pd.to_numeric(df_CAP['SO2 ppb'], errors='coerce') * 0.001).astype(float)
df_CAP['NO2'] = (pd.to_numeric(df_CAP['NO2 ppb'], errors='coerce') * 0.001).astype(float)
df_CAP['CO'] = (pd.to_numeric(df_CAP['CO ppb'], errors='coerce') * 0.001).astype(float)

# Eliminar las columnas originales en ppb
df_CAP = df_CAP.drop(['O3 ppb', 'SO2 ppb', 'NO2 ppb', 'CO ppb'], axis=1)

# Convert ppb to ppm (1 ppb = 10^-3 ppm)
# Ensure columns are numeric before multiplication
df_EPG['O3'] = (pd.to_numeric(df_EPG['O3 ppb'], errors='coerce') * 0.001).astype(float)
df_EPG['SO2'] = (pd.to_numeric(df_EPG['SO2 ppb'], errors='coerce') * 0.001).astype(float)
df_EPG['NO2'] = (pd.to_numeric(df_EPG['NO2 ppb'], errors='coerce') * 0.001).astype(float)
df_EPG['CO'] = (pd.to_numeric(df_EPG['CO ppb'], errors='coerce') * 0.001).astype(float)


# Eliminar las columnas originales en ppb
df_EPG = df_EPG.drop(['O3 ppb', 'SO2 ppb', 'NO2 ppb', 'CO ppb'], axis=1)

# Convert ppb to ppm (1 ppb = 10^-3 ppm)
# Ensure columns are numeric before multiplication
df_FEO['O3'] = (pd.to_numeric(df_FEO['O3 ppb'], errors='coerce') * 0.001).astype(float)
df_FEO['SO2'] = (pd.to_numeric(df_FEO['SO2 ppb'], errors='coerce') * 0.001).astype(float)
df_FEO['NO2'] = (pd.to_numeric(df_FEO['NO2 ppb'], errors='coerce') * 0.001).astype(float)
df_FEO['CO'] = (pd.to_numeric(df_FEO['CO ppb'], errors='coerce') * 0.001).astype(float)

# Eliminar las columnas originales en ppb
df_FEO = df_FEO.drop(['O3 ppb', 'SO2 ppb', 'NO2 ppb', 'CO ppb'], axis=1)

"""# ***Estación CAP***

## **Procesamiento de Datos**
"""

df_CAP.head()

df_CAP.info()

# Convert object type columns to float64 in df_CAP
for col in df_CAP.columns:
    if df_CAP[col].dtype == 'object':
        df_CAP[col] = pd.to_numeric(df_CAP[col], errors='coerce')

# Display data types to confirm conversion
print(df_CAP.dtypes)

df_CAP.dtypes

df_CAP.isnull().sum()

# Convert 'fecha' column to datetime objects
df_CAP['Fecha'] = pd.to_datetime(df_CAP['Fecha'])

# Set 'fecha' as the index for time series interpolation
df_CAP = df_CAP.set_index('Fecha')

# Perform time series interpolation for numerical columns
numerical_cols = df_CAP.select_dtypes(include=np.number).columns
df_CAP[numerical_cols] = df_CAP[numerical_cols].interpolate(method='time')

# Reset index if needed
df_CAP = df_CAP.reset_index()

# Verify that null values have been filled
df_CAP.isnull().sum()

df_CAP.describe()

df_CAP.head()

df_CAP.isnull().sum()

df_CAP.head()

df_CAP.describe()

"""## **Tendencia de los Datos**"""

plt.figure(figsize=(15, 10))

# Select only numerical columns for plotting histograms
numerical_cols = df_CAP.select_dtypes(include=np.number).columns

for i, col in enumerate(numerical_cols):
    plt.subplot(5, 2, i + 1)
    sns.histplot(df_CAP[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for i, col in enumerate(df_CAP.columns[1:]):
  plt.subplot(5, 2, i+1)
  sns.boxplot(x=df_CAP[col])
  plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()

"""## **Outliers**"""

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
df_co = df_CAP['CO'].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_co.quantile(0.25)
Q3 = df_co.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_cap_Clean_CO = df_co[~((df_co < (Q1 - 1.5 * IQR)) | (df_co > (Q3 + 1.5 * IQR)))]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean = df_cap_Clean_CO.mean()
cov_clean = df_cap_Clean_CO.var()

# Regularización de la matriz de covarianza
cov_clean = np.array([[cov_clean]])
cov_clean += np.eye(cov_clean.shape[0]) * 1e-6
inv_cov_clean = np.linalg.inv(cov_clean)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean = df_cap_Clean_CO - mean_clean
mahalanobis_dist_clean = (diff_clean ** 2) * inv_cov_clean[0][0]

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=1)
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_cap_Clean_CO = df_cap_Clean_CO[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_cap_Clean_CO))
df_cap_Clean_CO = df_cap_Clean_CO[(z_scores < 3)]

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
df_o3 = df_CAP['O3'].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_o3.quantile(0.25)
Q3 = df_o3.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_cap_Clean_O3 = df_o3[~((df_o3 < (Q1 - 1.5 * IQR)) | (df_o3 > (Q3 + 1.5 * IQR)))]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean_o3 = df_cap_Clean_O3.mean()
cov_clean_o3 = df_cap_Clean_O3.var()

# Regularización de la matriz de covarianza
cov_clean_o3 = np.array([[cov_clean_o3]])
cov_clean_o3 += np.eye(cov_clean_o3.shape[0]) * 1e-6
inv_cov_clean_o3 = np.linalg.inv(cov_clean_o3)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean_o3 = df_cap_Clean_O3 - mean_clean_o3
mahalanobis_dist_clean = (diff_clean_o3 ** 2) * inv_cov_clean_o3[0][0]

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=1)
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_cap_Clean_O3 = df_cap_Clean_O3[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_cap_Clean_O3))
df_cap_Clean_O3 = df_cap_Clean_O3[(z_scores < 3)]

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
variables = ['NO2', 'SO2']
df_gases = df_CAP[variables].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_gases.quantile(0.25)
Q3 = df_gases.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_cap_h_Clean = df_gases[~((df_gases < (Q1 - 1.5 * IQR)) | (df_gases > (Q3 + 1.5 * IQR))).any(axis=1)]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean = df_cap_h_Clean.mean().values
cov_clean = df_cap_h_Clean.cov().values

# Regularización de la matriz de covarianza
cov_clean += np.eye(len(variables)) * 1e-6
inv_cov_clean = np.linalg.inv(cov_clean)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean = df_cap_h_Clean - mean_clean
mahalanobis_dist_clean = np.sum(np.dot(diff_clean, inv_cov_clean) * diff_clean, axis=1)

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=len(variables))
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_cap_h_Clean = df_cap_h_Clean[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_cap_h_Clean))
df_cap_h_Clean = df_cap_h_Clean[(z_scores < 3).all(axis=1)]

df_pm = df_CAP[['PM10']].dropna()

# Configurar y entrenar el modelo
model = IsolationForest(
    contamination=0.05,          # Proporción esperada de outliers (ajustable)
    random_state=42,             # Semilla para reproducibilidad
    n_estimators=100             # Número de árboles (valor por defecto)
)

model.fit(df_pm)

# Predecir outliers (-1 = outlier, 1 = inlier)
predictions = model.predict(df_pm)

# Filtrar inliers y obtener DataFrame limpio
inlier_indices = df_pm.index[predictions == 1]
df_cap_Clean_PM = df_CAP.loc[inlier_indices]

variables_to_clean = ['TM', 'HR', 'VV', 'DV']

# Crear una copia del DataFrame original para la limpieza
df_CAP_cleaned_IQR = df_CAP.copy()

# Iterar sobre cada variable para eliminar outliers usando IQR
for col in variables_to_clean:
    if col in df_CAP_cleaned_IQR.columns:
        Q1 = df_CAP_cleaned_IQR[col].quantile(0.25)
        Q3 = df_CAP_cleaned_IQR[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definir los límites inferior y superior para la detección de outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Eliminar las filas donde el valor de la columna está fuera de los límites
        initial_rows = len(df_CAP_cleaned_IQR)
        df_CAP_cleaned_IQR = df_CAP_cleaned_IQR[(df_CAP_cleaned_IQR[col] >= lower_bound) & (df_CAP_cleaned_IQR[col] <= upper_bound)]
        rows_removed = initial_rows - len(df_CAP_cleaned_IQR)
        print(f"Variable: {col}, Filas eliminadas por IQR: {rows_removed}")
    else:
        print(f"La variable '{col}' no se encuentra en el DataFrame.")

# Crear el DataFrame dt_pm10
dt_pm10_cap = df_cap_Clean_PM[['Fecha', 'PM10']].copy()

# Crear el DataFrame dt_co
dt_co_cap = df_cap_Clean_PM[['Fecha', 'CO']].copy()

# Crear el DataFrame dt_o3
dt_o3_cap = df_cap_Clean_PM[['Fecha', 'O3']].copy()

df_CAP['Fecha'] = pd.to_datetime(df_CAP['Fecha'])

# Extract the hour from the 'Fecha' column
df_CAP['Hora'] = df_CAP['Fecha'].dt.floor('T')

# Merge the 'Hora' column into df_CAP_Clean
df_cap_h_Clean = df_cap_h_Clean.merge(df_CAP[['Hora']], left_index=True, right_index=True, how='left')

"""## **Antés y Después de Eliminar Outliers**"""

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['TM'], kde=True)
plt.title('Distribución de la Temperatura - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_CAP_cleaned_IQR['TM'], kde=True)
plt.title('Distribución de la Temperatura - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_CAP_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['HR'], kde=True)
plt.title('Distribución de la Humedad - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_CAP_cleaned_IQR['HR'], kde=True)
plt.title('Distribución de la Humedad - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_CAP_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['VV'], kde=True)
plt.title('Distribución de la Velocidad del Viento - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_CAP_cleaned_IQR['VV'], kde=True)
plt.title('Distribución de la Velocidad del Viento - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_CAP_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['DV'], kde=True)
plt.title('Distribución de la Dirección del Viento - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_CAP_cleaned_IQR['DV'], kde=True)
plt.title('Distribución de la Dirección del Viento - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_CAP_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['CO'], kde=True)
plt.title('Distribución del CO (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_co_cap['CO'], kde=True)
plt.title('Distribución del CO (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_co_cap.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['O3'], kde=True)
plt.title('Distribución del O3 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_o3_cap['O3'], kde=True)
plt.title('Distribución del O3 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_o3_cap.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['SO2'], kde=True)
plt.title('Distribución del SO2 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_cap_h_Clean['SO2'], kde=True)
plt.title('Distribución del SO2 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_cap_h_Clean.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['NO2'], kde=True)
plt.title('Distribución del NO2 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_cap_h_Clean['NO2'], kde=True)
plt.title('Distribución del NO2 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_cap_h_Clean.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_CAP['PM10'], kde=True)
plt.title('Distribución del PM 10 (ug/m3) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_pm10_cap['PM10'], kde=True)
plt.title('Distribución del PM 10 (ug/m3) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_CAP.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_pm10_cap.shape[0]}")

def get_max(df, date_column, value_columns):
        # Aseguramos que la columna de fecha sea datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Creamos una columna con solo la hora redondeada
    df['Hora'] = df[date_column].dt.floor('H')

    # Agrupamos por hora y calculamos el máximo para cada columna
    hourly_max = df.groupby('Hora')[value_columns].max().reset_index()

    return hourly_max

aqi_so2_cap = get_max(df_cap_h_Clean, 'Hora', 'SO2')

aqi_no2_cap = get_max(df_cap_h_Clean, 'Hora', 'NO2')

dt_co_cap.head()

dt_o3_cap.head()

aqi_no2_cap.head()

aqi_so2_cap.head()

dt_pm10_cap.head()

"""## **ICA**"""

def calculate_aqi(concentration, breakpoints):
    """Calcula el subíndice AQI para un contaminante dado."""
    for i in range(len(breakpoints) - 1):
        C_low, C_high = breakpoints[i][0], breakpoints[i + 1][0]
        I_low, I_high = breakpoints[i][1], breakpoints[i + 1][1]

        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low

    return None  # Fuera de rango

# Tabla de la EPA con los valores de referencia para cada contaminante
# Formato: [(C_low, I_low), (C_high, I_high)]
AQI_BREAKPOINTS = {
    "O3": [(0.000, 0), (0.055, 51), (0.071, 101), (0.086, 151), (0.106, 201), (0.201, 301)],
    #"PM2.5": [(0, 0), (9.1, 51), (35.5, 101), (55.5, 151), (125.5, 201), (225.5, 301)],
    "PM10": [(0, 0), (55, 51), (155, 101), (255, 151), (255, 201), (425, 301)],
    "CO": [(0.0, 0), (4.5, 51), (9.5, 101), (12.5, 151), (15.5, 210), (30.5, 301)],
    "SO2": [(0.000, 0), (0.036, 51), (0.076, 101), (0.186, 151), (0.305, 201), (0.605, 301)],
    "NO2": [(0.000, 0), (0.054, 51), (0.101, 101), (0.360, 151), (0.650, 201), (1.250, 301)]
}

def preprocess_data(df):
    """Promedia los datos según la recomendación de la EPA."""
    df = df.set_index('Hora')
    df.index = pd.to_datetime(df.index)

    df = df.resample('H').mean()  # Asegurar frecuencia horaria mínima
    df['O3'] = df['O3'].rolling('8H').mean()
    #df['PM2.5'] = df['PM2.5'].rolling('24H').mean()
    df['PM10'] = df['PM10'].rolling('24H').mean()
    df['CO'] = df['CO'].rolling('8H').mean()
    df['SO2'] = df['SO2'].rolling('1H').mean()
    df['NO2'] = df['NO2'].rolling('1H').mean()
    return df.dropna()

def get_aqi(df):
    """Calcula el AQI a partir de un DataFrame con concentraciones de contaminantes."""
    df = preprocess_data(df)
    aqi_values = df.apply(lambda row: max(calculate_aqi(row[col], AQI_BREAKPOINTS[col]) for col in AQI_BREAKPOINTS), axis=1)
    aqi_values = pd.DataFrame(aqi_values, columns=['AQI']) # DataFrame with AQI values
    aqi_values['AQI'] = aqi_values['AQI'].round(0)
    aqi_values = aqi_values.reset_index()
    return aqi_values

data_cap = pd.concat([dt_pm10_cap[['Fecha', 'PM10']], dt_co_cap[['CO']], dt_o3_cap[['O3']], aqi_no2_cap[['NO2']], aqi_so2_cap[['SO2']]], axis=1)

# Rename columns
data_cap.columns = ['Hora', 'PM10', 'CO', 'O3', 'NO2', 'SO2',]

# Reset index to start from 0
data_cap = data_cap.reset_index(drop=True)

data_cap.head()

# Calculate ICA for each component separately
def calculate_individual_ica(df, pollutant):
    """Calculates the ICA for a specific pollutant."""
    df = preprocess_data(df)
    aqi_values = df.apply(lambda row: calculate_aqi(row[pollutant], AQI_BREAKPOINTS[pollutant]), axis=1)
    aqi_values = pd.DataFrame(aqi_values, columns=[f'ICA_{pollutant}'])
    aqi_values[f'ICA_{pollutant}'] = aqi_values[f'ICA_{pollutant}'].round(0)
    aqi_values = aqi_values.reset_index()
    return aqi_values

ica_co_cap = calculate_individual_ica(data_cap, 'CO')
ica_o3_cap = calculate_individual_ica(data_cap, 'O3')
ica_pm10_cap = calculate_individual_ica(data_cap, 'PM10')
ica_no2_cap = calculate_individual_ica(data_cap, 'NO2')
ica_so2_cap = calculate_individual_ica(data_cap, 'SO2')

aqis_cap = get_aqi(data_cap)

aqis_cap.head()

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Buena"
    elif aqi >= 51 and aqi <= 100:
        return "Moderada"
    elif aqi >= 101 and aqi <= 150:
        return "Insalubre para Grupos Sensibles"
    elif aqi >= 151 and aqi <= 200:
        return "Insalubre"
    elif aqi >= 201 and aqi <= 300:
        return "Muy Insaublre"
    elif aqi >= 301:
        return "Peligroso"

aqis_cap['Categoria'] = aqis_cap['AQI'].apply(categorize_aqi)

aqis_cap.head()

ica_co_cap['Categoria'] = ica_co_cap['ICA_CO'].apply(categorize_aqi)

ica_co_cap.head()

ica_no2_cap['Categoria'] = ica_no2_cap['ICA_NO2'].apply(categorize_aqi)

ica_no2_cap.head()

ica_o3_cap['Categoria'] = ica_o3_cap['ICA_O3'].apply(categorize_aqi)

ica_o3_cap.head()

ica_pm10_cap['Categoria'] = ica_pm10_cap['ICA_PM10'].apply(categorize_aqi)

ica_pm10_cap.head()

ica_so2_cap['Categoria'] = ica_so2_cap['ICA_SO2'].apply(categorize_aqi)

ica_so2_cap.head()

aqis_cap['AQI'].plot(kind='line', figsize=(8, 4), title='AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_co_cap['ICA_CO'].plot(kind='line', figsize=(8, 4), title='CO AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_no2_cap['ICA_NO2'].plot(kind='line', figsize=(8, 4), title='NO2 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_o3_cap['ICA_O3'].plot(kind='line', figsize=(8, 4), title='O3 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_pm10_cap['ICA_PM10'].plot(kind='line', figsize=(8, 4), title='PM 10 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_so2_cap['ICA_SO2'].plot(kind='line', figsize=(8, 4), title='SO2 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

# Definir los rangos del AQI y sus colores
aqi_levels = [(0, 50, 'green', 'Bueno'),
              (51, 100, 'yellow', 'Moderado'),
              (101, 150, 'orange', 'No saludable para grupos sensibles'),
              (151, 200, 'red', 'No saludable'),
              (201, 300, 'purple', 'Muy no saludable'),
              (301, 500, 'maroon', 'Peligroso')]

# Graficar AQI
aqis_cap['AQI'].plot(kind='line', figsize=(8, 4), title='AQI', color='black', linewidth=2)

# Agregar bandas de colores
ax = plt.gca()
for low, high, color, label in aqi_levels:
    ax.axhspan(low, high, color=color, alpha=0.3, label=label)

# Ajustar el gráfico
plt.legend(loc='upper left', fontsize=8)
plt.ylabel('AQI')
plt.xlabel('Tiempo')
plt.grid(True, linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.show()

df_aqi_data_cap = pd.concat([aqis_cap[['Hora', 'AQI', 'Categoria']], ica_co_cap[['ICA_CO', 'Categoria']], ica_no2_cap[['ICA_NO2', 'Categoria']], ica_o3_cap[['ICA_O3', 'Categoria']], ica_so2_cap[['ICA_SO2', 'Categoria']], ica_pm10_cap[['ICA_PM10', 'Categoria']], df_CAP_cleaned_IQR[['TM', 'HR', 'VV', 'DV']]], axis=1)

# Rename columns
df_aqi_data_cap.columns = ['Hora', 'ICA_Total', 'Categoria_Total', 'ICA_CO', 'Categoria_CO', 'ICA_NO2', 'Categoria_NO2', 'ICA_O3', 'Categoria_O3', 'ICA_SO2', 'Categoria_SO2', 'ICA_PM', 'Categoria_PM', 'T', 'H', 'VV', 'DV']

# Reset index to start from 0
df_aqi_data_cap = df_aqi_data_cap.reset_index(drop=True)

df_aqi_data_cap.head()

df_aqi_data_cap.isnull().sum()

class DataInterpolationAnalyzer02:
    def __init__(self, df_cap, df_aqi):
        self.df_cap = df_cap.copy()
        self.df_aqi = df_aqi.copy()
        self.merged_df = None
        self.results = {}
        self.evaluation_results = {}
        self.interpolation_results = {}

    def analyze_missing_patterns02(self):
        self.df_cap['datetime'] = pd.to_datetime(self.df_cap['Fecha'])
        self.df_aqi['datetime'] = pd.to_datetime(self.df_aqi['Hora'])
        start_date = min(self.df_cap['datetime'].min(), self.df_aqi['datetime'].min())
        end_date = max(self.df_cap['datetime'].max(), self.df_aqi['datetime'].max())
        full_range = pd.date_range(start=start_date, end=end_date, freq='H')

        df_full = pd.DataFrame({'datetime': full_range})

        # Correct column selection for aqi_subset - needs 'Hora' mapped to 'datetime'
        # Assuming df_aqi_data_cap columns are 'Hora', 'ICA_Total', 'ICA_CO', etc.
        aqi_subset = self.df_aqi[['Hora', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']].copy()
        aqi_subset = aqi_subset.rename(columns={'Hora': 'datetime'})

        cap_subset = self.df_cap[['datetime', 'VV', 'DV', 'TM', 'HR']].copy()

        merged = df_full.merge(cap_subset, on='datetime', how='left')
        merged = merged.merge(aqi_subset, on='datetime', how='left')
        self.merged_df = merged.set_index('datetime')
        return self.merged_df

    def analyze_temporal_gaps02(self):
        """Analiza los gaps temporales en los datos"""
        print("\n=== ANÁLISIS DE GAPS TEMPORALES ===\n")

        # Identificar gaps consecutivos
        for col in ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']:
            if col in self.merged_df.columns:
                # Identificar secuencias de datos faltantes
                is_missing = self.merged_df[col].isna()
                gap_starts = is_missing & ~is_missing.shift(1).fillna(False)
                gap_ends = is_missing & ~is_missing.shift(-1).fillna(False)

                gaps = []
                start_idx = None
                for i, (start, end) in enumerate(zip(gap_starts, gap_ends)):
                    if start:
                        start_idx = i
                    if end and start_idx is not None:
                        gap_length = i - start_idx + 1
                        gaps.append(gap_length)
                        start_idx = None

                if gaps:
                    print(f"{col}:")
                    print(f"  - Número de gaps: {len(gaps)}")
                    print(f"  - Gap promedio: {np.mean(gaps):.1f} horas")
                    print(f"  - Gap máximo: {max(gaps)} horas")
                    print(f"  - Gap mínimo: {min(gaps)} horas")
                    print()

    def seasonal_decomposition_analysis02(self):
        """Analiza componentes estacionales para mejor interpolación"""
        print("=== ANÁLISIS DE COMPONENTES ESTACIONALES ===\n")

        # Añadir componentes temporales
        self.merged_df['hour'] = self.merged_df.index.hour
        self.merged_df['day_of_week'] = self.merged_df.index.dayofweek
        self.merged_df['month'] = self.merged_df.index.month
        self.merged_df['day_of_year'] = self.merged_df.index.dayofyear


        # Análisis de patrones horarios
        for col in ['VV', 'DV', 'TM', 'HR']:
            if col in self.merged_df.columns:
                hourly_pattern = self.merged_df.groupby('hour')[col].agg(['mean', 'std']).round(3)
                print(f"Patrón horario para {col}:")
                print(f"  - Variación promedio por hora: {hourly_pattern['mean'].std():.3f}")
                print(f"  - Hora con mayor valor promedio: {hourly_pattern['mean'].idxmax()}h ({hourly_pattern['mean'].max():.3f})")
                print(f"  - Hora con menor valor promedio: {hourly_pattern['mean'].idxmin()}h ({hourly_pattern['mean'].min():.3f})")
                print()

    def apply_interpolation_methods02(self):
        cols = ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']
        for col in cols:
            self.interpolation_results[col] = {}
            series = self.merged_df[col]
            self.interpolation_results[col]['linear'] = series.interpolate(method='linear')
            # Ensure enough non-NaN points for spline
            spline_interp = series.interpolate(method='spline', order=3) if series.dropna().shape[0] >= 4 else series
            self.interpolation_results[col]['spline'] = spline_interp
            self.interpolation_results[col]['time'] = series.interpolate(method='time')
            self.interpolation_results[col]['ffill_bfill'] = series.fillna(method='ffill').fillna(method='bfill')
        return self.interpolation_results

    def evaluate_interpolation_quality_blockwise02(self, col='VV', block_sizes=[3, 6, 12]):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        series = self.merged_df[col].copy()
        results = {}
        for block_size in block_sizes:
            mae_dict = {}
            for method in self.interpolation_results[col].keys():
                temp_series = series.copy()
                # Select a block to set to NaN
                # Ensure the block is within the valid index range
                if len(temp_series) > 100 + block_size:
                    start_idx = 100
                    indices_to_nan = temp_series.index[start_idx:start_idx+block_size]
                    true_vals = temp_series.loc[indices_to_nan].copy()
                    temp_series.loc[indices_to_nan] = np.nan

                    # Check if enough data for spline after setting NaNs
                    if method == 'spline' and temp_series.dropna().shape[0] < 4:
                        print(f"Skipping spline evaluation for '{col}' block size {block_size}: Insufficient data after setting NaNs.")
                        continue

                    try:
                        # Interpolate on the series with NaNs
                        if method == 'spline':
                            pred_vals = temp_series.interpolate(method='spline', order=3).loc[indices_to_nan]
                        else:
                             pred_vals = temp_series.interpolate(method=method).loc[indices_to_nan]


                        # Calculate metrics only on valid (non-NaN) true and predicted values
                        valid_indices = true_vals.dropna().index.intersection(pred_vals.dropna().index)

                        if not valid_indices.empty:
                            true_subset = true_vals.loc[valid_indices]
                            predicted_subset = pred_vals.loc[valid_indices]

                            mae = mean_absolute_error(true_subset, predicted_subset)
                            rmse = mean_squared_error(true_subset, predicted_subset, squared=False)
                            mae_dict[method] = {'MAE': mae, 'RMSE': rmse}
                        else:
                             print(f"Skipping {method} evaluation for '{col}' block size {block_size}: No valid points for comparison.")

                    except Exception as e:
                        print(f"Error during {method} evaluation for '{col}' block size {block_size}: {e}")

            results[block_size] = mae_dict
        return results

    def plot_interpolation02(self, col='VV', start='2022-04-01', end='2022-04-03'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,5))
        # Access data directly from merged_df using .loc based on index
        original = self.merged_df[col].loc[start:end]
        plt.plot(original.index, original, label='Original', linewidth=2)
        for method, series in self.interpolation_results.get(col, {}).items():
             # Access interpolated series using .loc based on index
             plt.plot(series.loc[start:end].index, series.loc[start:end], linestyle='--', label=method)
        plt.title(f"Interpolación de '{col}'")
        plt.xlabel("Fecha")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def apply_rf_interpolation02(self, col='ICA_Total'):
        from sklearn.ensemble import RandomForestRegressor
        predictors = ['VV', 'DV', 'TM', 'HR', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']
        df_rf = self.merged_df.copy()

        # Remove the target column from predictors if it's in the list
        if col in predictors:
            predictors.remove(col)

        # Ensure predictors are in the DataFrame
        predictors = [p for p in predictors if p in df_rf.columns]
        if not predictors:
            print(f"Error: No valid predictors found for Random Forest interpolation of '{col}'.")
            return self.merged_df[col] # Return original series if no predictors

        # Drop rows where predictors have NaNs (cannot be used for training)
        df_train = df_rf.dropna(subset=predictors + [col])

        if df_train.empty:
             print(f"Error: No complete cases found for training Random Forest model for '{col}'.")
             return self.merged_df[col] # Return original series if no training data


        X_train = df_train[predictors]
        y_train = df_train[col]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Identify rows where target is missing but predictors are available
        df_predict = df_rf[df_rf[col].isna()].dropna(subset=predictors)

        if not df_predict.empty:
            X_predict = df_predict[predictors]
            y_pred = model.predict(X_predict)
            # Assign predicted values using .loc and the index
            self.merged_df.loc[X_predict.index, col] = y_pred
            print(f"Random Forest interpolation applied to {len(y_pred)} missing values in '{col}'.")
        else:
            print(f"No missing values in '{col}' with valid predictors for Random Forest interpolation.")

        return self.merged_df[col]

    def create_final_dataset02(self, method_preferences=None):
         """Crea el dataset final con interpolaciones aplicadas"""
         print("=== CREACIÓN DE DATASET FINAL ===\n")

         if self.merged_df is None or not isinstance(self.merged_df.index, pd.DatetimeIndex):
              print("Error: El DataFrame fusionado no está preparado (sin DatetimeIndex). Ejecute analyze_missing_patterns02 primero.")
              return None

         final_df = self.merged_df.copy() # Start from the merged_df with DatetimeIndex

         # Drop the temporary index name if it was set during merge/indexing
         final_df.index.name = None

         target_cols = ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']


         if method_preferences is None:
             method_preferences = {}
             # Attempt to choose best method based on evaluation results if available
             if hasattr(self, 'evaluation_results') and self.evaluation_results:
                 print("Seleccionando métodos de interpolación basados en la evaluación de bloques simulados...")
                 for col in target_cols:
                     if col in self.evaluation_results:
                         # Find the best method across all block sizes
                         best_method = None
                         lowest_avg_rmse = float('inf')
                         method_rmse = {} # Store average RMSE per method

                         for block_size, metrics_dict in self.evaluation_results[col].items():
                              for method, metrics in metrics_dict.items():
                                  if metrics is not None and not np.isnan(metrics['RMSE']):
                                       if method not in method_rmse:
                                            method_rmse[method] = []
                                       method_rmse[method].append(metrics['RMSE'])

                         # Calculate average RMSE for each method
                         for method, rmses in method_rmse.items():
                              avg_rmse = np.mean(rmses) if rmses else float('inf')
                              if avg_rmse < lowest_avg_rmse:
                                   lowest_avg_rmse = avg_rmse
                                   best_method = method

                         if best_method:
                             method_preferences[col] = best_method
                             print(f"  - Mejor método para {col} (basado en RMSE promedio): {best_method}")
                         else:
                             # Fallback if evaluation didn't yield a best method
                             method_preferences[col] = 'linear'
                             print(f"  - No se encontró un mejor método basado en evaluación para {col}. Usando 'linear'.")
             else:
                  print("No se encontraron resultados de evaluación. Usando 'linear' como método por defecto.")
                  for col in target_cols:
                       method_preferences[col] = 'linear'


         for col in target_cols:
             if col in final_df.columns:
                 method = method_preferences.get(col, 'linear') # Get preferred method, default to linear

                 if col in self.interpolation_results and method in self.interpolation_results[col]:
                     # Use the pre-calculated interpolated Series if available
                     final_df[col] = self.interpolation_results[col][method]
                     print(f"Aplicando interpolación '{method}' (pre-calculada) para '{col}'.")
                 else:
                     # If pre-calculation wasn't done or method not found in results, interpolate now
                     print(f"Advertencia: Método '{method}' no encontrado o pre-calculado para '{col}'. Aplicando interpolación directa.")
                     try:
                          # Ensure index is datetime for interpolation methods that require it
                          if not isinstance(final_df.index, pd.DatetimeIndex):
                               print(f"Advertencia: Índice no es DatetimeIndex para interpolación directa de '{col}'. Intentando establecerlo.")
                               try:
                                   final_df.index = pd.to_datetime(final_df.index)
                               except Exception as e_idx:
                                   print(f"Error al establecer DatetimeIndex: {e_idx}. Interpolación directa podría fallar.")
                                   # Continue trying, but warn

                          if method == 'seasonal':
                              # Seasonal interpolation needs the seasonal columns, re-create if necessary
                              if not all(c in final_df.columns for c in ['hour', 'day_of_week', 'month']):
                                   print(f"Info: Añadiendo columnas estacionales temporales para interpolación 'seasonal' en '{col}'.")
                                   final_df['hour'] = final_df.index.hour
                                   final_df['day_of_week'] = final_df.index.dayofweek
                                   final_df['month'] = final_df.index.month

                              final_df[col] = self._seasonal_interpolation(final_df[col], col)
                              # Drop temporary seasonal columns after use
                              final_df = final_df.drop(columns=['hour', 'day_of_week', 'month'], errors='ignore')

                          elif method == 'spline':
                              if final_df[col].dropna().shape[0] >= 4:
                                   final_df[col] = final_df[col].interpolate(method='spline', order=3)
                              else:
                                  print(f"Advertencia: Insuficientes datos para interpolación spline directa en '{col}'. Dejando valores NaN.")

                          elif method in ['linear', 'time', 'ffill', 'bfill', 'ffill_bfill', 'rolling_mean']:
                               if method == 'ffill_bfill': # Handle combined method
                                   final_df[col] = final_df[col].fillna(method='ffill').fillna(method='bfill')
                               elif method == 'rolling_mean': # Handle rolling mean
                                   final_df[col] = final_df[col].fillna(
                                       final_df[col].rolling(window=24, center=True, min_periods=1).mean()
                                    )
                               else: # Simple methods
                                   final_df[col] = final_df[col].interpolate(method=method)
                          else:
                              print(f"Error: Método '{method}' no soportado para interpolación directa en '{col}'. Dejando valores NaN.")

                     except Exception as e:
                         print(f"Error durante la interpolación directa de '{col}' con método '{method}': {e}. Dejando valores NaN.")
                         # Leave NaNs if interpolation fails


         # Handle 'Categoria' column separately (categorical data) - assuming it's based on ICA_Total *after* interpolation
         # Re-calculate or forward/backward fill the category based on the interpolated ICA_Total
         if 'ICA_Total' in final_df.columns:
              # Re-calculate category if ICA_Total is interpolated
              def categorize_aqi_final(aqi):
                    if pd.isna(aqi): return None # Keep NaN if ICA_Total is NaN
                    if aqi <= 50: return "Buena"
                    elif 51 <= aqi <= 100: return "Moderada"
                    elif 101 <= aqi <= 150: return "Insalubre para Grupos Sensibles"
                    elif 151 <= aqi <= 200: return "Insalubre"
                    elif 201 <= aqi <= 300: return "Muy Insaublre"
                    elif aqi >= 301: return "Peligroso"
                    return None # Fallback

              final_df['Categoria'] = final_df['ICA_Total'].apply(categorize_aqi_final)
              # Optionally, fill remaining NaNs in Categoria with ffill/bfill if desired
              final_df['Categoria'] = final_df['Categoria'].fillna(method='ffill').fillna(method='bfill')
         elif 'Categoria' in final_df.columns:
             # If ICA_Total is not present, just ffill/bfill the existing category column
              final_df['Categoria'] = final_df['Categoria'].fillna(method='ffill').fillna(method='bfill')


         # Reset index to turn datetime index back into a column
         final_df = final_df.reset_index().rename(columns={'index': 'datetime'})

         # Drop temporary seasonal columns if they exist and weren't dropped by seasonal interpolation logic
         final_df = final_df.drop(columns=['hour', 'day_of_week', 'month', 'day_of_year'], errors='ignore')


         # Estadísticas finales
         print("ESTADÍSTICAS DEL DATASET FINAL:")
         print(f"- Total de registros: {len(final_df):,}")
         print(f"- Rango temporal: {final_df['datetime'].min()} a {final_df['datetime'].max()}")
         print(f"- Registros con datos completos (excepto Categoria que puede tener NaNs if ICA_Total is NaN): {final_df.drop(columns=['Categoria'], errors='ignore').dropna().shape[0]:,}")
         print(f"- Completitud (excluyendo Categoria): {(final_df.drop(columns=['Categoria'], errors='ignore').dropna().shape[0] / len(final_df)) * 100:.2f}%")

         for col in target_cols:
             if col in final_df.columns:
                 remaining_missing = final_df[col].isna().sum()
                 print(f"- {col} faltantes restantes: {remaining_missing} ({(remaining_missing/len(final_df))*100:.2f}%)")

         return final_df

# Paso 2: Crear instancia con los DataFrames originales
analyzer = DataInterpolationAnalyzer02(df_CAP_cleaned_IQR, df_aqi_data_cap)

# Paso 3: Análisis de datos faltantes y patrones temporales
missing_stats = analyzer.analyze_missing_patterns02()
analyzer.analyze_temporal_gaps02()
analyzer.seasonal_decomposition_analysis02()

# Paso 4: Aplicar múltiples métodos de interpolación
interpolation_results = analyzer.apply_interpolation_methods02()

# Paso 5: Evaluación de interpolaciones por bloques simulados
evaluation_blockwise = analyzer.evaluate_interpolation_quality_blockwise02()

# Paso 6: Visualización comparativa de interpolaciones
analyzer.plot_interpolation02(col='ICA_Total')  # Puedes cambiar a otra variable como 'TM', 'HR', etc.

# Paso 7: Interpolación predictiva con Random Forest para huecos medianos
analyzer.apply_rf_interpolation02()

# Paso 8: Generación del dataset final con interpolaciones aplicadas
final_dataset_cap = analyzer.create_final_dataset02()
final_dataset_cap.head()

final_dataset_cap.info()

final_dataset_cap.describe()

final_dataset_cap.isnull().sum()

"""# ***Estación EPG***

## **Procesamiento de Datos**
"""

# Convert object type columns to float64 in df_EPG
for col in df_EPG.columns:
    if df_EPG[col].dtype == 'object':
        df_EPG[col] = pd.to_numeric(df_EPG[col], errors='coerce')

# Display data types to confirm conversion
print(df_EPG.dtypes)

df_EPG.head()

df_EPG.info()

df_EPG.dtypes

df_EPG.isnull().sum()

# Convert 'fecha' column to datetime objects
df_EPG['Fecha'] = pd.to_datetime(df_EPG['Fecha'])

# Set 'fecha' as the index for time series interpolation
df_EPG = df_EPG.set_index('Fecha')

# Perform time series interpolation for numerical columns
numerical_cols = df_EPG.select_dtypes(include=np.number).columns
df_EPG[numerical_cols] = df_EPG[numerical_cols].interpolate(method='time')

# Reset index if needed
df_EPG = df_EPG.reset_index()

# Verify that null values have been filled
df_EPG.isnull().sum()

"""## **Tendencia de los Datos**"""

plt.figure(figsize=(15, 10))

for i, col in enumerate(df_EPG.columns[1:]): # Start from index 1 to exclude 'Fecha'
    plt.subplot(5, 2, i + 1)
    sns.histplot(df_EPG[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for i, col in enumerate(df_EPG.columns[1:]):
  plt.subplot(5, 2, i+1)
  sns.boxplot(x=df_EPG[col])
  plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()

"""## **Outliers**"""

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
df_co = df_EPG['CO'].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_co.quantile(0.25)
Q3 = df_co.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_epg_Clean_CO = df_co[~((df_co < (Q1 - 1.5 * IQR)) | (df_co > (Q3 + 1.5 * IQR)))]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean = df_epg_Clean_CO.mean()
cov_clean = df_epg_Clean_CO.var()

# Regularización de la matriz de covarianza
cov_clean = np.array([[cov_clean]])
cov_clean += np.eye(cov_clean.shape[0]) * 1e-6
inv_cov_clean = np.linalg.inv(cov_clean)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean = df_epg_Clean_CO - mean_clean
mahalanobis_dist_clean = (diff_clean ** 2) * inv_cov_clean[0][0]

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=1)
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_epg_Clean_CO = df_epg_Clean_CO[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_epg_Clean_CO))
df_epg_Clean_CO = df_epg_Clean_CO[(z_scores < 3)]

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
df_o3 = df_EPG['O3'].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_o3.quantile(0.25)
Q3 = df_o3.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_epg_Clean_O3 = df_o3[~((df_o3 < (Q1 - 1.5 * IQR)) | (df_o3 > (Q3 + 1.5 * IQR)))]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean_o3 = df_epg_Clean_O3.mean()
cov_clean_o3 = df_epg_Clean_O3.var()

# Regularización de la matriz de covarianza
cov_clean_o3 = np.array([[cov_clean_o3]])
cov_clean_o3 += np.eye(cov_clean_o3.shape[0]) * 1e-6
inv_cov_clean_o3 = np.linalg.inv(cov_clean_o3)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean_o3 = df_epg_Clean_O3 - mean_clean_o3
mahalanobis_dist_clean = (diff_clean_o3 ** 2) * inv_cov_clean_o3[0][0]

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=1)
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_epg_Clean_O3 = df_epg_Clean_O3[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_epg_Clean_O3))
df_epg_Clean_O3 = df_epg_Clean_O3[(z_scores < 3)]

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
variables = ['NO2', 'SO2']
df_gases = df_EPG[variables].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_gases.quantile(0.25)
Q3 = df_gases.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_epg_h_Clean = df_gases[~((df_gases < (Q1 - 1.5 * IQR)) | (df_gases > (Q3 + 1.5 * IQR))).any(axis=1)]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean = df_epg_h_Clean.mean().values
cov_clean = df_epg_h_Clean.cov().values

# Regularización de la matriz de covarianza
cov_clean += np.eye(len(variables)) * 1e-6
inv_cov_clean = np.linalg.inv(cov_clean)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean = df_epg_h_Clean - mean_clean
mahalanobis_dist_clean = np.sum(np.dot(diff_clean, inv_cov_clean) * diff_clean, axis=1)

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=len(variables))
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_epg_h_Clean = df_epg_h_Clean[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_epg_h_Clean))
df_epg_h_Clean = df_epg_h_Clean[(z_scores < 3).all(axis=1)]

df_pm = df_EPG[['PM2.5']].dropna()

# Configurar y entrenar el modelo
model = IsolationForest(
    contamination=0.05,          # Proporción esperada de outliers (ajustable)
    random_state=42,             # Semilla para reproducibilidad
    n_estimators=100             # Número de árboles (valor por defecto)
)

model.fit(df_pm)

# Predecir outliers (-1 = outlier, 1 = inlier)
predictions = model.predict(df_pm)

# Filtrar inliers y obtener DataFrame limpio
inlier_indices = df_pm.index[predictions == 1]
df_epg_Clean_PM = df_EPG.loc[inlier_indices]

variables_to_clean = ['TM', 'HR', 'VV', 'DV']

# Crear una copia del DataFrame original para la limpieza
df_EPG_cleaned_IQR = df_EPG.copy()

# Iterar sobre cada variable para eliminar outliers usando IQR
for col in variables_to_clean:
    if col in df_EPG_cleaned_IQR.columns:
        Q1 = df_EPG_cleaned_IQR[col].quantile(0.25)
        Q3 = df_EPG_cleaned_IQR[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definir los límites inferior y superior para la detección de outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Eliminar las filas donde el valor de la columna está fuera de los límites
        initial_rows = len(df_EPG_cleaned_IQR)
        df_EPG_cleaned_IQR = df_EPG_cleaned_IQR[(df_EPG_cleaned_IQR[col] >= lower_bound) & (df_EPG_cleaned_IQR[col] <= upper_bound)]
        rows_removed = initial_rows - len(df_EPG_cleaned_IQR)
        print(f"Variable: {col}, Filas eliminadas por IQR: {rows_removed}")
    else:
        print(f"La variable '{col}' no se encuentra en el DataFrame.")

# Crear el DataFrame dt_pm
dt_pm25_epg = df_epg_Clean_PM[['Fecha', 'PM2.5']].copy()

# Crear el DataFrame dt_co
dt_co_epg = df_epg_Clean_PM[['Fecha', 'CO']].copy()

# Crear el DataFrame dt_o3
dt_o3_epg = df_epg_Clean_PM[['Fecha', 'O3']].copy()

df_EPG['Fecha'] = pd.to_datetime(df_EPG['Fecha'])

# Extract the hour from the 'Fecha' column
df_EPG['Hora'] = df_EPG['Fecha'].dt.floor('T')

# Merge the 'Hora' column into df_epg_Clean
df_epg_h_Clean = df_epg_h_Clean.merge(df_EPG[['Hora']], left_index=True, right_index=True, how='left')

"""## **Antés y Después de Eliminar Outliers**"""

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['TM'], kde=True)
plt.title('Distribución de la Temperatura - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_EPG_cleaned_IQR['TM'], kde=True)
plt.title('Distribución de la Temperatura - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_EPG_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['HR'], kde=True)
plt.title('Distribución de la Humedad - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_EPG_cleaned_IQR['HR'], kde=True)
plt.title('Distribución de la Humedad - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_EPG_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['VV'], kde=True)
plt.title('Distribución de la Velocidad del Viento - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_EPG_cleaned_IQR['VV'], kde=True)
plt.title('Distribución de la Velocidad del Viento - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_EPG_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['DV'], kde=True)
plt.title('Distribución de la Dirección del Viento - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_EPG_cleaned_IQR['DV'], kde=True)
plt.title('Distribución de la Dirección del Viento - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_EPG_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['CO'], kde=True)
plt.title('Distribución del CO (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_co_epg['CO'], kde=True)
plt.title('Distribución del CO (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_co_epg.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['O3'], kde=True)
plt.title('Distribución del O3 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_o3_epg['O3'], kde=True)
plt.title('Distribución del O3 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_o3_epg.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['SO2'], kde=True)
plt.title('Distribución del SO2 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_epg_h_Clean['SO2'], kde=True)
plt.title('Distribución del SO2 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_epg_h_Clean.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['NO2'], kde=True)
plt.title('Distribución del NO2 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_epg_h_Clean['NO2'], kde=True)
plt.title('Distribución del NO2 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_epg_h_Clean.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_EPG['PM2.5'], kde=True)
plt.title('Distribución del PM2.5 (ug/m3) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_pm25_epg['PM2.5'], kde=True)
plt.title('Distribución del PM2.5 (ug/m3) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_EPG.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_pm25_epg.shape[0]}")

def get_max(df, date_column, value_columns):
        # Aseguramos que la columna de fecha sea datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Creamos una columna con solo la hora redondeada
    df['Hora'] = df[date_column].dt.floor('H')

    # Agrupamos por hora y calculamos el máximo para cada columna
    hourly_max = df.groupby('Hora')[value_columns].max().reset_index()

    return hourly_max

aqi_so2_epg = get_max(df_epg_h_Clean, 'Hora', 'SO2')

aqi_no2_epg = get_max(df_epg_h_Clean, 'Hora', 'NO2')

dt_co_epg.head()

dt_o3_epg.head()

aqi_no2_epg.head()

aqi_so2_epg.head()

dt_pm25_epg.head()

"""## **ICA**"""

def calculate_aqi(concentration, breakpoints):
    """Calcula el subíndice AQI para un contaminante dado."""
    for i in range(len(breakpoints) - 1):
        C_low, C_high = breakpoints[i][0], breakpoints[i + 1][0]
        I_low, I_high = breakpoints[i][1], breakpoints[i + 1][1]

        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low

    return None  # Fuera de rango

# Tabla de la EPA con los valores de referencia para cada contaminante
# Formato: [(C_low, I_low), (C_high, I_high)]
AQI_BREAKPOINTS = {
    "O3": [(0.000, 0), (0.055, 51), (0.071, 101), (0.086, 151), (0.106, 201), (0.201, 301)],
    "PM2.5": [(0, 0), (9.1, 51), (35.5, 101), (55.5, 151), (125.5, 201), (225.5, 301)],
    #"PM10": [(0, 0), (55, 51), (155, 101), (255, 151), (255, 201), (425, 301)],
    "CO": [(0.0, 0), (4.5, 51), (9.5, 101), (12.5, 151), (15.5, 210), (30.5, 301)],
    "SO2": [(0.000, 0), (0.036, 51), (0.076, 101), (0.186, 151), (0.305, 201), (0.605, 301)],
    "NO2": [(0.000, 0), (0.054, 51), (0.101, 101), (0.360, 151), (0.650, 201), (1.250, 301)]
}

def preprocess_data(df):
    """Promedia los datos según la recomendación de la EPA."""
    df = df.set_index('Hora')
    df.index = pd.to_datetime(df.index)

    df = df.resample('H').mean()  # Asegurar frecuencia horaria mínima
    df['O3'] = df['O3'].rolling('8H').mean()
    df['PM2.5'] = df['PM2.5'].rolling('24H').mean()
    #df['PM10'] = df['PM10'].rolling('24H').mean()
    df['CO'] = df['CO'].rolling('8H').mean()
    df['SO2'] = df['SO2'].rolling('1H').mean()
    df['NO2'] = df['NO2'].rolling('1H').mean()
    return df.dropna()

def get_aqi(df):
    """Calcula el AQI a partir de un DataFrame con concentraciones de contaminantes."""
    df = preprocess_data(df)
    aqi_values = df.apply(lambda row: max(calculate_aqi(row[col], AQI_BREAKPOINTS[col]) for col in AQI_BREAKPOINTS), axis=1)
    aqi_values = pd.DataFrame(aqi_values, columns=['AQI']) # DataFrame with AQI values
    aqi_values['AQI'] = aqi_values['AQI'].round(0)
    aqi_values = aqi_values.reset_index()
    return aqi_values

data_epg = pd.concat([dt_pm25_epg[['Fecha', 'PM2.5']], dt_co_epg[['CO']], dt_o3_epg[['O3']], aqi_no2_epg[['NO2']], aqi_so2_epg[['SO2']]], axis=1)

# Rename columns
data_epg.columns = ['Hora', 'PM2.5', 'CO', 'O3', 'NO2', 'SO2',]

# Reset index to start from 0
data_epg = data_epg.reset_index(drop=True)

data_epg.head()

# Calculate ICA for each component separately
def calculate_individual_ica(df, pollutant):
    """Calculates the ICA for a specific pollutant."""
    df = preprocess_data(df)
    aqi_values = df.apply(lambda row: calculate_aqi(row[pollutant], AQI_BREAKPOINTS[pollutant]), axis=1)
    aqi_values = pd.DataFrame(aqi_values, columns=[f'ICA_{pollutant}'])
    aqi_values[f'ICA_{pollutant}'] = aqi_values[f'ICA_{pollutant}'].round(0)
    aqi_values = aqi_values.reset_index()
    return aqi_values

ica_co_epg = calculate_individual_ica(data_epg, 'CO')
ica_o3_epg = calculate_individual_ica(data_epg, 'O3')
ica_pm25_epg = calculate_individual_ica(data_epg, 'PM2.5')
ica_no2_epg = calculate_individual_ica(data_epg, 'NO2')
ica_so2_epg = calculate_individual_ica(data_epg, 'SO2')

aqis_epg = get_aqi(data_epg)

aqis_epg.head()

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Buena"
    elif aqi >= 51 and aqi <= 100:
        return "Moderada"
    elif aqi >= 101 and aqi <= 150:
        return "Insalubre para Grupos Sensibles"
    elif aqi >= 151 and aqi <= 200:
        return "Insalubre"
    elif aqi >= 201 and aqi <= 300:
        return "Muy Insaublre"
    elif aqi >= 301:
        return "Peligroso"

aqis_epg['Categoria'] = aqis_epg['AQI'].apply(categorize_aqi)

aqis_epg.head()

ica_co_epg['Categoria'] = ica_co_epg['ICA_CO'].apply(categorize_aqi)

ica_co_epg.head()

ica_no2_epg['Categoria'] = ica_no2_epg['ICA_NO2'].apply(categorize_aqi)

ica_no2_epg.head()

ica_o3_epg['Categoria'] = ica_o3_epg['ICA_O3'].apply(categorize_aqi)

ica_o3_epg.head()

ica_pm25_epg['Categoria'] = ica_pm25_epg['ICA_PM2.5'].apply(categorize_aqi)

ica_pm25_epg.head()

ica_so2_epg['Categoria'] = ica_so2_epg['ICA_SO2'].apply(categorize_aqi)

ica_so2_epg.head()

aqis_epg['AQI'].plot(kind='line', figsize=(8, 4), title='AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_co_epg['ICA_CO'].plot(kind='line', figsize=(8, 4), title='CO AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_no2_epg['ICA_NO2'].plot(kind='line', figsize=(8, 4), title='NO2 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_o3_epg['ICA_O3'].plot(kind='line', figsize=(8, 4), title='O3 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_pm25_epg['ICA_PM2.5'].plot(kind='line', figsize=(8, 4), title='PM 2.5 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_so2_epg['ICA_SO2'].plot(kind='line', figsize=(8, 4), title='SO2 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

# Definir los rangos del AQI y sus colores
aqi_levels = [(0, 50, 'green', 'Bueno'),
              (51, 100, 'yellow', 'Moderado'),
              (101, 150, 'orange', 'No saludable para grupos sensibles'),
              (151, 200, 'red', 'No saludable'),
              (201, 300, 'purple', 'Muy no saludable'),
              (301, 500, 'maroon', 'Peligroso')]

# Graficar AQI
aqis_epg['AQI'].plot(kind='line', figsize=(8, 4), title='AQI', color='black', linewidth=2)

# Agregar bandas de colores
ax = plt.gca()
for low, high, color, label in aqi_levels:
    ax.axhspan(low, high, color=color, alpha=0.3, label=label)

# Ajustar el gráfico
plt.legend(loc='upper left', fontsize=8)
plt.ylabel('AQI')
plt.xlabel('Tiempo')
plt.grid(True, linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.show()

df_aqi_data_epg = pd.concat([aqis_epg[['Hora', 'AQI', 'Categoria']], ica_co_epg[['ICA_CO', 'Categoria']], ica_no2_epg[['ICA_NO2', 'Categoria']], ica_o3_epg[['ICA_O3', 'Categoria']], ica_so2_epg[['ICA_SO2', 'Categoria']], ica_pm25_epg[['ICA_PM2.5', 'Categoria']], df_EPG_cleaned_IQR[['TM', 'HR', 'VV', 'DV']]], axis=1)

# Rename columns
df_aqi_data_epg.columns = ['Hora', 'ICA_Total', 'Categoria_Total', 'ICA_CO', 'Categoria_CO', 'ICA_NO2', 'Categoria_NO2', 'ICA_O3', 'Categoria_O3', 'ICA_SO2', 'Categoria_SO2', 'ICA_PM', 'Categoria_PM', 'T', 'H', 'VV', 'DV']

# Reset index to start from 0
df_aqi_data_epg = df_aqi_data_epg.reset_index(drop=True)

df_aqi_data_epg.head()

df_aqi_data_epg.isnull().sum()

class DataInterpolationAnalyzer02:
    def __init__(self, df_cap, df_aqi):
        self.df_cap = df_cap.copy()
        self.df_aqi = df_aqi.copy()
        self.merged_df = None
        self.results = {}
        self.evaluation_results = {}
        self.interpolation_results = {}

    def analyze_missing_patterns02(self):
        self.df_cap['datetime'] = pd.to_datetime(self.df_cap['Fecha'])
        self.df_aqi['datetime'] = pd.to_datetime(self.df_aqi['Hora'])
        start_date = min(self.df_cap['datetime'].min(), self.df_aqi['datetime'].min())
        end_date = max(self.df_cap['datetime'].max(), self.df_aqi['datetime'].max())
        full_range = pd.date_range(start=start_date, end=end_date, freq='H')

        df_full = pd.DataFrame({'datetime': full_range})

        # Correct column selection for aqi_subset - needs 'Hora' mapped to 'datetime'
        # Assuming df_aqi_data_cap columns are 'Hora', 'ICA_Total', 'ICA_CO', etc.
        aqi_subset = self.df_aqi[['Hora', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']].copy()
        aqi_subset = aqi_subset.rename(columns={'Hora': 'datetime'})

        cap_subset = self.df_cap[['datetime', 'VV', 'DV', 'TM', 'HR']].copy()

        merged = df_full.merge(cap_subset, on='datetime', how='left')
        merged = merged.merge(aqi_subset, on='datetime', how='left')
        self.merged_df = merged.set_index('datetime')
        return self.merged_df

    def analyze_temporal_gaps02(self):
        """Analiza los gaps temporales en los datos"""
        print("\n=== ANÁLISIS DE GAPS TEMPORALES ===\n")

        # Identificar gaps consecutivos
        for col in ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']:
            if col in self.merged_df.columns:
                # Identificar secuencias de datos faltantes
                is_missing = self.merged_df[col].isna()
                gap_starts = is_missing & ~is_missing.shift(1).fillna(False)
                gap_ends = is_missing & ~is_missing.shift(-1).fillna(False)

                gaps = []
                start_idx = None
                for i, (start, end) in enumerate(zip(gap_starts, gap_ends)):
                    if start:
                        start_idx = i
                    if end and start_idx is not None:
                        gap_length = i - start_idx + 1
                        gaps.append(gap_length)
                        start_idx = None

                if gaps:
                    print(f"{col}:")
                    print(f"  - Número de gaps: {len(gaps)}")
                    print(f"  - Gap promedio: {np.mean(gaps):.1f} horas")
                    print(f"  - Gap máximo: {max(gaps)} horas")
                    print(f"  - Gap mínimo: {min(gaps)} horas")
                    print()

    def seasonal_decomposition_analysis02(self):
        """Analiza componentes estacionales para mejor interpolación"""
        print("=== ANÁLISIS DE COMPONENTES ESTACIONALES ===\n")

        # Añadir componentes temporales
        self.merged_df['hour'] = self.merged_df.index.hour
        self.merged_df['day_of_week'] = self.merged_df.index.dayofweek
        self.merged_df['month'] = self.merged_df.index.month
        self.merged_df['day_of_year'] = self.merged_df.index.dayofyear


        # Análisis de patrones horarios
        for col in ['VV', 'DV', 'TM', 'HR']:
            if col in self.merged_df.columns:
                hourly_pattern = self.merged_df.groupby('hour')[col].agg(['mean', 'std']).round(3)
                print(f"Patrón horario para {col}:")
                print(f"  - Variación promedio por hora: {hourly_pattern['mean'].std():.3f}")
                print(f"  - Hora con mayor valor promedio: {hourly_pattern['mean'].idxmax()}h ({hourly_pattern['mean'].max():.3f})")
                print(f"  - Hora con menor valor promedio: {hourly_pattern['mean'].idxmin()}h ({hourly_pattern['mean'].min():.3f})")
                print()

    def apply_interpolation_methods02(self):
        cols = ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']
        for col in cols:
            self.interpolation_results[col] = {}
            series = self.merged_df[col]
            self.interpolation_results[col]['linear'] = series.interpolate(method='linear')
            # Ensure enough non-NaN points for spline
            spline_interp = series.interpolate(method='spline', order=3) if series.dropna().shape[0] >= 4 else series
            self.interpolation_results[col]['spline'] = spline_interp
            self.interpolation_results[col]['time'] = series.interpolate(method='time')
            self.interpolation_results[col]['ffill_bfill'] = series.fillna(method='ffill').fillna(method='bfill')
        return self.interpolation_results

    def evaluate_interpolation_quality_blockwise02(self, col='VV', block_sizes=[3, 6, 12]):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        series = self.merged_df[col].copy()
        results = {}
        for block_size in block_sizes:
            mae_dict = {}
            for method in self.interpolation_results[col].keys():
                temp_series = series.copy()
                # Select a block to set to NaN
                # Ensure the block is within the valid index range
                if len(temp_series) > 100 + block_size:
                    start_idx = 100
                    indices_to_nan = temp_series.index[start_idx:start_idx+block_size]
                    true_vals = temp_series.loc[indices_to_nan].copy()
                    temp_series.loc[indices_to_nan] = np.nan

                    # Check if enough data for spline after setting NaNs
                    if method == 'spline' and temp_series.dropna().shape[0] < 4:
                        print(f"Skipping spline evaluation for '{col}' block size {block_size}: Insufficient data after setting NaNs.")
                        continue

                    try:
                        # Interpolate on the series with NaNs
                        if method == 'spline':
                            pred_vals = temp_series.interpolate(method='spline', order=3).loc[indices_to_nan]
                        else:
                             pred_vals = temp_series.interpolate(method=method).loc[indices_to_nan]


                        # Calculate metrics only on valid (non-NaN) true and predicted values
                        valid_indices = true_vals.dropna().index.intersection(pred_vals.dropna().index)

                        if not valid_indices.empty:
                            true_subset = true_vals.loc[valid_indices]
                            predicted_subset = pred_vals.loc[valid_indices]

                            mae = mean_absolute_error(true_subset, predicted_subset)
                            rmse = mean_squared_error(true_subset, predicted_subset, squared=False)
                            mae_dict[method] = {'MAE': mae, 'RMSE': rmse}
                        else:
                             print(f"Skipping {method} evaluation for '{col}' block size {block_size}: No valid points for comparison.")

                    except Exception as e:
                        print(f"Error during {method} evaluation for '{col}' block size {block_size}: {e}")

            results[block_size] = mae_dict
        return results

    def plot_interpolation02(self, col='VV', start='2022-04-01', end='2022-04-03'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,5))
        # Access data directly from merged_df using .loc based on index
        original = self.merged_df[col].loc[start:end]
        plt.plot(original.index, original, label='Original', linewidth=2)
        for method, series in self.interpolation_results.get(col, {}).items():
             # Access interpolated series using .loc based on index
             plt.plot(series.loc[start:end].index, series.loc[start:end], linestyle='--', label=method)
        plt.title(f"Interpolación de '{col}'")
        plt.xlabel("Fecha")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def apply_rf_interpolation02(self, col='ICA_Total'):
        from sklearn.ensemble import RandomForestRegressor
        predictors = ['VV', 'DV', 'TM', 'HR', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']
        df_rf = self.merged_df.copy()

        # Remove the target column from predictors if it's in the list
        if col in predictors:
            predictors.remove(col)

        # Ensure predictors are in the DataFrame
        predictors = [p for p in predictors if p in df_rf.columns]
        if not predictors:
            print(f"Error: No valid predictors found for Random Forest interpolation of '{col}'.")
            return self.merged_df[col] # Return original series if no predictors

        # Drop rows where predictors have NaNs (cannot be used for training)
        df_train = df_rf.dropna(subset=predictors + [col])

        if df_train.empty:
             print(f"Error: No complete cases found for training Random Forest model for '{col}'.")
             return self.merged_df[col] # Return original series if no training data


        X_train = df_train[predictors]
        y_train = df_train[col]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Identify rows where target is missing but predictors are available
        df_predict = df_rf[df_rf[col].isna()].dropna(subset=predictors)

        if not df_predict.empty:
            X_predict = df_predict[predictors]
            y_pred = model.predict(X_predict)
            # Assign predicted values using .loc and the index
            self.merged_df.loc[X_predict.index, col] = y_pred
            print(f"Random Forest interpolation applied to {len(y_pred)} missing values in '{col}'.")
        else:
            print(f"No missing values in '{col}' with valid predictors for Random Forest interpolation.")

        return self.merged_df[col]

    def create_final_dataset02(self, method_preferences=None):
         """Crea el dataset final con interpolaciones aplicadas"""
         print("=== CREACIÓN DE DATASET FINAL ===\n")

         if self.merged_df is None or not isinstance(self.merged_df.index, pd.DatetimeIndex):
              print("Error: El DataFrame fusionado no está preparado (sin DatetimeIndex). Ejecute analyze_missing_patterns02 primero.")
              return None

         final_df = self.merged_df.copy() # Start from the merged_df with DatetimeIndex

         # Drop the temporary index name if it was set during merge/indexing
         final_df.index.name = None

         target_cols = ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']


         if method_preferences is None:
             method_preferences = {}
             # Attempt to choose best method based on evaluation results if available
             if hasattr(self, 'evaluation_results') and self.evaluation_results:
                 print("Seleccionando métodos de interpolación basados en la evaluación de bloques simulados...")
                 for col in target_cols:
                     if col in self.evaluation_results:
                         # Find the best method across all block sizes
                         best_method = None
                         lowest_avg_rmse = float('inf')
                         method_rmse = {} # Store average RMSE per method

                         for block_size, metrics_dict in self.evaluation_results[col].items():
                              for method, metrics in metrics_dict.items():
                                  if metrics is not None and not np.isnan(metrics['RMSE']):
                                       if method not in method_rmse:
                                            method_rmse[method] = []
                                       method_rmse[method].append(metrics['RMSE'])

                         # Calculate average RMSE for each method
                         for method, rmses in method_rmse.items():
                              avg_rmse = np.mean(rmses) if rmses else float('inf')
                              if avg_rmse < lowest_avg_rmse:
                                   lowest_avg_rmse = avg_rmse
                                   best_method = method

                         if best_method:
                             method_preferences[col] = best_method
                             print(f"  - Mejor método para {col} (basado en RMSE promedio): {best_method}")
                         else:
                             # Fallback if evaluation didn't yield a best method
                             method_preferences[col] = 'linear'
                             print(f"  - No se encontró un mejor método basado en evaluación para {col}. Usando 'linear'.")
             else:
                  print("No se encontraron resultados de evaluación. Usando 'linear' como método por defecto.")
                  for col in target_cols:
                       method_preferences[col] = 'linear'


         for col in target_cols:
             if col in final_df.columns:
                 method = method_preferences.get(col, 'linear') # Get preferred method, default to linear

                 if col in self.interpolation_results and method in self.interpolation_results[col]:
                     # Use the pre-calculated interpolated Series if available
                     final_df[col] = self.interpolation_results[col][method]
                     print(f"Aplicando interpolación '{method}' (pre-calculada) para '{col}'.")
                 else:
                     # If pre-calculation wasn't done or method not found in results, interpolate now
                     print(f"Advertencia: Método '{method}' no encontrado o pre-calculado para '{col}'. Aplicando interpolación directa.")
                     try:
                          # Ensure index is datetime for interpolation methods that require it
                          if not isinstance(final_df.index, pd.DatetimeIndex):
                               print(f"Advertencia: Índice no es DatetimeIndex para interpolación directa de '{col}'. Intentando establecerlo.")
                               try:
                                   final_df.index = pd.to_datetime(final_df.index)
                               except Exception as e_idx:
                                   print(f"Error al establecer DatetimeIndex: {e_idx}. Interpolación directa podría fallar.")
                                   # Continue trying, but warn

                          if method == 'seasonal':
                              # Seasonal interpolation needs the seasonal columns, re-create if necessary
                              if not all(c in final_df.columns for c in ['hour', 'day_of_week', 'month']):
                                   print(f"Info: Añadiendo columnas estacionales temporales para interpolación 'seasonal' en '{col}'.")
                                   final_df['hour'] = final_df.index.hour
                                   final_df['day_of_week'] = final_df.index.dayofweek
                                   final_df['month'] = final_df.index.month

                              final_df[col] = self._seasonal_interpolation(final_df[col], col)
                              # Drop temporary seasonal columns after use
                              final_df = final_df.drop(columns=['hour', 'day_of_week', 'month'], errors='ignore')

                          elif method == 'spline':
                              if final_df[col].dropna().shape[0] >= 4:
                                   final_df[col] = final_df[col].interpolate(method='spline', order=3)
                              else:
                                  print(f"Advertencia: Insuficientes datos para interpolación spline directa en '{col}'. Dejando valores NaN.")

                          elif method in ['linear', 'time', 'ffill', 'bfill', 'ffill_bfill', 'rolling_mean']:
                               if method == 'ffill_bfill': # Handle combined method
                                   final_df[col] = final_df[col].fillna(method='ffill').fillna(method='bfill')
                               elif method == 'rolling_mean': # Handle rolling mean
                                   final_df[col] = final_df[col].fillna(
                                       final_df[col].rolling(window=24, center=True, min_periods=1).mean()
                                    )
                               else: # Simple methods
                                   final_df[col] = final_df[col].interpolate(method=method)
                          else:
                              print(f"Error: Método '{method}' no soportado para interpolación directa en '{col}'. Dejando valores NaN.")

                     except Exception as e:
                         print(f"Error durante la interpolación directa de '{col}' con método '{method}': {e}. Dejando valores NaN.")
                         # Leave NaNs if interpolation fails


         # Handle 'Categoria' column separately (categorical data) - assuming it's based on ICA_Total *after* interpolation
         # Re-calculate or forward/backward fill the category based on the interpolated ICA_Total
         if 'ICA_Total' in final_df.columns:
              # Re-calculate category if ICA_Total is interpolated
              def categorize_aqi_final(aqi):
                    if pd.isna(aqi): return None # Keep NaN if ICA_Total is NaN
                    if aqi <= 50: return "Buena"
                    elif 51 <= aqi <= 100: return "Moderada"
                    elif 101 <= aqi <= 150: return "Insalubre para Grupos Sensibles"
                    elif 151 <= aqi <= 200: return "Insalubre"
                    elif 201 <= aqi <= 300: return "Muy Insaublre"
                    elif aqi >= 301: return "Peligroso"
                    return None # Fallback

              final_df['Categoria'] = final_df['ICA_Total'].apply(categorize_aqi_final)
              # Optionally, fill remaining NaNs in Categoria with ffill/bfill if desired
              final_df['Categoria'] = final_df['Categoria'].fillna(method='ffill').fillna(method='bfill')
         elif 'Categoria' in final_df.columns:
             # If ICA_Total is not present, just ffill/bfill the existing category column
              final_df['Categoria'] = final_df['Categoria'].fillna(method='ffill').fillna(method='bfill')


         # Reset index to turn datetime index back into a column
         final_df = final_df.reset_index().rename(columns={'index': 'datetime'})

         # Drop temporary seasonal columns if they exist and weren't dropped by seasonal interpolation logic
         final_df = final_df.drop(columns=['hour', 'day_of_week', 'month', 'day_of_year'], errors='ignore')


         # Estadísticas finales
         print("ESTADÍSTICAS DEL DATASET FINAL:")
         print(f"- Total de registros: {len(final_df):,}")
         print(f"- Rango temporal: {final_df['datetime'].min()} a {final_df['datetime'].max()}")
         print(f"- Registros con datos completos (excepto Categoria que puede tener NaNs if ICA_Total is NaN): {final_df.drop(columns=['Categoria'], errors='ignore').dropna().shape[0]:,}")
         print(f"- Completitud (excluyendo Categoria): {(final_df.drop(columns=['Categoria'], errors='ignore').dropna().shape[0] / len(final_df)) * 100:.2f}%")

         for col in target_cols:
             if col in final_df.columns:
                 remaining_missing = final_df[col].isna().sum()
                 print(f"- {col} faltantes restantes: {remaining_missing} ({(remaining_missing/len(final_df))*100:.2f}%)")

         return final_df

# Paso 2: Crear instancia con los DataFrames originales
analyzer = DataInterpolationAnalyzer02(df_EPG_cleaned_IQR, df_aqi_data_epg)

# Paso 3: Análisis de datos faltantes y patrones temporales
missing_stats = analyzer.analyze_missing_patterns02()
analyzer.analyze_temporal_gaps02()
analyzer.seasonal_decomposition_analysis02()

# Paso 4: Aplicar múltiples métodos de interpolación
interpolation_results = analyzer.apply_interpolation_methods02()

# Paso 5: Evaluación de interpolaciones por bloques simulados
evaluation_blockwise = analyzer.evaluate_interpolation_quality_blockwise02()

# Paso 6: Visualización comparativa de interpolaciones
analyzer.plot_interpolation02(col='ICA_Total')  # Puedes cambiar a otra variable como 'TM', 'HR', etc.

# Paso 7: Interpolación predictiva con Random Forest para huecos medianos
analyzer.apply_rf_interpolation02()

# Paso 8: Generación del dataset final con interpolaciones aplicadas
final_dataset_epg = analyzer.create_final_dataset02()
final_dataset_epg.head()

final_dataset_epg.info()

final_dataset_epg.describe()

final_dataset_epg.isnull().sum()

"""# ***Estación FEO***

## **Procesamiento de Datos**
"""

# Convert object type columns to float64 in df_FEO
for col in df_FEO.columns:
    if df_FEO[col].dtype == 'object':
        df_FEO[col] = pd.to_numeric(df_FEO[col], errors='coerce')

# Display data types to confirm conversion
print(df_FEO.dtypes)

df_FEO.head()

df_FEO.info()

df_FEO.describe()

df_FEO.dtypes

df_FEO.isnull().sum()

# Convert 'fecha' column to datetime objects
df_FEO['Fecha'] = pd.to_datetime(df_FEO['Fecha'])

# Set 'fecha' as the index for time series interpolation
df_FEO = df_FEO.set_index('Fecha')

# Perform time series interpolation for numerical columns
numerical_cols = df_FEO.select_dtypes(include=np.number).columns
df_FEO[numerical_cols] = df_FEO[numerical_cols].interpolate(method='time')

# Reset index if needed
df_FEO = df_FEO.reset_index()

# Verify that null values have been filled
df_FEO.isnull().sum()

df_FEO.info()

# Aplicar extrapolación hacia adelante en la columna PM2.5
df_FEO['PM2.5'] = df_FEO['PM2.5'].interpolate(method='linear', limit_direction='forward')

# Aplicar extrapolación hacia atrás en la columna PM2.5 para los nulos restantes
df_FEO['PM2.5'] = df_FEO['PM2.5'].interpolate(method='linear', limit_direction='backward')

# Verificar si aún quedan valores nulos en PM2.5
print(df_FEO['PM2.5'].isnull().sum())

# Verificar rango de valores realistas
print(f"Rango PM2.5: {df_FEO['PM2.5'].min():.2f} - {df_FEO['PM2.5'].max():.2f}")

df_FEO.info()

"""## **Tendencia de los Datos**"""

plt.figure(figsize=(15, 10))

for i, col in enumerate(df_FEO.columns[1:]): # Start from index 1 to exclude 'Fecha'
    plt.subplot(5, 2, i + 1)
    sns.histplot(df_FEO[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for i, col in enumerate(df_FEO.columns[1:]):
  plt.subplot(5, 2, i+1)
  sns.boxplot(x=df_FEO[col])
  plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()

"""## **Outliers**"""

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
df_co = df_FEO['CO'].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_co.quantile(0.25)
Q3 = df_co.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_feo_Clean_CO = df_co[~((df_co < (Q1 - 1.5 * IQR)) | (df_co > (Q3 + 1.5 * IQR)))]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean = df_feo_Clean_CO.mean()
cov_clean = df_feo_Clean_CO.var()

# Regularización de la matriz de covarianza
cov_clean = np.array([[cov_clean]])
cov_clean += np.eye(cov_clean.shape[0]) * 1e-6
inv_cov_clean = np.linalg.inv(cov_clean)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean = df_feo_Clean_CO - mean_clean
mahalanobis_dist_clean = (diff_clean ** 2) * inv_cov_clean[0][0]

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=1)
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_feo_Clean_CO = df_feo_Clean_CO[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_feo_Clean_CO))
df_feo_Clean_CO = df_feo_Clean_CO[(z_scores < 3)]

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
df_o3 = df_FEO['O3'].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_o3.quantile(0.25)
Q3 = df_o3.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_feo_Clean_O3 = df_o3[~((df_o3 < (Q1 - 1.5 * IQR)) | (df_o3 > (Q3 + 1.5 * IQR)))]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean_o3 = df_feo_Clean_O3.mean()
cov_clean_o3 = df_feo_Clean_O3.var()

# Regularización de la matriz de covarianza
cov_clean_o3 = np.array([[cov_clean_o3]])
cov_clean_o3 += np.eye(cov_clean_o3.shape[0]) * 1e-6
inv_cov_clean_o3 = np.linalg.inv(cov_clean_o3)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean_o3 = df_feo_Clean_O3 - mean_clean_o3
mahalanobis_dist_clean = (diff_clean_o3 ** 2) * inv_cov_clean_o3[0][0]

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=1)
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_feo_Clean_O3 = df_feo_Clean_O3[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_feo_Clean_O3))
df_feo_Clean_O3 = df_feo_Clean_O3[(z_scores < 3)]

# Seleccionar las variables relevantes y eliminar filas con valores faltantes
variables = ['NO2', 'SO2']
df_gases = df_FEO[variables].dropna()

# Recalcular métricas estadísticas para limpieza
Q1 = df_gases.quantile(0.25)
Q3 = df_gases.quantile(0.75)
IQR = Q3 - Q1

# 1. Aplicar IQR para eliminar valores extremos univariados
df_feo_h_Clean = df_gases[~((df_gases < (Q1 - 1.5 * IQR)) | (df_gases > (Q3 + 1.5 * IQR))).any(axis=1)]

# 2. Aplicar distancia de Mahalanobis (99%) en los datos filtrados
mean_clean = df_feo_h_Clean.mean().values
cov_clean = df_feo_h_Clean.cov().values

# Regularización de la matriz de covarianza
cov_clean += np.eye(len(variables)) * 1e-6
inv_cov_clean = np.linalg.inv(cov_clean)

# Recalcular Mahalanobis en datos sin outliers univariados
diff_clean = df_feo_h_Clean - mean_clean
mahalanobis_dist_clean = np.sum(np.dot(diff_clean, inv_cov_clean) * diff_clean, axis=1)

# Aplicar umbral del 99%
threshold_mahalanobis = chi2.ppf(0.99, df=len(variables))
outliers_mahalanobis = mahalanobis_dist_clean > threshold_mahalanobis

# Filtrar datos con Mahalanobis
df_feo_h_Clean = df_feo_h_Clean[~outliers_mahalanobis]

# 3. Aplicar Z-score (>3) en los datos restantes
z_scores = np.abs(zscore(df_feo_h_Clean))
df_feo_h_Clean = df_feo_h_Clean[(z_scores < 3).all(axis=1)]

df_pm = df_FEO[['PM2.5']].dropna()

# Configurar y entrenar el modelo
model = IsolationForest(
    contamination=0.05,          # Proporción esperada de outliers (ajustable)
    random_state=42,             # Semilla para reproducibilidad
    n_estimators=100             # Número de árboles (valor por defecto)
)

model.fit(df_pm)

# Predecir outliers (-1 = outlier, 1 = inlier)
predictions = model.predict(df_pm)

# Filtrar inliers y obtener DataFrame limpio
inlier_indices = df_pm.index[predictions == 1]
df_feo_Clean_PM = df_FEO.loc[inlier_indices]

variables_to_clean = ['TM', 'HR', 'VV', 'DV']

# Crear una copia del DataFrame original para la limpieza
df_FEO_cleaned_IQR = df_FEO.copy()

# Iterar sobre cada variable para eliminar outliers usando IQR
for col in variables_to_clean:
    if col in df_FEO_cleaned_IQR.columns:
        Q1 = df_FEO_cleaned_IQR[col].quantile(0.25)
        Q3 = df_FEO_cleaned_IQR[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definir los límites inferior y superior para la detección de outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Eliminar las filas donde el valor de la columna está fuera de los límites
        initial_rows = len(df_FEO_cleaned_IQR)
        df_FEO_cleaned_IQR = df_FEO_cleaned_IQR[(df_FEO_cleaned_IQR[col] >= lower_bound) & (df_FEO_cleaned_IQR[col] <= upper_bound)]
        rows_removed = initial_rows - len(df_FEO_cleaned_IQR)
        print(f"Variable: {col}, Filas eliminadas por IQR: {rows_removed}")
    else:
        print(f"La variable '{col}' no se encuentra en el DataFrame.")

# Crear el DataFrame dt_pm
dt_pm25_feo = df_feo_Clean_PM[['Fecha', 'PM2.5']].copy()

# Crear el DataFrame dt_co
dt_co_feo = df_feo_Clean_PM[['Fecha', 'CO']].copy()

# Crear el DataFrame dt_o3
dt_o3_feo = df_feo_Clean_PM[['Fecha', 'O3']].copy()

df_FEO['Fecha'] = pd.to_datetime(df_FEO['Fecha'])

# Extract the hour from the 'Fecha' column
df_FEO['Hora'] = df_FEO['Fecha'].dt.floor('T')

# Merge the 'Hora' column into df_feo_Clean
df_feo_h_Clean = df_feo_h_Clean.merge(df_FEO[['Hora']], left_index=True, right_index=True, how='left')

"""## **Antés y Después de Eliminar Outliers**"""

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['TM'], kde=True)
plt.title('Distribución de la Temperatura - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_FEO_cleaned_IQR['TM'], kde=True)
plt.title('Distribución de la Temperatura - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_FEO_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['HR'], kde=True)
plt.title('Distribución de la Humedad - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_FEO_cleaned_IQR['HR'], kde=True)
plt.title('Distribución de la Humedad - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_FEO_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['VV'], kde=True)
plt.title('Distribución de la Velocidad del Viento - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_FEO_cleaned_IQR['VV'], kde=True)
plt.title('Distribución de la Velocidad del Viento - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_FEO_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['DV'], kde=True)
plt.title('Distribución de la Dirección del Viento - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_FEO_cleaned_IQR['DV'], kde=True)
plt.title('Distribución de la Dirección del Viento - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_FEO_cleaned_IQR.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['CO'], kde=True)
plt.title('Distribución del CO (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_co_feo['CO'], kde=True)
plt.title('Distribución del CO (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_co_feo.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['O3'], kde=True)
plt.title('Distribución del O3 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_o3_feo['O3'], kde=True)
plt.title('Distribución del O3 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_o3_feo.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['SO2'], kde=True)
plt.title('Distribución del SO2 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_feo_h_Clean['SO2'], kde=True)
plt.title('Distribución del SO2 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_feo_h_Clean.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['NO2'], kde=True)
plt.title('Distribución del NO2 (ppm) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(df_feo_h_Clean['NO2'], kde=True)
plt.title('Distribución del NO2 (ppm) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {df_feo_h_Clean.shape[0]}")

# Visualize the distributions of the target variable before and after outlier removal.
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(df_FEO['PM2.5'], kde=True)
plt.title('Distribución del PM2.5 (ug/m3) - Datos Originales')

plt.subplot(1, 2, 2)
sns.histplot(dt_pm25_feo['PM2.5'], kde=True)
plt.title('Distribución del PM2.5 (ug/m3) - Después de Remover los Outliers')

plt.tight_layout()
plt.show()

print(f"Número de Columnas removidas antes del procesamineto de Outliers: {df_FEO.shape[0]}")
print(f"Número de Columnas removidas después del procesamineto de Outliers: {dt_pm25_feo.shape[0]}")

def get_max(df, date_column, value_columns):
        # Aseguramos que la columna de fecha sea datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Creamos una columna con solo la hora redondeada
    df['Hora'] = df[date_column].dt.floor('H')

    # Agrupamos por hora y calculamos el máximo para cada columna
    hourly_max = df.groupby('Hora')[value_columns].max().reset_index()

    return hourly_max

aqi_so2_feo = get_max(df_feo_h_Clean, 'Hora', 'SO2')

aqi_no2_feo = get_max(df_feo_h_Clean, 'Hora', 'NO2')

dt_co_feo.head()

dt_o3_feo.head()

aqi_no2_feo.head()

aqi_so2_feo.head()

dt_pm25_feo.head()

"""## **ICA**"""

def calculate_aqi(concentration, breakpoints):
    """Calcula el subíndice AQI para un contaminante dado."""
    for i in range(len(breakpoints) - 1):
        C_low, C_high = breakpoints[i][0], breakpoints[i + 1][0]
        I_low, I_high = breakpoints[i][1], breakpoints[i + 1][1]

        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low

    return None  # Fuera de rango

# Tabla de la EPA con los valores de referencia para cada contaminante
# Formato: [(C_low, I_low), (C_high, I_high)]
AQI_BREAKPOINTS = {
    "O3": [(0.000, 0), (0.055, 51), (0.071, 101), (0.086, 151), (0.106, 201), (0.201, 301)],
    "PM2.5": [(0, 0), (9.1, 51), (35.5, 101), (55.5, 151), (125.5, 201), (225.5, 301)],
    #"PM10": [(0, 0), (55, 51), (155, 101), (255, 151), (255, 201), (425, 301)],
    "CO": [(0.0, 0), (4.5, 51), (9.5, 101), (12.5, 151), (15.5, 210), (30.5, 301)],
    "SO2": [(0.000, 0), (0.036, 51), (0.076, 101), (0.186, 151), (0.305, 201), (0.605, 301)],
    "NO2": [(0.000, 0), (0.054, 51), (0.101, 101), (0.360, 151), (0.650, 201), (1.250, 301)]
}

def preprocess_data(df):
    """Promedia los datos según la recomendación de la EPA."""
    df = df.set_index('Hora')
    df.index = pd.to_datetime(df.index)

    df = df.resample('H').mean()  # Asegurar frecuencia horaria mínima
    df['O3'] = df['O3'].rolling('8H').mean()
    df['PM2.5'] = df['PM2.5'].rolling('24H').mean()
    #df['PM10'] = df['PM10'].rolling('24H').mean()
    df['CO'] = df['CO'].rolling('8H').mean()
    df['SO2'] = df['SO2'].rolling('1H').mean()
    df['NO2'] = df['NO2'].rolling('1H').mean()
    return df.dropna()

def get_aqi(df):
    """Calcula el AQI a partir de un DataFrame con concentraciones de contaminantes."""
    df = preprocess_data(df)
    aqi_values = df.apply(lambda row: max(calculate_aqi(row[col], AQI_BREAKPOINTS[col]) for col in AQI_BREAKPOINTS), axis=1)
    aqi_values = pd.DataFrame(aqi_values, columns=['AQI']) # DataFrame with AQI values
    aqi_values['AQI'] = aqi_values['AQI'].round(0)
    aqi_values = aqi_values.reset_index()
    return aqi_values

data_feo = pd.concat([dt_pm25_feo[['Fecha', 'PM2.5']], dt_co_feo[['CO']], dt_o3_feo[['O3']], aqi_no2_feo[['NO2']], aqi_so2_feo[['SO2']]], axis=1)

# Rename columns
data_feo.columns = ['Hora', 'PM2.5', 'CO', 'O3', 'NO2', 'SO2',]

# Reset index to start from 0
data_feo = data_feo.reset_index(drop=True)

data_feo.head()

# Calculate ICA for each component separately
def calculate_individual_ica(df, pollutant):
    """Calculates the ICA for a specific pollutant."""
    df = preprocess_data(df)
    aqi_values = df.apply(lambda row: calculate_aqi(row[pollutant], AQI_BREAKPOINTS[pollutant]), axis=1)
    aqi_values = pd.DataFrame(aqi_values, columns=[f'ICA_{pollutant}'])
    aqi_values[f'ICA_{pollutant}'] = aqi_values[f'ICA_{pollutant}'].round(0)
    aqi_values = aqi_values.reset_index()
    return aqi_values

ica_co_feo = calculate_individual_ica(data_feo, 'CO')
ica_o3_feo = calculate_individual_ica(data_feo, 'O3')
ica_pm25_feo = calculate_individual_ica(data_feo, 'PM2.5')
ica_no2_feo = calculate_individual_ica(data_feo, 'NO2')
ica_so2_feo = calculate_individual_ica(data_feo, 'SO2')

aqis_feo = get_aqi(data_feo)

aqis_feo.head()

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Buena"
    elif aqi >= 51 and aqi <= 100:
        return "Moderada"
    elif aqi >= 101 and aqi <= 150:
        return "Insalubre para Grupos Sensibles"
    elif aqi >= 151 and aqi <= 200:
        return "Insalubre"
    elif aqi >= 201 and aqi <= 300:
        return "Muy Insaublre"
    elif aqi >= 301:
        return "Peligroso"

aqis_feo['Categoria'] = aqis_feo['AQI'].apply(categorize_aqi)

aqis_feo.head()

ica_co_feo['Categoria'] = ica_co_feo['ICA_CO'].apply(categorize_aqi)

ica_co_feo.head()

ica_no2_feo['Categoria'] = ica_no2_feo['ICA_NO2'].apply(categorize_aqi)

ica_no2_feo.head()

ica_o3_feo['Categoria'] = ica_o3_feo['ICA_O3'].apply(categorize_aqi)

ica_o3_feo.head()

ica_pm25_feo['Categoria'] = ica_pm25_feo['ICA_PM2.5'].apply(categorize_aqi)

ica_pm25_feo.head()

ica_so2_feo['Categoria'] = ica_so2_feo['ICA_SO2'].apply(categorize_aqi)

ica_so2_feo.head()

aqis_feo['AQI'].plot(kind='line', figsize=(8, 4), title='AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_co_feo['ICA_CO'].plot(kind='line', figsize=(8, 4), title='CO AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_no2_feo['ICA_NO2'].plot(kind='line', figsize=(8, 4), title='NO2 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_o3_feo['ICA_O3'].plot(kind='line', figsize=(8, 4), title='O3 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_pm25_feo['ICA_PM2.5'].plot(kind='line', figsize=(8, 4), title='PM 2.5 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

ica_so2_feo['ICA_SO2'].plot(kind='line', figsize=(8, 4), title='SO2 AQI')
plt.gca().spines[['top', 'right']].set_visible(False)

# Definir los rangos del AQI y sus colores
aqi_levels = [(0, 50, 'green', 'Bueno'),
              (51, 100, 'yellow', 'Moderado'),
              (101, 150, 'orange', 'No saludable para grupos sensibles'),
              (151, 200, 'red', 'No saludable'),
              (201, 300, 'purple', 'Muy no saludable'),
              (301, 500, 'maroon', 'Peligroso')]

# Graficar AQI
aqis_feo['AQI'].plot(kind='line', figsize=(8, 4), title='AQI', color='black', linewidth=2)

# Agregar bandas de colores
ax = plt.gca()
for low, high, color, label in aqi_levels:
    ax.axhspan(low, high, color=color, alpha=0.3, label=label)

# Ajustar el gráfico
plt.legend(loc='upper left', fontsize=8)
plt.ylabel('AQI')
plt.xlabel('Tiempo')
plt.grid(True, linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.show()

df_aqi_data_feo = pd.concat([aqis_feo[['Hora', 'AQI', 'Categoria']], ica_co_feo[['ICA_CO', 'Categoria']], ica_no2_feo[['ICA_NO2', 'Categoria']], ica_o3_feo[['ICA_O3', 'Categoria']], ica_so2_feo[['ICA_SO2', 'Categoria']], ica_pm25_feo[['ICA_PM2.5', 'Categoria']], df_FEO_cleaned_IQR[['TM', 'HR', 'VV', 'DV']]], axis=1)

# Rename columns
df_aqi_data_feo.columns = ['Hora', 'ICA_Total', 'Categoria_Total', 'ICA_CO', 'Categoria_CO', 'ICA_NO2', 'Categoria_NO2', 'ICA_O3', 'Categoria_O3', 'ICA_SO2', 'Categoria_SO2', 'ICA_PM', 'Categoria_PM', 'T', 'H', 'VV', 'DV']

# Reset index to start from 0
df_aqi_data_feo = df_aqi_data_feo.reset_index(drop=True)

df_aqi_data_feo.head()

df_aqi_data_feo.isnull().sum()

class DataInterpolationAnalyzer02:
    def __init__(self, df_cap, df_aqi):
        self.df_cap = df_cap.copy()
        self.df_aqi = df_aqi.copy()
        self.merged_df = None
        self.results = {}
        self.evaluation_results = {}
        self.interpolation_results = {}

    def analyze_missing_patterns02(self):
        self.df_cap['datetime'] = pd.to_datetime(self.df_cap['Fecha'])
        self.df_aqi['datetime'] = pd.to_datetime(self.df_aqi['Hora'])
        start_date = min(self.df_cap['datetime'].min(), self.df_aqi['datetime'].min())
        end_date = max(self.df_cap['datetime'].max(), self.df_aqi['datetime'].max())
        full_range = pd.date_range(start=start_date, end=end_date, freq='H')

        df_full = pd.DataFrame({'datetime': full_range})

        # Correct column selection for aqi_subset - needs 'Hora' mapped to 'datetime'
        # Assuming df_aqi_data_cap columns are 'Hora', 'ICA_Total', 'ICA_CO', etc.
        aqi_subset = self.df_aqi[['Hora', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']].copy()
        aqi_subset = aqi_subset.rename(columns={'Hora': 'datetime'})

        cap_subset = self.df_cap[['datetime', 'VV', 'DV', 'TM', 'HR']].copy()

        merged = df_full.merge(cap_subset, on='datetime', how='left')
        merged = merged.merge(aqi_subset, on='datetime', how='left')
        self.merged_df = merged.set_index('datetime')
        return self.merged_df

    def analyze_temporal_gaps02(self):
        """Analiza los gaps temporales en los datos"""
        print("\n=== ANÁLISIS DE GAPS TEMPORALES ===\n")

        # Identificar gaps consecutivos
        for col in ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']:
            if col in self.merged_df.columns:
                # Identificar secuencias de datos faltantes
                is_missing = self.merged_df[col].isna()
                gap_starts = is_missing & ~is_missing.shift(1).fillna(False)
                gap_ends = is_missing & ~is_missing.shift(-1).fillna(False)

                gaps = []
                start_idx = None
                for i, (start, end) in enumerate(zip(gap_starts, gap_ends)):
                    if start:
                        start_idx = i
                    if end and start_idx is not None:
                        gap_length = i - start_idx + 1
                        gaps.append(gap_length)
                        start_idx = None

                if gaps:
                    print(f"{col}:")
                    print(f"  - Número de gaps: {len(gaps)}")
                    print(f"  - Gap promedio: {np.mean(gaps):.1f} horas")
                    print(f"  - Gap máximo: {max(gaps)} horas")
                    print(f"  - Gap mínimo: {min(gaps)} horas")
                    print()

    def seasonal_decomposition_analysis02(self):
        """Analiza componentes estacionales para mejor interpolación"""
        print("=== ANÁLISIS DE COMPONENTES ESTACIONALES ===\n")

        # Añadir componentes temporales
        self.merged_df['hour'] = self.merged_df.index.hour
        self.merged_df['day_of_week'] = self.merged_df.index.dayofweek
        self.merged_df['month'] = self.merged_df.index.month
        self.merged_df['day_of_year'] = self.merged_df.index.dayofyear


        # Análisis de patrones horarios
        for col in ['VV', 'DV', 'TM', 'HR']:
            if col in self.merged_df.columns:
                hourly_pattern = self.merged_df.groupby('hour')[col].agg(['mean', 'std']).round(3)
                print(f"Patrón horario para {col}:")
                print(f"  - Variación promedio por hora: {hourly_pattern['mean'].std():.3f}")
                print(f"  - Hora con mayor valor promedio: {hourly_pattern['mean'].idxmax()}h ({hourly_pattern['mean'].max():.3f})")
                print(f"  - Hora con menor valor promedio: {hourly_pattern['mean'].idxmin()}h ({hourly_pattern['mean'].min():.3f})")
                print()

    def apply_interpolation_methods02(self):
        cols = ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']
        for col in cols:
            self.interpolation_results[col] = {}
            series = self.merged_df[col]
            self.interpolation_results[col]['linear'] = series.interpolate(method='linear')
            # Ensure enough non-NaN points for spline
            spline_interp = series.interpolate(method='spline', order=3) if series.dropna().shape[0] >= 4 else series
            self.interpolation_results[col]['spline'] = spline_interp
            self.interpolation_results[col]['time'] = series.interpolate(method='time')
            self.interpolation_results[col]['ffill_bfill'] = series.fillna(method='ffill').fillna(method='bfill')
        return self.interpolation_results

    def evaluate_interpolation_quality_blockwise02(self, col='VV', block_sizes=[3, 6, 12]):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        series = self.merged_df[col].copy()
        results = {}
        for block_size in block_sizes:
            mae_dict = {}
            for method in self.interpolation_results[col].keys():
                temp_series = series.copy()
                # Select a block to set to NaN
                # Ensure the block is within the valid index range
                if len(temp_series) > 100 + block_size:
                    start_idx = 100
                    indices_to_nan = temp_series.index[start_idx:start_idx+block_size]
                    true_vals = temp_series.loc[indices_to_nan].copy()
                    temp_series.loc[indices_to_nan] = np.nan

                    # Check if enough data for spline after setting NaNs
                    if method == 'spline' and temp_series.dropna().shape[0] < 4:
                        print(f"Skipping spline evaluation for '{col}' block size {block_size}: Insufficient data after setting NaNs.")
                        continue

                    try:
                        # Interpolate on the series with NaNs
                        if method == 'spline':
                            pred_vals = temp_series.interpolate(method='spline', order=3).loc[indices_to_nan]
                        else:
                             pred_vals = temp_series.interpolate(method=method).loc[indices_to_nan]


                        # Calculate metrics only on valid (non-NaN) true and predicted values
                        valid_indices = true_vals.dropna().index.intersection(pred_vals.dropna().index)

                        if not valid_indices.empty:
                            true_subset = true_vals.loc[valid_indices]
                            predicted_subset = pred_vals.loc[valid_indices]

                            mae = mean_absolute_error(true_subset, predicted_subset)
                            rmse = mean_squared_error(true_subset, predicted_subset, squared=False)
                            mae_dict[method] = {'MAE': mae, 'RMSE': rmse}
                        else:
                             print(f"Skipping {method} evaluation for '{col}' block size {block_size}: No valid points for comparison.")

                    except Exception as e:
                        print(f"Error during {method} evaluation for '{col}' block size {block_size}: {e}")

            results[block_size] = mae_dict
        return results

    def plot_interpolation02(self, col='VV', start='2022-04-01', end='2022-04-03'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,5))
        # Access data directly from merged_df using .loc based on index
        original = self.merged_df[col].loc[start:end]
        plt.plot(original.index, original, label='Original', linewidth=2)
        for method, series in self.interpolation_results.get(col, {}).items():
             # Access interpolated series using .loc based on index
             plt.plot(series.loc[start:end].index, series.loc[start:end], linestyle='--', label=method)
        plt.title(f"Interpolación de '{col}'")
        plt.xlabel("Fecha")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def apply_rf_interpolation02(self, col='ICA_Total'):
        from sklearn.ensemble import RandomForestRegressor
        predictors = ['VV', 'DV', 'TM', 'HR', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']
        df_rf = self.merged_df.copy()

        # Remove the target column from predictors if it's in the list
        if col in predictors:
            predictors.remove(col)

        # Ensure predictors are in the DataFrame
        predictors = [p for p in predictors if p in df_rf.columns]
        if not predictors:
            print(f"Error: No valid predictors found for Random Forest interpolation of '{col}'.")
            return self.merged_df[col] # Return original series if no predictors

        # Drop rows where predictors have NaNs (cannot be used for training)
        df_train = df_rf.dropna(subset=predictors + [col])

        if df_train.empty:
             print(f"Error: No complete cases found for training Random Forest model for '{col}'.")
             return self.merged_df[col] # Return original series if no training data


        X_train = df_train[predictors]
        y_train = df_train[col]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Identify rows where target is missing but predictors are available
        df_predict = df_rf[df_rf[col].isna()].dropna(subset=predictors)

        if not df_predict.empty:
            X_predict = df_predict[predictors]
            y_pred = model.predict(X_predict)
            # Assign predicted values using .loc and the index
            self.merged_df.loc[X_predict.index, col] = y_pred
            print(f"Random Forest interpolation applied to {len(y_pred)} missing values in '{col}'.")
        else:
            print(f"No missing values in '{col}' with valid predictors for Random Forest interpolation.")

        return self.merged_df[col]

    def create_final_dataset02(self, method_preferences=None):
         """Crea el dataset final con interpolaciones aplicadas"""
         print("=== CREACIÓN DE DATASET FINAL ===\n")

         if self.merged_df is None or not isinstance(self.merged_df.index, pd.DatetimeIndex):
              print("Error: El DataFrame fusionado no está preparado (sin DatetimeIndex). Ejecute analyze_missing_patterns02 primero.")
              return None

         final_df = self.merged_df.copy() # Start from the merged_df with DatetimeIndex

         # Drop the temporary index name if it was set during merge/indexing
         final_df.index.name = None

         target_cols = ['VV', 'DV', 'TM', 'HR', 'ICA_Total', 'ICA_CO', 'ICA_NO2', 'ICA_O3', 'ICA_SO2', 'ICA_PM']


         if method_preferences is None:
             method_preferences = {}
             # Attempt to choose best method based on evaluation results if available
             if hasattr(self, 'evaluation_results') and self.evaluation_results:
                 print("Seleccionando métodos de interpolación basados en la evaluación de bloques simulados...")
                 for col in target_cols:
                     if col in self.evaluation_results:
                         # Find the best method across all block sizes
                         best_method = None
                         lowest_avg_rmse = float('inf')
                         method_rmse = {} # Store average RMSE per method

                         for block_size, metrics_dict in self.evaluation_results[col].items():
                              for method, metrics in metrics_dict.items():
                                  if metrics is not None and not np.isnan(metrics['RMSE']):
                                       if method not in method_rmse:
                                            method_rmse[method] = []
                                       method_rmse[method].append(metrics['RMSE'])

                         # Calculate average RMSE for each method
                         for method, rmses in method_rmse.items():
                              avg_rmse = np.mean(rmses) if rmses else float('inf')
                              if avg_rmse < lowest_avg_rmse:
                                   lowest_avg_rmse = avg_rmse
                                   best_method = method

                         if best_method:
                             method_preferences[col] = best_method
                             print(f"  - Mejor método para {col} (basado en RMSE promedio): {best_method}")
                         else:
                             # Fallback if evaluation didn't yield a best method
                             method_preferences[col] = 'linear'
                             print(f"  - No se encontró un mejor método basado en evaluación para {col}. Usando 'linear'.")
             else:
                  print("No se encontraron resultados de evaluación. Usando 'linear' como método por defecto.")
                  for col in target_cols:
                       method_preferences[col] = 'linear'


         for col in target_cols:
             if col in final_df.columns:
                 method = method_preferences.get(col, 'linear') # Get preferred method, default to linear

                 if col in self.interpolation_results and method in self.interpolation_results[col]:
                     # Use the pre-calculated interpolated Series if available
                     final_df[col] = self.interpolation_results[col][method]
                     print(f"Aplicando interpolación '{method}' (pre-calculada) para '{col}'.")
                 else:
                     # If pre-calculation wasn't done or method not found in results, interpolate now
                     print(f"Advertencia: Método '{method}' no encontrado o pre-calculado para '{col}'. Aplicando interpolación directa.")
                     try:
                          # Ensure index is datetime for interpolation methods that require it
                          if not isinstance(final_df.index, pd.DatetimeIndex):
                               print(f"Advertencia: Índice no es DatetimeIndex para interpolación directa de '{col}'. Intentando establecerlo.")
                               try:
                                   final_df.index = pd.to_datetime(final_df.index)
                               except Exception as e_idx:
                                   print(f"Error al establecer DatetimeIndex: {e_idx}. Interpolación directa podría fallar.")
                                   # Continue trying, but warn

                          if method == 'seasonal':
                              # Seasonal interpolation needs the seasonal columns, re-create if necessary
                              if not all(c in final_df.columns for c in ['hour', 'day_of_week', 'month']):
                                   print(f"Info: Añadiendo columnas estacionales temporales para interpolación 'seasonal' en '{col}'.")
                                   final_df['hour'] = final_df.index.hour
                                   final_df['day_of_week'] = final_df.index.dayofweek
                                   final_df['month'] = final_df.index.month

                              final_df[col] = self._seasonal_interpolation(final_df[col], col)
                              # Drop temporary seasonal columns after use
                              final_df = final_df.drop(columns=['hour', 'day_of_week', 'month'], errors='ignore')

                          elif method == 'spline':
                              if final_df[col].dropna().shape[0] >= 4:
                                   final_df[col] = final_df[col].interpolate(method='spline', order=3)
                              else:
                                  print(f"Advertencia: Insuficientes datos para interpolación spline directa en '{col}'. Dejando valores NaN.")

                          elif method in ['linear', 'time', 'ffill', 'bfill', 'ffill_bfill', 'rolling_mean']:
                               if method == 'ffill_bfill': # Handle combined method
                                   final_df[col] = final_df[col].fillna(method='ffill').fillna(method='bfill')
                               elif method == 'rolling_mean': # Handle rolling mean
                                   final_df[col] = final_df[col].fillna(
                                       final_df[col].rolling(window=24, center=True, min_periods=1).mean()
                                    )
                               else: # Simple methods
                                   final_df[col] = final_df[col].interpolate(method=method)
                          else:
                              print(f"Error: Método '{method}' no soportado para interpolación directa en '{col}'. Dejando valores NaN.")

                     except Exception as e:
                         print(f"Error durante la interpolación directa de '{col}' con método '{method}': {e}. Dejando valores NaN.")
                         # Leave NaNs if interpolation fails


         # Handle 'Categoria' column separately (categorical data) - assuming it's based on ICA_Total *after* interpolation
         # Re-calculate or forward/backward fill the category based on the interpolated ICA_Total
         if 'ICA_Total' in final_df.columns:
              # Re-calculate category if ICA_Total is interpolated
              def categorize_aqi_final(aqi):
                    if pd.isna(aqi): return None # Keep NaN if ICA_Total is NaN
                    if aqi <= 50: return "Buena"
                    elif 51 <= aqi <= 100: return "Moderada"
                    elif 101 <= aqi <= 150: return "Insalubre para Grupos Sensibles"
                    elif 151 <= aqi <= 200: return "Insalubre"
                    elif 201 <= aqi <= 300: return "Muy Insaublre"
                    elif aqi >= 301: return "Peligroso"
                    return None # Fallback

              final_df['Categoria'] = final_df['ICA_Total'].apply(categorize_aqi_final)
              # Optionally, fill remaining NaNs in Categoria with ffill/bfill if desired
              final_df['Categoria'] = final_df['Categoria'].fillna(method='ffill').fillna(method='bfill')
         elif 'Categoria' in final_df.columns:
             # If ICA_Total is not present, just ffill/bfill the existing category column
              final_df['Categoria'] = final_df['Categoria'].fillna(method='ffill').fillna(method='bfill')


         # Reset index to turn datetime index back into a column
         final_df = final_df.reset_index().rename(columns={'index': 'datetime'})

         # Drop temporary seasonal columns if they exist and weren't dropped by seasonal interpolation logic
         final_df = final_df.drop(columns=['hour', 'day_of_week', 'month', 'day_of_year'], errors='ignore')


         # Estadísticas finales
         print("ESTADÍSTICAS DEL DATASET FINAL:")
         print(f"- Total de registros: {len(final_df):,}")
         print(f"- Rango temporal: {final_df['datetime'].min()} a {final_df['datetime'].max()}")
         print(f"- Registros con datos completos (excepto Categoria que puede tener NaNs if ICA_Total is NaN): {final_df.drop(columns=['Categoria'], errors='ignore').dropna().shape[0]:,}")
         print(f"- Completitud (excluyendo Categoria): {(final_df.drop(columns=['Categoria'], errors='ignore').dropna().shape[0] / len(final_df)) * 100:.2f}%")

         for col in target_cols:
             if col in final_df.columns:
                 remaining_missing = final_df[col].isna().sum()
                 print(f"- {col} faltantes restantes: {remaining_missing} ({(remaining_missing/len(final_df))*100:.2f}%)")

         return final_df

# Paso 2: Crear instancia con los DataFrames originales
analyzer = DataInterpolationAnalyzer02(df_FEO_cleaned_IQR, df_aqi_data_feo)

# Paso 3: Análisis de datos faltantes y patrones temporales
missing_stats = analyzer.analyze_missing_patterns02()
analyzer.analyze_temporal_gaps02()
analyzer.seasonal_decomposition_analysis02()

# Paso 4: Aplicar múltiples métodos de interpolación
interpolation_results = analyzer.apply_interpolation_methods02()

# Paso 5: Evaluación de interpolaciones por bloques simulados
evaluation_blockwise = analyzer.evaluate_interpolation_quality_blockwise02()

# Paso 6: Visualización comparativa de interpolaciones
analyzer.plot_interpolation02(col='ICA_Total')  # Puedes cambiar a otra variable como 'TM', 'HR', etc.

# Paso 7: Interpolación predictiva con Random Forest para huecos medianos
analyzer.apply_rf_interpolation02()

# Paso 8: Generación del dataset final con interpolaciones aplicadas
final_dataset_feo = analyzer.create_final_dataset02()
final_dataset_feo.head()

final_dataset_feo.info()

final_dataset_feo.describe()

final_dataset_feo.isnull().sum()

"""# **Datos Procesados**"""

final_dataset_cap.to_excel("ICA_datos_procesados_Validacion_CAP.xlsx", index=False)

final_dataset_epg.to_excel("ICA_datos_procesados_Validacion_EPG.xlsx", index=False)

final_dataset_feo.to_excel("ICA_datos_procesados_Validacion_FEO.xlsx", index=False)
