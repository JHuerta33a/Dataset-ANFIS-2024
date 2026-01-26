# Desarrollo de un Sistema Experto para la Generación de Mapas de Interpolación de Índices de Calidad del Aire Proveniente de Múltiples Estaciones 

### Repositorio de Material Suplementario

Este repositorio contiene la base de datos completa, el código fuente y los resultados de validación pertenecientes a la tesis presentada por **Juan Manuel Huerta Ordaz** para la obtención del título de **Maestro en Tecnología Avanzada** en **CICATA Unidad Querétaro**.

## 📄 Contenido del Repositorio

El proyecto implementa un enfoque híbrido utilizando **Python** para el procesamiento de datos y **MATLAB** para el entrenamiento de redes neuronales difusas (ANFIS).

### 1. Bases de Datos (`/data`)
Se adjuntan los datasets originales en formato digital para garantizar la reproducibilidad del estudio:
* **Entrenamiento:** Excel con **26,300 registros** distribuidos en múltiples hojas.
* **Validación:** Excel con **3,340 registros** para pruebas finales.

### 2. Código Fuente (`/code`)
* **MATLAB:** Scripts principales para la configuración y entrenamiento del sistema ANFIS (*Adaptive Network-based Fuzzy Inference System*).
* **Python:** Scripts auxiliares para la limpieza, estructuración de los datos y pre-análisis.

### 3. Resultados y Métricas (`/results`)
* **Comparación de Folds:** Archivo Excel detallado (`comparacion_folds_anfis.xlsx`) que documenta el rendimiento del modelo en cada iteración de la validación cruzada, permitiendo verificar la estabilidad del entrenamiento.

## 🚀 Requisitos Técnicos
Para ejecutar los scripts contenidos en este repositorio se requiere:
* MATLAB R2025a.
* Python 3.12.12.

## cite-as (Cómo citar)
Si utilizas estos datos o código, por favor cita la tesis original:
> Huerta Ordaz, Juan Manuel. (2025). Desarrollo de un Sistema Experto para la Generación de Mapas de Interpolación de Índices de Calidad del Aire Proveniente de Múltiples Estaciones. Maestría en Tecnología Avanzada, CICATA Unidad Querétaro. Repositorio GitHub: https://github.com/JHuerta33a/Dataset-ANFIS-2024

---
**Contacto:** jhuertao2400@alumno.ipn.mx || Asesor: Antonio Hernandez Zavala, anhernandezz@ipn.mx
