TFM - Sistema de Predicción Energética con IA
Descripción
TFM sobre la creación de un sistema de predicción energética con un servidor MCP para su acceso basado en IA.
Este proyecto implementa un sistema completo de análisis y predicción del consumo energético doméstico utilizando técnicas de machine learning y explicabilidad de modelos (SHAP). El sistema permite realizar predicciones precisas, analizar patrones históricos y generar recomendaciones personalizadas de optimización energética.
Estructura del Proyecto
Este código incluye todas las funciones utilizadas para la creación, subida y prueba de modelos en la clase modelos_v3. El código del servidor MCP aparece en server.py.
Archivos Principales

modelos_v3.py - Clase principal con funciones para creación, entrenamiento y evaluación de modelos
server.py - Servidor MCP que expone las funcionalidades del sistema
data/ - Datasets de entrenamiento y validación
models/ - Modelos entrenados guardados
shap/outputs - Valores SHAP por modelo y horizonte

Características

Predicción energética utilizando modelos ensemble optimizados
Análisis histórico con breakdown por electrodomésticos
Explicabilidad de modelos mediante valores SHAP
Integración con APIs de precios eléctricos (ESIOS)
Servidor MCP para acceso conversacional via IA
Visualizaciones interactivas de consumo y predicciones
Recomendaciones personalizadas de optimización

Tecnologías Utilizadas

Python 3.8+
Scikit-learn - Modelos de machine learning
XGBoost/LightGBM - Algoritmos ensemble
SHAP - Explicabilidad de modelos
FastAPI - Framework para servidor MCP
Pandas/NumPy - Manipulación de datos
