David González Labrada
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

Para conectar con el MCP, se requiere tener un token de acceso a la API de ESIOS y escribir la siguiente configuración en el archivo config del LLM a utilizar que permita la conexión con MCPs mediante STDIO:


# ⚡ Sistema de Predicción Energética con IA

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-Academic-red.svg)
![Status](https://img.shields.io/badge/status-TFM-green.svg)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-orange.svg)

*TFM sobre la creación de un sistema de predicción energética con un servidor MCP para su acceso basado en IA*

</div>

---

## 🎯 Descripción

Este proyecto implementa un **sistema completo de análisis y predicción del consumo energético doméstico** utilizando técnicas de machine learning y explicabilidad de modelos (SHAP). El sistema permite realizar predicciones precisas, analizar patrones históricos y generar recomendaciones personalizadas de optimización energética a través de una interfaz conversacional con IA.

### ✨ Características Principales

- 🔮 **Predicción energética** utilizando modelos ensemble optimizados
- 📊 **Análisis histórico** con breakdown por electrodomésticos  
- 🧠 **Explicabilidad de modelos** mediante valores SHAP
- 💰 **Integración con APIs** de precios eléctricos (ESIOS)
- 🤖 **Servidor MCP** para acceso conversacional via Claude AI
- 📈 **Visualizaciones interactivas** de consumo y predicciones
- 💡 **Recomendaciones personalizadas** de optimización energética

---

### 🔧 Archivos Principales

- **`modelos_v3.py`** - Clase principal con todas las funciones para creación, entrenamiento y evaluación de modelos
- **`server.py`** - Servidor MCP que expone las funcionalidades del sistema para integración con Claude AI

---

## 🛠️ Tecnologías Utilizadas

<div align="center">

| Categoría | Tecnologías |
|-----------|-------------|
| **🤖 Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **🧠 Explicabilidad** | SHAP (SHapley Additive exPlanations) |
| **🌐 Backend** | FastAPI, Python 3.8+ |
| **📊 Datos** | Pandas, NumPy, APIs ESIOS |
| **📈 Visualización** | Matplotlib |
| **🔗 Integración** | MCP (Model Context Protocol) |

</div>

---

## 🚀 Instalación y Configuración

### 📋 Requisitos Previos

- Python 3.8 o superior
- Cuenta en ESIOS (Red Eléctrica de España)
- Acceso a Claude AI con soporte MCP

---

## ▶️ Uso del Sistema

### 🚀 Iniciar Servidor MCP
'''

{
  "mcpServers": {
       "mcp-david-TFM": {
            "command": "uv",
            "args": [
                "--directory",
                "RUTA DEL PROYECTO",
                "run",
                "-m",
                "davidElectric"
            ],
            "env": {
                "ESIOS_API_TOKEN": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            }
        }
  }
}
'''



### 🤖 Funciones Disponibles via MCP

| Función | Descripción | Parámetros |
|---------|-------------|------------|
| `predict_consumption()` | Predicciones de consumo futuro | `init_date`, `end_date` |
| `get_consumption_analysis()` | Análisis histórico detallado | `init_date`, `end_date` |
| `explain_predictions()` | Explicabilidad con SHAP | `appliance`, `horizon` |
| `get_precio()` | Consulta precios eléctricos | `init_date`, `end_date`, `price_type` |
| `get_precio_inteligente()` | Precio con fallback automático | `target_date` |

---

## 🔒 Licencia y Términos Legales

### ⚖️ AVISO LEGAL IMPORTANTE

> **Cualquier distribución ilegal del contenido de este repositorio privado será perseguida legalmente hasta las últimas consecuencias.**

Este material está protegido por **derechos de autor** y constituye **propiedad intelectual** del autor. Su uso está limitado exclusivamente a:

✅ **Permitido:**
- Evaluación académica por el tribunal del TFM
- Consulta de referencia con autorización previa  
- Fines educativos no comerciales (con cita obligatoria)

❌ **Prohibido:**
- Uso comercial sin licencia
- Redistribución sin autorización
- Plagio o apropiación indebida
- Modificación de autoría

Para solicitar permisos de uso, contactar al autor.

