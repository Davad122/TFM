import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.base import clone
import warnings
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf
from xgboost import XGBRegressor
import pickle
import joblib
import json
import os
import shap


warnings.filterwarnings("ignore")

class EnergyConsumptionPredictor:

    def __init__(self):
        #Inicializar variables de clase para el predictor energético
        #Horizontes a considerar (de momento máximo 7 días)
        self.horizons = [1, 2, 3, 4, 5, 6, 7]
        #Almacenar electrodomésticos estables
        self.is_stable = []
        #Se almacenarán los modelos en el formato {appliance: {horizon: model_info}}
        self.models = {}
        #Dataset a utilizar, ya medido en Wh
        self.data = pd.read_csv("./data/House_1_pre_real.csv")
        #Auxiliar para no perder las últimas fechas
        self.data_aux = self.data.copy()
        #Atributos de los electrodomésticos y la energía total consumida
        self.appliances = self.data.columns.tolist()
        #Quitar columnas que no son electrodomésticos
        self.appliances.remove('Date')
        self.appliances.remove('tavg')
        self.feature_importance = {}
        self.evaluation_results = {}
        self.recommendations = {}
        self.models_trained = False
        self.models_metadata = {}
        self.models_by_horizon = {}
        self.shap_importance = {}


    def preprocess_data(self):
        """Cargar y preprocesar los datos añadiendo features"""
        #Convertir fecha y ordenar (por si no está ordenado)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        #Añadir features temporales especiales
        self.data['day_of_week'] = self.data['Date'].dt.dayofweek
        self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        self.data['month'] = self.data['Date'].dt.month
        self.data['day_of_month'] = self.data['Date'].dt.day
        self.data['quarter'] = self.data['Date'].dt.quarter
        #Features de temperatura para los próximos 7 y 30 días (media)
        self.data['tavg_rolling_7'] = self.data['tavg'].rolling(window=7, min_periods=1).mean()
        self.data['tavg_rolling_30'] = self.data['tavg'].rolling(window=30, min_periods=1).mean()
        #Variables lag (consumo histórico)
        self.recommendations = self.analyse_and_create_features()
        self.data_aux = self.data.copy()
        #Crear targets (variables a predecir) para 1 semana
        for appliance in self.appliances:
            for day in range(1,  8):
                target_col = f'{appliance}_target_day_{day}'
                self.data[target_col] = self.data[appliance].shift(-day)
        #Eliminar filas con NaN para eliminar primeros y últimos valores nulos
        self.data = self.data.dropna()
        #Filtrar valores extremos, en este caso
        self.data = self.data[self.data['Aggregate'] <= 20000]
        print(f"Datos procesados: {len(self.data)} registros válidos")
        print(f"Rango de fechas: {self.data['Date'].min().date()} a {self.data['Date'].max().date()}")
        print(self.data.columns.tolist())
        return self.data

    def analyse_and_create_features(self, max_lags=30, threshold=0.10):
        """
        Analiza autocorrelación y crea automáticamente las features recomendadas
        """
        #Se almacenanarán recomendaciones de features
        all_recommendations = {}
        for appliance in self.appliances:
            #Análisis de autocorrelación
            autocorr = acf(self.data[appliance].dropna(), nlags=max_lags, fft=True)
            #Encontrar lags significativos, se debe excluir el lag 0
            significant_lags = []
            for i, corr in enumerate(autocorr[1:], 1):
                if abs(corr) > threshold:
                    #Se almacena si supera el umbral
                    significant_lags.append(i)
            #Almacenar electrodoméstico y sus lags
            recommendations = {
                'appliance': appliance,
                'significant_lags': significant_lags,
                'category': '',
                'features_created': []
            }
            #Si no hay lags significativos es un electrodoméstico estable
            if len(significant_lags) == 0:
                recommendations['category'] = 'stable'
                #Añadir a is_stable
                self.is_stable.append(appliance)
                #Crear features relacionads con su media móvil y temperatura
                self.data[f'{appliance}_rolling_7_mean'] = self.data[appliance].rolling(7, min_periods=1).mean()
                #Añadir al diccionario de recomendaciones
                recommendations['features_created'].append(f'{appliance}_rolling_7_mean')
            #Si hay pocos lags significativos, se considera un electrodoméstico de uso esporádico
            elif len(significant_lags) <= 5:
                recommendations['category'] = 'sporadic'
                #Crear features relacionados con su media móvil y con sus lags significativos
                self.data[f'{appliance}_rolling_7_mean'] = self.data[appliance].rolling(7, min_periods=1).mean()
                recommendations['features_created'].append(f'{appliance}_rolling_7_mean')
                for lag in significant_lags:
                    self.data[f'{appliance}_lag_{lag}'] = self.data[appliance].shift(lag)
                    recommendations['features_created'].append(f'{appliance}_lag_{lag}')
            #Caso especial de aggregate, donde hay mucha autocorrelación y es la variable más importante de predecir
            elif appliance.lower() == 'aggregate':
                recommendations['category'] = 'aggregate'
                #Lags clave
                key_lags = [1, 2, 3, 4, 5, 6, 7]
                #key_lags = [1, 2, 3, 7, 14, 21, 28]
                important_lags = [lag for lag in key_lags if lag in significant_lags]
                #Media semanal, quincenal y desviación semanal
                self.data[f'{appliance}_rolling_7_mean'] = self.data[appliance].rolling(7, min_periods=1).mean()
                self.data[f'{appliance}_rolling_7_std'] = self.data[appliance].rolling(7, min_periods=1).std()
                #self.data[f'{appliance}_rolling_14_mean'] = self.data[appliance].rolling(14, min_periods=1).mean()
                #Almacenar features creadas de media
                recommendations['features_created'].extend([f'{appliance}_rolling_7_mean', f'{appliance}_rolling_7_std', f'{appliance}_rolling_14_mean'])
                #Lags importantes, crear y almacenar
                for lag in important_lags:
                    self.data[f'{appliance}_lag_{lag}'] = self.data[appliance].shift(lag)
                    recommendations['features_created'].append(f'{appliance}_lag_{lag}')
            #Caso de uso regular (más de 5 lags significativos)
            else:
                recommendations['category'] = 'regular'
                #Lags clave considerados
                key_lags = [1, 2, 3, 4, 5, 6, 7]
                important_lags = [lag for lag in key_lags if lag in significant_lags]
                #Media de los últimos 7 días
                self.data[f'{appliance}_rolling_7_mean'] = self.data[appliance].rolling(7, min_periods=1).mean()
                recommendations['features_created'].append(f'{appliance}_rolling_7_mean')
                #Lags importantes y almacenar
                for lag in important_lags:
                    self.data[f'{appliance}_lag_{lag}'] = self.data[appliance].shift(lag)
                    recommendations['features_created'].append(f'{appliance}_lag_{lag}')
            #Guardar recomendaciones finales
            all_recommendations[appliance] = recommendations
            #Imprimir resumen para depurar
            print(f"\n{appliance}:")
            print(f"  Categoría: {recommendations['category']}")
            print(f"  Lags significativos: {recommendations['significant_lags']}")
            print(f"  Features creadas: {recommendations['features_created']}")
            print(self.data)

        return all_recommendations

    def prepare_features(self, appliance, horizon_days):
        """Features específicas para un electrodoméstico incluyendo cross-correlations"""

        # Features base (existentes)
        base_features_stable = [
            #'day_of_week'
        ]
        aggregate_features = [
            'day_of_week', 'month', 'tavg_rolling_7'
        ]
        base_features = [
            'day_of_week', 'month',
            'tavg', 'tavg_rolling_7', 'tavg_rolling_30'
        ]
        appliance_features = []

        #Features que fueron creadas por analyze_and_create_features
        possible_features = [
            f'{appliance}_rolling_7_mean',
            f'{appliance}_rolling_7_std',
            f'{appliance}_rolling_14_mean',
            f'{appliance}_temp_interaction'
        ]
        #Añadir solo las que existen
        for feature in possible_features:
            if feature in self.data.columns:
                appliance_features.append(feature)
        #Obtener diccionario de recomendación para el electrodoméstico
        #Lags específicos del electrodoméstico
        for col in self.data.columns:
            if col.startswith(f'{appliance}_lag_'):
                category = self.recommendations.get(appliance, {}).get('category')
                if (appliance.lower() == 'aggregate' or category == 'regular') and (col.endswith(f'_lag_{horizon_days}')):
                    #Si es Aggregate, añadir el lag del horizonte
                    appliance_features.append(col)
                elif appliance.lower() == 'aggregate' or category == 'regular':
                    continue
                else:
                    appliance_features.append(col)
        # Combinar todas las features

        all_features = base_features + appliance_features if appliance not in self.is_stable and appliance != 'Aggregate' else base_features_stable + appliance_features if appliance != 'Aggregate' else aggregate_features + appliance_features
        print(all_features)
        # Filtrar features que existen en los datos
        available_features = [f for f in all_features if f in self.data.columns]
        print(available_features)
        return available_features

    def temporal_split(self, test_size=0.2):
        """División temporal de datos sin shuffle"""
        split_idx = int(len(self.data) * (1 - test_size))
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()

        print("División temporal:")
        print(f"Entrenamiento: {len(train_data)} registros ({train_data['Date'].min().date()} - {train_data['Date'].max().date()})")
        print(f"Test: {len(test_data)} registros ({test_data['Date'].min().date()} - {test_data['Date'].max().date()})")
        return train_data, test_data

    def train_models(self, test_size=0.2):
        """Entrenar modelos para todos los electrodomésticos y horizontes"""
        print("\n🤖 Iniciando entrenamiento de modelos...")
        train_data, test_data = self.temporal_split(test_size)
        #Configurar modelos a probar
        model_configs = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0)
        }
        model_configs = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
            #'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            #'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        }
        results = {}
        for horizon in ['1', '2', '3', '4', '5', '6', '7']:
            print(f"\nEntrenando modelos para horizonte {horizon}...")
            horizon_models = {}
            horizon_results = {}
            for appliance in self.appliances:
                print(f"   🔧 {appliance}...")
                features = self.prepare_features(appliance, horizon)
                print(features)
                target_col = f'{appliance}_target_day_{horizon}'
                #Preparar datos
                X_train = train_data[features]
                y_train = train_data[target_col]
                X_test = test_data[features]
                y_test = test_data[target_col]
                # Probar diferentes modelos
                best_model = None
                best_score = float('inf')
                best_model_name = None
                for model_name, model_template in model_configs.items():
                    model = clone(model_template)
                    #Validación cruzada temporal
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = -cross_val_score(model, X_train, y_train,cv=tscv, scoring='neg_mean_absolute_error')
                    avg_cv_score = np.mean(cv_scores)
                    #Ir actualizando el mejor modelo
                    if avg_cv_score < best_score:
                        best_score = avg_cv_score
                        best_model = model
                        best_model_name = model_name
                #Entrenar mejor modelo
                best_model.fit(X_train, y_train)
                #Evaluar en test
                y_pred = best_model.predict(X_test)
                #Métricas
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                try:
                    explainer = shap.Explainer(best_model, X_train)  # sin check_additivity
                    shap_values = explainer(X_test)
                    shap_importance = np.abs(shap_values.values).mean(axis=0)
                    shap_importance_dict = dict(zip(features, shap_importance))
                    self.shap_importance[f'{appliance}_{horizon}'] = shap_importance_dict
                except Exception as e:
                    print(f"⚠ No se pudo calcular SHAP para {best_model_name}: {e}")
                #Guardar modelo y resultados
                horizon_models[appliance] = {
                    'model': best_model,
                    'features': features.copy(),
                    'model_name': best_model_name
                }

                horizon_results[appliance] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_score': best_score,
                    'model_name': best_model_name
                }
                #Feature importance (si es posible)
                if hasattr(best_model, 'feature_importances_'):
                    importance = dict(zip(features, best_model.feature_importances_))
                    self.feature_importance[f'{appliance}_{horizon}'] = importance
            #Guardar modelos y resultados por horizonte
            self.models[horizon] = horizon_models
            results[horizon] = horizon_results

        self.evaluation_results = results
        return results

    def predict_future_date(self, target_date):
        """Predecir consumo para una fecha futura específica"""
        #Fecha que se quiere predecir
        target_date = pd.to_datetime(target_date)
        #Extraer features temporales de la fecha objetivo
        base_features_dict = {
            'day_of_week': target_date.dayofweek,
            'month': target_date.month,
        }
        horizon_days = (target_date - self.data_aux['Date'].max()).days
        #print(horizon_days)
        if horizon_days < 1:
            return "La fecha no es válida para predicción, debe ser posterior a la última fecha del dataset."
        elif horizon_days > 7:
            horizon_days = 7
        #Estimar temperatura usando promedio histórico para ese mes
        historical_temp = self.data_aux[self.data_aux['Date'].dt.month == target_date.month]['tavg'].mean()
        base_features_dict['tavg'] = historical_temp
        base_features_dict['tavg_rolling_7'] = historical_temp
        base_features_dict['tavg_rolling_30'] = historical_temp

        #Obtener los datos más recientes disponibles del auxiliar
        latest_data = self.data_aux.iloc[-horizon_days:]  #Últimos N días según horizonte
        #print(f"Usando datos desde: {latest_data['Date'].min().date()} hasta: {latest_data['Date'].max().date()}")
        #Ver si existe modelo para ese horizonte
        horizon_str = str(horizon_days)
        #print(self.models)
        if horizon_str not in self.models:
            raise ValueError(f"No hay modelos entrenados para horizonte {horizon_days} días")
        horizon_models = self.models[horizon_str]
        #print(horizon_models)
        # Inicializar predicciones
        predictions = {}
        #Ver si existen todos los modelos
        for appliance in self.appliances:
            #Verificar que existe modelo para este electrodoméstico
            if appliance not in horizon_models:
                #print(f"⚠ No hay modelo entrenado para {appliance}")
                continue
            #Extraer características del modelo
            model_info = horizon_models[appliance]
            model = model_info['model']
            features = model_info['features']
            #print(f"\nPrediciendo {appliance}")
            #print(f"Features requeridas: {features}")
            #Crear diccionario de features específico para este modelo
            features_dict = base_features_dict.copy()
            #Añadir features específicas del electrodoméstico
            for feature in features:
                #Caos en el que ya esté
                if feature in base_features_dict:
                    continue
                #Media móvil de 7 días
                elif '_rolling_7_mean' in feature:
                    appliance_name = feature.replace('_rolling_7_mean', '')
                    if appliance_name in self.data_aux.columns:
                        rolling_mean = self.data_aux[appliance_name].tail(7).mean()
                        features_dict[feature] = rolling_mean
                        #print(f" {feature} = {rolling_mean:.2f}")
                #Desviación estándar móvil de 7 días
                elif '_rolling_7_std' in feature:
                    appliance_name = feature.replace('_rolling_7_std', '')
                    if appliance_name in self.data_aux.columns:
                        rolling_std = self.data_aux[appliance_name].tail(7).std()
                        features_dict[feature] = rolling_std if not pd.isna(rolling_std) else 0
                        #print(f" {feature} = {rolling_std:.2f}")
                #Media móvil de 14 días
                elif '_rolling_14_mean' in feature:
                    appliance_name = feature.replace('_rolling_14_mean', '')
                    if appliance_name in self.data_aux.columns:
                        rolling_mean = self.data_aux[appliance_name].tail(14).mean()
                        features_dict[feature] = rolling_mean
                        #print(f"  {feature} = {rolling_mean:.2f}")
                #Rezagos
                elif '_lag_' in feature:
                    #Separar por _lag_
                    parts = feature.split('_lag_')
                    if len(parts) == 2:
                        #Hallar nombre electrodoméstico
                        appliance_name = parts[0]
                        #Hallar número de días de rezago
                        lag_days = int(parts[1])
                        if appliance_name in self.data_aux.columns and len(self.data_aux) >= lag_days:
                            lag_value = self.data_aux[appliance_name].iloc[-lag_days]
                            features_dict[feature] = lag_value
                            #print(f" {feature} = {lag_value:.2f}")
                        else:
                            print(f" No se puede obtener {feature}")

            #Verificar que todas las features requeridas están disponibles
            missing_features = [f for f in features if f not in features_dict]
            if missing_features:
                #print(f"⚠ Features faltantes para {appliance}: {missing_features}")
                #Rellenar con valores por defecto o promedio histórico
                for missing_feature in missing_features:
                    if missing_feature in self.data_aux.columns:
                        #Si es una columna del dataset, usar el último valor
                        features_dict[missing_feature] = self.data_aux[missing_feature].iloc[-1]
                        #print(
                            #f"Rellenado {missing_feature} con último valor: {features_dict[missing_feature]:.2f}")
                    else:
                        #Valor por defecto
                        features_dict[missing_feature] = 0
                        #print(f" Rellenado {missing_feature} con 0")

            #Preparar features para predicción en el orden correcto
            try:
                X_pred = pd.DataFrame([features_dict])[features]
                #print(f"Shape X_pred: {X_pred.shape}")
                #print(f"Valores: {X_pred.iloc[0].to_dict()}")

                #Realizar predicción
                prediction = model.predict(X_pred)[0]
                #Asegurar que la predicción no sea negativa, en ese caso 0
                predictions[appliance] = max(0, prediction)
                #print(f" Predicción {appliance}: {prediction:.2f} Wh")
            except Exception as e:
                #print(f"Error prediciendo {appliance}: {str(e)}")
                continue
        return {
            'date': target_date.date(),
            'horizon_days': horizon_days,
            'predictions': predictions,
            'estimated_temperature': historical_temp,
            'total_predicted_consumption': (predictions['Aggregate'])
        }

    def save_models(self, models_dir="models/", use_joblib=True):
        """
        Guardar modelos multi-horizonte y metadatos en archivos
        """
        os.makedirs(models_dir, exist_ok=True)

        print(f"Guardando modelos multi-horizonte en {models_dir}...")

        # Preparar metadatos
        metadata = {
            'training_date': datetime.now().isoformat(),
            'appliances': self.appliances,
            'horizons': self.horizons,
            'is_stable': self.is_stable,
            'recommendations': self.recommendations,
            'feature_importance': self.feature_importance,
            'evaluation_results': self.evaluation_results,
            'data_shape': self.data.shape,
            'data_date_range': {
                'start': self.data['Date'].min().isoformat(),
                'end': self.data['Date'].max().isoformat()
            },
            'model_structure': 'multi_horizon'  # Indicador de estructura
        }

        # Guardar metadatos
        with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # CAMBIO: Guardar modelos organizados por horizonte
        models_structure = {}

        for horizon in ['1', '2', '3', '4', '5', '6', '7']:
            horizon_dir = os.path.join(models_dir, f"horizon_{horizon}")
            os.makedirs(horizon_dir, exist_ok=True)

            horizon_models = {}

            # Verificar si tenemos modelos para este horizonte
            if horizon in self.evaluation_results:
                for appliance in self.appliances:
                    # Buscar el modelo en la estructura actual
                    model_info = None
                    if hasattr(self, 'models') and appliance in self.models[horizon]:
                        model_info = self.models[horizon][appliance]
                    elif hasattr(self, f'models_{horizon}d') and appliance in getattr(self, f'models_{horizon}d'):
                        model_info = getattr(self, f'models_{horizon}d')[appliance]

                    if model_info:
                        # Guardar modelo individual
                        model_filename = f"model_{appliance.replace(' ', '_')}_day_{horizon}.pkl"
                        model_path = os.path.join(horizon_dir, model_filename)

                        if use_joblib:
                            joblib.dump(model_info['model'], model_path)
                        else:
                            with open(model_path, 'wb') as f:
                                pickle.dump(model_info['model'], f)

                        # Guardar info del modelo
                        horizon_models[appliance] = {
                            'features': model_info['features'],
                            'model_name': model_info['model_name'],
                            'model_file': model_filename,
                            'horizon': horizon
                        }

            models_structure[f'horizon_{horizon}'] = horizon_models

        # Guardar estructura completa de modelos
        with open(os.path.join(models_dir, 'models_structure.json'), 'w') as f:
            json.dump(models_structure, f, indent=2)

        # Guardar datos auxiliares
        self.data_aux.to_csv(os.path.join(models_dir, 'data_aux.csv'), index=False)

        total_models = sum(len(horizon_models) for horizon_models in models_structure.values())
        print(f"✅ Modelos multi-horizonte guardados:")
        print(f"   - {total_models} modelos totales")
        print(f"   - {len(self.appliances)} electrodomésticos")
        print(f"   - {len(self.horizons)} horizontes (1-7 días)")

        return models_dir

    def load_models(self, models_dir="models/", use_joblib=True):
        """
        Cargar modelos multi-horizonte desde archivos
        """
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {models_dir}")

        #print(f"Cargando modelos multi-horizonte desde {models_dir}...")

        # Cargar metadatos
        metadata_path = os.path.join(models_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.appliances = metadata['appliances']
            self.horizons = metadata['horizons']
            self.is_stable = metadata['is_stable']
            self.recommendations = metadata['recommendations']
            self.feature_importance = metadata['feature_importance']
            self.evaluation_results = metadata['evaluation_results']
            self.models_metadata = metadata

            #print(f" Metadatos cargados (entrenado: {metadata['training_date']})")

        # Cargar estructura de modelos
        models_structure_path = os.path.join(models_dir, 'models_structure.json')
        with open(models_structure_path, 'r') as f:
            models_structure = json.load(f)

        # CAMBIO: Cargar modelos por horizonte
        self.models = {}
        total_loaded = 0

        for horizon_key, horizon_models in models_structure.items():
            horizon = horizon_key.replace('horizon_', '')
            horizon_dir = os.path.join(models_dir, horizon_key)

            self.models[horizon] = {}

            for appliance, info in horizon_models.items():
                model_path = os.path.join(horizon_dir, info['model_file'])

                if os.path.exists(model_path):
                    if use_joblib:
                        model = joblib.load(model_path)
                    else:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)

                    self.models[horizon][appliance] = {
                        'model': model,
                        'features': info['features'],
                        'model_name': info['model_name'],
                        'horizon': horizon
                    }
                    total_loaded += 1

        # Cargar datos auxiliares
        data_aux_path = os.path.join(models_dir, 'data_aux.csv')
        if os.path.exists(data_aux_path):
            self.data_aux = pd.read_csv(data_aux_path)
            self.data_aux['Date'] = pd.to_datetime(self.data_aux['Date'])
            #print(f" Datos auxiliares cargados: {len(self.data_aux)} registros")

        self.models_trained = True
        #print(f" {total_loaded} modelos cargados exitosamente")
        #print(f" Horizontes disponibles: {list(self.models.keys())}")

        return True


    def get_model_info(self):
        """Obtener información sobre los modelos cargados"""
        if not self.models_trained:
            return {"error": "No hay modelos cargados"}

        info = {
            "models_loaded": len(self.models),
            "appliances": self.appliances,
            "training_date": self.models_metadata.get('training_date', 'Unknown'),
            "data_range": self.models_metadata.get('data_date_range', {}),
            "model_details": {}
        }

        for appliance, model_info in self.models.items():
            info["model_details"][appliance] = {
                "model_type": model_info['model_name'],
                "features_count": len(model_info['features']),
                "features": model_info['features']
            }

            # Añadir métricas si están disponibles
            if appliance in self.evaluation_results.get('1', {}):
                metrics = self.evaluation_results['1'][appliance]
                info["model_details"][appliance]["metrics"] = {
                    "mae": round(metrics.get('mae', 0), 2),
                    "rmse": round(metrics.get('rmse', 0), 2),
                    "r2": round(metrics.get('r2', 0), 3)
                }

        return info

    def train_and_save_models(self, models_dir="models/", test_size=0.2):
        """
        Entrenar modelos y guardarlos automáticamente
        """
        print("Entrenando y guardando modelos...")

        # Entrenar modelos
        results = self.train_models(test_size=test_size)

        # Guardar automáticamente
        saved_path = self.save_models(models_dir=models_dir)

        print(f"Proceso completado. Modelos disponibles en: {saved_path}")
        return results, saved_path

    def predict_with_loaded_models(self, target_date):
        """
        Predecir usando modelos cargados (versión optimizada para MCP)
        """
        if not self.models_trained:
            raise ValueError("No hay modelos cargados. Usa load_models() primero.")

        return self.predict_future_date(target_date)


    def evaluate_models(self):
        """Mostrar evaluación detallada de todos los modelos"""
        print("\nEVALUACIÓN DE MODELOS")
        print("=" * 80)

        for horizon in self.horizons:
            print(f"\nHORIZONTE {horizon} DÍAS")
            print("-" * 50)

            results_df = pd.DataFrame(self.evaluation_results[f'{horizon}']).T
            results_df = results_df.round(3)
            print(results_df.to_string())

            # Mejores modelos por métrica
            print(f"\n🏆 Mejores modelos ({horizon}):")
            print(f"   Menor MAE: {results_df['mae'].idxmin()} (MAE: {results_df['mae'].min():.3f})")
            print(f"   Mayor R²: {results_df['r2'].idxmax()} (R²: {results_df['r2'].max():.3f})")

        print(self.evaluation_results)


    def generate_tables_by_appliance(self):
        """
        Returns a dictionary where each key is an appliance name,
        and each value is a DataFrame with the metrics for each day.
        """
        appliances = set()
        for day_data in self.evaluation_results.values():
            appliances.update(day_data.keys())

        tables_by_appliance = {}

        for appliance in appliances:
            rows = []
            for day_str, day_data in self.evaluation_results.items():
                day = int(day_str)
                metrics = day_data[appliance]
                rows.append({
                    'day': day,
                    'mae': metrics.get('mae'),
                    'rmse': metrics.get('rmse'),
                    'r2': metrics.get('r2'),
                    'cv_score': metrics.get('cv_score'),
                    'model': metrics.get('model_name')
                })
                table = pd.DataFrame(rows).sort_values('day')
                tables_by_appliance[appliance] = table
        return tables_by_appliance

    def plot_mae_by_day(self, tables_by_appliance):
        """
        Generates a line plot of MAE by day for each appliance.
        """
        plt.figure(figsize=(10, 6))

        for appliance, table in tables_by_appliance.items():
            plt.plot(table['day'], table['mae'], marker='o', label=appliance)

        plt.title('MAE por día y electrodoméstico')
        plt.xlabel('Día')
        plt.ylabel('MAE')
        plt.xticks(range(1, 8))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def restructure_tables_by_metric(self, tables_by_appliance):
        """
        Restructura las tablas por métrica para representarlas
        """
        maes = {}
        rmses = {}
        r2s = {}

        for appliance, df in tables_by_appliance.items():
            df_sorted = df.sort_values('day')  # Asegurar orden
            maes[appliance] = df_sorted['mae'].values
            rmses[appliance] = df_sorted['rmse'].values
            r2s[appliance] = df_sorted['r2'].values

        days = df_sorted['day'].values
        mae_df = pd.DataFrame(maes, index=days)
        rmse_df = pd.DataFrame(rmses, index=days)
        r2_df = pd.DataFrame(r2s, index=days)

        return mae_df, rmse_df, r2_df

    def plot_metric_subplots(self, mae_table, rmse_table, r2_table):
        '''
        Representa MAE, RMSE y R² en subgráficas
        '''
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        metric_tables = {
            'MAE': mae_table,
            'RMSE': rmse_table,
            'R²': r2_table
        }

        for ax, (metric_name, table) in zip(axes, metric_tables.items()):
            for appliance in table.columns:
                ax.plot(table.index, table[appliance], marker='o', label=appliance)

            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} por día')
            ax.grid(True)
            ax.legend(loc='best', fontsize='small')

        axes[-1].set_xlabel('Día')

        plt.tight_layout()
        plt.show()

    def plot_metrics_per_appliance(self, tables_by_appliance):
        '''
        Genera gráficos de líneas para MAE, RMSE y R² por día para cada electrodoméstico.
        '''
        for appliance, df in tables_by_appliance.items():
            df = df.sort_values('day')
            days = df['day']

            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'Métricas para {appliance}', fontsize=16)

            axs[0].plot(days, df['mae'], marker='o', color='blue')
            axs[0].set_ylabel('MAE')
            axs[0].grid(True)

            axs[1].plot(days, df['rmse'], marker='o', color='green')
            axs[1].set_ylabel('RMSE')
            axs[1].grid(True)

            axs[2].plot(days, df['r2'], marker='o', color='red')
            axs[2].set_ylabel('R2')
            axs[2].set_xlabel('Día')
            axs[2].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_feature_importance(self, appliance, horizon='1d', top_n=10):
        """Visualizar importancia de features"""
        key = f'{appliance}_{horizon}'
        if key not in self.feature_importance:
            print(f"No hay datos de importancia para {appliance} en horizonte {horizon}")
            return

        importance = self.feature_importance[key]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        features, values = zip(*sorted_features)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importancia')
        plt.title(f'Feature Importance - {appliance} ({horizon})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def batch_predict_future(self, start_date, end_date):
        """Predecir para un rango de fechas futuras"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        predictions_list = []

        for date in date_range:
            pred = self.predict_future_date(date)
            predictions_list.append(pred)

        return predictions_list


'''
Main para pruebas, entrenar modelos,...
'''
if __name__ == "__main__":
    predictor = EnergyConsumptionPredictor()
    predictor.preprocess_data()
    predictor.train_models()
    print(predictor.shap_importance)
    os.makedirs('shap_outputs', exist_ok=True)
    with open('shap_outputs/all_shap_importance.json', 'w') as f:
        json.dump(predictor.shap_importance, f, indent=4)
    #predictor.evaluate_models()
    print(predictor.predict_future_date("2015-07-19"))

    tables = predictor.generate_tables_by_appliance()
    mae_df, rmse_df, r2_df = predictor.restructure_tables_by_metric(tables)
    predictor.plot_mae_by_day(tables)
