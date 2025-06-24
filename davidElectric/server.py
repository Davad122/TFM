from typing import Any, Dict, List, Optional
import httpx
from mcp.server.fastmcp import FastMCP
import os
import json
import pandas as pd
from davidElectric.modelos_v3 import EnergyConsumptionPredictor
import pickle
from datetime import datetime, timedelta

import warnings
import pandas as pd

warnings.filterwarnings('ignore')

mcp = FastMCP("DavElectric")

BASE_URL = "https://api.esios.ree.es"
ESIOS_API_TOKEN = os.getenv("ESIOS_API_TOKEN")


#Instancia global del predictor
df = pd.read_csv("data/House_1_pre.csv")
predictor = EnergyConsumptionPredictor()
with open("shap_outputs/all_shap_importance.json", "r") as f:
    shaps = json.load(f)


async def make_get_request(url: str) -> dict[str, Any] | None:
    """Realiza petición GET a la API de ESIOS"""
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": ESIOS_API_TOKEN,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            if response.status_code == 200:
                return response.json()
            else:
                return f"Error HTTP {response.status_code}: {response.text}"
        except httpx.HTTPStatusError as e:
            return (f"Error HTTP: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            return (f"Error de conexión: {e}")
        except Exception as e:
            return (f"Error inesperado: {e}, {ESIOS_API_TOKEN}")
        except Exception:
            return None


@mcp.tool()
async def get_consumption_analysis(init_date: str, end_date: str) -> dict[str, Any]:
    """
    Analyze the energy consumption between two dates, generate an artifact and give the user some tips for optimization.
    Args:
    :param init_date: Start date (format: 2025-01-01)
    :param end_date: End date (format: 2025-01-01)
    """
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = pd.to_datetime(init_date)
    end_date = pd.to_datetime(end_date)
    if(start_date> end_date):
        return {"error": "La fecha de inicio debe ser anterior a la fecha de fin."}
    elif(start_date < df['Date'].min() or end_date > df['Date'].max()):
        return {"error": "Las fechas deben estar dentro del rango de datos disponibles."}
    else:
        filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        return filtered.to_dict(orient='records')


@mcp.tool()
async def predict_consumption(init_date: str, end_date: str) -> dict[str, Any]:
    """
    Predict energy consumption between two dates. If the user only spicifies one date, init_date and end_date are the same. Predictions are in Wh.
    Create an artifact with the predictions and give the user some tips for optimization.

    Args:
        init_date: Start date (format: 2025-01-01)
        end_date: End date (format: 2025-01-01)
    """
    df_predict = pd.read_csv("models/data_aux.csv")
    df_predict['Date'] = pd.to_datetime(df_predict['Date'])
    start_date = pd.to_datetime(init_date)
    end_date = pd.to_datetime(end_date)

    if(start_date > end_date):
        return {"error": "La fecha de inicio debe ser anterior a la fecha de fin."}
    elif(start_date < df_predict['Date'].max()):
        return {"error": "Las fechas deben ser futuras para predecir, debe usar get_consumption_analysis."}

    predictor.load_models()
    if (start_date == end_date):
        prediction = predictor.predict_future_date(start_date)
    else:
        prediction = predictor.batch_predict_future(start_date, end_date)
    return prediction


@mcp.tool()
async def get_precio(init_date: str, end_date: str, price_type: str = "actual") -> dict[str, Any]:
    """
    Get electricity prices from ESIOS API for specified dates.

    Args:
        init_date: Start date (format: 2025-01-01)
        end_date: End date (format: 2025-01-01)
        price_type: Type of price to get:
            - "actual": Real prices (PVPC) - indicator 1001
            - "market": Market prices (OMIE) - indicator 600
            - "forecast": Price forecasts when available - indicator 10209
            - "pvpc": PVPC consumer prices - indicator 1001
            - "marginal": Marginal system price - indicator 805
    """

    # Mapeo de tipos de precio a indicadores ESIOS
    price_indicators = {
        "actual": 1001,  # PVPC - Precio voluntario pequeño consumidor
        "pvpc": 1001,  # Mismo que actual
        "market": 600,  # Precio mercado diario OMIE
        "forecast": 10209,  # Previsión de precios (cuando está disponible)
        "marginal": 805,  # Precio marginal del sistema
    }

    indicator = price_indicators.get(price_type, 1001)

    url = (
        f"{BASE_URL}/indicators/{indicator}"
        f"?locale=es&date_from={init_date}T00:00:00Z&date_to={end_date}T23:59:59Z"
    )

    response = await make_get_request(url)

    if isinstance(response, dict):
        # Añadir información sobre el tipo de precio
        response["price_type"] = price_type
        response["indicator_id"] = indicator
        response["note"] = {
            "actual": "Precios reales PVPC",
            "pvpc": "Precios reales PVPC",
            "market": "Precios mercado OMIE",
            "forecast": "Previsión de precios",
            "marginal": "Precio marginal del sistema"
        }.get(price_type, "Precio no especificado")

    return response


@mcp.tool()
async def get_precio_inteligente(target_date: str) -> dict[str, Any]:
    """
    Get electricity price with intelligent fallback for future dates.

    Args:
        target_date: Target date for price (format: 2025-01-01)
    """

    try:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        today = datetime.now()

        # Si es fecha pasada o hoy, usar precios reales
        if target <= today:
            price_data = await get_precio(target_date, target_date, "actual")

            if isinstance(price_data, dict) and "indicator" in price_data:
                return {
                    "date": target_date,
                    "price_data": price_data,
                    "method": "real_prices",
                    "confidence": "high"
                }

        # Para fechas futuras, intentar previsión
        forecast_data = await get_precio(target_date, target_date, "forecast")

        if isinstance(forecast_data, dict) and "indicator" in forecast_data and forecast_data["indicator"]["values"]:
            return {
                "date": target_date,
                "price_data": forecast_data,
                "method": "forecast",
                "confidence": "medium"
            }

        # Fallback: estimar basándose en histórico reciente
        # Usar últimos 7 días disponibles
        historical_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        historical_start = (today - timedelta(days=8)).strftime("%Y-%m-%d")

        historical_data = await get_precio(historical_start, historical_end, "actual")

        if isinstance(historical_data, dict) and "indicator" in historical_data:
            values = historical_data["indicator"]["values"]
            if values:
                # Calcular precio promedio y patrones
                prices = [float(v["value"]) for v in values if v["value"]]
                avg_price = sum(prices) / len(prices)

                # Aplicar factor estacional simple (opcional)
                day_of_week = target.weekday()  # 0=Lunes, 6=Domingo
                weekend_factor = 0.95 if day_of_week >= 5 else 1.0  # Fines de semana más baratos

                estimated_price = avg_price * weekend_factor

                return {
                    "date": target_date,
                    "estimated_price_eur_mwh": round(estimated_price, 2),
                    "estimated_price_eur_kwh": round(estimated_price / 1000, 4),
                    "method": "historical_estimation",
                    "confidence": "low",
                    "note": f"Estimación basada en promedio de últimos 7 días ({len(prices)} valores)",
                    "historical_period": f"{historical_start} to {historical_end}",
                    "weekend_adjustment": weekend_factor != 1.0
                }

        return {
            "date": target_date,
            "error": "No se pudo obtener ni estimar el precio para esta fecha",
            "recommendation": "Use fechas más cercanas al presente"
        }

    except Exception as e:
        return {"error": f"Error obteniendo precio inteligente: {e}"}



@mcp.tool()
async def explain_predictions(appliance: str, horizon: str = '7') -> Dict[str, Any]:
    """
    Explain the predictions for a specific appliance over a given horizon. You receive the SHAP values for the appliance in the horizon.
    Your mission is to explain the user which ones are the most influential values, do not show the numbers.

    Args:
        appliance: Name of the appliance to explain (e.g., "Fridge", "Washing_Machine", etc.)
        horizon: Number of days to explain (default is 7)
    """
    # Load the model and explain predictions
    shap_explainer = appliance+'_'+horizon
    appliances = list(df.columns)
    appliances.remove('Date')
    if appliance not in appliances:
        return {"error": f" '{appliance}' not available. The available ones are {appliances}."}
    return shaps[shap_explainer]


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')