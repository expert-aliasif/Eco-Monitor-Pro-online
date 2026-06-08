import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_aqi(lat: float, lon: float) -> dict:
    """
    grabs the air quality so we know if it's safe to breathe outside
    
    Args:
        lat: Latitude of the target location
        lon: Longitude of the target location
    
    Returns:
        dict: AQI data including main AQI value and pollutant components
    
    Raises:
        requests.HTTPError: If the API returns a non-200 status
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_current_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather for given coordinates.
    
    Args:
        lat: Latitude of the target location
        lon: Longitude of the target location
    
    Returns:
        dict: Current weather data
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_weather_forecast(lat: float, lon: float) -> dict:
    """
    Fetch 5-day / 3-hour forecast for given coordinates.
    
    Args:
        lat: Latitude of the target location
        lon: Longitude of the target location
    
    Returns:
        dict: 5-day forecast data
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_open_meteo_daily(lat: float, lon: float) -> dict:
    """
    pulling today's high and low temps so the ML model has a starting point.
    
    Args:
        lat: Latitude of the target location
        lon: Longitude of the target location
    
    Returns:
        dict: Daily weather summaries, renamed feature keys for compatibility.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&timezone=auto"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Map windspeed_10m_max to wind_speed_10m_max for downstream consistency
        if "daily" in data and "windspeed_10m_max" in data["daily"]:
            data["daily"]["wind_speed_10m_max"] = data["daily"].pop("windspeed_10m_max")
            
        return data
    except requests.exceptions.RequestException:
        # Return empty dict if API fails (e.g. 429 Too Many Requests)
        return {}
