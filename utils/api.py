import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def _fetch_with_offline_cache(url: str, cache_filename: str) -> dict:
    cache_path = os.path.join("data", cache_filename)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Save to cache for offline use
        os.makedirs("data", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
            
        return data
    except requests.exceptions.RequestException:
        # If offline or API fails, try to load from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

def get_aqi(lat: float, lon: float) -> dict:
    """
    grabs the air quality so we know if it's safe to breathe outside
    
    Args:
        lat: Latitude of the target location
        lon: Longitude of the target location
    
    Returns:
        dict: AQI data including main AQI value and pollutant components
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    return _fetch_with_offline_cache(url, f"aqi_{lat}_{lon}.json")

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
    return _fetch_with_offline_cache(url, f"current_weather_{lat}_{lon}.json")

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
    return _fetch_with_offline_cache(url, f"forecast_{lat}_{lon}.json")
