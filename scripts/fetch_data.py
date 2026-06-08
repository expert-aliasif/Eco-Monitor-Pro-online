import requests
import pandas as pd
import time
import os

# The cities we actually care about
CITIES = {
    "Toba Tek Singh": {"lat": 30.9713, "lon": 72.4827},  # My hometown — gotta lead with this
    "Lahore": {"lat": 31.5497, "lon": 74.3436},
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Rajana": {"lat": 30.8252, "lon": 72.5694},
    "Pirmahal": {"lat": 30.7675, "lon": 72.4347},
    "Kamalia": {"lat": 30.7258, "lon": 72.6447}
}

# Parameters for Open-Meteo Historical API
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

def fetch_weather_data(city: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch historical daily weather data from Open-Meteo.
    
    Args:
        city: Name of the city
        lat: Latitude
        lon: Longitude
        
    Returns:
        pd.DataFrame: DataFrame containing daily historical features
    """
    print(f"Fetching data for {city}...")
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&timezone=auto"
    
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    
    if "daily" not in data:
        print(f"No daily data found for {city}")
        return pd.DataFrame()
        
    df = pd.DataFrame(data["daily"])
    
    if "windspeed_10m_max" in df.columns:
        df.rename(columns={"windspeed_10m_max": "wind_speed_10m_max"}, inplace=True)
        
    df["city"] = city
    return df

def main():
    """Main execution function to fetch all city data and save to CSV."""
    all_data = []
    
    os.makedirs("data", exist_ok=True)
    
    for city, coords in CITIES.items():
        df = fetch_weather_data(city, coords["lat"], coords["lon"])
        if not df.empty:
            all_data.append(df)
        time.sleep(1) # playing nice with the free API so we don't get banned
        
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Rename columns to be cleaner
        final_df.rename(columns={
            "time": "date",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precipitation",
            "wind_speed_10m_max": "wind_speed"
        }, inplace=True)
        
        output_path = "data/city_daily_weather.csv"
        final_df.to_csv(output_path, index=False)
        print(f"✅ Data successfully saved to {output_path} ({len(final_df)} rows)")
    else:
        print("❌ Failed to compile dataset.")

if __name__ == "__main__":
    main()
