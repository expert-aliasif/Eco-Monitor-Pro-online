import pandas as pd
import json

file_path = r"C:\Users\Kratos\.gemini\antigravity\brain\1e45b6ab-9d28-4786-a7b4-6d14c01514bb\.system_generated\steps\83\content.md"
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

json_str = lines[4].strip()
data = json.loads(json_str)

df = pd.DataFrame(data["daily"])
df.rename(columns={
    "time": "date",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "precipitation",
    "windspeed_10m_max": "wind_speed"
}, inplace=True)
df["city"] = "Toba Tek Singh"

output_path = "data/city_daily_weather.csv"
existing_df = pd.read_csv(output_path)

# Drop any existing "Toba Tek Singh" just in case
existing_df = existing_df[existing_df["city"] != "Toba Tek Singh"]

final_df = pd.concat([existing_df, df], ignore_index=True)
final_df.to_csv(output_path, index=False)
print("Successfully appended Toba Tek Singh data!")
