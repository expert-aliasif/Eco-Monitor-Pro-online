import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import json
import os

from utils.api import get_aqi, get_weather_forecast

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Eco-Monitor Pro", layout="wide", page_icon="🌍", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0.5rem !important;}
    
    div[data-testid="stAppViewContainer"] > div:first-child {
        padding-top: 0 !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }
        .stMarkdown p {
            font-size: 14px !important;
        }
        [data-testid="stPlotlyChart"] {
            width: 100% !important;
        }
        .main .block-container {
            padding: 1rem !important;
        }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- CUSTOM PREMIUM CSS & ANIMATIONS -----------------
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stApp { animation: fadeIn 0.8s ease-out; }
        [data-testid="metric-container"] {
            background: rgba(128, 128, 128, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.2);
            border: 1px solid rgba(0, 200, 255, 0.5);
        }
        h1, h2, h3 {
            background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            color: black;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            box-shadow: 0px 0px 15px rgba(0, 201, 255, 0.5);
            transform: scale(1.02);
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- CACHED API WRAPPERS -----------------


@st.cache_data(ttl=3600)
def fetch_weather_forecast(lat: float, lon: float) -> dict:
    return get_weather_forecast(lat, lon)

@st.cache_data(ttl=3600)
def fetch_aqi(lat: float, lon: float) -> dict:
    return get_aqi(lat, lon)

# ----------------- HELPER FUNCTIONS -----------------
def get_temp_color(temp: float) -> str:
    """picking colors based on how hot it is outside"""
    if temp >= 35: return "#FF3D00"
    elif temp >= 25: return "#FFEA00"
    elif temp >= 15: return "#00E676"
    else: return "#00C9FF"

def get_weather_icon(condition_desc: str) -> str:
    """giving the user a nice emoji instead of boring text"""
    desc = condition_desc.lower()
    if "rain" in desc or "drizzle" in desc: return "🌧️"
    elif "cloud" in desc: return "☁️"
    elif "clear" in desc or "sun" in desc: return "☀️"
    elif "snow" in desc: return "❄️"
    elif "thunder" in desc or "storm" in desc: return "⛈️"
    else: return "🌤️"

# ----------------- LOAD MODELS -----------------
@st.cache_resource
def load_models():
    """Load machine learning models from disk. Cached to prevent reload lag."""
    try:
        rainfall_model = pickle.load(open("model/model.pkl", "rb"))
        daily_model = pickle.load(open("model/daily_weather_model.pkl", "rb"))
        city_encoder = pickle.load(open("model/city_encoder.pkl", "rb"))
        return rainfall_model, daily_model, city_encoder
    except FileNotFoundError:
        st.error("Model files not found! Please ensure training scripts have been run in `scripts/`.")
        return None, None, None

rainfall_model, daily_model, city_encoder = load_models()

# ----------------- CONSTANTS -----------------
CITIES = {
    "Toba Tek Singh": {"lat": 30.9713, "lon": 72.4827},  # My hometown — lead with this!
    "Lahore": {"lat": 31.5497, "lon": 74.3436},
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Rajana": {"lat": 30.8252, "lon": 72.5694},
    "Pirmahal": {"lat": 30.7675, "lon": 72.4347},
    "Kamalia": {"lat": 30.7258, "lon": 72.6447}
}

# ----------------- REUSABLE COMPONENTS -----------------
def render_weather_forecast(city: str):
    """Renders the hybrid ML Delta vs API weather forecast."""
    coords = CITIES[city]
    with st.spinner(f"Pulling live weather data for {city}..."):
        forecast = fetch_weather_forecast(coords["lat"], coords["lon"])
        
        if str(forecast.get("cod")) == "200":
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_items = [item for item in forecast["list"] if item["dt_txt"].startswith(today_str)]
            
            if not today_items:
                today_items = [forecast["list"][0]]
                
            temp_max_today = max([item["main"]["temp_max"] for item in today_items])
            temp_min_today = min([item["main"]["temp_min"] for item in today_items])
            wind_speed_today = max([item["wind"]["speed"] for item in today_items])
            precip_today = sum([item.get("rain", {}).get("3h", 0) for item in today_items])
            
            tomorrow = datetime.now() + timedelta(days=1)
            month = tomorrow.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            day_of_year = tomorrow.timetuple().tm_yday
            
            try:
                city_enc = city_encoder.transform([city])[0]
                X_pred = np.array([[city_enc, month_sin, month_cos, day_of_year, temp_max_today, temp_min_today, precip_today, wind_speed_today]])
                
                # dumping data into the ML model to see what it thinks
                ml_delta = daily_model.predict(X_pred)[0]
                delta_max, delta_min, delta_precip = ml_delta
                
                # figuring out how confused the model is (confidence intervals)
                preds_max = np.array([tree.predict(X_pred)[0][0] for tree in daily_model.estimators_])
                preds_min = np.array([tree.predict(X_pred)[0][1] for tree in daily_model.estimators_])
                std_max = preds_max.std()
                std_min = preds_min.std()
                
                ml_temp_max = temp_max_today + delta_max
                ml_temp_min = temp_min_today + delta_min
                
                if ml_temp_max > 42:
                    st.warning("⚠️ Heatwave Alert: Predicted temperature exceeds 42°C. Stay hydrated and avoid outdoor exposure between 11am–4pm.")
                
                max_color = get_temp_color(ml_temp_max)
                min_color = get_temp_color(ml_temp_min)
                
                tomorrow_items = [item for item in forecast["list"] if item["dt_txt"].startswith(tomorrow.strftime("%Y-%m-%d"))]
                api_temp_max = max([item["main"]["temp_max"] for item in tomorrow_items]) if tomorrow_items else "N/A"
                condition_desc = tomorrow_items[0]["weather"][0]["main"] if tomorrow_items else "Clear"
                weather_icon = get_weather_icon(condition_desc)
                
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("### 🤖 ML Delta Prediction")
                    st.markdown(f"**Tomorrow's High:** <span style='font-size:2em; font-weight:bold; color:{max_color};'>{ml_temp_max:.1f} °C ± {std_max:.1f}°C {weather_icon}</span>", unsafe_allow_html=True)
                    st.caption(f"{delta_max:+.1f} °C from today's {temp_max_today}°C")
                    st.markdown(f"**Tomorrow's Low:** <span style='font-size:2em; font-weight:bold; color:{min_color};'>{ml_temp_min:.1f} °C ± {std_min:.1f}°C</span>", unsafe_allow_html=True)
                    st.caption(f"{delta_min:+.1f} °C from today's {temp_min_today}°C")
                    
                with res_col2:
                    st.markdown("### ☁️ OpenWeather Forecast")
                    api_color = get_temp_color(api_temp_max) if isinstance(api_temp_max, float) else "#FFF"
                    st.markdown(f"**Tomorrow's High:** <span style='font-size:2em; font-weight:bold; color:{api_color};'>{api_temp_max} °C {weather_icon}</span>", unsafe_allow_html=True)
                    st.caption("Commercial API")
                    
                st.markdown("---")
                st.markdown("### 📈 5-Day Live Weather Trend")
                df_trend = pd.DataFrame([
                    {"Time": pd.to_datetime(item["dt_txt"]), "Temperature": item["main"]["temp"]} 
                    for item in forecast["list"]
                ])
                fig = px.area(df_trend, x="Time", y="Temperature", color_discrete_sequence=['#00C9FF'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Export Functionality
                csv = df_trend.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download 5-Day Forecast as CSV",
                    data=csv,
                    file_name='5_day_forecast.csv',
                    mime='text/csv',
                )
                
            except ValueError as e:
                if "unseen" in str(e).lower() or "unrecognized" in str(e).lower() or "previously unseen" in str(e).lower():
                    st.error(f"'{city}' was not recognized by the model. Please retrain with this city included.")
                else:
                    st.error(f"Prediction error: {e}")
        else:
            st.error("API Connectivity Failure.")

def render_rainfall_simulation(year: int):
    """Renders the national long-term rainfall prediction."""
    if rainfall_model:
        with st.spinner("Running 12-month simulation..."):
            months = np.arange(1, 13)
            months_data = pd.DataFrame({
                "year": [year]*12,
                "month_sin": np.sin(2 * np.pi * months / 12),
                "month_cos": np.cos(2 * np.pi * months / 12)
            })
            
            predictions = rainfall_model.predict(months_data)
            predictions = [max(0, p) for p in predictions]
            
            total_rainfall = sum(predictions)
            max_month_idx = np.argmax(predictions)
            max_month_name = calendar.month_name[max_month_idx + 1]
            
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Total Expected Annual Rainfall", f"{total_rainfall:.2f} mm")
            mcol2.metric("Wettest Expected Month", max_month_name, f"{predictions[max_month_idx]:.2f} mm")
                
            st.markdown("---")
            st.markdown("### 📊 Annual Rainfall Distribution")
            df_rain = pd.DataFrame({
                "Month": list(calendar.month_abbr)[1:],
                "Rainfall (mm)": predictions
            })
            fig = px.bar(df_rain, x="Month", y="Rainfall (mm)", color="Rainfall (mm)", color_continuous_scale="Blues", text_auto='.1f')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=40, b=60)
            )
            fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

def render_aqi_radar(city: str):
    """Renders the Air Quality Index radar chart and stats."""
    coords = CITIES[city]
    with st.spinner("Analyzing atmospheric composition..."):
        data = fetch_aqi(coords["lat"], coords["lon"])
        
        if "list" in data:
            aqi_val = data['list'][0]['main']['aqi']
            components = data['list'][0]['components']
            
            aqi_info = {
                1: ("Excellent", "rgba(0, 230, 118, 0.6)", "#00E676"),
                2: ("Fair", "rgba(255, 234, 0, 0.6)", "#FFEA00"),
                3: ("Moderate", "rgba(255, 145, 0, 0.6)", "#FF9100"),
                4: ("Poor", "rgba(255, 61, 0, 0.6)", "#FF3D00"),
                5: ("Hazardous", "rgba(213, 0, 0, 0.6)", "#D50000")
            }
            label, fill_color, text_color = aqi_info.get(aqi_val, ("Unknown", "rgba(128, 128, 128, 0.5)", "#FFF"))
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"### Status: <span style='color:{text_color}'>{label}</span>", unsafe_allow_html=True)
                with st.expander("🔬 Pollutant Lexicon"):
                    st.markdown("""**PM2.5**: Deep lung penetration hazard.  \n**PM10**: Respiratory tract irritant.  \n**O3**: Causes shortness of breath.  \n**NO2**: Auto-exhaust pollutant.  \n**SO2**: Industrial smog component.  \n**CO**: Reduces blood oxygen.""")
            with col_b:
                categories = list(components.keys())
                values = list(components.values())
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', fillcolor=fill_color, line=dict(color=text_color)))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            tickfont=dict(size=10, color="black"),
                            tickangle=45,
                            gridcolor="rgba(0,0,0,0.2)",
                            linecolor="rgba(0,0,0,0.2)"
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=12, color="white")
                        ),
                        bgcolor="white"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400,
                    margin=dict(l=60, r=60, t=60, b=60)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Atmospheric scan failed.")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<h1>🌍 Eco-Monitor</h1>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Dashboard Home", "🌤️ Weather Engine", "🌧️ Climate Rainfall", "🌫️ AQI Radar", "📈 Data & EDA"], label_visibility="collapsed")
    st.markdown("---")
    with st.expander("📊 Model Info & Academic Metrics"):
        try:
            with open("model/metrics.json", "r") as f:
                metrics = json.load(f)
            st.markdown(f"**Daily Model MAE**: {metrics['daily_model']['mae']} °C")
            st.markdown(f"**Naive Baseline MAE**: {metrics['daily_model'].get('baseline_mae', 'N/A')} °C")
            st.markdown(f"**Daily Model R²**: {metrics['daily_model']['r2_score']}")
            st.markdown(f"**Data Span**: {metrics['dataset']['date_range']}")
            st.markdown(f"**Samples**: {metrics['dataset']['total_samples']}")
            st.markdown(f"**Split**: {metrics['dataset'].get('split', 'N/A')}")
        except FileNotFoundError:
            st.caption("Metrics not available.")

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <p style='color: gray; font-size: 0.8em; margin-bottom: 4px;'>Developed by</p>
            <p style='font-weight: bold; font-size: 1.1em; margin: 0;'>Ali Asif</p>
            <p style='color: gray; font-size: 0.8em; margin: 4px 0;'>BS Computer Science | FYP 2026</p>
            <div style='margin-top: 8px;'>
                <a href='mailto:Expert.aliasif@gmail.com' style='color: #00C9FF; text-decoration: none; font-size: 0.8em;'>📧 Email</a>
                &nbsp;|&nbsp;
                <a href='https://www.linkedin.com/in/expert-ali-asif/' target='_blank' style='color: #00C9FF; text-decoration: none; font-size: 0.8em;'>💼 LinkedIn</a>
                &nbsp;|&nbsp;
                <a href='https://github.com/expert-aliasif' target='_blank' style='color: #00C9FF; text-decoration: none; font-size: 0.8em;'>🐙 GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- PAGES -----------------
if page == "🏠 Dashboard Home":
    st.markdown("<h1>Welcome to Eco-Monitor Pro</h1>", unsafe_allow_html=True)
    st.markdown("An advanced, machine-learning-driven environmental forecasting system built for absolute precision.")
    
    st.markdown("---")
    dash_col1, dash_col2 = st.columns(2)
    with dash_col1:
        dash_city = st.selectbox("Dashboard Target City", list(CITIES.keys()), index=0)
    with dash_col2:
        dash_year = st.number_input("Dashboard Target Year", min_value=1900, max_value=2100, value=datetime.now().year, step=1)
        
    st.markdown(f"## 🌤️ Hybrid Weather Engine ({dash_city})")
    render_weather_forecast(dash_city)
    
    st.markdown(f"## 🌧️ Climate AI ({dash_year})")
    render_rainfall_simulation(dash_year)
    
    st.markdown(f"## 🌫️ AQI Radar ({dash_city})")
    render_aqi_radar(dash_city)

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px; color: gray; font-size: 0.85em;'>
            Eco-Monitor Pro &nbsp;|&nbsp; Developed by <strong>Ali Asif</strong> &nbsp;|&nbsp; BS CS FYP 2026<br>
            <a href='mailto:Expert.aliasif@gmail.com' style='color: #00C9FF; text-decoration: none;'>Expert.aliasif@gmail.com</a>
            &nbsp;|&nbsp;
            <a href='https://www.linkedin.com/in/expert-ali-asif/' target='_blank' style='color: #00C9FF; text-decoration: none;'>LinkedIn</a>
            &nbsp;|&nbsp;
            <a href='https://github.com/expert-aliasif' target='_blank' style='color: #00C9FF; text-decoration: none;'>GitHub</a>
        </div>
    """, unsafe_allow_html=True)

elif page == "🌤️ Weather Engine":
    st.markdown("<h1>Hybrid Weather Forecaster</h1>", unsafe_allow_html=True)
    city = st.selectbox("Select Target City", list(CITIES.keys()))
    if st.button("Predict Tomorrow's Weather", use_container_width=True):
        render_weather_forecast(city)

elif page == "🌧️ Climate Rainfall":
    st.markdown("<h1>Historical Climate Prediction</h1>", unsafe_allow_html=True)
    st.info("Note: Rainfall model uses National Level Aggregated Data (1901-2016). Output applies nationwide.")
    year = st.number_input("Select Target Year", min_value=1900, max_value=2100, value=datetime.now().year, step=1)
    if st.button("Show Yearly Rainfall Forecast", use_container_width=True):
        render_rainfall_simulation(year)

elif page == "🌫️ AQI Radar":
    st.markdown("<h1>Environmental Hazard Radar</h1>", unsafe_allow_html=True)
    city = st.selectbox("Select Target Region", list(CITIES.keys()))
    if st.button("Check Air Quality", use_container_width=True):
        render_aqi_radar(city)

elif page == "📈 Data & EDA":
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    # 1. Map Visualization
    st.markdown("### 🗺️ Target Network Map")
    map_data = []
    for c, coords in CITIES.items():
        map_data.append({"City": c, "lat": coords["lat"], "lon": coords["lon"]})
    df_map = pd.DataFrame(map_data)
    fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", hover_name="City", zoom=5, height=400)
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        # 2. Raw Data View
        st.markdown("### 📁 Raw Historical Data")
        if os.path.exists("data/city_daily_weather.csv"):
            df_hist = pd.read_csv("data/city_daily_weather.csv")
            st.dataframe(df_hist.tail(100), height=300)
        else:
            st.warning("Data file not generated yet.")
            
    with col2:
        # 3. Correlation Heatmap
        st.markdown("### 📉 Feature Correlation")
        if os.path.exists("data/city_daily_weather.csv"):
            corr = df_hist[['temp_max', 'temp_min', 'precipitation', 'wind_speed']].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)
            
    st.markdown("---")
    # 4. Feature Importance
    st.markdown("### 🧠 Model Feature Importance")
    if os.path.exists("model/feature_importances.json"):
        with open("model/feature_importances.json", "r") as f:
            fi = json.load(f)
        df_fi = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        df_fi = df_fi.sort_values(by="Importance", ascending=True)
        fig_fi = px.bar(df_fi, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis")
        fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Train the daily model to view Feature Importances.")