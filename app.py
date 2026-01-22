"""
TrafficSafe AI – Intelligent Road Risk & Design Optimization Platform
A production-grade traffic safety analysis platform with ML-powered insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import random

from utils import (
    load_data, clean_data, extract_time_features,
    encode_categorical_features, calculate_risk_index,
    get_risk_level, identify_hotspots,
    auto_optimize_data, get_global_data_sources
)
from train_model import train_and_evaluate_models
from advanced_ml import TrafficSafeML, get_priority_recommendations
from traffic_api import TomTomTrafficAPI, MAJOR_CITIES, get_demo_traffic_data

st.set_page_config(
    page_title="TrafficSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    * {
        font-family: 'Rajdhani', sans-serif;
    }
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00fff9 0%, #ff00ff 50%, #fffc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 30px rgba(0,255,249,0.5);
        letter-spacing: 2px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #00fff9;
        margin-bottom: 1.5rem;
        font-weight: 400;
        text-shadow: 0 0 10px rgba(0,255,249,0.3);
    }
    
    .section-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: #ff00ff;
        margin-bottom: 15px;
        text-shadow: 0 0 15px rgba(255,0,255,0.5);
        letter-spacing: 1px;
    }
    
    .view-all {
        font-size: 0.8rem;
        color: #00fff9;
        cursor: pointer;
        text-shadow: 0 0 5px rgba(0,255,249,0.5);
    }
    
    .overview-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0,255,249,0.2), inset 0 0 20px rgba(0,0,0,0.5);
        border: 1px solid rgba(0,255,249,0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .overview-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00fff9, transparent);
    }
    
    .circular-progress {
        position: relative;
        width: 80px;
        height: 80px;
        margin: 0 auto 10px auto;
    }
    
    .circular-progress svg {
        transform: rotate(-90deg);
        filter: drop-shadow(0 0 8px currentColor);
    }
    
    .circular-progress .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-family: 'Orbitron', monospace;
        font-size: 1rem;
        font-weight: 700;
        text-shadow: 0 0 10px currentColor;
    }
    
    .stat-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: #fff;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
    }
    
    .stat-highlight {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 8px;
        text-shadow: 0 0 8px currentColor;
    }
    
    .stat-red { color: #ff0040; text-shadow: 0 0 10px rgba(255,0,64,0.8); }
    .stat-green { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.8); }
    .stat-orange { color: #fffc00; text-shadow: 0 0 10px rgba(255,252,0,0.8); }
    .stat-purple { color: #ff00ff; text-shadow: 0 0 10px rgba(255,0,255,0.8); }
    
    .road-card {
        background: linear-gradient(135deg, rgba(0,0,0,0.8) 0%, rgba(20,20,35,0.9) 100%);
        border-radius: 10px;
        padding: 12px;
        color: white;
        min-height: 100px;
        display: flex;
        align-items: flex-start;
        border: 1px solid rgba(255,0,255,0.3);
        box-shadow: 0 0 15px rgba(255,0,255,0.2);
    }
    
    .road-card-title {
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        background: rgba(0,255,249,0.2);
        color: #00fff9;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid rgba(0,255,249,0.5);
    }
    
    .status-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 20px rgba(255,0,255,0.15);
        border: 1px solid rgba(255,0,255,0.3);
        margin-bottom: 15px;
    }
    
    .donut-legend {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.6);
    }
    
    .donut-legend-value {
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        text-shadow: 0 0 8px currentColor;
    }
    
    .toggle-btn {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        cursor: pointer;
        border: none;
    }
    
    .toggle-active {
        background: linear-gradient(90deg, #ff00ff, #00fff9);
        color: #0a0a0f;
    }
    
    .toggle-inactive {
        background: rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.5);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .chart-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 25px rgba(0,255,249,0.1);
        border: 1px solid rgba(0,255,249,0.2);
    }
    
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        margin-right: 15px;
    }
    
    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        box-shadow: 0 0 8px currentColor;
    }
    
    .ai-summary {
        background: linear-gradient(135deg, rgba(255,0,255,0.1) 0%, rgba(10,10,20,0.95) 100%);
        border-left: 4px solid #ff00ff;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 15px 0;
        box-shadow: 0 0 20px rgba(255,0,255,0.2);
    }
    
    .ai-summary-title {
        font-family: 'Orbitron', monospace;
        font-size: 0.9rem;
        color: #ff00ff;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        text-shadow: 0 0 15px rgba(255,0,255,0.8);
    }
    
    .ai-summary p, .ai-summary div {
        color: rgba(255,255,255,0.85);
    }
    
    .recommendation-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid rgba(0,255,249,0.2);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        border-color: #00fff9;
        box-shadow: 0 0 25px rgba(0,255,249,0.3);
        transform: translateY(-2px);
    }
    
    .priority-critical { border-left: 3px solid #ff0040; box-shadow: inset 3px 0 15px rgba(255,0,64,0.3); }
    .priority-high { border-left: 3px solid #fffc00; box-shadow: inset 3px 0 15px rgba(255,252,0,0.3); }
    .priority-medium { border-left: 3px solid #00fff9; box-shadow: inset 3px 0 15px rgba(0,255,249,0.3); }
    .priority-low { border-left: 3px solid #00ff88; box-shadow: inset 3px 0 15px rgba(0,255,136,0.3); }
    
    .impact-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'Orbitron', monospace;
    }
    
    .impact-high { background: rgba(255,0,64,0.2); color: #ff0040; border: 1px solid rgba(255,0,64,0.5); }
    .impact-medium { background: rgba(255,252,0,0.2); color: #fffc00; border: 1px solid rgba(255,252,0,0.5); }
    .impact-low { background: rgba(0,255,136,0.2); color: #00ff88; border: 1px solid rgba(0,255,136,0.5); }
    
    .simulation-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,0,255,0.15);
        border: 1px solid rgba(255,0,255,0.3);
    }
    
    .stButton>button {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(135deg, #ff00ff 0%, #00fff9 100%);
        color: #0a0a0f;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255,0,255,0.4);
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 35px rgba(0,255,249,0.6), 0 0 60px rgba(255,0,255,0.4);
        transform: translateY(-2px);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        color: #00fff9;
        text-shadow: 0 0 15px rgba(0,255,249,0.5);
    }
    
    div[data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.7);
    }
    
    .risk-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 25px rgba(0,255,249,0.15);
        border: 1px solid rgba(0,255,249,0.3);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        text-shadow: 0 0 20px currentColor;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    
    .risk-low { color: #00ff88; }
    .risk-medium { color: #fffc00; }
    .risk-high { color: #ff0040; }
    
    .traffic-light {
        width: 60px;
        height: 150px;
        background: linear-gradient(180deg, #1a1a2e 0%, #0a0a0f 100%);
        border-radius: 30px;
        padding: 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        margin: 0 auto;
        border: 2px solid rgba(255,255,255,0.2);
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    
    .light {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        opacity: 0.2;
        transition: all 0.5s ease;
    }
    
    .light.red { background: #ff0040; }
    .light.yellow { background: #fffc00; }
    .light.green { background: #00ff88; }
    
    .light.active {
        opacity: 1;
        box-shadow: 0 0 30px currentColor, 0 0 60px currentColor;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px currentColor; }
        50% { box-shadow: 0 0 40px currentColor, 0 0 80px currentColor; }
    }
    
    .hotspot-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'Orbitron', monospace;
    }
    
    .hotspot-high { background: rgba(255,0,64,0.2); color: #ff0040; border: 1px solid rgba(255,0,64,0.5); }
    .hotspot-medium { background: rgba(255,252,0,0.2); color: #fffc00; border: 1px solid rgba(255,252,0,0.5); }
    .hotspot-low { background: rgba(0,255,136,0.2); color: #00ff88; border: 1px solid rgba(0,255,136,0.5); }
    
    .feature-card {
        background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%);
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        border: 1px solid rgba(0,255,249,0.2);
        display: flex;
        align-items: center;
        gap: 10px;
        color: rgba(255,255,255,0.85);
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,0,64,0.2);
        padding: 5px 15px;
        border-radius: 20px;
        border: 1px solid rgba(255,0,64,0.5);
        font-family: 'Orbitron', monospace;
        font-size: 0.75rem;
        color: #ff0040;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #ff0040;
        border-radius: 50%;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .cyber-grid {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(rgba(0,255,249,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,249,0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: -1;
    }
    
    .neon-text-cyan { color: #00fff9; text-shadow: 0 0 10px rgba(0,255,249,0.8); }
    .neon-text-pink { color: #ff00ff; text-shadow: 0 0 10px rgba(255,0,255,0.8); }
    .neon-text-yellow { color: #fffc00; text-shadow: 0 0 10px rgba(255,252,0,0.8); }
    .neon-text-red { color: #ff0040; text-shadow: 0 0 10px rgba(255,0,64,0.8); }
    .neon-text-green { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.8); }
    
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #00fff9 !important;
        font-family: 'Orbitron', monospace;
    }
    
    .stSelectbox > div > div {
        background: rgba(20,20,35,0.9) !important;
        border: 1px solid rgba(0,255,249,0.3) !important;
        color: white !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #fff !important;
    }
    
    p, span, div {
        color: rgba(255,255,255,0.85);
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%);
    }
    
    .stSidebar .stRadio label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    .stSidebar .stRadio label:hover {
        color: #00fff9 !important;
    }
    
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(0,255,249,0.2);
    }
    
    .incident-card {
        background: linear-gradient(145deg, rgba(30,30,45,0.9) 0%, rgba(15,15,25,0.95) 100%);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid;
        transition: all 0.3s ease;
    }
    
    .incident-card:hover {
        transform: translateX(5px);
        box-shadow: 0 0 20px rgba(0,255,249,0.2);
    }
    
    .incident-critical { border-color: #ff0040; }
    .incident-major { border-color: #fffc00; }
    .incident-moderate { border-color: #00fff9; }
    .incident-minor { border-color: #00ff88; }
</style>
""", unsafe_allow_html=True)


def load_and_process_data(file_path: str = 'data/sample_accidents.csv'):
    """Load and process accident data"""
    df = load_data(file_path)
    df = clean_data(df)
    df = extract_time_features(df)
    df, encodings = encode_categorical_features(df)
    df = calculate_risk_index(df)
    return df, encodings


def get_active_data():
    """Get either uploaded data or sample data"""
    if 'using_custom_data' in st.session_state and st.session_state.using_custom_data:
        if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
            return st.session_state.uploaded_data, st.session_state.get('uploaded_encodings', {})
    return load_and_process_data()


def process_uploaded_data(uploaded_file):
    """Process uploaded CSV file"""
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    df = extract_time_features(df)
    df, encodings = encode_categorical_features(df)
    df = calculate_risk_index(df)
    return df, encodings


@st.cache_resource
def get_ml_pipeline():
    """Get cached ML pipeline instance"""
    return TrafficSafeML()


def render_traffic_light(level: str):
    """Render traffic light indicator"""
    red_active = "active" if level == "High" else ""
    yellow_active = "active" if level == "Medium" else ""
    green_active = "active" if level == "Low" else ""
    
    st.markdown(f"""
    <div class="traffic-light">
        <div class="light red {red_active}"></div>
        <div class="light yellow {yellow_active}"></div>
        <div class="light green {green_active}"></div>
    </div>
    """, unsafe_allow_html=True)


def render_circular_progress(percent, color, label, stats, highlight_text, highlight_color):
    """Render a circular progress indicator card with cyberpunk styling"""
    circumference = 2 * 3.14159 * 35
    offset = circumference - (percent / 100) * circumference
    
    st.markdown(f"""
    <div class="overview-card">
        <div class="circular-progress">
            <svg width="80" height="80" viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="35" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="6"/>
                <circle cx="40" cy="40" r="35" fill="none" stroke="{color}" stroke-width="6" 
                        stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round"
                        style="filter: drop-shadow(0 0 8px {color});"/>
            </svg>
            <div class="progress-text" style="color: {color}; text-shadow: 0 0 10px {color};">{percent}%</div>
        </div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{stats}</div>
        <div class="stat-highlight" style="color: {highlight_color}; text-shadow: 0 0 8px {highlight_color};">{highlight_text}</div>
    </div>
    """, unsafe_allow_html=True)


def get_live_traffic():
    """Get live traffic data from API or demo"""
    api = TomTomTrafficAPI()
    selected_city = st.session_state.get('selected_city', 'New York')
    
    if api.is_configured() and selected_city in MAJOR_CITIES:
        coords = MAJOR_CITIES[selected_city]
        return api.get_city_traffic_summary(coords)
    else:
        return get_demo_traffic_data()


def executive_dashboard():
    """Render the executive dashboard - main landing page"""
    st.markdown('<div class="cyber-grid"></div>', unsafe_allow_html=True)
    
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<h1 class="main-header">NEURAL TRAFFIC COMMAND</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-Time Traffic Intelligence System</p>', unsafe_allow_html=True)
    with header_col2:
        st.markdown("""
        <div class="live-indicator">
            <div class="live-dot"></div>
            LIVE DATA STREAM
        </div>
        """, unsafe_allow_html=True)
    
    city_col1, city_col2 = st.columns([1, 3])
    with city_col1:
        if 'selected_city' not in st.session_state:
            st.session_state.selected_city = 'New York'
        selected_city = st.selectbox("Select City", list(MAJOR_CITIES.keys()), 
                                     index=list(MAJOR_CITIES.keys()).index(st.session_state.selected_city),
                                     key='city_select')
        st.session_state.selected_city = selected_city
    
    live_data = get_live_traffic()
    
    df, _ = get_active_data()
    ml = get_ml_pipeline()
    
    risk_data = ml.calculate_city_risk_score(df)
    hotspots_df, hotspot_stats = ml.detect_hotspots_dbscan(df)
    risk_24h = ml.predict_24h_risk_change(df)
    
    total_incidents = live_data.get('total_incidents', len(df))
    accidents = live_data.get('accidents', (df['severity'] == 'Fatal').sum())
    jams = live_data.get('jams', (df['severity'] == 'Serious').sum())
    congestion = int(live_data.get('congestion_percent', risk_data['score']))
    
    fatal_count = (df['severity'] == 'Fatal').sum()
    serious_count = (df['severity'] == 'Serious').sum()
    minor_count = (df['severity'] == 'Minor').sum()
    casualties = df['num_casualties'].sum() if 'num_casualties' in df.columns else 0
    high_risk_zones = live_data.get('critical_incidents', 0) + live_data.get('major_incidents', 0)
    
    risk_percent = min(100, congestion)
    incident_percent = min(100, total_incidents * 5)
    accident_percent = min(100, accidents * 15)
    forecast_percent = max(0, min(100, 50 + int(risk_24h['change'] * 2)))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_circular_progress(
            risk_percent, 
            '#ff0040', 
            'CONGESTION',
            f"Level: {live_data.get('risk_level', 'Unknown')}",
            f"{high_risk_zones} Critical Zones",
            '#ff0040'
        )
    
    with col2:
        render_circular_progress(
            incident_percent,
            '#00fff9',
            'INCIDENTS',
            f"Active: {total_incidents}",
            f"{accidents} Accidents | {jams} Jams",
            '#ff00ff'
        )
    
    with col3:
        flow = live_data.get('flow_data', {})
        current_speed = flow.get('current_speed', 45) if flow else 45
        render_circular_progress(
            min(100, int(current_speed)),
            '#fffc00',
            'AVG SPEED',
            f"{current_speed} km/h",
            f"Free Flow: {flow.get('free_flow_speed', 65) if flow else 65} km/h",
            '#00ff88'
        )
    
    with col4:
        forecast_color = '#ff0040' if risk_24h['change'] > 0 else '#00ff88' if risk_24h['change'] < 0 else '#00fff9'
        render_circular_progress(
            forecast_percent,
            forecast_color,
            '24H FORECAST',
            f"Change: {'+' if risk_24h['change'] > 0 else ''}{risk_24h['change']}%",
            f"{risk_24h['confidence']} Confidence",
            '#00ff88' if risk_24h['confidence'] == 'High' else '#fffc00'
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2.5, 1])
    
    with col_left:
        st.markdown('<div class="section-title">// ACTIVE INCIDENTS</div>', unsafe_allow_html=True)
        
        incidents = live_data.get('incidents', [])[:6]
        if incidents:
            for i, inc in enumerate(incidents):
                severity_class = f"incident-{inc.get('severity', 'minor').lower()}"
                inc_type = inc.get('type', 'Unknown')
                desc = inc.get('description', 'Traffic incident')[:60]
                delay = inc.get('delay_seconds', 0)
                delay_min = delay // 60 if delay else 0
                
                st.markdown(f"""
                <div class="incident-card {severity_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-family: 'Orbitron', monospace; font-size: 0.8rem; color: #00fff9;">
                            {inc_type.upper()}
                        </span>
                        <span class="hotspot-{inc.get('severity', 'minor').lower()}" style="font-size: 0.7rem;">
                            {inc.get('severity', 'Minor')}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; margin-top: 5px; color: rgba(255,255,255,0.8);">
                        {desc}
                    </div>
                    <div style="font-size: 0.75rem; margin-top: 5px; color: #fffc00;">
                        Delay: {delay_min} min
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            colors_bg = ['#ff0040', '#fffc00', '#00fff9']
            names = ['SECTOR-A // CRITICAL', 'SECTOR-B // ELEVATED', 'SECTOR-C // MONITORED']
            for i, col in enumerate(st.columns(3)):
                with col:
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, rgba(20,20,35,0.9) 0%, rgba(10,10,20,0.95) 100%); 
                                border-radius: 10px; padding: 15px; min-height: 90px;
                                border: 1px solid {colors_bg[i]}; box-shadow: 0 0 15px {colors_bg[i]}33;">
                        <div style="font-family: 'Orbitron', monospace; font-size: 0.7rem; font-weight: 600; 
                                    color: {colors_bg[i]}; text-shadow: 0 0 10px {colors_bg[i]};">
                            {names[i]}
                        </div>
                        <div style="margin-top: 35px; font-size: 0.75rem; color: rgba(255,255,255,0.6);">
                            Scanning...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="section-title">// SYSTEM STATUS</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;"><span style="font-family: Orbitron, monospace; font-weight: 500; font-size: 0.8rem; color: #00fff9;">THREAT MATRIX</span><span class="view-all">EXPAND</span></div>', unsafe_allow_html=True)
        
        fig1, ax1 = plt.subplots(figsize=(3, 2.5), facecolor='#0a0a0f')
        ax1.set_facecolor('#0a0a0f')
        sizes = [minor_count, serious_count, fatal_count]
        colors_pie = ['#00fff9', '#fffc00', '#ff0040']
        
        wedges, texts = ax1.pie(sizes if sum(sizes) > 0 else [1,1,1], colors=colors_pie, startangle=90, 
                                wedgeprops=dict(width=0.4, edgecolor='#0a0a0f'))
        ax1.axis('equal')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f'<div class="donut-legend-value" style="color: #00fff9;">{minor_count}</div><div class="donut-legend">Minor</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="donut-legend-value" style="color: #fffc00;">{serious_count}</div><div class="donut-legend">Serious</div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="donut-legend-value" style="color: #ff0040;">{fatal_count}</div><div class="donut-legend">Fatal</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;"><span style="font-family: Orbitron, monospace; font-weight: 500; font-size: 0.8rem; color: #ff00ff;">ZONE SCAN</span><span class="view-all">EXPAND</span></div>', unsafe_allow_html=True)
        
        total_hotspots = len(hotspots_df) if not hotspots_df.empty else 0
        medium_risk = len(hotspots_df[hotspots_df['risk_level'] == 'Medium']) if not hotspots_df.empty and 'risk_level' in hotspots_df.columns else 0
        low_risk = len(hotspots_df[hotspots_df['risk_level'] == 'Low']) if not hotspots_df.empty and 'risk_level' in hotspots_df.columns else 0
        
        fig2, ax2 = plt.subplots(figsize=(3, 2.5), facecolor='#0a0a0f')
        ax2.set_facecolor('#0a0a0f')
        sizes2 = [high_risk_zones, medium_risk, low_risk]
        colors_pie2 = ['#ff0040', '#fffc00', '#00ff88']
        
        wedges2, texts2 = ax2.pie(sizes2 if sum(sizes2) > 0 else [1,1,1], colors=colors_pie2, startangle=90,
                                  wedgeprops=dict(width=0.4, edgecolor='#0a0a0f'))
        ax2.axis('equal')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.markdown(f'<div class="donut-legend-value" style="color: #ff0040;">{high_risk_zones}</div><div class="donut-legend">Critical</div>', unsafe_allow_html=True)
        with col_e:
            st.markdown(f'<div class="donut-legend-value" style="color: #fffc00;">{medium_risk}</div><div class="donut-legend">Elevated</div>', unsafe_allow_html=True)
        with col_f:
            st.markdown(f'<div class="donut-legend-value" style="color: #00ff88;">{low_risk}</div><div class="donut-legend">Normal</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">// TEMPORAL ANALYSIS</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    if 'month' in df.columns:
        monthly_data = df.groupby('month').agg({
            'severity': 'count',
            'num_casualties': 'sum' if 'num_casualties' in df.columns else 'count'
        }).reindex(range(1, 13), fill_value=0)
        
        minor_monthly = df[df['severity'] == 'Minor'].groupby('month').size().reindex(range(1, 13), fill_value=0)
        serious_monthly = df[df['severity'] == 'Serious'].groupby('month').size().reindex(range(1, 13), fill_value=0)
        fatal_monthly = df[df['severity'] == 'Fatal'].groupby('month').size().reindex(range(1, 13), fill_value=0)
    else:
        np.random.seed(42)
        minor_monthly = pd.Series(np.random.randint(5, 15, 12))
        serious_monthly = pd.Series(np.random.randint(3, 10, 12))
        fatal_monthly = pd.Series(np.random.randint(0, 3, 12))
    
    legend_html = """
    <div style="display: flex; justify-content: flex-end; gap: 20px; margin-bottom: 10px;">
        <span class="legend-item"><span class="legend-dot" style="background: #00fff9; box-shadow: 0 0 8px #00fff9;"></span> Minor</span>
        <span class="legend-item"><span class="legend-dot" style="background: #fffc00; box-shadow: 0 0 8px #fffc00;"></span> Serious</span>
        <span class="legend-item"><span class="legend-dot" style="background: #ff0040; box-shadow: 0 0 8px #ff0040;"></span> Fatal</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    ax.plot(months, minor_monthly.values, marker='o', linewidth=2, color='#00fff9', label='Minor', markersize=6)
    ax.plot(months, serious_monthly.values, marker='o', linewidth=2, color='#fffc00', label='Serious', markersize=6)
    ax.plot(months, fatal_monthly.values, marker='o', linewidth=2, color='#ff0040', label='Fatal', markersize=6)
    
    ax.set_ylim(0, max(minor_monthly.max(), serious_monthly.max(), fatal_monthly.max()) + 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333')
    ax.spines['bottom'].set_color('#333')
    ax.tick_params(colors='#00fff9', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#00fff9')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">// CASUALTY METRICS</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    
    legend_html2 = """
    <div style="display: flex; justify-content: flex-end; gap: 20px; margin-bottom: 10px;">
        <span class="legend-item"><span class="legend-dot" style="background: #00fff9; box-shadow: 0 0 8px #00fff9;"></span> Injured</span>
        <span class="legend-item"><span class="legend-dot" style="background: #ff0040; box-shadow: 0 0 8px #ff0040;"></span> Deaths</span>
    </div>
    """
    st.markdown(legend_html2, unsafe_allow_html=True)
    
    if 'month' in df.columns:
        injured_monthly = df.groupby('month')['num_casualties'].sum().reindex(range(1, 13), fill_value=0) if 'num_casualties' in df.columns else pd.Series(np.random.randint(20, 80, 12))
        deaths_monthly = df[df['severity'] == 'Fatal'].groupby('month').size().reindex(range(1, 13), fill_value=0)
    else:
        np.random.seed(43)
        injured_monthly = pd.Series(np.random.randint(30, 80, 12))
        deaths_monthly = pd.Series(np.random.randint(5, 25, 12))
    
    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor='#0a0a0f')
    ax2.set_facecolor('#0a0a0f')
    
    x = np.arange(len(months))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, injured_monthly.values, width, label='Injured', color='#00fff9', edgecolor='#00fff9', linewidth=1)
    bars2 = ax2.bar(x + width/2, deaths_monthly.values, width, label='Deaths', color='#ff0040', edgecolor='#ff0040', linewidth=1)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(months)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#333')
    ax2.spines['bottom'].set_color('#333')
    ax2.tick_params(colors='#00fff9', labelsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.2, color='#ff00ff')
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    is_demo = live_data.get('is_demo', True)
    demo_notice = " [DEMO MODE - Add TomTom API key for live data]" if is_demo else ""
    
    ai_summary = ml.generate_ai_summary(df, risk_data, hotspots_df)
    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-summary-title">
            ⚡ NEURAL ANALYSIS ENGINE{demo_notice}
        </div>
        <div style="line-height: 1.6; font-size: 0.9rem;">
            {ai_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)


def data_upload_page():
    """Render the data upload page with form-based entry"""
    st.markdown('<h1 class="main-header">📤 Data Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload CSV data or enter accidents manually</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📁 CSV Upload", "✏️ Manual Entry", "🌐 Global Data Sources"])
    
    with tab1:
        st.markdown("### Upload Your Accident Data")
        st.markdown("Data is **automatically optimized** - columns renamed, values standardized, missing data filled.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload accident data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                
                st.success(f"✅ File uploaded: {len(df_raw)} records found")
                
                with st.expander("View Original Data", expanded=False):
                    st.dataframe(df_raw.head(10), use_container_width=True)
                
                with st.spinner("🔄 Optimizing data..."):
                    df_optimized, opt_report = auto_optimize_data(df_raw)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Rows", opt_report['original_rows'])
                with col2:
                    st.metric("Optimized Rows", opt_report['final_rows'])
                with col3:
                    st.metric("Quality Score", f"{opt_report['quality_score']:.0f}%")
                
                if opt_report['changes']:
                    with st.expander("🔧 Optimization Details", expanded=True):
                        for change in opt_report['changes']:
                            st.markdown(f"• {change}")
                
                auto_retrain = st.checkbox("🔄 Auto-retrain models when data is loaded", value=True, 
                                          help="Automatically train ML models when new data is uploaded")
                
                if st.button("✅ Use This Data & Process", type="primary", use_container_width=True):
                    with st.spinner("Processing and optimizing data..."):
                        df_processed, encodings = process_uploaded_data(uploaded_file)
                        df_processed, _ = auto_optimize_data(df_processed)
                        st.session_state.uploaded_data = df_processed
                        st.session_state.uploaded_encodings = encodings
                        st.session_state.using_custom_data = True
                        
                        df_processed.to_csv('data/custom_accidents.csv', index=False)
                        
                        if auto_retrain:
                            st.info("🤖 Auto-retraining models on new data...")
                            ml = get_ml_pipeline()
                            xgb_result = ml.train_severity_model(df_processed)
                            poisson_result = ml.fit_poisson_model(df_processed)
                            negbin_result = ml.fit_negative_binomial(df_processed)
                            
                            if xgb_result.get('trained'):
                                st.session_state.xgb_result = xgb_result
                                st.success(f"✅ XGBoost trained! Accuracy: {xgb_result['accuracy']:.1%}")
                            if poisson_result.get('fitted'):
                                st.success(f"✅ Poisson model fitted! Pseudo R²: {poisson_result['pseudo_r2']:.3f}")
                            if negbin_result.get('fitted'):
                                st.success(f"✅ Negative Binomial fitted!")
                    
                    st.success("✅ Data loaded and models updated! Navigate to other pages to analyze.")
                    st.rerun()
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.markdown("### Add New Accident Record")
        
        with st.form("accident_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                date = st.date_input("Date", datetime.now())
                time = st.time_input("Time", datetime.now().time())
                latitude = st.number_input("Latitude", value=28.6139, format="%.4f")
                longitude = st.number_input("Longitude", value=77.2090, format="%.4f")
                severity = st.selectbox("Severity", ["Minor", "Serious", "Fatal"])
                weather = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Snow"])
            
            with col2:
                road_type = st.selectbox("Road Type", ["Urban", "Rural", "Motorway"])
                speed_limit = st.number_input("Speed Limit (km/h)", value=50, min_value=10, max_value=150)
                num_vehicles = st.number_input("Vehicles Involved", value=2, min_value=1, max_value=10)
                num_casualties = st.number_input("Casualties", value=0, min_value=0, max_value=50)
                light_conditions = st.selectbox("Light Conditions", ["Daylight", "Dark - lights lit", "Dark - no lights"])
                road_surface = st.selectbox("Road Surface", ["Dry", "Wet", "Ice", "Snow"])
            
            submitted = st.form_submit_button("➕ Add Record", use_container_width=True)
            
            if submitted:
                new_record = {
                    'accident_id': f"ACC{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'date': date.strftime('%Y-%m-%d'),
                    'time': time.strftime('%H:%M'),
                    'latitude': latitude,
                    'longitude': longitude,
                    'severity': severity,
                    'weather': weather,
                    'road_type': road_type,
                    'speed_limit': speed_limit,
                    'num_vehicles': num_vehicles,
                    'num_casualties': num_casualties,
                    'light_conditions': light_conditions,
                    'road_surface': road_surface
                }
                
                df, _ = get_active_data()
                new_df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
                st.session_state.uploaded_data = new_df
                st.session_state.using_custom_data = True
                
                new_df.to_csv('data/custom_accidents.csv', index=False)
                ml = get_ml_pipeline()
                xgb_result = ml.train_severity_model(new_df)
                if xgb_result.get('trained'):
                    st.session_state.xgb_result = xgb_result
                
                st.success("✅ Record added and models auto-retrained!")
                st.rerun()
    
    with tab3:
        st.markdown("### Free Global Accident Data Sources")
        
        data_sources = get_global_data_sources()
        
        cols = st.columns(2)
        for idx, source in enumerate(data_sources):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon" style="background: #E8F0FE;">🌍</div>
                    <div>
                        <strong>{source['name']}</strong><br>
                        <small style="color: #5f6368;">{source['country']} • {source['format']}</small><br>
                        <a href="{source['url']}" target="_blank">Download Data →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def ml_insights_page():
    """Render ML insights and explainability page"""
    st.markdown('<h1 class="main-header">🧠 AI Insights & Explainability</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand how our AI makes predictions</p>', unsafe_allow_html=True)
    
    df, _ = get_active_data()
    ml = get_ml_pipeline()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "🔍 Hotspot Detection", "⚠️ Anomaly Detection", "📖 Explanations"])
    
    with tab1:
        st.markdown("### Train Advanced ML Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Severity Prediction (XGBoost)")
            if st.button("🚀 Train XGBoost Model", use_container_width=True):
                with st.spinner("Training XGBoost model..."):
                    result = ml.train_severity_model(df)
                
                if result.get('trained'):
                    st.success(f"✅ Model trained successfully!")
                    st.metric("Accuracy", f"{result['accuracy']:.1%}")
                    st.metric("F1 Score", f"{result['f1_score']:.1%}")
                    st.session_state.xgb_result = result
                else:
                    st.error(f"Training failed: {result.get('error')}")
        
        with col2:
            st.markdown("#### Frequency Models (Statistical)")
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("📈 Poisson Model", use_container_width=True):
                    with st.spinner("Fitting Poisson regression..."):
                        result = ml.fit_poisson_model(df)
                    
                    if result.get('fitted'):
                        st.success("✅ Poisson model fitted!")
                        st.metric("Pseudo R²", f"{result['pseudo_r2']:.3f}")
                        st.session_state.poisson_result = result
                    else:
                        st.error(f"Fitting failed: {result.get('error')}")
            
            with col2b:
                if st.button("📊 Neg. Binomial", use_container_width=True):
                    with st.spinner("Fitting Negative Binomial..."):
                        result = ml.fit_negative_binomial(df)
                    
                    if result.get('fitted'):
                        st.success("✅ Neg. Binomial fitted!")
                        st.metric("Pseudo R²", f"{result['pseudo_r2']:.3f}")
                        st.session_state.negbin_result = result
                    else:
                        st.error(f"Fitting failed: {result.get('error')}")
        
        if 'xgb_result' in st.session_state:
            st.markdown("---")
            st.markdown("### Feature Importance")
            
            result = st.session_state.xgb_result
            importance = result.get('feature_importance', {})
            
            if importance:
                fig, ax = plt.subplots(figsize=(10, 5))
                features = list(importance.keys())
                values = list(importance.values())
                colors = ['#4285F4' if v > 0.15 else '#34A853' if v > 0.1 else '#FBBC05' for v in values]
                
                bars = ax.barh(features, values, color=colors)
                ax.set_xlabel('Importance Score')
                ax.set_title('What Factors Matter Most?')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        st.markdown("### DBSCAN Hotspot Detection")
        st.markdown("Automatically identifies accident clusters using density-based clustering.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            eps = st.slider("Cluster Radius (degrees)", 0.005, 0.05, 0.01, 0.005)
            min_samples = st.slider("Minimum Points", 2, 10, 3)
            
            if st.button("🎯 Detect Hotspots", use_container_width=True):
                with st.spinner("Clustering accidents..."):
                    hotspots, stats = ml.detect_hotspots_dbscan(df, eps=eps, min_samples=min_samples)
                    st.session_state.hotspots = hotspots
                    st.session_state.hotspot_stats = stats
        
        with col2:
            if 'hotspots' in st.session_state and not st.session_state.hotspots.empty:
                stats = st.session_state.hotspot_stats
                st.metric("Clusters Found", stats['n_clusters'])
                st.metric("Noise Points", stats['noise_points'])
                
                st.dataframe(st.session_state.hotspots, use_container_width=True)
    
    with tab3:
        st.markdown("### Isolation Forest Anomaly Detection")
        st.markdown("Identifies unusual accidents that deviate from normal patterns.")
        
        contamination = st.slider("Expected Anomaly Rate", 0.01, 0.2, 0.1)
        
        if st.button("🔎 Detect Anomalies", use_container_width=True):
            with st.spinner("Detecting anomalies..."):
                df_anomalies, anomaly_stats = ml.detect_anomalies(df, contamination=contamination)
            
            st.metric("Anomalies Detected", anomaly_stats['n_anomalies'])
            st.metric("Percentage", f"{anomaly_stats['percentage']}%")
            
            if anomaly_stats['n_anomalies'] > 0:
                st.markdown("#### Anomalous Accidents")
                anomaly_df = df_anomalies[df_anomalies['is_anomaly'] == True]
                st.dataframe(anomaly_df[['accident_id', 'severity', 'weather', 'road_type', 'num_casualties']].head(10),
                           use_container_width=True)
    
    with tab4:
        st.markdown("### 📖 AI Explainability Dashboard")
        st.markdown("""
        <div class="ai-summary" style="margin-bottom: 20px;">
            <div class="ai-summary-title">🔬 How This Works</div>
            Our AI uses <strong>gradient-based feature importance</strong> from XGBoost to identify which factors 
            most influence accident severity predictions. Each factor is analyzed and explained in plain English 
            so you can understand exactly what the model has learned from your data.
        </div>
        """, unsafe_allow_html=True)
        
        if 'xgb_result' in st.session_state:
            result = st.session_state.xgb_result
            explanations = ml.get_feature_explanations(result.get('feature_importance', {}))
            
            st.markdown(f"""
            <div style="background: #E6F4EA; padding: 15px 20px; border-radius: 12px; margin-bottom: 20px;">
                <strong style="color: #137333;">Model Version:</strong> {result.get('model_version', 'N/A')} | 
                <strong style="color: #137333;">Accuracy:</strong> {result.get('accuracy', 0):.1%} | 
                <strong style="color: #137333;">Features:</strong> {len(result.get('features_used', []))}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### What Factors Drive Accident Severity?")
            
            for exp in explanations:
                impact_class = 'impact-high' if exp['impact'] in ['Very High', 'High'] else 'impact-medium' if exp['impact'] == 'Moderate' else 'impact-low'
                icon = "🔴" if exp['impact'] == 'Very High' else "🟠" if exp['impact'] == 'High' else "🟡" if exp['impact'] == 'Moderate' else "🟢"
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.1rem;">{icon} {exp['feature']}</strong>
                            <p style="color: #5f6368; margin: 5px 0;">{exp['explanation']}</p>
                        </div>
                        <span class="impact-badge {impact_class}">{exp['importance']}% importance</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Feature Importance Visualization")
            
            importance = result.get('feature_importance', {})
            if importance:
                fig, ax = plt.subplots(figsize=(10, 4))
                sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])
                features = [f.replace('_encoded', '').replace('_', ' ').title() for f in sorted_importance.keys()]
                values = list(sorted_importance.values())
                colors = ['#EA4335' if v > 0.2 else '#FBBC05' if v > 0.1 else '#34A853' for v in values]
                
                bars = ax.barh(features, [v*100 for v in values], color=colors, height=0.6)
                ax.set_xlabel('Importance (%)', fontsize=11)
                ax.set_xlim(0, 100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                for bar, val in zip(bars, values):
                    ax.text(val*100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', 
                           va='center', fontsize=9, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("👆 Train a model first in the 'Model Training' tab to see AI explanations.")


def road_design_page():
    """Render road design recommendations page"""
    st.markdown('<h1 class="main-header">🛣️ Road Design Optimization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-driven recommendations for safer roads</p>', unsafe_allow_html=True)
    
    df, _ = get_active_data()
    ml = get_ml_pipeline()
    
    hotspots_df, _ = ml.detect_hotspots_dbscan(df)
    recommendations = get_priority_recommendations(df, hotspots_df)
    
    st.markdown("### 📋 Prioritized Recommendations")
    
    for i, rec in enumerate(recommendations):
        priority_class = f"priority-{rec['priority'].lower()}"
        impact_class = 'impact-high' if rec['impact_score'] > 70 else 'impact-medium' if rec['impact_score'] > 50 else 'impact-low'
        
        with st.expander(f"**{i+1}. {rec['category']}** — {rec['priority']} Priority", expanded=(i < 3)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Recommendation:** {rec['recommendation']}")
                st.markdown(f"**Specific Action:** {rec['specific_action']}")
                st.markdown(f"**Expected Reduction:** {rec['estimated_reduction']}")
                st.markdown(f"**Cost Level:** {rec['cost_level']}")
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 2.5rem; font-weight: bold; color: {'#EA4335' if rec['impact_score'] > 70 else '#FBBC05' if rec['impact_score'] > 50 else '#34A853'};">
                        {rec['impact_score']}
                    </div>
                    <div style="color: #5f6368; font-size: 0.9rem;">Impact Score</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Impact Analysis")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    categories = [r['category'] for r in recommendations]
    impact_scores = [r['impact_score'] for r in recommendations]
    colors = ['#EA4335' if s > 70 else '#FBBC05' if s > 50 else '#34A853' for s in impact_scores]
    
    bars = ax.barh(categories, impact_scores, color=colors)
    ax.set_xlabel('Impact Score (%)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, score in zip(bars, impact_scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score}%', 
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def whatif_simulation_page():
    """Render what-if simulation page"""
    st.markdown('<h1 class="main-header">🔮 What-If Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Forecast the impact of safety interventions</p>', unsafe_allow_html=True)
    
    df, _ = get_active_data()
    ml = get_ml_pipeline()
    
    st.markdown("### Select Intervention Type")
    
    intervention = st.selectbox(
        "Choose an intervention to simulate",
        [
            ("speed_reduction", "🚗 Speed Limit Reduction (10 km/h lower in high-risk zones)"),
            ("improved_lighting", "💡 Improved Street Lighting"),
            ("junction_redesign", "🔄 Junction Redesign (Convert to Roundabouts)"),
            ("weather_warning_system", "🌧️ Real-Time Weather Warning Systems"),
            ("comprehensive", "✅ Comprehensive (All Interventions)")
        ],
        format_func=lambda x: x[1]
    )
    
    if st.button("🔮 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating intervention impact..."):
            result = ml.simulate_intervention(df, intervention[0])
        
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
            st.markdown("#### Before Intervention")
            
            before = result['before']
            risk_color = '#EA4335' if before['risk_level'] == 'High' else '#FBBC05' if before['risk_level'] == 'Medium' else '#34A853'
            
            st.markdown(f"""
            <div style="font-size: 3rem; font-weight: bold; color: {risk_color};">
                {before['risk_score']}
            </div>
            <div style="color: #5f6368;">Risk Score</div>
            """, unsafe_allow_html=True)
            
            st.metric("Fatal Accidents", before['fatal_count'])
            st.metric("Serious Accidents", before['serious_count'])
            st.metric("Total Casualties", int(before['total_casualties']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 100%;">', unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <div style="font-size: 4rem; color: #34A853;">→</div>
                <div style="background: #E6F4EA; color: #137333; padding: 10px 20px; border-radius: 20px; font-weight: bold;">
                    -{reduction}%
                </div>
            </div>
            """.format(reduction=result['reduction']['risk_percentage_reduction']), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="simulation-card">', unsafe_allow_html=True)
            st.markdown("#### After Intervention")
            
            after = result['after']
            risk_color = '#EA4335' if after['risk_level'] == 'High' else '#FBBC05' if after['risk_level'] == 'Medium' else '#34A853'
            
            st.markdown(f"""
            <div style="font-size: 3rem; font-weight: bold; color: {risk_color};">
                {after['risk_score']}
            </div>
            <div style="color: #5f6368;">Risk Score</div>
            """, unsafe_allow_html=True)
            
            delta_fatal = result['reduction']['fatal_reduction']
            st.metric("Fatal Accidents", after['fatal_count'], delta=f"-{delta_fatal}" if delta_fatal > 0 else None, delta_color="inverse")
            st.metric("Serious Accidents", after['serious_count'])
            st.metric("Total Casualties", int(after['total_casualties']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        lives_saved = result['reduction'].get('lives_saved_estimate', 1)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #E6F4EA 0%, #CEEAD6 100%); 
                    border-radius: 16px; padding: 30px; text-align: center; margin-bottom: 20px;">
            <div style="font-size: 0.9rem; color: #137333; text-transform: uppercase; letter-spacing: 1px;">Estimated Lives Saved</div>
            <div style="font-size: 4rem; font-weight: 700; color: #137333;">{lives_saved}</div>
            <div style="font-size: 0.85rem; color: #5f6368;">Based on historical accident severity patterns</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ai-summary">
            <div class="ai-summary-title">📋 Intervention Summary</div>
            <strong>{result['description']}</strong><br><br>
            This intervention is projected to reduce the city's risk score by <strong>{result['reduction']['risk_percentage_reduction']}%</strong>, 
            potentially preventing <strong>{result['reduction']['fatal_reduction']}</strong> fatal accidents, 
            reducing <strong>{int(result['reduction']['casualty_reduction'])}</strong> casualties, 
            and saving an estimated <strong>{lives_saved} lives</strong>.
        </div>
        """, unsafe_allow_html=True)


def accident_map_page():
    """Render interactive accident map"""
    st.markdown('<h1 class="main-header">🗺️ Interactive Accident Map</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore accident locations and hotspots</p>', unsafe_allow_html=True)
    
    df, _ = get_active_data()
    ml = get_ml_pipeline()
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        from utils import get_global_data_sources
        
        known_locations = [
            {"name": "Delhi, India", "lat": 28.6139, "lon": 77.2090, "radius": 0.5},
            {"name": "Mumbai, India", "lat": 19.0760, "lon": 72.8777, "radius": 0.5},
            {"name": "London, UK", "lat": 51.5074, "lon": -0.1278, "radius": 0.5},
            {"name": "New York, USA", "lat": 40.7128, "lon": -74.0060, "radius": 0.5},
        ]
        
        location_name = "Unknown Location"
        for loc in known_locations:
            if abs(center_lat - loc["lat"]) < loc["radius"] and abs(center_lon - loc["lon"]) < loc["radius"]:
                location_name = loc["name"]
                break
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #4285F4 0%, #34A853 100%); 
                    padding: 15px 25px; border-radius: 12px; margin-bottom: 20px;">
            <h3 style="color: white; margin: 0;">📍 Currently Viewing: {location_name}</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">
                {len(df)} accidents • Center: {center_lat:.4f}, {center_lon:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.multiselect(
            "Severity",
            options=df['severity'].unique().tolist(),
            default=df['severity'].unique().tolist()
        )
    
    with col2:
        weather_filter = st.multiselect(
            "Weather",
            options=df['weather'].unique().tolist(),
            default=df['weather'].unique().tolist()
        )
    
    with col3:
        road_filter = st.multiselect(
            "Road Type",
            options=df['road_type'].unique().tolist(),
            default=df['road_type'].unique().tolist()
        )
    
    filtered_df = df[
        (df['severity'].isin(severity_filter)) &
        (df['weather'].isin(weather_filter)) &
        (df['road_type'].isin(road_filter))
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} accidents**")
    
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns and len(filtered_df) > 0:
        m = folium.Map(
            location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()],
            zoom_start=12,
            tiles='cartodbpositron'
        )
        
        severity_colors = {'Minor': '#34A853', 'Serious': '#FBBC05', 'Fatal': '#EA4335'}
        
        for _, row in filtered_df.iterrows():
            color = severity_colors.get(row['severity'], '#4285F4')
            
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; min-width: 200px;">
                <b>ID:</b> {row.get('accident_id', 'N/A')}<br>
                <b>Severity:</b> <span style="color: {color}; font-weight: bold;">{row['severity']}</span><br>
                <b>Weather:</b> {row['weather']}<br>
                <b>Road Type:</b> {row['road_type']}<br>
                <b>Casualties:</b> {row.get('num_casualties', 0)}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['severity']} - {row['weather']}"
            ).add_to(m)
        
        folium_static(m, width=1200, height=600)
        
        st.markdown("---")
        st.markdown("### 🔥 Hotspot Heatmap")
        
        hotspots_df, stats = ml.detect_hotspots_dbscan(filtered_df)
        
        if not hotspots_df.empty:
            m_heat = folium.Map(
                location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()],
                zoom_start=12,
                tiles='cartodbdark_matter'
            )
            
            from folium.plugins import HeatMap
            heat_data = [[row['latitude'], row['longitude']] for _, row in filtered_df.iterrows()]
            HeatMap(heat_data, radius=15, blur=10).add_to(m_heat)
            
            folium_static(m_heat, width=1200, height=400)


def main():
    """Main application entry point"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #4285F4; margin: 0;">🛡️ TrafficSafe AI</h2>
        <p style="color: #5f6368; font-size: 0.8rem;">Intelligent Road Safety Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Executive Dashboard": executive_dashboard,
        "📤 Data Management": data_upload_page,
        "🧠 AI Insights": ml_insights_page,
        "🛣️ Road Design": road_design_page,
        "🔮 What-If Simulation": whatif_simulation_page,
        "🗺️ Accident Map": accident_map_page
    }
    
    selection = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📊 Data Status")
    if 'using_custom_data' in st.session_state and st.session_state.using_custom_data:
        st.sidebar.success("✅ Using uploaded data")
        if st.sidebar.button("Reset to Sample", use_container_width=True):
            st.session_state.uploaded_data = None
            st.session_state.using_custom_data = False
            st.rerun()
    else:
        st.sidebar.info("📁 Using sample data")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Status")
    
    if os.path.exists('models/xgb_severity_latest.joblib'):
        st.sidebar.success("✅ XGBoost Ready")
    else:
        st.sidebar.warning("⚠️ Model not trained")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #9aa0a6; font-size: 0.75rem;">
        Powered by Machine Learning<br>
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)
    
    pages[selection]()


if __name__ == "__main__":
    main()
