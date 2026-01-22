"""
Utility functions for Traffic Accident Risk & Road Design Optimization
Contains data processing, feature engineering, and helper functions
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load traffic accident data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the accident data
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform automated data cleaning on the accident dataset
    
    Args:
        df: Raw accident DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    df_clean = df_clean.drop_duplicates()
    
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown')
    
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df_clean


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from date and time columns
    
    Args:
        df: DataFrame with date and time columns
        
    Returns:
        DataFrame with additional time features
    """
    df_time = df.copy()
    
    if 'date' in df_time.columns:
        df_time['date'] = pd.to_datetime(df_time['date'])
        df_time['day_of_week'] = df_time['date'].dt.dayofweek
        df_time['month'] = df_time['date'].dt.month
        df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
    
    if 'time' in df_time.columns:
        df_time['hour'] = pd.to_datetime(df_time['time'], format='%H:%M').dt.hour
        df_time['time_of_day'] = df_time['hour'].apply(categorize_time_of_day)
    
    return df_time


def categorize_time_of_day(hour: int) -> str:
    """
    Categorize hour into time of day periods
    
    Args:
        hour: Hour of the day (0-23)
        
    Returns:
        Time of day category
    """
    if 6 <= hour < 10:
        return 'Morning Rush'
    elif 10 <= hour < 16:
        return 'Midday'
    elif 16 <= hour < 20:
        return 'Evening Rush'
    else:
        return 'Night'


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical features for ML models
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        Tuple of (encoded DataFrame, encoding mappings)
    """
    df_encoded = df.copy()
    encodings = {}
    
    categorical_columns = ['weather', 'road_type', 'light_conditions', 
                          'road_surface', 'junction_type', 'time_of_day']
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            unique_values = df_encoded[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            encodings[col] = mapping
            df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
    
    return df_encoded, encodings


def calculate_severity_score(row: pd.Series) -> float:
    """
    Calculate accident severity score based on multiple factors
    
    Args:
        row: DataFrame row containing accident data
        
    Returns:
        Severity score (0-100)
    """
    score = 0
    
    severity_mapping = {'Minor': 20, 'Serious': 60, 'Fatal': 100}
    if 'severity' in row.index:
        score += severity_mapping.get(row['severity'], 30)
    
    if 'num_casualties' in row.index:
        score += min(row['num_casualties'] * 10, 30)
    
    if 'num_vehicles' in row.index:
        score += min((row['num_vehicles'] - 1) * 5, 15)
    
    weather_risk = {'Clear': 0, 'Rain': 10, 'Fog': 15, 'Snow': 20}
    if 'weather' in row.index:
        score += weather_risk.get(row['weather'], 5)
    
    return min(score, 100)


def calculate_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate traffic risk index for each accident record
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        DataFrame with risk index column
    """
    df_risk = df.copy()
    
    df_risk['severity_score'] = df_risk.apply(calculate_severity_score, axis=1)
    
    road_risk = {'Urban': 1.0, 'Rural': 1.2, 'Motorway': 1.5}
    df_risk['road_risk_factor'] = df_risk['road_type'].map(road_risk).fillna(1.0)
    
    light_risk = {'Daylight': 1.0, 'Dark': 1.5}
    df_risk['light_risk_factor'] = df_risk['light_conditions'].map(light_risk).fillna(1.2)
    
    surface_risk = {'Dry': 1.0, 'Wet': 1.3, 'Ice': 1.8}
    df_risk['surface_risk_factor'] = df_risk['road_surface'].map(surface_risk).fillna(1.2)
    
    df_risk['risk_index'] = (
        df_risk['severity_score'] * 
        df_risk['road_risk_factor'] * 
        df_risk['light_risk_factor'] * 
        df_risk['surface_risk_factor']
    ) / 10
    
    max_risk = df_risk['risk_index'].max()
    if max_risk > 0:
        df_risk['risk_index'] = (df_risk['risk_index'] / max_risk) * 100
    
    return df_risk


def get_risk_level(risk_index: float) -> str:
    """
    Convert risk index to risk level category
    
    Args:
        risk_index: Numeric risk index (0-100)
        
    Returns:
        Risk level category
    """
    if risk_index < 25:
        return 'Low'
    elif risk_index < 50:
        return 'Medium'
    elif risk_index < 75:
        return 'High'
    else:
        return 'Critical'


def prepare_features_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target variable for model training
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Tuple of (feature matrix X, target variable y)
    """
    feature_columns = [
        'speed_limit', 'num_vehicles', 'num_casualties',
        'is_weekend', 'hour', 'day_of_week', 'month'
    ]
    
    encoded_columns = [col for col in df.columns if col.endswith('_encoded')]
    feature_columns.extend(encoded_columns)
    
    available_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[available_columns].copy()
    
    severity_mapping = {'Minor': 0, 'Serious': 1, 'Fatal': 2}
    y = df['severity'].map(severity_mapping)
    
    return X, y


def get_road_design_recommendations(df: pd.DataFrame) -> list:
    """
    Generate data-driven road design optimization recommendations
    
    Args:
        df: Processed accident DataFrame
        
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    if 'junction_type' in df.columns:
        junction_severity = df.groupby('junction_type')['severity_score'].mean()
        high_risk_junctions = junction_severity[junction_severity > junction_severity.median()].index.tolist()
        if high_risk_junctions:
            recommendations.append({
                'category': 'Junction Design',
                'priority': 'High',
                'issue': f'High-risk junction types: {", ".join(high_risk_junctions)}',
                'recommendation': 'Consider installing traffic signals, improving visibility, or adding dedicated turn lanes at these junction types.',
                'expected_impact': 'Potential 20-30% reduction in junction-related accidents'
            })
    
    if 'road_surface' in df.columns and 'severity_score' in df.columns:
        wet_accidents = df[df['road_surface'].isin(['Wet', 'Ice'])]
        if len(wet_accidents) > len(df) * 0.3:
            recommendations.append({
                'category': 'Road Surface',
                'priority': 'High',
                'issue': f'{len(wet_accidents)} accidents ({len(wet_accidents)/len(df)*100:.1f}%) occurred on wet/icy surfaces',
                'recommendation': 'Improve drainage systems, apply anti-skid treatments, and install weather warning signs.',
                'expected_impact': 'Potential 15-25% reduction in weather-related accidents'
            })
    
    if 'light_conditions' in df.columns:
        dark_accidents = df[df['light_conditions'] == 'Dark']
        if len(dark_accidents) > len(df) * 0.35:
            recommendations.append({
                'category': 'Lighting Infrastructure',
                'priority': 'Medium',
                'issue': f'{len(dark_accidents)} accidents ({len(dark_accidents)/len(df)*100:.1f}%) occurred in dark conditions',
                'recommendation': 'Enhance street lighting, install reflective road markings, and add LED road studs.',
                'expected_impact': 'Potential 10-20% reduction in night-time accidents'
            })
    
    if 'speed_limit' in df.columns and 'severity_score' in df.columns:
        high_speed = df[df['speed_limit'] >= 60]
        avg_severity_high = high_speed['severity_score'].mean() if len(high_speed) > 0 else 0
        low_speed = df[df['speed_limit'] < 60]
        avg_severity_low = low_speed['severity_score'].mean() if len(low_speed) > 0 else 0
        
        if avg_severity_high > avg_severity_low * 1.3:
            recommendations.append({
                'category': 'Speed Management',
                'priority': 'High',
                'issue': f'High-speed roads show {((avg_severity_high/avg_severity_low)-1)*100:.1f}% higher severity scores',
                'recommendation': 'Implement variable speed limits, install speed cameras, and add traffic calming measures.',
                'expected_impact': 'Potential 25-35% reduction in severe accidents on high-speed roads'
            })
    
    if 'weather' in df.columns:
        fog_accidents = df[df['weather'] == 'Fog']
        if len(fog_accidents) > 5:
            avg_fog_severity = fog_accidents['severity_score'].mean() if 'severity_score' in df.columns else 0
            recommendations.append({
                'category': 'Weather Warning Systems',
                'priority': 'Medium',
                'issue': f'{len(fog_accidents)} fog-related accidents with avg severity of {avg_fog_severity:.1f}',
                'recommendation': 'Install fog detection sensors, variable message signs, and automatic speed reduction systems.',
                'expected_impact': 'Potential 20-30% reduction in fog-related accidents'
            })
    
    if 'hour' in df.columns:
        rush_hour = df[df['hour'].isin([7, 8, 9, 16, 17, 18])]
        if len(rush_hour) > len(df) * 0.4:
            recommendations.append({
                'category': 'Traffic Flow Management',
                'priority': 'Medium',
                'issue': f'{len(rush_hour)} accidents ({len(rush_hour)/len(df)*100:.1f}%) during rush hours',
                'recommendation': 'Implement intelligent traffic signals, create dedicated bus lanes, and improve intersection timing.',
                'expected_impact': 'Potential 15-20% reduction in rush hour accidents'
            })
    
    return recommendations


def identify_hotspots(df: pd.DataFrame, grid_size: float = 0.01) -> pd.DataFrame:
    """
    Identify accident hotspots based on geographic clustering
    
    Args:
        df: DataFrame with latitude and longitude columns
        grid_size: Size of grid cells for clustering
        
    Returns:
        DataFrame with hotspot information
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return pd.DataFrame()
    
    df_hotspot = df.copy()
    df_hotspot['lat_grid'] = (df_hotspot['latitude'] / grid_size).round() * grid_size
    df_hotspot['lon_grid'] = (df_hotspot['longitude'] / grid_size).round() * grid_size
    
    hotspots = df_hotspot.groupby(['lat_grid', 'lon_grid']).agg({
        'accident_id': 'count',
        'severity_score': 'mean',
        'num_casualties': 'sum'
    }).reset_index()
    
    hotspots.columns = ['latitude', 'longitude', 'accident_count', 'avg_severity', 'total_casualties']
    
    hotspots = hotspots.sort_values('accident_count', ascending=False)
    
    return hotspots


def get_feature_importance_analysis(feature_names: list, importances: np.ndarray) -> pd.DataFrame:
    """
    Create a DataFrame with feature importance analysis
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importance values
        
    Returns:
        DataFrame with sorted feature importances
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    
    return importance_df


def auto_optimize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Automatically optimize uploaded data for ML training and analysis
    
    Args:
        df: Raw uploaded DataFrame
        
    Returns:
        Tuple of (optimized DataFrame, optimization report)
    """
    report = {
        'original_rows': len(df),
        'original_cols': len(df.columns),
        'changes': [],
        'warnings': [],
        'quality_score': 100
    }
    
    df_opt = df.copy()
    
    df_opt.columns = df_opt.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    column_mappings = {
        'lat': 'latitude', 'lng': 'longitude', 'lon': 'longitude',
        'accident_severity': 'severity', 'severity_level': 'severity',
        'weather_condition': 'weather', 'weather_conditions': 'weather',
        'road_condition': 'road_surface', 'surface_condition': 'road_surface',
        'number_of_vehicles': 'num_vehicles', 'vehicles': 'num_vehicles',
        'number_of_casualties': 'num_casualties', 'casualties': 'num_casualties',
        'light_condition': 'light_conditions', 'lighting': 'light_conditions',
        'junction': 'junction_type', 'junction_detail': 'junction_type',
        'accident_date': 'date', 'crash_date': 'date',
        'accident_time': 'time', 'crash_time': 'time',
        'speed': 'speed_limit', 'limit': 'speed_limit'
    }
    
    renamed = []
    for old_name, new_name in column_mappings.items():
        if old_name in df_opt.columns and new_name not in df_opt.columns:
            df_opt = df_opt.rename(columns={old_name: new_name})
            renamed.append(f"{old_name} -> {new_name}")
    
    if renamed:
        report['changes'].append(f"Renamed columns: {', '.join(renamed)}")
    
    original_len = len(df_opt)
    df_opt = df_opt.drop_duplicates()
    if len(df_opt) < original_len:
        removed = original_len - len(df_opt)
        report['changes'].append(f"Removed {removed} duplicate rows")
        report['quality_score'] -= min(10, removed / original_len * 100)
    
    for col in df_opt.select_dtypes(include=[np.number]).columns:
        missing = df_opt[col].isna().sum()
        if missing > 0:
            df_opt[col] = df_opt[col].fillna(df_opt[col].median())
            report['changes'].append(f"Filled {missing} missing values in '{col}' with median")
    
    for col in df_opt.select_dtypes(include=['object']).columns:
        missing = df_opt[col].isna().sum()
        if missing > 0:
            mode_val = df_opt[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_opt[col] = df_opt[col].fillna(fill_val)
            report['changes'].append(f"Filled {missing} missing values in '{col}' with mode")
    
    if 'severity' in df_opt.columns:
        severity_mappings = {
            '1': 'Minor', '2': 'Serious', '3': 'Fatal',
            'slight': 'Minor', 'minor': 'Minor', 'low': 'Minor',
            'serious': 'Serious', 'moderate': 'Serious', 'medium': 'Serious',
            'fatal': 'Fatal', 'severe': 'Fatal', 'high': 'Fatal', 'killed': 'Fatal'
        }
        df_opt['severity'] = df_opt['severity'].astype(str).str.lower().str.strip()
        df_opt['severity'] = df_opt['severity'].map(
            lambda x: severity_mappings.get(x, x.title())
        )
        valid_severities = ['Minor', 'Serious', 'Fatal']
        invalid_count = (~df_opt['severity'].isin(valid_severities)).sum()
        if invalid_count > 0:
            df_opt.loc[~df_opt['severity'].isin(valid_severities), 'severity'] = 'Minor'
            report['changes'].append(f"Standardized {invalid_count} severity values")
    
    if 'weather' in df_opt.columns:
        weather_mappings = {
            'fine': 'Clear', 'clear': 'Clear', 'dry': 'Clear', 'sunny': 'Clear',
            'rain': 'Rain', 'raining': 'Rain', 'wet': 'Rain', 'rainy': 'Rain',
            'fog': 'Fog', 'foggy': 'Fog', 'mist': 'Fog', 'haze': 'Fog',
            'snow': 'Snow', 'snowing': 'Snow', 'icy': 'Snow', 'ice': 'Snow'
        }
        df_opt['weather'] = df_opt['weather'].astype(str).str.lower().str.strip()
        df_opt['weather'] = df_opt['weather'].map(
            lambda x: weather_mappings.get(x, x.title())
        )
    
    if 'road_type' in df_opt.columns:
        road_mappings = {
            'urban': 'Urban', 'city': 'Urban', 'town': 'Urban',
            'rural': 'Rural', 'country': 'Rural', 'village': 'Rural',
            'highway': 'Motorway', 'motorway': 'Motorway', 'expressway': 'Motorway', 
            'freeway': 'Motorway', 'national_highway': 'Motorway'
        }
        df_opt['road_type'] = df_opt['road_type'].astype(str).str.lower().str.strip()
        df_opt['road_type'] = df_opt['road_type'].map(
            lambda x: road_mappings.get(x.replace(' ', '_'), x.title())
        )
    
    if 'latitude' in df_opt.columns:
        invalid_lat = (df_opt['latitude'] < -90) | (df_opt['latitude'] > 90)
        if invalid_lat.sum() > 0:
            report['warnings'].append(f"{invalid_lat.sum()} rows have invalid latitude values")
            report['quality_score'] -= 5
    
    if 'longitude' in df_opt.columns:
        invalid_lon = (df_opt['longitude'] < -180) | (df_opt['longitude'] > 180)
        if invalid_lon.sum() > 0:
            report['warnings'].append(f"{invalid_lon.sum()} rows have invalid longitude values")
            report['quality_score'] -= 5
    
    if 'accident_id' not in df_opt.columns:
        df_opt['accident_id'] = [f'ACC{str(i+1).zfill(6)}' for i in range(len(df_opt))]
        report['changes'].append("Generated accident IDs")
    
    report['final_rows'] = len(df_opt)
    report['final_cols'] = len(df_opt.columns)
    report['quality_score'] = max(0, report['quality_score'])
    
    return df_opt, report


def get_global_data_sources() -> list:
    """
    Returns a list of free global traffic accident data sources
    """
    return [
        {
            "name": "UK Road Safety Data",
            "country": "United Kingdom",
            "url": "https://data.gov.uk/dataset/road-accidents-safety-data",
            "description": "Official UK government accident data (1979-2022)",
            "format": "CSV"
        },
        {
            "name": "US NHTSA FARS",
            "country": "United States",
            "url": "https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars",
            "description": "Fatal accident data from NHTSA",
            "format": "CSV/SAS"
        },
        {
            "name": "India Open Data Portal",
            "country": "India",
            "url": "https://data.gov.in/search?keywords=road+accident",
            "description": "Indian government road accident statistics",
            "format": "CSV/Excel"
        },
        {
            "name": "Australia Road Deaths",
            "country": "Australia",
            "url": "https://www.bitre.gov.au/statistics/safety",
            "description": "Australian road crash data",
            "format": "CSV/Excel"
        },
        {
            "name": "European Road Safety Observatory",
            "country": "Europe",
            "url": "https://road-safety.transport.ec.europa.eu/statistics-and-analysis_en",
            "description": "EU-wide accident statistics",
            "format": "Various"
        },
        {
            "name": "Kaggle Datasets",
            "country": "Various",
            "url": "https://www.kaggle.com/datasets?search=traffic+accident",
            "description": "Community-contributed accident datasets",
            "format": "CSV"
        },
        {
            "name": "NYC Open Data - Crashes",
            "country": "USA (NYC)",
            "url": "https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95",
            "description": "New York City motor vehicle collision data",
            "format": "CSV/API"
        },
        {
            "name": "Brazil Traffic Data",
            "country": "Brazil",
            "url": "https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos",
            "description": "Brazilian federal highway accident data",
            "format": "CSV"
        }
    ]
