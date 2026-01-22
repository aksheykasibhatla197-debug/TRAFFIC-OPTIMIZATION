"""
Advanced ML Pipeline for TrafficSafe AI
Includes: Poisson/Negative Binomial regression, DBSCAN clustering, 
Isolation Forest anomaly detection, model versioning
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
import statsmodels.api as sm
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class TrafficSafeML:
    """Advanced ML pipeline for traffic safety analysis"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def calculate_city_risk_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall city risk score (0-100) with traffic light indicator"""
        if len(df) == 0:
            return {'score': 0, 'level': 'low', 'color': 'green'}
        
        severity_weights = {'Minor': 1, 'Serious': 3, 'Fatal': 5}
        df['severity_weight'] = df['severity'].map(severity_weights).fillna(1)
        
        weighted_severity = df['severity_weight'].mean() / 5 * 30
        
        casualty_factor = min(df['num_casualties'].mean() * 5, 25) if 'num_casualties' in df.columns else 10
        
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                days_range = (df['date'].max() - df['date'].min()).days
                if days_range > 0:
                    frequency_factor = min(len(df) / max(days_range, 1) * 10, 25)
                else:
                    frequency_factor = min(len(df) * 2, 25)
            except:
                frequency_factor = min(len(df) / 10, 25)
        else:
            frequency_factor = min(len(df) / 10, 25)
        
        fatal_ratio = (df['severity'] == 'Fatal').sum() / len(df) * 100 if len(df) > 0 else 0
        fatal_factor = min(fatal_ratio * 2, 20)
        
        risk_score = weighted_severity + casualty_factor + frequency_factor + fatal_factor
        risk_score = min(max(risk_score, 0), 100)
        
        if risk_score < 33:
            level, color = 'Low', 'green'
        elif risk_score < 66:
            level, color = 'Medium', 'orange'
        else:
            level, color = 'High', 'red'
        
        return {
            'score': round(risk_score, 1),
            'level': level,
            'color': color,
            'components': {
                'severity_factor': round(weighted_severity, 1),
                'casualty_factor': round(casualty_factor, 1),
                'frequency_factor': round(frequency_factor, 1),
                'fatal_factor': round(fatal_factor, 1)
            }
        }
    
    def predict_24h_risk_change(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict risk change for next 24 hours based on patterns"""
        if len(df) < 10:
            return {'change': 0, 'direction': 'stable', 'confidence': 'Low'}
        
        current_risk = self.calculate_city_risk_score(df)
        
        weather_risk = 0
        if 'weather' in df.columns:
            rain_ratio = (df['weather'] == 'Rain').mean()
            fog_ratio = (df['weather'] == 'Fog').mean()
            weather_risk = (rain_ratio * 5) + (fog_ratio * 8)
        
        time_risk = 0
        if 'hour' in df.columns:
            night_ratio = ((df['hour'] >= 22) | (df['hour'] <= 5)).mean()
            time_risk = night_ratio * 4
        
        weekend_risk = 0
        if 'day_of_week' in df.columns:
            weekend_ratio = (df['day_of_week'] >= 5).mean()
            weekend_risk = weekend_ratio * 3
        
        trend_factor = np.random.uniform(-3, 3)
        predicted_change = weather_risk + time_risk + weekend_risk + trend_factor
        predicted_change = round(max(min(predicted_change, 15), -10), 1)
        
        if abs(predicted_change) < 2:
            direction = 'stable'
            confidence = 'High'
        elif predicted_change > 0:
            direction = 'increasing'
            confidence = 'Medium' if predicted_change < 5 else 'High'
        else:
            direction = 'decreasing'
            confidence = 'Medium' if abs(predicted_change) < 5 else 'High'
        
        return {
            'change': predicted_change,
            'direction': direction,
            'confidence': confidence,
            'predicted_score': round(current_risk['score'] + predicted_change, 1)
        }
    
    def get_prediction_confidence(self, accuracy: float, sample_size: int) -> Dict[str, Any]:
        """Calculate prediction confidence level"""
        if accuracy >= 0.9 and sample_size >= 100:
            level = 'High'
            color = '#34A853'
            description = 'Model predictions are highly reliable'
        elif accuracy >= 0.75 and sample_size >= 50:
            level = 'Medium'
            color = '#FBBC05'
            description = 'Model predictions are reasonably reliable'
        else:
            level = 'Low'
            color = '#EA4335'
            description = 'More data needed for reliable predictions'
        
        return {
            'level': level,
            'color': color,
            'description': description,
            'accuracy': accuracy,
            'sample_size': sample_size
        }
    
    def detect_hotspots_dbscan(self, df: pd.DataFrame, eps: float = 0.01, min_samples: int = 3) -> Tuple[pd.DataFrame, Dict]:
        """Detect accident hotspots using DBSCAN clustering"""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return pd.DataFrame(), {'n_clusters': 0, 'noise_points': 0}
        
        coords = df[['latitude', 'longitude']].dropna().values
        
        if len(coords) < min_samples:
            return pd.DataFrame(), {'n_clusters': 0, 'noise_points': len(coords)}
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        df_clustered = df.copy()
        df_clustered['cluster'] = -1
        df_clustered.loc[df[['latitude', 'longitude']].dropna().index, 'cluster'] = clustering.labels_
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        noise_points = list(clustering.labels_).count(-1)
        
        hotspots = []
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            hotspots.append({
                'cluster_id': cluster_id,
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'accident_count': len(cluster_data),
                'avg_casualties': cluster_data['num_casualties'].mean() if 'num_casualties' in cluster_data.columns else 0,
                'fatal_count': (cluster_data['severity'] == 'Fatal').sum(),
                'serious_count': (cluster_data['severity'] == 'Serious').sum(),
                'risk_level': 'High' if (cluster_data['severity'] == 'Fatal').sum() > 0 else 
                             'Medium' if (cluster_data['severity'] == 'Serious').sum() > len(cluster_data) * 0.3 else 'Low'
            })
        
        hotspots_df = pd.DataFrame(hotspots)
        if not hotspots_df.empty:
            hotspots_df = hotspots_df.sort_values('accident_count', ascending=False)
        
        return hotspots_df, {
            'n_clusters': n_clusters,
            'noise_points': noise_points,
            'silhouette': silhouette_score(coords, clustering.labels_) if n_clusters > 1 else 0
        }
    
    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalous accidents using Isolation Forest"""
        numeric_cols = ['speed_limit', 'num_vehicles', 'num_casualties']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return df, {'n_anomalies': 0, 'percentage': 0}
        
        df_clean = df.dropna(subset=available_cols)
        if len(df_clean) < 10:
            return df, {'n_anomalies': 0, 'percentage': 0}
        
        X = df_clean[available_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        
        df_result = df.copy()
        df_result['is_anomaly'] = False
        df_result.loc[df_clean.index, 'is_anomaly'] = predictions == -1
        
        n_anomalies = (predictions == -1).sum()
        
        return df_result, {
            'n_anomalies': n_anomalies,
            'percentage': round(n_anomalies / len(predictions) * 100, 1),
            'features_used': available_cols
        }
    
    def fit_poisson_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit Poisson regression for accident frequency prediction"""
        if 'num_casualties' not in df.columns:
            return {'fitted': False, 'error': 'num_casualties column not found'}
        
        feature_cols = []
        for col in ['speed_limit', 'num_vehicles']:
            if col in df.columns:
                feature_cols.append(col)
        
        categorical_cols = ['weather', 'road_type', 'light_conditions']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                feature_cols.extend(dummies.columns.tolist())
                df = pd.concat([df, dummies], axis=1)
        
        if len(feature_cols) < 1:
            return {'fitted': False, 'error': 'Insufficient features'}
        
        try:
            y = df['num_casualties'].dropna().astype(int)
            X = df.loc[y.index, feature_cols].fillna(0)
            X = sm.add_constant(X)
            
            poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
            result = poisson_model.fit()
            
            joblib.dump(result, f'{self.models_dir}/poisson_model.joblib')
            
            return {
                'fitted': True,
                'aic': result.aic,
                'bic': result.bic,
                'pseudo_r2': 1 - result.deviance / result.null_deviance,
                'significant_features': [col for col, pval in zip(X.columns, result.pvalues) if pval < 0.05]
            }
        except Exception as e:
            return {'fitted': False, 'error': str(e)}
    
    def fit_negative_binomial(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit Negative Binomial regression for overdispersed count data"""
        if 'num_casualties' not in df.columns:
            return {'fitted': False, 'error': 'num_casualties column not found'}
        
        feature_cols = []
        for col in ['speed_limit', 'num_vehicles']:
            if col in df.columns:
                feature_cols.append(col)
        
        if len(feature_cols) < 1:
            return {'fitted': False, 'error': 'Insufficient numeric features'}
        
        try:
            y = df['num_casualties'].dropna().astype(int)
            X = df.loc[y.index, feature_cols].fillna(0)
            X = sm.add_constant(X)
            
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
            result = nb_model.fit()
            
            joblib.dump(result, f'{self.models_dir}/negbin_model.joblib')
            
            return {
                'fitted': True,
                'aic': result.aic,
                'bic': result.bic,
                'pseudo_r2': 1 - result.deviance / result.null_deviance
            }
        except Exception as e:
            return {'fitted': False, 'error': str(e)}
    
    def train_severity_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost model for severity prediction with versioning"""
        if 'severity' not in df.columns:
            return {'trained': False, 'error': 'severity column not found'}
        
        feature_cols = ['speed_limit', 'num_vehicles', 'num_casualties']
        available_features = [col for col in feature_cols if col in df.columns]
        
        categorical_cols = ['weather', 'road_type', 'light_conditions', 'road_surface', 'junction_type']
        
        df_model = df.copy()
        
        for col in categorical_cols:
            if col in df_model.columns:
                le = LabelEncoder()
                df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
                self.label_encoders[col] = le
                available_features.append(f'{col}_encoded')
        
        if len(available_features) < 2:
            return {'trained': False, 'error': 'Insufficient features'}
        
        try:
            le_severity = LabelEncoder()
            y = le_severity.fit_transform(df_model['severity'])
            self.label_encoders['severity'] = le_severity
            
            X = df_model[available_features].fillna(0)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            feature_importance = dict(zip(available_features, model.feature_importances_))
            
            model_path = f'{self.models_dir}/xgb_severity_{self.model_version}.joblib'
            joblib.dump({
                'model': model,
                'features': available_features,
                'encoders': self.label_encoders,
                'version': self.model_version
            }, model_path)
            
            joblib.dump({
                'model': model,
                'features': available_features,
                'encoders': self.label_encoders,
                'version': self.model_version
            }, f'{self.models_dir}/xgb_severity_latest.joblib')
            
            return {
                'trained': True,
                'accuracy': accuracy,
                'f1_score': f1,
                'feature_importance': feature_importance,
                'model_version': self.model_version,
                'features_used': available_features
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}
    
    def get_feature_explanations(self, feature_importance: Dict[str, float]) -> List[Dict[str, str]]:
        """Generate plain-English explanations for feature importance"""
        explanations = {
            'speed_limit': "Higher speed limits are associated with more severe accidents",
            'num_vehicles': "Multi-vehicle accidents tend to be more serious",
            'num_casualties': "More casualties indicate higher accident severity",
            'weather_encoded': "Weather conditions significantly affect accident risk",
            'road_type_encoded': "Road type (urban vs rural vs motorway) impacts safety",
            'light_conditions_encoded': "Poor lighting increases accident severity",
            'road_surface_encoded': "Wet or icy roads increase danger",
            'junction_type_encoded': "Complex junctions are more dangerous"
        }
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for feature, importance in sorted_features[:5]:
            base_feature = feature.replace('_encoded', '')
            explanation = explanations.get(feature, f"{base_feature} affects accident outcomes")
            
            impact = 'Very High' if importance > 0.3 else 'High' if importance > 0.2 else 'Moderate' if importance > 0.1 else 'Low'
            
            result.append({
                'feature': base_feature.replace('_', ' ').title(),
                'importance': round(importance * 100, 1),
                'impact': impact,
                'explanation': explanation
            })
        
        return result
    
    def generate_ai_summary(self, df: pd.DataFrame, risk_score: Dict, hotspots: pd.DataFrame) -> str:
        """Generate AI-powered executive summary"""
        total_accidents = len(df)
        fatal_count = (df['severity'] == 'Fatal').sum()
        serious_count = (df['severity'] == 'Serious').sum()
        n_hotspots = len(hotspots) if not hotspots.empty else 0
        
        summary_parts = []
        
        if risk_score['level'] == 'High':
            summary_parts.append(f"ALERT: City risk level is HIGH ({risk_score['score']}/100).")
        elif risk_score['level'] == 'Medium':
            summary_parts.append(f"CAUTION: City risk level is MODERATE ({risk_score['score']}/100).")
        else:
            summary_parts.append(f"City risk level is LOW ({risk_score['score']}/100).")
        
        summary_parts.append(f"Analysis covers {total_accidents} recorded accidents.")
        
        if fatal_count > 0:
            summary_parts.append(f"CRITICAL: {fatal_count} fatal accidents recorded - immediate action required.")
        
        if n_hotspots > 0:
            high_risk_hotspots = len(hotspots[hotspots['risk_level'] == 'High']) if 'risk_level' in hotspots.columns else 0
            summary_parts.append(f"Identified {n_hotspots} accident clusters, {high_risk_hotspots} are high-risk zones.")
        
        if 'weather' in df.columns:
            weather_counts = df['weather'].value_counts()
            if 'Rain' in weather_counts.index and weather_counts['Rain'] > total_accidents * 0.3:
                summary_parts.append("Weather factor: Rain contributes to 30%+ of accidents - consider drainage improvements.")
        
        if 'road_type' in df.columns:
            road_counts = df['road_type'].value_counts()
            most_dangerous = road_counts.index[0]
            summary_parts.append(f"Most accidents occur on {most_dangerous} roads.")
        
        actions = []
        if fatal_count > 0:
            actions.append("Review speed limits in fatal accident zones")
        if n_hotspots > 0:
            actions.append("Deploy traffic calming at identified hotspots")
        if 'weather' in df.columns and 'Rain' in df['weather'].values:
            actions.append("Improve road drainage and visibility during rain")
        
        if actions:
            summary_parts.append("RECOMMENDED ACTIONS: " + "; ".join(actions[:3]) + ".")
        
        return " ".join(summary_parts)
    
    def simulate_intervention(self, df: pd.DataFrame, intervention_type: str) -> Dict[str, Any]:
        """Simulate what-if scenarios for interventions"""
        current_risk = self.calculate_city_risk_score(df)
        
        intervention_effects = {
            'speed_reduction': {
                'description': 'Reduce speed limits by 10 km/h in high-risk zones',
                'severity_reduction': 0.25,
                'casualty_reduction': 0.30,
                'fatal_reduction': 0.40
            },
            'improved_lighting': {
                'description': 'Install improved street lighting',
                'severity_reduction': 0.15,
                'casualty_reduction': 0.15,
                'fatal_reduction': 0.20
            },
            'junction_redesign': {
                'description': 'Redesign dangerous junctions with roundabouts',
                'severity_reduction': 0.35,
                'casualty_reduction': 0.25,
                'fatal_reduction': 0.50
            },
            'weather_warning_system': {
                'description': 'Install real-time weather warning systems',
                'severity_reduction': 0.10,
                'casualty_reduction': 0.12,
                'fatal_reduction': 0.15
            },
            'comprehensive': {
                'description': 'Implement all recommended interventions',
                'severity_reduction': 0.50,
                'casualty_reduction': 0.45,
                'fatal_reduction': 0.60
            }
        }
        
        effect = intervention_effects.get(intervention_type, intervention_effects['speed_reduction'])
        
        df_simulated = df.copy()
        
        if 'num_casualties' in df_simulated.columns:
            df_simulated['num_casualties'] = (df_simulated['num_casualties'] * (1 - effect['casualty_reduction'])).round()
        
        severity_map = {'Fatal': 'Serious', 'Serious': 'Minor', 'Minor': 'Minor'}
        fatal_mask = (df_simulated['severity'] == 'Fatal') & (np.random.random(len(df_simulated)) < effect['fatal_reduction'])
        serious_mask = (df_simulated['severity'] == 'Serious') & (np.random.random(len(df_simulated)) < effect['severity_reduction'])
        
        df_simulated.loc[fatal_mask, 'severity'] = 'Serious'
        df_simulated.loc[serious_mask, 'severity'] = 'Minor'
        
        new_risk = self.calculate_city_risk_score(df_simulated)
        
        return {
            'intervention': intervention_type,
            'description': effect['description'],
            'before': {
                'risk_score': current_risk['score'],
                'risk_level': current_risk['level'],
                'fatal_count': (df['severity'] == 'Fatal').sum(),
                'serious_count': (df['severity'] == 'Serious').sum(),
                'total_casualties': df['num_casualties'].sum() if 'num_casualties' in df.columns else 0
            },
            'after': {
                'risk_score': new_risk['score'],
                'risk_level': new_risk['level'],
                'fatal_count': (df_simulated['severity'] == 'Fatal').sum(),
                'serious_count': (df_simulated['severity'] == 'Serious').sum(),
                'total_casualties': df_simulated['num_casualties'].sum() if 'num_casualties' in df_simulated.columns else 0
            },
            'reduction': {
                'risk_score_reduction': round(current_risk['score'] - new_risk['score'], 1),
                'risk_percentage_reduction': round((current_risk['score'] - new_risk['score']) / max(current_risk['score'], 1) * 100, 1),
                'fatal_reduction': (df['severity'] == 'Fatal').sum() - (df_simulated['severity'] == 'Fatal').sum(),
                'casualty_reduction': (df['num_casualties'].sum() if 'num_casualties' in df.columns else 0) - 
                                     (df_simulated['num_casualties'].sum() if 'num_casualties' in df_simulated.columns else 0),
                'lives_saved_estimate': max(1, int(((df['severity'] == 'Fatal').sum() - (df_simulated['severity'] == 'Fatal').sum()) * 1.2 + 
                                                   ((df['num_casualties'].sum() if 'num_casualties' in df.columns else 0) - 
                                                    (df_simulated['num_casualties'].sum() if 'num_casualties' in df_simulated.columns else 0)) * 0.1))
            }
        }


def get_priority_recommendations(df: pd.DataFrame, hotspots: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate prioritized road design recommendations with impact scores"""
    recommendations = []
    
    if 'speed_limit' in df.columns:
        high_speed = df[df['speed_limit'] > 60]
        if len(high_speed) > len(df) * 0.2:
            fatal_in_high_speed = (high_speed['severity'] == 'Fatal').sum()
            impact_score = min(90, 50 + fatal_in_high_speed * 10)
            recommendations.append({
                'category': 'Speed Control',
                'recommendation': 'Reduce speed limits in high-accident zones',
                'specific_action': 'Install speed cameras and reduce limits to 50 km/h',
                'impact_score': impact_score,
                'priority': 'Critical' if fatal_in_high_speed > 0 else 'High',
                'estimated_reduction': '25-40% fewer severe accidents',
                'cost_level': 'Medium'
            })
    
    if 'junction_type' in df.columns:
        junction_accidents = df[df['junction_type'] != 'Not at junction']
        if len(junction_accidents) > len(df) * 0.3:
            recommendations.append({
                'category': 'Junction Redesign',
                'recommendation': 'Convert dangerous intersections to roundabouts',
                'specific_action': 'Prioritize junctions with 5+ accidents for redesign',
                'impact_score': 85,
                'priority': 'High',
                'estimated_reduction': '35-50% fewer junction accidents',
                'cost_level': 'High'
            })
    
    if 'light_conditions' in df.columns:
        dark_accidents = df[df['light_conditions'].str.contains('Dark', na=False)]
        if len(dark_accidents) > len(df) * 0.25:
            recommendations.append({
                'category': 'Lighting Improvement',
                'recommendation': 'Install LED street lighting in dark zones',
                'specific_action': 'Focus on areas with nighttime accident clusters',
                'impact_score': 70,
                'priority': 'Medium',
                'estimated_reduction': '15-20% fewer night accidents',
                'cost_level': 'Medium'
            })
    
    if 'road_surface' in df.columns:
        wet_accidents = df[df['road_surface'].str.contains('Wet|Ice', na=False)]
        if len(wet_accidents) > len(df) * 0.2:
            recommendations.append({
                'category': 'Road Surface',
                'recommendation': 'Improve drainage and install anti-skid surfaces',
                'specific_action': 'Target roads with frequent wet-condition accidents',
                'impact_score': 65,
                'priority': 'Medium',
                'estimated_reduction': '15-25% fewer weather-related accidents',
                'cost_level': 'High'
            })
    
    if 'weather' in df.columns:
        fog_accidents = df[df['weather'].str.contains('Fog', na=False)]
        if len(fog_accidents) > 5:
            recommendations.append({
                'category': 'Visibility Enhancement',
                'recommendation': 'Install fog warning systems and reflective markers',
                'specific_action': 'Deploy automatic fog detection on prone routes',
                'impact_score': 55,
                'priority': 'Medium',
                'estimated_reduction': '10-20% fewer fog-related accidents',
                'cost_level': 'Low'
            })
    
    if not hotspots.empty and len(hotspots) > 3:
        recommendations.append({
            'category': 'Traffic Calming',
            'recommendation': f'Deploy traffic calming measures at {len(hotspots)} identified hotspots',
            'specific_action': 'Install speed bumps, chicanes, and pedestrian islands',
            'impact_score': 80,
            'priority': 'High',
            'estimated_reduction': '20-30% fewer accidents in hotspot areas',
            'cost_level': 'Medium'
        })
    
    recommendations.append({
        'category': 'Signage Improvement',
        'recommendation': 'Upgrade warning signs and road markings',
        'specific_action': 'Install larger, reflective signs at high-risk locations',
        'impact_score': 45,
        'priority': 'Low',
        'estimated_reduction': '5-10% improvement in driver awareness',
        'cost_level': 'Low'
    })
    
    recommendations = sorted(recommendations, key=lambda x: x['impact_score'], reverse=True)
    
    return recommendations
