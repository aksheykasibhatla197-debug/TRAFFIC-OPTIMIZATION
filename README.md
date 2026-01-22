# TrafficSafe AI – Intelligent Road Risk & Design Optimization Platform

A production-grade, Google-quality traffic safety analysis platform with advanced machine learning, explainability, and what-if simulation capabilities.

## Overview

TrafficSafe AI transforms raw accident data into actionable insights for city planners, traffic engineers, and safety officials. The platform uses advanced ML algorithms to predict accident severity, detect high-risk zones, identify anomalies, and simulate the impact of safety interventions.

## Key Features

### 1. Executive Dashboard (Landing Page)
- **Circular Progress Indicators**: 4 animated ring charts showing Risk Level, Accident Data, Severity Analysis, and 24h Forecast
- **Most Affected Areas**: Color-coded cards (purple/orange/cyan) highlighting high-risk zones
- **Status Panel**: Donut charts for Risk Distribution (Minor/Serious/Fatal) and Hotspot Analysis (High/Medium/Low)
- **Accident Statistics**: Line chart tracking Minor, Serious, and Fatal accidents by month
- **Accident Overview**: Bar chart comparing Injured vs Deaths monthly trends
- **AI-Generated Summary**: Plain-English executive briefing with recommended actions
- **24-Hour Risk Prediction**: Forecast with confidence levels (High/Medium/Low)

### 2. Data Management
- **CSV Upload**: Upload your own accident data with automatic optimization
- **Form-Based Entry**: Manually add individual accident records
- **Auto Feature Engineering**: Columns renamed, values standardized, missing data filled
- **Auto-Retraining**: ML models automatically retrain when new data is added
- **Global Data Sources**: Links to free accident datasets from UK, US, India, Australia, EU, NYC, Brazil

### 3. AI Insights & Explainability
- **XGBoost Severity Prediction**: Classifies accidents as Minor/Serious/Fatal (95%+ accuracy)
- **Poisson Regression**: Models accident frequency rates
- **Negative Binomial Regression**: Handles overdispersed count data
- **DBSCAN Hotspot Detection**: Automatically clusters accident locations
- **Isolation Forest Anomaly Detection**: Identifies unusual accident patterns
- **Plain-English Explanations**: Each feature's impact explained simply (e.g., "Higher speed limits are associated with more severe accidents")
- **Feature Importance Visualization**: Color-coded charts showing what factors matter most

### 4. Road Design Optimization
- **Prioritized Recommendations**: Ranked list of safety improvements
- **Impact Scores**: 0-100 score for each recommendation's potential effect
- **Categories**: Speed Control, Junction Redesign, Lighting, Road Surface, Traffic Calming, Signage
- **Cost Estimates**: Low/Medium/High implementation costs
- **Expected Reduction**: Projected accident reduction percentages

### 5. What-If Simulation
- **Intervention Types**:
  - Speed Limit Reduction (10 km/h lower in high-risk zones)
  - Improved Street Lighting
  - Junction Redesign (convert to roundabouts)
  - Real-Time Weather Warning Systems
  - Comprehensive (all interventions combined)
- **Before/After Comparison**: Side-by-side risk score comparison
- **Projected Savings**: Estimated fatal accidents and casualties prevented
- **Lives Saved Estimate**: Prominently displayed calculation of lives saved

### 6. Interactive Accident Map
- **Filterable View**: Filter by severity, weather, road type
- **Hotspot Visualization**: Circles around high-risk clusters
- **Heatmap Mode**: Density visualization of accident locations
- **Location Detection**: Automatically identifies city (Delhi, Mumbai, London, New York, etc.)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Streamlit |
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualizations | Matplotlib, Seaborn |
| ML Classification | XGBoost, Scikit-learn |
| Statistical Models | Statsmodels (Poisson, Negative Binomial) |
| Clustering | DBSCAN (Scikit-learn) |
| Anomaly Detection | Isolation Forest |
| Model Persistence | Joblib (with versioning) |
| Interactive Maps | Folium, Streamlit-Folium |

## Project Structure

```
├── app.py                      # Main Streamlit app (6 pages)
├── advanced_ml.py              # Advanced ML pipeline
│   ├── TrafficSafeML class     # City risk score, DBSCAN, Isolation Forest
│   ├── Poisson/NegBin models   # Frequency modeling
│   ├── XGBoost training        # Severity prediction with versioning
│   └── What-if simulation      # Intervention impact forecasting
├── train_model.py              # Basic ML training pipeline
├── utils.py                    # Data processing & optimization
├── data/
│   └── sample_accidents.csv    # Sample dataset (100 records)
├── models/                     # Trained model artifacts
│   ├── xgb_severity_latest.joblib
│   ├── poisson_model.joblib
│   └── negbin_model.joblib
├── .streamlit/config.toml      # Streamlit configuration
└── README.md                   # This file
```

## How to Run

### On Replit
1. Click the **Run** button
2. The app opens automatically at port 5000

### Locally
```bash
# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels folium streamlit-folium joblib

# Run the app
streamlit run app.py --server.port 5000
```

## Machine Learning Pipeline

### Severity Prediction (XGBoost)
- **Input Features**: Speed limit, vehicles involved, casualties, weather, road type, lighting, surface conditions, junction type
- **Output**: Minor / Serious / Fatal classification
- **Accuracy**: ~95%
- **Model Versioning**: Automatic timestamped saves (YYYYMMDD_HHMMSS format)

### Frequency Modeling
- **Poisson Regression**: For modeling accident counts
- **Negative Binomial**: For overdispersed data (variance > mean)
- **Output**: Pseudo R-squared, significant predictors

### Hotspot Detection (DBSCAN)
- **Parameters**: Configurable cluster radius and minimum points
- **Output**: Cluster centers, accident counts, risk levels
- **Silhouette Score**: Clustering quality metric

### Anomaly Detection (Isolation Forest)
- **Contamination Rate**: Adjustable expected anomaly percentage
- **Output**: Flagged unusual accidents for investigation

## Sample Dataset

The included sample dataset contains 100 synthetic accident records with:

| Field | Description |
|-------|-------------|
| accident_id | Unique identifier |
| date, time | When the accident occurred |
| latitude, longitude | Location coordinates |
| severity | Minor, Serious, or Fatal |
| weather | Clear, Rain, Fog, Snow |
| road_type | Urban, Rural, Motorway |
| speed_limit | Posted speed limit (km/h) |
| num_vehicles | Vehicles involved |
| num_casualties | People injured/killed |
| light_conditions | Daylight, Dark with/without lights |
| road_surface | Dry, Wet, Ice, Snow |
| junction_type | Type of intersection |

## Professional UI Design

- **Modern Dashboard**: Circular progress indicators with animated ring charts
- **Color Scheme**: Marvel Universe theme - Red (#ED1D24), Gold (#FFD700), Blue (#0072C6)
- **Poppins Font**: Clean, modern typography throughout
- **Donut Charts**: Risk distribution and hotspot analysis visualizations
- **Line & Bar Charts**: Monthly accident trends with smooth styling
- **Card-Based Layout**: Soft shadows, rounded corners, subtle borders
- **Impact Badges**: Visual priority markers on recommendations
- **Plain Language**: Technical jargon translated for non-technical users

## API Reference

### TrafficSafeML Class
```python
from advanced_ml import TrafficSafeML

ml = TrafficSafeML()

# Calculate city-wide risk score
risk = ml.calculate_city_risk_score(df)
# Returns: {'score': 56.5, 'level': 'Medium', 'color': 'orange', 'components': {...}}

# Detect hotspots
hotspots, stats = ml.detect_hotspots_dbscan(df, eps=0.01, min_samples=3)

# Detect anomalies
df_anomalies, stats = ml.detect_anomalies(df, contamination=0.1)

# Train severity model
result = ml.train_severity_model(df)
# Returns: {'trained': True, 'accuracy': 0.95, 'f1_score': 0.95, 'feature_importance': {...}}

# Simulate intervention
simulation = ml.simulate_intervention(df, 'speed_reduction')
# Returns: {'before': {...}, 'after': {...}, 'reduction': {...}}
```

## Deployment

### Replit (Recommended)
1. Fork this project
2. Click **Run**
3. Use the **Publish** button to deploy publicly

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["streamlit", "run", "app.py", "--server.port=5000"]
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Traffic safety research community
- Open government data initiatives worldwide
- Streamlit, Scikit-learn, XGBoost, and Statsmodels teams
