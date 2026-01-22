# TrafficSafe AI – Intelligent Road Risk & Design Optimization Platform

## Project Overview
A production-grade, Google-quality traffic safety analysis platform with advanced ML, explainability, and what-if simulation capabilities.

## Tech Stack
- Python 3.11
- Streamlit (Web UI on port 5000)
- Pandas, NumPy (Data processing)
- Matplotlib, Seaborn (Visualizations)
- Scikit-learn, XGBoost (ML models)
- Statsmodels (Poisson/Negative Binomial regression)
- Joblib (Model persistence & versioning)
- Folium, Streamlit-Folium (Interactive maps)
- DBSCAN (Hotspot clustering)
- Isolation Forest (Anomaly detection)

## Project Structure
```
├── app.py                      # Main Streamlit app (6 pages)
├── advanced_ml.py              # Advanced ML pipeline (DBSCAN, Isolation Forest, simulations)
├── train_model.py              # Basic ML training pipeline
├── utils.py                    # Data processing & optimization utilities
├── data/
│   └── sample_accidents.csv    # Sample dataset (100 records)
├── models/                     # Trained model artifacts with versioning
├── .streamlit/config.toml      # Streamlit configuration
└── README.md                   # Documentation
```

## How to Run
```bash
streamlit run app.py
```

## Key Features
1. **Executive Dashboard**: City risk score (0-100), traffic-light indicator, high-risk zones, AI summary
2. **Data Management**: CSV upload, form-based manual entry, auto-optimization, global data sources
3. **AI Insights**: XGBoost training, DBSCAN hotspot detection, Isolation Forest anomaly detection, plain-English explanations
4. **Road Design Optimization**: Prioritized recommendations with impact scores
5. **What-If Simulation**: Before/after risk comparison for interventions (speed reduction, lighting, junctions)
6. **Accident Map**: Interactive world map with heatmaps, location detection

## Advanced ML Pipeline
- **Severity Prediction**: XGBoost classifier with feature importance
- **Frequency Modeling**: Poisson & Negative Binomial regression (statsmodels)
- **Hotspot Detection**: DBSCAN clustering with configurable parameters
- **Anomaly Detection**: Isolation Forest for unusual accident patterns
- **Model Versioning**: Automatic timestamped model saves in models/registry/
- **Auto-retraining**: Automatic model retraining when new data is uploaded or entered manually
- **Explainability**: Gradient-based feature importance with plain-English explanations
- **24-Hour Risk Prediction**: Forecast risk changes with confidence levels (High/Medium/Low)
- **Lives Saved Estimation**: What-If simulation calculates estimated lives saved from interventions

## Professional UI
- Cyberpunk-themed dashboard with neon colors (cyan #00fff9, magenta #ff00ff, yellow #fffc00, red #ff0040)
- Dark background with glowing effects and animated elements
- Orbitron + Rajdhani fonts for futuristic look
- Real-time traffic data integration (TomTom API)
- Live incident feed with severity indicators
- Donut charts for risk distribution and hotspot analysis
- Line/bar charts with dark theme and neon accents
- Neural Analysis Engine AI summaries
- Impact score badges with glow effects
- Before/after simulation comparisons

## Recent Changes
- January 2026: Cyberpunk UI redesign with neon colors and dark theme
- January 2026: Added TomTom real-time traffic API integration
- January 2026: Added live incident feed with city selector (20+ cities worldwide)
- January 2026: Complete redesign to TrafficSafe AI with Google-quality UI
- January 2026: Added What-If simulation with intervention forecasting
- January 2026: Added advanced ML: DBSCAN, Isolation Forest, Poisson regression
- January 2026: Added AI-generated executive summaries
- January 2026: Added form-based manual accident entry
- January 2026: Added feature importance with plain-English explanations
- January 2026: Added automatic data optimization
- January 2026: Initial application created
