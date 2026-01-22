"""
Model Training Module for Traffic Accident Risk Prediction
Trains multiple ML models and selects the best one based on performance metrics
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils import (
    load_data, clean_data, extract_time_features,
    encode_categorical_features, calculate_risk_index,
    prepare_features_for_training
)


def preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete data preprocessing pipeline
    
    Args:
        file_path: Path to the CSV data file
        
    Returns:
        Tuple of (processed DataFrame, encoding mappings)
    """
    df = load_data(file_path)
    
    df = clean_data(df)
    
    df = extract_time_features(df)
    
    df, encodings = encode_categorical_features(df)
    
    df = calculate_risk_index(df)
    
    return df, encodings


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train an XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained XGBClassifier model
    """
    if not XGBOOST_AVAILABLE:
        return None
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, float]:
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred
    }


def select_best_model(results: list) -> Dict:
    """
    Select the best model based on F1 score
    
    Args:
        results: List of model evaluation results
        
    Returns:
        Best model result dictionary
    """
    best_result = max(results, key=lambda x: x['f1_score'])
    return best_result


def train_and_evaluate_models(file_path: str = 'data/sample_accidents.csv') -> Dict[str, Any]:
    """
    Main training pipeline - trains multiple models and selects the best
    
    Args:
        file_path: Path to the training data
        
    Returns:
        Dictionary with training results and best model
    """
    print("Loading and preprocessing data...")
    df, encodings = preprocess_data(file_path)
    
    print("Preparing features for training...")
    X, y = prepare_features_for_training(df)
    
    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    models = {}
    results = []
    
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    models['Logistic Regression'] = lr_model
    lr_result = evaluate_model(lr_model, X_test_scaled, y_test, 'Logistic Regression')
    results.append(lr_result)
    print(f"  Accuracy: {lr_result['accuracy']:.4f}, F1 Score: {lr_result['f1_score']:.4f}")
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    rf_result = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    results.append(rf_result)
    print(f"  Accuracy: {rf_result['accuracy']:.4f}, F1 Score: {rf_result['f1_score']:.4f}")
    
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        if xgb_model is not None:
            models['XGBoost'] = xgb_model
            xgb_result = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
            results.append(xgb_result)
            print(f"  Accuracy: {xgb_result['accuracy']:.4f}, F1 Score: {xgb_result['f1_score']:.4f}")
    else:
        print("XGBoost not available, skipping...")
    
    best_result = select_best_model(results)
    best_model = models[best_result['model_name']]
    
    print(f"\nBest Model: {best_result['model_name']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 Score: {best_result['f1_score']:.4f}")
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/best_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")
    
    scaler_path = 'models/scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    encodings_path = 'models/encodings.joblib'
    joblib.dump(encodings, encodings_path)
    print(f"Encodings saved to {encodings_path}")
    
    feature_names_path = 'models/feature_names.joblib'
    joblib.dump(X.columns.tolist(), feature_names_path)
    print(f"Feature names saved to {feature_names_path}")
    
    feature_importances = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importances = np.abs(best_model.coef_).mean(axis=0)
    
    return {
        'best_model_name': best_result['model_name'],
        'accuracy': best_result['accuracy'],
        'f1_score': best_result['f1_score'],
        'all_results': results,
        'feature_names': X.columns.tolist(),
        'feature_importances': feature_importances,
        'encodings': encodings,
        'processed_data': df
    }


def load_trained_model() -> Tuple[Any, Any, Dict, list]:
    """
    Load the trained model and associated artifacts
    
    Returns:
        Tuple of (model, scaler, encodings, feature_names)
    """
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    encodings = joblib.load('models/encodings.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    return model, scaler, encodings, feature_names


def predict_risk(input_data: Dict, model: Any, scaler: Any, 
                encodings: Dict, feature_names: list) -> Dict:
    """
    Make risk prediction for new input data
    
    Args:
        input_data: Dictionary with input features
        model: Trained model
        scaler: Fitted scaler
        encodings: Feature encodings
        feature_names: List of feature names
        
    Returns:
        Dictionary with prediction results
    """
    input_df = pd.DataFrame([input_data])
    
    for col, mapping in encodings.items():
        if col in input_df.columns:
            input_df[f'{col}_encoded'] = input_df[col].map(mapping).fillna(0)
    
    available_features = [f for f in feature_names if f in input_df.columns]
    missing_features = [f for f in feature_names if f not in input_df.columns]
    
    for f in missing_features:
        input_df[f] = 0
    
    X_input = input_df[feature_names]
    
    model_name = type(model).__name__
    if model_name == 'LogisticRegression':
        X_input = scaler.transform(X_input)
    
    prediction = model.predict(X_input)[0]
    
    severity_labels = {0: 'Minor', 1: 'Serious', 2: 'Fatal'}
    
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_input)[0]
    
    return {
        'predicted_severity': severity_labels.get(prediction, 'Unknown'),
        'severity_code': int(prediction),
        'probabilities': probabilities
    }


if __name__ == '__main__':
    results = train_and_evaluate_models()
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
