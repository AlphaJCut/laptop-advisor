"""
Utility functions for Smart Laptop Advisor
"""
import pandas as pd
import numpy as np
import pickle
import os


def load_models(model_dir: str):
    """Load all saved models and preprocessors."""
    models = {}
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, '..', 'data', 'processed', 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            models['preprocessor'] = pickle.load(f)
    
    # Load price model
    price_model_path = os.path.join(model_dir, 'price_model.pkl')
    if os.path.exists(price_model_path):
        with open(price_model_path, 'rb') as f:
            models['price_model'] = pickle.load(f)
    
    # Load recommender
    recommender_path = os.path.join(model_dir, 'recommender.pkl')
    if os.path.exists(recommender_path):
        with open(recommender_path, 'rb') as f:
            models['recommender'] = pickle.load(f)
    
    return models


def prepare_user_input(user_specs: dict, preprocessor: dict) -> np.ndarray:
    """Prepare user input for prediction."""
    # Create a dataframe with user input
    df_input = pd.DataFrame([user_specs])
    
    # Add derived features
    processor_scores = {
        'Intel Core i3': 3, 'Intel Core i5': 5, 'Intel Core i7': 7,
        'Intel Core i9': 9, 'AMD Ryzen 5': 5, 'AMD Ryzen 7': 7,
        'AMD Ryzen 9': 9, 'Apple M1': 7, 'Apple M2': 8, 'Apple M3': 9
    }
    
    gpu_scores = {
        'Integrated': 1, 'NVIDIA GTX 1650': 4, 'NVIDIA GTX 1660': 5,
        'NVIDIA RTX 3050': 5, 'NVIDIA RTX 3060': 6, 'NVIDIA RTX 3070': 7,
        'NVIDIA RTX 3080': 8, 'NVIDIA RTX 4060': 7, 'NVIDIA RTX 4070': 8,
        'NVIDIA RTX 4080': 9, 'AMD Radeon RX 6600': 5, 'AMD Radeon RX 6700': 6
    }
    
    resolution_pixels = {
        '1366x768': 1366 * 768,
        '1920x1080': 1920 * 1080,
        '2560x1440': 2560 * 1440,
        '3840x2160': 3840 * 2160
    }
    
    # Add scores
    df_input['processor_score'] = df_input['processor'].map(processor_scores).fillna(5)
    df_input['gpu_score'] = df_input['gpu'].map(gpu_scores).fillna(3)
    df_input['total_pixels'] = df_input['resolution'].map(resolution_pixels).fillna(1920*1080)
    df_input['ppi'] = np.sqrt(df_input['total_pixels']) / df_input['screen_size']
    
    df_input['performance_score'] = (
        df_input['processor_score'] * 0.4 +
        df_input['gpu_score'] * 0.3 +
        (df_input['ram_gb'] / 64) * 10 * 0.2 +
        (df_input['storage_gb'] / 2048) * 10 * 0.1
    )
    
    df_input['portability_score'] = 10 - (df_input['weight_kg'] * 2)
    df_input['portability_score'] = df_input['portability_score'].clip(0, 10)
    
    df_input['is_gaming'] = (
        (df_input['laptop_type'] == 'Gaming') |
        (df_input['gpu'].str.contains('RTX|GTX', na=False))
    ).astype(int)
    
    df_input['is_ultraportable'] = (
        (df_input['weight_kg'] < 1.5) &
        (df_input['screen_size'] <= 14)
    ).astype(int)
    
    # Encode categorical features
    label_encoders = preprocessor['label_encoders']
    categorical_columns = preprocessor['categorical_columns']
    
    for col in categorical_columns:
        if col in df_input.columns and col in label_encoders:
            le = label_encoders[col]
            val = df_input[col].iloc[0]
            if val in le.classes_:
                df_input[f'{col}_encoded'] = le.transform([val])[0]
            else:
                df_input[f'{col}_encoded'] = 0
    
    # Get feature columns
    feature_columns = preprocessor['feature_columns']
    
    # Ensure all columns exist
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Select and order features
    X = df_input[feature_columns].values
    
    # Scale features
    scaler = preprocessor['scaler']
    X_scaled = scaler.transform(X)
    
    return X_scaled


def format_price(price: float) -> str:
    """Format price for display."""
    return f"${price:,.2f}"


def get_use_case_description(use_case: str) -> str:
    """Get description for a use case."""
    descriptions = {
        'gaming': '🎮 Gaming - High-performance for modern games',
        'office': '💼 Office - Business and productivity',
        'creative': '🎨 Creative - Design, video editing, 3D work',
        'student': '📚 Student - Budget-friendly for studies',
        'ultraportable': '✈️ Ultraportable - Lightweight for travel',
        'all_rounder': '🌟 All-Rounder - Balanced for everything'
    }
    return descriptions.get(use_case, use_case)


def create_radar_chart_data(laptop_data: dict) -> dict:
    """Create data for radar chart visualization."""
    categories = ['Performance', 'Portability', 'Value', 'Display', 'Storage']
    
    values = [
        laptop_data.get('performance_score', 5) * 10,
        laptop_data.get('portability_score', 5) * 10,
        laptop_data.get('value_score', 5) * 10,
        min(laptop_data.get('total_pixels', 2073600) / 8294400 * 100, 100),
        min(laptop_data.get('storage_gb', 512) / 2048 * 100, 100)
    ]
    
    return {'categories': categories, 'values': values}
