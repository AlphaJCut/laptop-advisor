"""
Price Prediction Model for Smart Laptop Advisor
Sử dụng 3 thuật toán cơ bản phù hợp cho AI Fresher:
- Linear Regression
- Random Forest
- Gradient Boosting
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class PricePredictor:
    """Laptop price prediction model."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        """Initialize 3 models to compare (suitable for Fresher)."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        return self.models
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate a single model."""
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='neg_root_mean_squared_error'
        )
        metrics['cv_rmse_mean'] = -cv_scores.mean()
        metrics['cv_rmse_std'] = cv_scores.std()
        
        return metrics
    
    def compare_models(self, X_train, X_test, y_train, y_test):
        """Compare all models and select the best one."""
        self.initialize_models()
        
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            self.results[name] = metrics
            
            print(f"  Test RMSE: ${metrics['test_rmse']:.2f}")
            print(f"  Test MAE:  ${metrics['test_mae']:.2f}")
            print(f"  Test R²:   {metrics['test_r2']:.4f}")
            print(f"  CV RMSE:   ${metrics['cv_rmse_mean']:.2f} (+/- ${metrics['cv_rmse_std']:.2f})")
        
        # Find best model based on test RMSE
        best_name = min(self.results.keys(), key=lambda x: self.results[x]['test_rmse'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_name}")
        print(f"Test RMSE: ${self.results[best_name]['test_rmse']:.2f}")
        print(f"Test R²: {self.results[best_name]['test_r2']:.4f}")
        print("=" * 60)
        
        return self.results
    
    def tune_hyperparameters(self, X_train, y_train, model_name='Random Forest'):
        """Tune hyperparameters for the selected model."""
        print(f"\nTuning hyperparameters for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            base_model = GradientBoostingRegressor(random_state=42)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Use smaller grid for faster tuning
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: ${-grid_search.best_score_:.2f}")
        
        self.best_model = grid_search.best_estimator_
        self.best_model_name = f"{model_name}_tuned"
        
        return grid_search.best_estimator_
    
    def train_final_model(self, X_train, y_train, model=None):
        """Train the final model."""
        if model is not None:
            self.best_model = model
        
        if self.best_model is None:
            # Default to Gradient Boosting with good parameters
            self.best_model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.best_model_name = 'Gradient Boosting (default)'
        
        self.best_model.fit(X_train, y_train)
        print(f"Final model trained: {self.best_model_name}")
        return self.best_model
    
    def predict(self, X):
        """Make predictions."""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        return self.best_model.predict(X)
    
    def predict_with_confidence(self, X, confidence=0.95):
        """Make predictions with confidence intervals (for tree-based models)."""
        predictions = self.predict(X)
        
        # Estimate confidence interval based on training error
        if hasattr(self, 'train_rmse'):
            margin = self.train_rmse * 1.96  # ~95% confidence
        else:
            margin = predictions * 0.15  # Default 15% margin
        
        lower = predictions - margin
        upper = predictions + margin
        
        return predictions, lower, upper
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the model."""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'results': self.results
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.results = model_data.get('results', {})
        print(f"Model loaded: {self.best_model_name}")


def train_price_model(data_dir: str, model_dir: str):
    """Complete training pipeline."""
    # Load processed data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize predictor
    predictor = PricePredictor()
    
    # Compare models
    results = predictor.compare_models(X_train, X_test, y_train, y_test)
    
    # Train final model (use best from comparison)
    predictor.train_final_model(X_train, y_train)
    
    # Final evaluation
    y_pred = predictor.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== FINAL MODEL PERFORMANCE ===")
    print(f"Test RMSE: ${final_rmse:.2f}")
    print(f"Test R²: {final_r2:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    predictor.save_model(os.path.join(model_dir, 'price_model.pkl'))
    
    # Get feature importance
    with open(os.path.join(data_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor_data = pickle.load(f)
    feature_names = preprocessor_data['feature_columns']
    
    importance_df = predictor.get_feature_importance(feature_names)
    if importance_df is not None:
        print("\n=== TOP 10 IMPORTANT FEATURES ===")
        print(importance_df.head(10).to_string(index=False))
        importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    return predictor, results


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_dir, "data", "processed")
    model_dir = os.path.join(project_dir, "models")
    
    # Check if processed data exists
    if not os.path.exists(os.path.join(data_dir, 'X_train.npy')):
        print("=" * 60)
        print("❌ Processed data not found!")
        print(f"   Expected: {data_dir}")
        print("\n💡 Please run data_preprocessing.py first:")
        print("   python src/data_preprocessing.py")
        print("=" * 60)
    else:
        predictor, results = train_price_model(data_dir, model_dir)
        
        # Create results summary
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(2)
        print("\n=== MODEL COMPARISON SUMMARY ===")
        print(results_df[['test_rmse', 'test_mae', 'test_r2', 'cv_rmse_mean']].to_string())
        results_df.to_csv(os.path.join(model_dir, 'model_comparison.csv'))
