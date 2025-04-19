#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Module for FinLLM-Insight
This module trains machine learning models to predict stock returns based on LLM features.
"""
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import json
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from JSON file
    
    Args:
        config_path (str): Path to config JSON file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def load_data(features_path, targets_path):
    """
    Load and merge features and targets
    
    Args:
        features_path (str): Path to LLM features CSV
        targets_path (str): Path to targets CSV
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    try:
        # Load features
        features_df = pd.read_csv(features_path, encoding='utf-8-sig')
        logger.info(f"Loaded features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
        
        # Load targets
        targets_df = pd.read_csv(targets_path, encoding='utf-8-sig')
        logger.info(f"Loaded targets: {targets_df.shape[0]} rows, {targets_df.shape[1]} columns")
        
        # Merge on company code and report year
        merged_df = pd.merge(
            features_df,
            targets_df,
            how='inner',
            left_on=['company_code', 'report_year'],
            right_on=['stock_code', 'report_year']
        )
        
        logger.info(f"Merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Check if merge was successful
        if merged_df.shape[0] == 0:
            # Try alternative column names
            logger.warning("First merge attempt failed. Trying alternative column mappings...")
            merged_df = pd.merge(
                features_df,
                targets_df,
                how='inner',
                left_on=['company_code', 'report_year'],
                right_on=['stock_code', 'year']
            )
            
            if merged_df.shape[0] == 0:
                logger.error("Failed to merge features and targets. Check column names.")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """
    Preprocess data for model training
    
    Args:
        df (pd.DataFrame): Merged dataset
        target_column (str): Target column name
        test_size (float): Proportion of test split
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    try:
        # Remove rows with missing target
        df = df.dropna(subset=[target_column])
        
        # Define feature columns - numeric scores from LLM
        numeric_features = [col for col in df.columns if col.endswith('_score')]
        
        # Define categorical features
        categorical_features = [col for col in df.columns if col.endswith('_category')]
        
        # Combined features
        feature_columns = numeric_features + categorical_features
        
        # Check if we have features
        if not feature_columns:
            logger.error("No feature columns identified. Check feature naming patterns.")
            raise ValueError("No feature columns found")
        
        # Create feature matrix and target vector
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Drop rows with missing features
        missing_mask = X.isnull().any(axis=1)
        if missing_mask.sum() > 0:
            logger.warning(f"Dropping {missing_mask.sum()} rows with missing features")
            X = X[~missing_mask]
            y = y[~missing_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def create_pipeline(model_type='regression', model_name='random_forest'):
    """
    Create a model training pipeline
    
    Args:
        model_type (str): 'regression' or 'classification'
        model_name (str): Type of model to use
        
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    try:
        # Define preprocessors for different column types
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Define models
        if model_type == 'regression':
            if model_name == 'linear':
                model = LinearRegression()
            elif model_name == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42
                )
            elif model_name == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                logger.warning(f"Unknown regression model: {model_name}. Using RandomForest.")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif model_type == 'classification':
            if model_name == 'logistic':
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42
                )
            else:
                logger.warning(f"Unknown classification model: {model_name}. Using RandomForest.")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise

def train_model(X_train, y_train, model, cv=5):
    """
    Train a model with cross-validation
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        model: Scikit-learn model
        cv (int): Number of cross-validation folds
        
    Returns:
        tuple: (trained_model, cv_scores)
    """
    try:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train model on full training set
        model.fit(X_train, y_train)
        
        return model, cv_scores
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_type='regression'):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        model_type (str): 'regression' or 'classification'
        
    Returns:
        dict: Performance metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        if model_type == 'regression':
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            
            logger.info(f"Test MSE: {metrics['mse']:.4f}")
            logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
            logger.info(f"Test MAE: {metrics['mae']:.4f}")
            logger.info(f"Test R²: {metrics['r2']:.4f}")
        
        elif model_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test Precision: {metrics['precision']:.4f}")
            logger.info(f"Test Recall: {metrics['recall']:.4f}")
            logger.info(f"Test F1: {metrics['f1']:.4f}")
        
        return metrics, y_pred
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def feature_importance_analysis(model, feature_names, output_dir):
    """
    Analyze and visualize feature importance
    
    Args:
        model: Trained model
        feature_names (list): Feature column names
        output_dir (str): Directory to save plots
    """
    try:
        # Create visualization directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)
        else:
            logger.warning("Model doesn't provide feature importance information")
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save to CSV
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
        plt.close()
        
        logger.info(f"Feature importance analysis complete. Results saved to {importance_path}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")

def create_prediction_vs_actual_plot(y_test, y_pred, output_dir):
    """
    Create a plot of predicted vs. actual values
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.array): Predicted values
        output_dir (str): Directory to save plot
    """
    try:
        # Create visualization directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Predicted vs. Actual Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(viz_dir, 'predicted_vs_actual.png'))
        plt.close()
        
        logger.info(f"Prediction vs. actual plot saved to {viz_dir}")
    
    except Exception as e:
        logger.error(f"Error creating prediction vs. actual plot: {e}")

def save_model(model, model_info, output_dir, model_name='model'):
    """
    Save trained model and related information
    
    Args:
        model: Trained model
        model_info (dict): Model metadata and metrics
        output_dir (str): Directory to save model
        model_name (str): Base name for model files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model info
        info_path = os.path.join(output_dir, f"{model_name}_info.json")
        
        # Convert numpy values to Python native types for JSON serialization
        for key, value in model_info.items():
            if isinstance(value, np.ndarray):
                model_info[key] = value.tolist()
            elif isinstance(value, np.integer):
                model_info[key] = int(value)
            elif isinstance(value, np.floating):
                model_info[key] = float(value)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model info saved to {info_path}")
        
        return model_path, info_path
    
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def train_return_prediction_model(features_path, targets_path, output_dir, target_window=60, 
                                model_type='regression', model_name='random_forest'):
    """
    Main function to train a return prediction model
    
    Args:
        features_path (str): Path to LLM features CSV
        targets_path (str): Path to targets CSV
        output_dir (str): Directory to save model and results
        target_window (int): Time window for return prediction (in days)
        model_type (str): 'regression' or 'classification'
        model_name (str): Type of model to use
        
    Returns:
        tuple: (model_path, metrics)
    """
    try:
        # Load and merge data
        data = load_data(features_path, targets_path)
        
        # Determine target column based on target_window and model_type
        if model_type == 'regression':
            target_column = f'future_return_{target_window}d'
        else:
            target_column = f'future_up_{target_window}d'
        
        logger.info(f"Using target column: {target_column}")
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            data, target_column, test_size=0.2
        )
        
        # Create model
        model = create_pipeline(model_type, model_name)
        
        # Train model
        trained_model, cv_scores = train_model(X_train, y_train, model)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(trained_model, X_test, y_test, model_type)
        
        # Feature importance analysis
        importance_df = feature_importance_analysis(trained_model, feature_names, output_dir)
        
        # Create prediction vs. actual plot (for regression)
        if model_type == 'regression':
            create_prediction_vs_actual_plot(y_test, y_pred, output_dir)
        
        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file_name = f"{model_type}_{model_name}_{target_window}d_{timestamp}"
        
        model_info = {
            "model_type": model_type,
            "model_name": model_name,
            "target_column": target_column,
            "target_window": target_window,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_names),
            "features": feature_names,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "metrics": metrics,
            "top_features": importance_df.head(10).to_dict('records') if importance_df is not None else None,
            "timestamp": timestamp
        }
        
        model_path, info_path = save_model(
            trained_model, model_info, output_dir, model_file_name
        )
        
        return model_path, metrics
    
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise

def main():
    """Main function to run the model training process"""
    parser = argparse.ArgumentParser(description='Train models for stock return prediction')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--model_type', type=str, default='regression', 
                        choices=['regression', 'classification'],
                        help='Type of model to train')
    parser.add_argument('--model_name', type=str, default='random_forest',
                        choices=['linear', 'logistic', 'random_forest', 'gradient_boosting'],
                        help='Type of algorithm to use')
    parser.add_argument('--target_window', type=int, default=None,
                        help='Time window for return prediction (in days)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get parameters
    features_dir = config.get('features_directory', './data/processed/features')
    targets_dir = config.get('targets_directory', './data/processed/targets')
    models_dir = config.get('models_directory', './models')
    
    # Use command line args or config for target window
    target_window = args.target_window or config.get('target_window', 60)
    
    # File paths
    features_path = os.path.join(features_dir, 'llm_features.csv')
    targets_path = os.path.join(targets_dir, 'stock_targets.csv')
    
    # Train model
    model_path, metrics = train_return_prediction_model(
        features_path=features_path,
        targets_path=targets_path,
        output_dir=models_dir,
        target_window=target_window,
        model_type=args.model_type,
        model_name=args.model_name
    )
    
    logger.info(f"Model training complete. Model saved to {model_path}")
    logger.info(f"Model metrics: {metrics}")

if __name__ == "__main__":
    main()
