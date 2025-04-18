#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Prediction Module for FinLLM-Insight
This module uses trained models to make stock return predictions based on LLM features.
"""

import os
import sys
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_prediction.log"),
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

def load_model(model_path):
    """
    Load a trained model from pickle file
    
    Args:
        model_path (str): Path to model pickle file
        
    Returns:
        object: Trained model
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def load_model_info(info_path):
    """
    Load model information from JSON file
    
    Args:
        info_path (str): Path to model info JSON file
        
    Returns:
        dict: Model information
    """
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        raise

def load_features(features_path):
    """
    Load LLM features
    
    Args:
        features_path (str): Path to features CSV
        
    Returns:
        pd.DataFrame: Features dataframe
    """
    try:
        features_df = pd.read_csv(features_path, encoding='utf-8-sig')
        logger.info(f"Loaded features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
        return features_df
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        raise

def find_latest_model(models_dir, model_type='regression', target_window=60):
    """
    Find the latest model of specified type
    
    Args:
        models_dir (str): Directory containing models
        model_type (str): Type of model ('regression' or 'classification')
        target_window (int): Target prediction window
        
    Returns:
        tuple: (model_path, info_path)
    """
    try:
        # List all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        # Filter by model type and target window
        filtered_files = [
            f for f in model_files 
            if f.startswith(f"{model_type}_") and f"{target_window}d_" in f
        ]
        
        if not filtered_files:
            logger.error(f"No {model_type} models found for {target_window}d window")
            return None, None
        
        # Sort by timestamp (assuming format: model_type_model_name_window_timestamp.pkl)
        sorted_files = sorted(filtered_files, reverse=True)
        latest_model = sorted_files[0]
        
        # Get corresponding info file
        info_file = latest_model.replace('.pkl', '_info.json')
        
        model_path = os.path.join(models_dir, latest_model)
        info_path = os.path.join(models_dir, info_file)
        
        if not os.path.exists(info_path):
            logger.warning(f"Model info file not found: {info_path}")
            info_path = None
        
        return model_path, info_path
    
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return None, None

def make_predictions(model, features_df, model_info=None):
    """
    Make predictions using a trained model
    
    Args:
        model: Trained model
        features_df (pd.DataFrame): Features
        model_info (dict): Model information
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    try:
        # Get feature columns from model info if available
        if model_info and 'features' in model_info:
            feature_columns = model_info['features']
            logger.info(f"Using {len(feature_columns)} features from model info")
        else:
            # Otherwise, use common patterns to identify feature columns
            numeric_features = [col for col in features_df.columns if col.endswith('_score')]
            categorical_features = [col for col in features_df.columns if col.endswith('_category')]
            feature_columns = numeric_features + categorical_features
            logger.info(f"Using {len(feature_columns)} features identified from column names")
        
        # Check if we have all required features
        missing_features = [f for f in feature_columns if f not in features_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Only use available features
            feature_columns = [f for f in feature_columns if f in features_df.columns]
        
        # Create feature matrix
        X = features_df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create results dataframe
        results_df = features_df[['company_code', 'report_year']].copy()
        
        # Add prediction column
        if model_info and 'model_type' in model_info:
            model_type = model_info['model_type']
            target_window = model_info.get('target_window', 60)
            
            if model_type == 'regression':
                results_df[f'predicted_return_{target_window}d'] = predictions
            else:
                results_df[f'predicted_up_{target_window}d'] = predictions
        else:
            results_df['prediction'] = predictions
        
        # Calculate prediction statistics
        if model_info and model_info.get('model_type') == 'regression':
            results_df['prediction_percentile'] = results_df[f'predicted_return_{target_window}d'].rank(pct=True)
            
            # Add prediction category based on percentile
            def categorize_prediction(percentile):
                if percentile < 0.25:
                    return "It is likely to fall!!"
                elif percentile < 0.5:
                    return "Possible to fall slightly!"
                elif percentile < 0.75:
                    return "Probably a slight increase!"
                else:
                    return "It's likely to rise!!"
            
            results_df['prediction_category'] = results_df['prediction_percentile'].apply(categorize_prediction)
        
        logger.info(f"Generated predictions for {len(results_df)} companies")
        return results_df
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def save_predictions(predictions_df, output_dir, timestamp=None):
    """
    Save predictions to file
    
    Args:
        predictions_df (pd.DataFrame): Predictions dataframe
        output_dir (str): Output directory
        timestamp (str): Timestamp string
        
    Returns:
        str: Path to saved file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig
