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
            target_window = model_info.get('target_window', 60)
            results_df['prediction_percentile'] = results_df[f'predicted_return_{target_window}d'].rank(pct=True)
            
            # Add prediction category based on percentile
            def categorize_prediction(percentile):
                if percentile < 0.25:
                    return "很可能下跌"
                elif percentile < 0.5:
                    return "可能小幅下跌"
                elif percentile < 0.75:
                    return "可能小幅上涨"
                else:
                    return "很可能上涨"
            
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
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Saved predictions to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise

def create_prediction_summary(predictions_df, model_info, output_dir, timestamp=None):
    """
    Create prediction summary visualizations
    
    Args:
        predictions_df (pd.DataFrame): Predictions dataframe
        model_info (dict): Model information
        output_dir (str): Output directory
        timestamp (str): Timestamp string
        
    Returns:
        str: Path to summary file
    """
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualization directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get prediction column name
        model_type = model_info.get('model_type', 'regression')
        target_window = model_info.get('target_window', 60)
        
        if model_type == 'regression':
            prediction_col = f'predicted_return_{target_window}d'
        else:
            prediction_col = f'predicted_up_{target_window}d'
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        if model_type == 'regression':
            # Distribution of predicted returns
            sns.histplot(predictions_df[prediction_col], kde=True)
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f'{target_window}日预测收益率分布')
            plt.xlabel('预测收益率 (%)')
        else:
            # Distribution of prediction classes
            sns.countplot(x=prediction_col, data=predictions_df)
            plt.title(f'{target_window}日预测涨跌分布')
            plt.xlabel('预测涨跌')
            plt.xticks([0, 1], ['下跌', '上涨'])
        
        plt.tight_layout()
        dist_plot_path = os.path.join(viz_dir, f'prediction_distribution_{timestamp}.png')
        plt.savefig(dist_plot_path)
        plt.close()
        
        # Create category distribution plot (for regression models)
        if model_type == 'regression' and 'prediction_category' in predictions_df.columns:
            plt.figure(figsize=(10, 6))
            category_counts = predictions_df['prediction_category'].value_counts()
            
            # Sort categories in logical order
            category_order = ["很可能下跌", "可能小幅下跌", "可能小幅上涨", "很可能上涨"]
            category_counts = category_counts.reindex(category_order)
            
            sns.barplot(x=category_counts.index, y=category_counts.values)
            plt.title(f'{target_window}日预测分类分布')
            plt.ylabel('公司数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            cat_plot_path = os.path.join(viz_dir, f'prediction_categories_{timestamp}.png')
            plt.savefig(cat_plot_path)
            plt.close()
        
        # Create top companies plot
        plt.figure(figsize=(12, 6))
        
        if model_type == 'regression':
            # Top companies by predicted return
            top_positive = predictions_df.nlargest(10, prediction_col)
            top_negative = predictions_df.nsmallest(10, prediction_col)
            
            # Combine and sort
            top_combined = pd.concat([top_positive, top_negative])
            top_combined = top_combined.sort_values(prediction_col)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bar_colors = ['red' if x < 0 else 'green' for x in top_combined[prediction_col]]
            
            ax = sns.barplot(x=prediction_col, y='company_code', data=top_combined, palette=bar_colors)
            plt.title(f'预测收益率最高和最低的公司（{target_window}日）')
            plt.xlabel('预测收益率 (%)')
            plt.ylabel('公司代码')
            
            # Add values to bars
            for i, v in enumerate(top_combined[prediction_col]):
                ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}%", 
                        va='center', ha='left' if v >= 0 else 'right')
        
        else:
            # Show companies with highest confidence
            if hasattr(model, 'predict_proba'):
                # Get probability predictions
                proba = model.predict_proba(predictions_df[feature_columns])
                predictions_df['prediction_confidence'] = np.max(proba, axis=1)
                
                # Get top confident predictions for each class
                top_pos = predictions_df[predictions_df[prediction_col] == 1].nlargest(10, 'prediction_confidence')
                top_neg = predictions_df[predictions_df[prediction_col] == 0].nlargest(10, 'prediction_confidence')
                
                # Combine and plot
                top_combined = pd.concat([top_pos, top_neg])
                sns.barplot(x='prediction_confidence', y='company_code', hue=prediction_col, data=top_combined)
                plt.title(f'预测置信度最高的公司（{target_window}日）')
                plt.xlabel('预测置信度')
                plt.ylabel('公司代码')
                plt.legend(['下跌', '上涨'])
            else:
                # If no probabilities available, just show predictions
                plt.text(0.5, 0.5, "无置信度数据可用", ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        top_plot_path = os.path.join(viz_dir, f'top_predictions_{timestamp}.png')
        plt.savefig(top_plot_path)
        plt.close()
        
        # Create summary HTML
        summary_path = os.path.join(output_dir, f"prediction_summary_{timestamp}.html")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <html>
            <head>
                <title>股票预测结果汇总 - {datetime.now().strftime("%Y-%m-%d")}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .section {{ margin-bottom: 30px; }}
                    .img-container {{ text-align: center; margin: 20px 0; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    th {{ background-color: #2c3e50; color: white; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>股票预测结果汇总</h1>
                <div class="section">
                    <h2>预测概述</h2>
                    <p>模型类型: {model_info.get('model_type', 'Unknown')}</p>
                    <p>模型算法: {model_info.get('model_name', 'Unknown')}</p>
                    <p>预测时间窗口: {model_info.get('target_window', 60)}天</p>
                    <p>预测生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>预测公司总数: {len(predictions_df)}</p>
                </div>
                
                <div class="section">
                    <h2>预测分布</h2>
                    <div class="img-container">
                        <img src="visualizations/{os.path.basename(dist_plot_path)}" alt="预测分布">
                    </div>
            """)
            
            # Add category distribution if available
            if model_type == 'regression' and 'prediction_category' in predictions_df.columns:
                f.write(f"""
                    <div class="img-container">
                        <img src="visualizations/{os.path.basename(cat_plot_path)}" alt="预测类别分布">
                    </div>
                """)
            
            f.write(f"""
                </div>
                
                <div class="section">
                    <h2>最突出的预测</h2>
                    <div class="img-container">
                        <img src="visualizations/{os.path.basename(top_plot_path)}" alt="最突出的预测">
                    </div>
                </div>
                
                <div class="section">
                    <h2>预测涨幅最高的公司</h2>
                    <table>
                        <tr>
                            <th>公司代码</th>
                            <th>年份</th>
                            <th>预测收益率</th>
                            <th>预测类别</th>
                        </tr>
            """)
            
            # Add top positive predictions
            if model_type == 'regression':
                for _, row in predictions_df.nlargest(20, prediction_col).iterrows():
                    f.write(f"""
                        <tr>
                            <td>{row['company_code']}</td>
                            <td>{row['report_year']}</td>
                            <td class="positive">{row[prediction_col]:.2f}%</td>
                            <td>{row.get('prediction_category', '')}</td>
                        </tr>
                    """)
            else:
                # For classification models, show top confidence for positive class
                if 'prediction_confidence' in predictions_df.columns:
                    top_positive = predictions_df[predictions_df[prediction_col] == 1].nlargest(20, 'prediction_confidence')
                    for _, row in top_positive.iterrows():
                        f.write(f"""
                            <tr>
                                <td>{row['company_code']}</td>
                                <td>{row['report_year']}</td>
                                <td class="positive">{row['prediction_confidence']:.2f}</td>
                                <td>上涨</td>
                            </tr>
                        """)
            
            f.write(f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>预测跌幅最大的公司</h2>
                    <table>
                        <tr>
                            <th>公司代码</th>
                            <th>年份</th>
                            <th>预测收益率</th>
                            <th>预测类别</th>
                        </tr>
            """)
            
            # Add top negative predictions
            if model_type == 'regression':
                for _, row in predictions_df.nsmallest(20, prediction_col).iterrows():
                    f.write(f"""
                        <tr>
                            <td>{row['company_code']}</td>
                            <td>{row['report_year']}</td>
                            <td class="negative">{row[prediction_col]:.2f}%</td>
                            <td>{row.get('prediction_category', '')}</td>
                        </tr>
                    """)
            else:
                # For classification models, show top confidence for negative class
                if 'prediction_confidence' in predictions_df.columns:
                    top_negative = predictions_df[predictions_df[prediction_col] == 0].nlargest(20, 'prediction_confidence')
                    for _, row in top_negative.iterrows():
                        f.write(f"""
                            <tr>
                                <td>{row['company_code']}</td>
                                <td>{row['report_year']}</td>
                                <td class="negative">{row['prediction_confidence']:.2f}</td>
                                <td>下跌</td>
                            </tr>
                        """)
            
            f.write(f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>完整预测结果</h2>
                    <p>完整的预测结果保存在 <a href="{os.path.basename(output_path)}">{os.path.basename(output_path)}</a></p>
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"Created prediction summary at {summary_path}")
        return summary_path
    
    except Exception as e:
        logger.error(f"Error creating prediction summary: {e}")
        return None

def predict_returns(features_path, model_path=None, info_path=None, output_dir=None, 
                  model_type='regression', target_window=60, create_summary=True):
    """
    Main prediction function
    
    Args:
        features_path (str): Path to features CSV
        model_path (str): Path to model file, or None to use latest
        info_path (str): Path to model info file, or None to use latest
        output_dir (str): Directory to save results
        model_type (str): Model type to use if finding latest
        target_window (int): Target window to use if finding latest
        create_summary (bool): Whether to create summary HTML
        
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    try:
        # Load features
        features_df = load_features(features_path)
        
        # If model_path not provided, find latest model
        if model_path is None:
            if output_dir is None:
                logger.error("Either model_path or output_dir must be provided")
                return None
            
            models_dir = os.path.dirname(output_dir)
            model_path, info_path = find_latest_model(models_dir, model_type, target_window)
            
            if model_path is None:
                logger.error("No suitable model found")
                return None
        
        # Load model
        model = load_model(model_path)
        
        # Load model info if available
        model_info = None
        if info_path and os.path.exists(info_path):
            model_info = load_model_info(info_path)
        
        # Make predictions
        predictions_df = make_predictions(model, features_df, model_info)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        if output_dir:
            output_path = save_predictions(predictions_df, output_dir, timestamp)
            
            # Create summary
            if create_summary and model_info:
                summary_path = create_prediction_summary(
                    predictions_df, model_info, output_dir, timestamp
                )
                logger.info(f"Prediction process complete. Summary available at {summary_path}")
        
        return predictions_df
    
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        return None

def main():
    """Main function to run the prediction process"""
    parser = argparse.ArgumentParser(description='Make stock return predictions with trained models')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model file (optional, will use latest if not provided)')
    parser.add_argument('--info_path', type=str, default=None,
                        help='Path to model info file (optional)')
    parser.add_argument('--model_type', type=str, default='regression', 
                        choices=['regression', 'classification'],
                        help='Type of model to use when finding latest')
    parser.add_argument('--target_window', type=int, default=None,
                        help='Target window for prediction when finding latest model')
    parser.add_argument('--no_summary', action='store_true',
                        help='Disable creation of HTML summary')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get parameters
    features_dir = config.get('features_directory', './data/processed/features')
    models_dir = config.get('models_directory', './models')
    predictions_dir = config.get('predictions_directory', './predictions')
    
    # Use command line args or config for target window
    target_window = args.target_window or config.get('target_window', 60)
    
    # File paths
    features_path = os.path.join(features_dir, 'llm_features.csv')
    
    # Run prediction
    predictions_df = predict_returns(
        features_path=features_path,
        model_path=args.model_path,
        info_path=args.info_path,
        output_dir=predictions_dir,
        model_type=args.model_type,
        target_window=target_window,
        create_summary=not args.no_summary
    )
    
    if predictions_df is not None:
        logger.info(f"Prediction successful with {len(predictions_df)} companies")
    else:
        logger.error("Prediction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
