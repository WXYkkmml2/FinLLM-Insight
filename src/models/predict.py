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
from sklearn.preprocessing import OneHotEncoder

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
    """使用模型进行预测"""
    try:
        # 获取特征列
        if model_info and 'features' in model_info:
            feature_columns = model_info['features']
        else:
            # 如果没有提供特征列表，使用所有数值列
            feature_columns = [col for col in features_df.columns if col not in ['company_code', 'report_year']]
        
        logger.info(f"Using features: {feature_columns}")
        logger.info(f"Available columns in features_df: {features_df.columns.tolist()}")
        
        # 创建特征矩阵
        X = features_df.copy()
        
        # 处理缺失值
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
        
        # 处理分类特征
        categorical_features = X.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            logger.info(f"Categorical features: {categorical_features.tolist()}")
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[categorical_features])
            encoded_df = pd.DataFrame(encoded_features, 
                                    columns=encoder.get_feature_names_out(categorical_features))
            X = pd.concat([X.drop(categorical_features, axis=1), encoded_df], axis=1)
            logger.info(f"Encoded features: {X.columns.tolist()}")
        
        # 确保所有训练时使用的特征都存在
        if model_info and 'features' in model_info:
            required_features = model_info['features']
            missing_features = [f for f in required_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # 为缺失的特征添加零值列
                for feature in missing_features:
                    X[feature] = 0
            # 确保特征顺序与训练时一致
            X = X[required_features]
        
        # 进行预测
        predictions = model.predict(X)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'company_code': features_df['company_code'],
            'report_year': features_df['report_year']
        })
        
        # 根据模型类型添加预测列
        if model_info and 'model_type' in model_info and model_info['model_type'] == 'classification':
            results['predicted_class'] = predictions
            results['prediction_confidence'] = model.predict_proba(X).max(axis=1)
        else:
            results['predicted_return'] = predictions
        
        # 计算预测统计信息
        if 'predicted_return' in results.columns:
            results['prediction_percentile'] = results['predicted_return'].rank(pct=True) * 100
            results['prediction_category'] = pd.cut(results['predicted_return'],
                                                 bins=[-float('inf'), 0, float('inf')],
                                                 labels=['Negative', 'Positive'])
        
        return results
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)}")
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
        
        # 保存预测结果到CSV
        predictions_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved predictions to {predictions_path}")
        
        # Create visualization directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get prediction column name
        model_type = model_info.get('model_type', 'regression')
        target_window = model_info.get('target_window', 60)
        
        if model_type == 'regression':
            prediction_col = 'predicted_return'
        else:
            prediction_col = 'predicted_class'
        
        # 确保预测值为数值类型
        predictions_df[prediction_col] = pd.to_numeric(predictions_df[prediction_col], errors='coerce')
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        if model_type == 'regression':
            # Distribution of predicted returns
            sns.histplot(data=predictions_df, x=prediction_col, kde=True)
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f'{target_window}-day Predicted Return Distribution')
            plt.xlabel('Predicted Return (%)')
        else:
            # Distribution of prediction classes
            sns.countplot(data=predictions_df, x=prediction_col)
            plt.title(f'{target_window}-day Prediction Distribution')
            plt.xlabel('Predicted Direction')
            plt.xticks([0, 1], ['Down', 'Up'])
        
        plt.tight_layout()
        dist_plot_path = os.path.join(viz_dir, f'prediction_distribution_{timestamp}.png')
        plt.savefig(dist_plot_path)
        plt.close()
        
        # Create category distribution plot (for regression models)
        if model_type == 'regression' and 'prediction_category' in predictions_df.columns:
            plt.figure(figsize=(10, 6))
            category_counts = predictions_df['prediction_category'].value_counts()
            
            # Sort categories in logical order
            category_order = ["Negative", "Positive"]
            category_counts = category_counts.reindex(category_order)
            
            sns.barplot(x=category_counts.index, y=category_counts.values)
            plt.title(f'{target_window}-day Prediction Category Distribution')
            plt.ylabel('Number of Companies')
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
            
            # 创建颜色映射
            top_combined['is_positive'] = top_combined[prediction_col] >= 0
            
            # 使用单一颜色
            color = 'green' if top_combined['is_positive'].iloc[0] else 'red'
            sns.barplot(data=top_combined, x=prediction_col, y='company_code', 
                       color=color)
            
            plt.title(f'Companies with Highest and Lowest Predicted Returns ({target_window}-day)')
            plt.xlabel('Predicted Return (%)')
            plt.ylabel('Company Code')
            
            # Add values to bars
            for i, v in enumerate(top_combined[prediction_col]):
                plt.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}%", 
                        va='center', ha='left' if v >= 0 else 'right')
        
        else:
            # Show companies with highest confidence
            if hasattr(model, 'predict_proba'):
                # Get feature columns from model info
                feature_columns = model_info.get('features', [])
                
                if feature_columns and all(col in predictions_df.columns for col in feature_columns):
                    # Get probability predictions
                    proba = model.predict_proba(predictions_df[feature_columns])
                    predictions_df['prediction_confidence'] = np.max(proba, axis=1)
                    
                    # Get top confident predictions for each class
                    predictions_df[prediction_col] = pd.to_numeric(predictions_df[prediction_col], errors='coerce')
                    top_pos = predictions_df[predictions_df[prediction_col] == 1].nlargest(10, 'prediction_confidence')
                    top_neg = predictions_df[predictions_df[prediction_col] == 0].nlargest(10, 'prediction_confidence')
                    
                    # Combine and plot
                    top_combined = pd.concat([top_pos, top_neg])
                    sns.barplot(data=top_combined, x='prediction_confidence', y='company_code', 
                              hue=prediction_col, palette=['red', 'green'])
                    plt.title(f'Companies with Highest Prediction Confidence ({target_window}-day)')
                    plt.xlabel('Prediction Confidence')
                    plt.ylabel('Company Code')
                    plt.legend(['Down', 'Up'])
                else:
                    # If features not available in dataframe
                    plt.text(0.5, 0.5, "Cannot calculate confidence: Missing feature columns", ha='center', va='center', fontsize=14)
            else:
                # If no probabilities available, just show predictions
                plt.text(0.5, 0.5, "No confidence data available", ha='center', va='center', fontsize=14)
        
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
                <title>Stock Prediction Summary - {datetime.now().strftime("%Y-%m-%d")}</title>
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
                <h1>Stock Prediction Summary</h1>
                <div class="section">
                    <h2>Prediction Overview</h2>
                    <p>Model Type: {model_info.get('model_type', 'Unknown')}</p>
                    <p>Model Algorithm: {model_info.get('model_name', 'Unknown')}</p>
                    <p>Prediction Window: {model_info.get('target_window', 60)} days</p>
                    <p>Generation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>Total Companies: {len(predictions_df)}</p>
                </div>
                
                <div class="section">
                    <h2>Prediction Distribution</h2>
                    <div class="img-container">
                        <img src="visualizations/{os.path.basename(dist_plot_path)}" alt="Prediction Distribution">
                    </div>
            """)
            
            # Add category distribution if available
            if model_type == 'regression' and 'prediction_category' in predictions_df.columns:
                f.write(f"""
                    <div class="img-container">
                        <img src="visualizations/{os.path.basename(cat_plot_path)}" alt="Prediction Category Distribution">
                    </div>
                """)
            
            f.write(f"""
                </div>
                
                <div class="section">
                    <h2>Top Predictions</h2>
                    <div class="img-container">
                        <img src="visualizations/{os.path.basename(top_plot_path)}" alt="Top Predictions">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Companies with Highest Predicted Returns</h2>
                    <table>
                        <tr>
                            <th>Company Code</th>
                            <th>Year</th>
                            <th>Predicted Return</th>
                            <th>Prediction Category</th>
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
                                <td>Up</td>
                            </tr>
                        """)
            
            f.write(f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>Companies with Lowest Predicted Returns</h2>
                    <table>
                        <tr>
                            <th>Company Code</th>
                            <th>Year</th>
                            <th>Predicted Return</th>
                            <th>Prediction Category</th>
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
                                <td>Down</td>
                            </tr>
                        """)
            
            f.write(f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>Complete Results</h2>
                    <p>Complete prediction results are saved in <a href="{os.path.basename(predictions_path)}">{os.path.basename(predictions_path)}</a></p>
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
    try:
        # Load features
        features_df = load_features(features_path)
        
        # If model_path not provided, find latest model
        if model_path is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'models')
            if not os.path.exists(models_dir):
                # If the models directory cannot be found by default path, try using output_dir's parent directory
                models_dir = os.path.dirname(output_dir)
                if not os.path.exists(os.path.join(models_dir, 'models')):
                    #If the parent directory does not have a models subdirectory, then use the models directory in the current directory directly
                    models_dir = './models'
            
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
def predict_current_returns(model, features_df, model_info=None, current_year=2025):
    """为当前年份的公司生成未来预测"""
    
    # 筛选当前年份的公司
    current_companies = features_df[features_df['report_year'] == current_year].copy()
    
    if current_companies.empty:
        logger.warning(f"No company data found for year {current_year}")
        return None
    
    # 使用模型进行预测
    predictions_df = make_predictions(model, current_companies, model_info)
    
    # 添加预测时间戳
    predictions_df['prediction_date'] = datetime.now().strftime("%Y-%m-%d")
    predictions_df['prediction_type'] = 'future_forecast'
    
    return predictions_df
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
    
    os.makedirs(predictions_dir, exist_ok=True)
    
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
