#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Module for FinLLM-Insight
This module trains machine learning models to predict stock returns based on LLM features.
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

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置日志
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
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def load_data(features_path, targets_path):
    """加载并合并特征和目标数据"""
    try:
        # 加载特征
        features_df = pd.read_csv(features_path, encoding='utf-8-sig')
        logger.info(f"Loaded features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
        
        # 加载目标
        targets_df = pd.read_csv(targets_path, encoding='utf-8-sig')
        logger.info(f"Loaded targets: {targets_df.shape[0]} rows, {targets_df.shape[1]} columns")
        
        # 合并数据
        merged_df = pd.merge(
            features_df,
            targets_df,
            how='inner',
            left_on=['company_code', 'report_year'],
            right_on=['stock_code', 'report_year']
        )
        
        logger.info(f"Merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """预处理数据"""
    try:
        # 检查目标列是否存在
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} does not exist")
        
        # 移除目标值为空的行
        df = df.dropna(subset=[target_column])
        logger.info(f"After removing null targets: {df.shape[0]} rows")
        
        # 定义特征列 - LLM 生成的数值分数
        numeric_features = [col for col in df.columns if col.endswith('_score')]
        logger.info(f"Found {len(numeric_features)} numeric features")
        
        # 创建特征矩阵和目标向量
        X = df[numeric_features].copy()
        y = df[target_column].copy()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, numeric_features
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def create_model(model_type='regression', model_name='random_forest'):
    """创建模型"""
    try:
        if model_type == 'regression':
            if model_name == 'linear':
                model = LinearRegression()
            elif model_name == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=4,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:
                logger.warning(f"Unknown regression model: {model_name}. Using RandomForest.")
                model = RandomForestRegressor(n_estimators=200, random_state=42)
        
        elif model_type == 'classification':
            if model_name == 'logistic':
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=4,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                logger.warning(f"Unknown classification model: {model_name}. Using RandomForest.")
                model = RandomForestClassifier(n_estimators=200, random_state=42)
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def train_model(X_train, y_train, model, cv=2):
    """训练模型"""
    try:
        # 检查数据量
        if len(X_train) < 10:
            logger.warning(f"Training data too small ({len(X_train)} samples)")
            model.fit(X_train, y_train)
            return model, np.array([0.0])
        
        # 执行交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        logger.info(f"CV R² scores: {cv_scores}")
        logger.info(f"Mean CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 在完整训练集上训练模型
        model.fit(X_train, y_train)
        
        return model, cv_scores
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_type='regression'):
    """评估模型"""
    try:
        # 进行预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = {}
        
        if model_type == 'regression':
            # 回归指标
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            
            # 计算相对误差
            metrics['relative_error'] = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
            
            logger.info(f"Test MSE: {metrics['mse']:.4f}")
            logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
            logger.info(f"Test MAE: {metrics['mae']:.4f}")
            logger.info(f"Test R²: {metrics['r2']:.4f}")
            logger.info(f"Relative Error: {metrics['relative_error']:.2f}%")
        
        return metrics, y_pred
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def feature_importance_analysis(model, feature_names, output_dir):
    """分析特征重要性"""
    try:
        # 创建可视化目录
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)
        else:
            logger.warning("Model does not provide feature importance")
            return None
        
        # 创建特征重要性 DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 保存到 CSV
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
        plt.close()
        
        logger.info(f"Feature importance analysis completed. Results saved to {importance_path}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")
        return None

def main():
    """主函数"""
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
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 获取参数
    features_dir = config.get('features_directory', './data/processed/features')
    targets_dir = config.get('targets_directory', './data/processed/targets')
    models_dir = config.get('models_directory', './models')
    
    # 使用命令行参数或配置中的目标窗口
    target_window = args.target_window or config.get('target_window', 60)
    logger.info(f"Target window: {target_window} days")
    
    # 文件路径
    features_path = os.path.join(features_dir, 'llm_features.csv')
    targets_path = os.path.join(targets_dir, 'stock_targets.csv')
    
    logger.info(f"Features file: {features_path}")
    logger.info(f"Targets file: {targets_path}")
    
    # 加载数据
    data = load_data(features_path, targets_path)
    
    # 根据目标窗口和模型类型确定目标列
    if args.model_type == 'regression':
        target_column = f'future_return_{target_window}d'
    else:
        target_column = f'future_up_{target_window}d'
    
    logger.info(f"Using target column: {target_column}")
    
    # 预处理数据
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        data, target_column, test_size=0.2
    )
    
    # 创建模型
    model = create_model(args.model_type, args.model_name)
    
    # 训练模型
    trained_model, cv_scores = train_model(X_train, y_train, model)
    
    # 评估模型
    metrics, y_pred = evaluate_model(trained_model, X_test, y_test, args.model_type)
    
    # 特征重要性分析
    importance_df = feature_importance_analysis(trained_model, feature_names, models_dir)
    
    # 保存模型和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file_name = f"{args.model_type}_{args.model_name}_{target_window}d_{timestamp}"
    
    # 创建模型目录
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(models_dir, f"{model_file_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(trained_model, f)
    
    # 保存模型信息
    model_info = {
        "model_type": args.model_type,
        "model_name": args.model_name,
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
    
    info_path = os.path.join(models_dir, f"{model_file_name}_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Model training completed successfully")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Model metrics: {metrics}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
