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
    """加载并合并特征和目标数据"""
    try:
        # 加载特征
        features_df = pd.read_csv(features_path, encoding='utf-8-sig')
        logger.info(f"加载特征: {features_df.shape[0]}行, {features_df.shape[1]}列")
        
        # 加载目标
        targets_df = pd.read_csv(targets_path, encoding='utf-8-sig')
        logger.info(f"加载目标: {targets_df.shape[0]}行, {targets_df.shape[1]}列")
        
        # 打印列名，便于调试
        logger.debug(f"特征数据列名: {features_df.columns.tolist()}")
        logger.debug(f"目标数据列名: {targets_df.columns.tolist()}")
        
        # 确定公司代码列名
        company_code_cols = ['company_code', 'stock_code', '公司代码', '股票代码', '代码']
        report_year_cols = ['report_year', 'year', '报告年份', '年份']
        
        # 在特征数据中查找公司代码列
        feature_company_col = None
        for col in company_code_cols:
            if col in features_df.columns:
                feature_company_col = col
                break
        
        if not feature_company_col:
            raise ValueError(f"在特征数据中找不到公司代码列: {company_code_cols}")
        
        # 在特征数据中查找报告年份列
        feature_year_col = None
        for col in report_year_cols:
            if col in features_df.columns:
                feature_year_col = col
                break
        
        if not feature_year_col:
            raise ValueError(f"在特征数据中找不到报告年份列: {report_year_cols}")
        
        # 在目标数据中查找公司代码列
        target_company_col = None
        for col in company_code_cols:
            if col in targets_df.columns:
                target_company_col = col
                break
        
        if not target_company_col:
            raise ValueError(f"在目标数据中找不到公司代码列: {company_code_cols}")
        
        # 在目标数据中查找报告年份列
        target_year_col = None
        for col in report_year_cols:
            if col in targets_df.columns:
                target_year_col = col
                break
        
        if not target_year_col:
            raise ValueError(f"在目标数据中找不到报告年份列: {report_year_cols}")
        
        # 合并数据
        merged_df = pd.merge(
            features_df,
            targets_df,
            how='inner',
            left_on=[feature_company_col, feature_year_col],
            right_on=[target_company_col, target_year_col]
        )
        
        logger.info(f"合并后数据: {merged_df.shape[0]}行, {merged_df.shape[1]}列")
        
        # 检查合并是否成功
        if merged_df.shape[0] == 0:
            logger.error(f"合并后数据为空，请检查公司代码和报告年份列是否匹配")
            logger.error(f"特征数据示例: {features_df[[feature_company_col, feature_year_col]].head()}")
            logger.error(f"目标数据示例: {targets_df[[target_company_col, target_year_col]].head()}")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"加载数据错误: {e}")
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
        # 检查目标列是否存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 {target_column} 不存在")
        
        # 移除目标值为空的行
        df = df.dropna(subset=[target_column])
        logger.info(f"移除目标值为空的行后: {df.shape[0]}行")
        
        # 定义特征列 - LLM 生成的数值分数
        numeric_features = [col for col in df.columns if col.endswith('_score')]
        logger.info(f"找到 {len(numeric_features)} 个数值特征: {numeric_features}")
        
        # 定义分类特征
        categorical_features = [col for col in df.columns if col.endswith('_category')]
        logger.info(f"找到 {len(categorical_features)} 个分类特征: {categorical_features}")
        
        # 组合特征
        feature_columns = numeric_features + categorical_features
        
        # 检查是否有特征
        if not feature_columns:
            logger.error("没有找到特征列。请检查特征命名模式。")
            raise ValueError("没有找到特征列")
        
        # 创建特征矩阵和目标向量
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # 移除特征值为空的行
        missing_mask = X.isnull().any(axis=1)
        if missing_mask.sum() > 0:
            logger.warning(f"移除 {missing_mask.sum()} 行特征值为空的数据")
            X = X[~missing_mask]
            y = y[~missing_mask]
        
        # 编码分类特征
        if categorical_features:
            logger.info(f"编码 {len(categorical_features)} 个分类特征")
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_features = encoder.fit_transform(X[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            
            # 创建包含编码特征的 DataFrame
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            
            # 删除原始分类列并添加编码后的列
            X = X.drop(columns=categorical_features)
            X = pd.concat([X, encoded_df], axis=1)
            
            # 更新特征列列表
            feature_columns = numeric_features + list(encoded_feature_names)
        
        # 检查数据量是否足够
        if len(X) < 10:
            logger.warning(f"数据量太少（{len(X)}个样本），建议增加数据量")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"训练数据: {X_train.shape[0]}个样本, 测试数据: {X_test.shape[0]}个样本")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    except Exception as e:
        logger.error(f"预处理数据时出错: {e}")
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
                    n_estimators=200,  # 增加树的数量
                    max_depth=4,      # 增加树的深度
                    min_samples_split=3,  # 减少分裂所需的最小样本数
                    min_samples_leaf=2,   # 减少叶节点所需的最小样本数
                    max_features='sqrt',  # 使用sqrt特征数
                    bootstrap=True,       # 使用bootstrap采样
                    random_state=42,
                    oob_score=True,       # 使用袋外样本评估
                    n_jobs=-1            # 使用所有CPU核心
                )
            elif model_name == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=200,  # 增加树的数量
                    max_depth=4,      # 增加树的深度
                    learning_rate=0.05,  # 降低学习率
                    min_samples_split=3,  # 减少分裂所需的最小样本数
                    min_samples_leaf=2,   # 减少叶节点所需的最小样本数
                    random_state=42
                )
            else:
                logger.warning(f"未知的回归模型: {model_name}. 使用 RandomForest.")
                model = RandomForestRegressor(n_estimators=200, random_state=42)
        
        elif model_type == 'classification':
            if model_name == 'logistic':
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=200,  # 增加树的数量
                    max_depth=4,      # 增加树的深度
                    min_samples_split=3,  # 减少分裂所需的最小样本数
                    min_samples_leaf=2,   # 减少叶节点所需的最小样本数
                    max_features='sqrt',  # 使用sqrt特征数
                    bootstrap=True,       # 使用bootstrap采样
                    random_state=42,
                    oob_score=True,       # 使用袋外样本评估
                    n_jobs=-1            # 使用所有CPU核心
                )
            else:
                logger.warning(f"未知的分类模型: {model_name}. 使用 RandomForest.")
                model = RandomForestClassifier(n_estimators=200, random_state=42)
        
        else:
            logger.error(f"未知的模型类型: {model_type}")
            raise ValueError(f"未知的模型类型: {model_type}")
        
        return model
    
    except Exception as e:
        logger.error(f"创建管道时出错: {e}")
        raise

def train_model(X_train, y_train, model, cv=2):
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
        # 检查数据量
        if len(X_train) < 10:
            logger.warning(f"训练数据量太少（{len(X_train)}个样本），建议增加数据量")
            # 如果数据量太少，不使用交叉验证
            model.fit(X_train, y_train)
            return model, np.array([0.0])
        
        # 根据数据量调整交叉验证折数
        cv = min(cv, len(X_train) // 2)
        if cv < 2:
            logger.warning("数据量太少，不使用交叉验证")
            model.fit(X_train, y_train)
            return model, np.array([0.0])
        
        # 执行交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        logger.info(f"交叉验证 R² 分数: {cv_scores}")
        logger.info(f"平均 CV R² 分数: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 在完整训练集上训练模型
        model.fit(X_train, y_train)
        
        return model, cv_scores
    
    except Exception as e:
        logger.error(f"训练模型时出错: {e}")
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
        # 检查数据量
        if len(X_test) < 2:
            logger.warning(f"测试数据量太少（{len(X_test)}个样本），无法计算评估指标")
            return {
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan
            }
        
        # 进行预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = {}
        
        if model_type == 'regression':
            # 计算基本指标
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            
            # 对于小数据集，使用更稳健的R²计算
            if len(y_test) < 10:
                # 使用调整后的R²
                n = len(y_test)
                p = X_test.shape[1]
                r2 = r2_score(y_test, y_pred)
                metrics['r2'] = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            else:
                metrics['r2'] = r2_score(y_test, y_pred)
            
            # 添加相对误差指标
            metrics['relative_error'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            logger.info(f"测试 MSE: {metrics['mse']:.4f}")
            logger.info(f"测试 RMSE: {metrics['rmse']:.4f}")
            logger.info(f"测试 MAE: {metrics['mae']:.4f}")
            logger.info(f"测试 R²: {metrics['r2']:.4f}")
            logger.info(f"相对误差: {metrics['relative_error']:.2f}%")
        
        elif model_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            logger.info(f"测试准确率: {metrics['accuracy']:.4f}")
            logger.info(f"测试精确率: {metrics['precision']:.4f}")
            logger.info(f"测试召回率: {metrics['recall']:.4f}")
            logger.info(f"测试 F1: {metrics['f1']:.4f}")
        
        return metrics, y_pred
    
    except Exception as e:
        logger.error(f"评估模型时出错: {e}")
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
            logger.warning("模型不提供特征重要性信息")
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
        plt.title('特征重要性')
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
        plt.close()
        
        # 绘制特征重要性分布图
        plt.figure(figsize=(10, 6))
        plt.title('特征重要性分布')
        sns.histplot(importance_df['Importance'], bins=20)
        plt.xlabel('重要性')
        plt.ylabel('特征数量')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance_distribution.png'))
        plt.close()
        
        # 绘制特征重要性累积图
        plt.figure(figsize=(10, 6))
        plt.title('特征重要性累积')
        cumulative_importance = np.cumsum(importance_df['Importance'])
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance_cumulative.png'))
        plt.close()
        
        logger.info(f"特征重要性分析完成。结果保存到 {importance_path}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"分析特征重要性时出错: {e}")
        return None

def create_prediction_vs_actual_plot(y_test, y_pred, output_dir):
    """
    Create a plot of predicted vs. actual values
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.array): Predicted values
        output_dir (str): Directory to save plot
    """
    try:
        # 创建可视化目录
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 创建散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # 添加完美预测线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('预测值 vs. 实际值')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存散点图
        plt.savefig(os.path.join(viz_dir, 'predicted_vs_actual.png'))
        plt.close()
        
        # 创建残差图
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('残差图')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存残差图
        plt.savefig(os.path.join(viz_dir, 'residuals.png'))
        plt.close()
        
        # 创建残差分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=20)
        plt.title('残差分布')
        plt.xlabel('残差')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存残差分布图
        plt.savefig(os.path.join(viz_dir, 'residuals_distribution.png'))
        plt.close()
        
        logger.info(f"预测 vs. 实际值图保存到 {viz_dir}")
    
    except Exception as e:
        logger.error(f"创建预测 vs. 实际值图时出错: {e}")

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
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存模型信息
        info_path = os.path.join(output_dir, f"{model_name}_info.json")
        
        # 将 numpy 值转换为 Python 原生类型以便 JSON 序列化
        for key, value in model_info.items():
            if isinstance(value, np.ndarray):
                model_info[key] = value.tolist()
            elif isinstance(value, np.integer):
                model_info[key] = int(value)
            elif isinstance(value, np.floating):
                model_info[key] = float(value)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型保存到 {model_path}")
        logger.info(f"模型信息保存到 {info_path}")
        
        # 打印模型信息摘要
        logger.info("模型信息摘要:")
        logger.info(f"模型类型: {model_info.get('model_type', '未知')}")
        logger.info(f"模型名称: {model_info.get('model_name', '未知')}")
        logger.info(f"目标列: {model_info.get('target_column', '未知')}")
        logger.info(f"目标窗口: {model_info.get('target_window', '未知')}")
        logger.info(f"训练样本数: {model_info.get('training_samples', '未知')}")
        logger.info(f"测试样本数: {model_info.get('test_samples', '未知')}")
        logger.info(f"特征数量: {model_info.get('feature_count', '未知')}")
        
        if 'metrics' in model_info:
            logger.info("模型指标:")
            for metric_name, metric_value in model_info['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    logger.info(f"{metric_name}: {metric_value:.4f}")
        
        return model_path, info_path
    
    except Exception as e:
        logger.error(f"保存模型时出错: {e}")
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
        logger.info(f"开始训练模型: {model_type} - {model_name}")
        logger.info(f"目标窗口: {target_window}天")
        
        # 加载并合并数据
        data = load_data(features_path, targets_path)
        
        # 根据目标窗口和模型类型确定目标列
        if model_type == 'regression':
            target_column = f'future_return_{target_window}d'
        else:
            target_column = f'future_up_{target_window}d'
        
        logger.info(f"使用目标列: {target_column}")
        
        # 预处理数据
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            data, target_column, test_size=0.2
        )
        
        # 创建模型
        model = create_pipeline(model_type, model_name)
        
        # 训练模型
        trained_model, cv_scores = train_model(X_train, y_train, model)
        
        # 评估模型
        metrics, y_pred = evaluate_model(trained_model, X_test, y_test, model_type)
        
        # 特征重要性分析
        importance_df = feature_importance_analysis(trained_model, feature_names, output_dir)
        
        # 创建预测 vs. 实际值图（仅用于回归）
        if model_type == 'regression':
            create_prediction_vs_actual_plot(y_test, y_pred, output_dir)
        
        # 保存模型和结果
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
        
        logger.info(f"模型训练完成。模型保存到 {model_path}")
        logger.info(f"模型指标: {metrics}")
        
        return model_path, metrics
    
    except Exception as e:
        logger.error(f"模型训练过程中出错: {e}")
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
    
    logger.info(f"开始模型训练过程")
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"模型名称: {args.model_name}")
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 获取参数
    features_dir = config.get('features_directory', './data/processed/features')
    targets_dir = config.get('targets_directory', './data/processed/targets')
    models_dir = config.get('models_directory', './models')
    
    # 使用命令行参数或配置中的目标窗口
    target_window = args.target_window or config.get('target_window', 60)
    logger.info(f"目标窗口: {target_window}天")
    
    # 文件路径
    features_path = os.path.join(features_dir, 'llm_features.csv')
    targets_path = os.path.join(targets_dir, 'stock_targets.csv')
    
    logger.info(f"特征文件: {features_path}")
    logger.info(f"目标文件: {targets_path}")
    
    # 训练模型
    model_path, metrics = train_return_prediction_model(
        features_path=features_path,
        targets_path=targets_path,
        output_dir=models_dir,
        target_window=target_window,
        model_type=args.model_type,
        model_name=args.model_name
    )
    
    logger.info(f"模型训练完成。模型保存到 {model_path}")
    logger.info(f"模型指标: {metrics}")

if __name__ == "__main__":
    main()
