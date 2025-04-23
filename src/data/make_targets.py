#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Target Generation Module for FinLLM-Insight
This module creates target variables for model training based on stock price data.
"""

import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import json
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("make_targets.log"),
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

def get_stock_price_history(stock_code, start_date, end_date):
    """获取股票历史价格数据"""
    try:
        # 确定市场代码
        if stock_code.startswith('6'):
            market = 'sh'
        else:
            market = 'sz'
        
        # 格式化股票代码
        symbol = f"{market}{stock_code}"
        
        # 获取股票历史价格数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权价格
        )
        
        # 打印原始列名，便于调试
        logger.debug(f"原始数据列名: {df.columns.tolist()}")
        
        # 定义可能的列名映射（更全面）
        column_mappings = {
            'date': ['日期', '交易日期', 'date', 'trade_date'],
            'open': ['开盘', '开盘价', 'open', 'open_price'],
            'close': ['收盘', '收盘价', 'close', 'close_price'],
            'high': ['最高', '最高价', 'high', 'high_price'],
            'low': ['最低', '最低价', 'low', 'low_price'],
            'volume': ['成交量', '成交股数', 'volume', 'vol'],
            'amount': ['成交额', '成交金额', 'amount'],
            'pct_change': ['涨跌幅', '变动率', 'pct_chg', 'change_pct'],
            'change': ['涨跌额', '价格变动', 'price_change'],
            'turnover': ['换手率', 'turnover']
        }
        
        # 创建重命名映射
        rename_map = {}
        for target_col, possible_names in column_mappings.items():
            for col_name in possible_names:
                if col_name in df.columns:
                    rename_map[col_name] = target_col
                    break
        
        # 重命名列
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"重命名列: {rename_map}")
        
        # 确保日期列是datetime类型
        date_col = 'date'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            # 如果找不到date列，尝试找其他可能的日期列
            for col in df.columns:
                if '日期' in col or 'date' in col.lower():
                    logger.warning(f"使用替代日期列: {col}")
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
        
        return df
    
    except Exception as e:
        logger.error(f"获取{stock_code}价格历史失败: {e}")
        return None

def calculate_future_returns(price_df, windows=[1, 5, 20, 60, 120]):
    """
    Calculate future returns for different time windows
    
    Args:
        price_df (pd.DataFrame): DataFrame with price history
        windows (list): List of time windows (in trading days)
        
    Returns:
        pd.DataFrame: DataFrame with future returns
    """
    if price_df is None or len(price_df) == 0:
        return None
    
    # Make a copy to avoid modifying the original
    df = price_df.copy()
    
    # Calculate future returns for each window
    for window in windows:
        # Calculate future price
        future_price = df['close'].shift(-window)
        
        # Calculate future return (%)
        df[f'future_return_{window}d'] = (future_price - df['close']) / df['close'] * 100
        
        # Create binary target (1 if return is positive, 0 otherwise)
        df[f'future_up_{window}d'] = (df[f'future_return_{window}d'] > 0).astype(int)
        
        # Create categorical target based on return quartiles
        # Bins: bottom 25%, 25-50%, 50-75%, top 25%
        df[f'future_category_{window}d'] = pd.qcut(
            df[f'future_return_{window}d'],
            q=4,
            labels=['poor', 'below_avg', 'above_avg', 'excellent']
        )
    
    return df

def create_report_date_mapping(reports_dir):
    """
    创建从股票代码到报告发布日期的映射
    
    Args:
        reports_dir (str): 包含报告的目录
        
    Returns:
        dict: 从股票代码到报告日期的映射
    """
    mapping = {}
    
    # 检查下载结果文件是否存在
    results_path = os.path.join(reports_dir, 'download_results.csv')
    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path, encoding='utf-8-sig')
            
            # 打印列名，便于调试
            logger.debug(f"下载结果文件列名: {results_df.columns.tolist()}")
            
            # 定义可能的列名
            stock_code_cols = ['stock_code', 'company_code', '股票代码', '公司代码', '代码']
            year_cols = ['year', 'report_year', '年份', '报告年份']
            
            # 确定实际使用的列名
            stock_code_col = None
            for col in stock_code_cols:
                if col in results_df.columns:
                    stock_code_col = col
                    break
            
            year_col = None
            for col in year_cols:
                if col in results_df.columns:
                    year_col = col
                    break
            
            if not stock_code_col or not year_col:
                logger.warning(f"结果文件中找不到股票代码或年份列，可用列: {results_df.columns.tolist()}")
            else:
                # 按股票代码和年份分组
                for _, row in results_df.iterrows():
                    if pd.notna(row[stock_code_col]) and pd.notna(row[year_col]):
                        stock_code = row[stock_code_col]
                        try:
                            year = int(row[year_col])
                        except:
                            year = row[year_col]  # 如果转换失败，直接使用原值
                        
                        # 提取报告日期
                        if 'file_path' in row and pd.notna(row['file_path']) and os.path.exists(row['file_path']):
                            file_date = datetime.fromtimestamp(os.path.getmtime(row['file_path']))
                            
                            # 结构: {stock_code: {year: date}}
                            if stock_code not in mapping:
                                mapping[stock_code] = {}
                            
                            mapping[stock_code][year] = file_date.strftime('%Y%m%d')
            
            return mapping
            
        except Exception as e:
            logger.error(f"从结果文件加载报告日期时出错: {e}")
    
    # 如果没有结果文件或出错，遍历目录
    try:
        for year in os.listdir(reports_dir):
            year_dir = os.path.join(reports_dir, year)
            if os.path.isdir(year_dir) and year.isdigit():
                year_int = int(year)
                
                for file in os.listdir(year_dir):
                    if file.endswith('.pdf') or file.endswith('.txt'):
                        # 从文件名提取股票代码（假设格式：stock_code_year_*.pdf/txt）
                        parts = file.split('_')
                        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].isdigit():
                            stock_code = parts[0]
                            
                            # 使用文件修改日期
                            file_path = os.path.join(year_dir, file)
                            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            if stock_code not in mapping:
                                mapping[stock_code] = {}
                            
                            mapping[stock_code][year_int] = file_date.strftime('%Y%m%d')
        
        return mapping
    
    except Exception as e:
        logger.error(f"创建报告日期映射时出错: {e}")
        return {}
def generate_targets(reports_dir, output_dir, price_start_date=None, price_end_date=None):
    """
    Generate target variables for all stocks with reports
    
    Args:
        reports_dir (str): Directory containing reports
        output_dir (str): Directory to save target files
        price_start_date (str): Start date for price data, format 'YYYYMMDD'
        price_end_date (str): End date for price data, format 'YYYYMMDD'
        
    Returns:
        pd.DataFrame: Combined targets for all stocks
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default date range if not provided
    if price_end_date is None:
        price_end_date = datetime.now().strftime('%Y%m%d')
    
    if price_start_date is None:
        # Default to 5 years before end date
        start_date = datetime.now() - timedelta(days=5*365)
        price_start_date = start_date.strftime('%Y%m%d')
    
    # Get mapping from stock codes to report dates
    report_dates = create_report_date_mapping(reports_dir)
    
    if not report_dates:
        logger.error("No report dates found. Cannot generate targets.")
        return None
    
    all_targets = []
    
    # Process each stock
    for stock_code, years in tqdm(report_dates.items(), desc="Generating targets"):
        try:
            # Get price history for this stock
            price_history = get_stock_price_history(
                stock_code=stock_code,
                start_date=price_start_date,
                end_date=price_end_date
            )
            
            if price_history is None or len(price_history) < 60:  # Need at least 60 days for meaningful analysis
                logger.warning(f"Insufficient price data for {stock_code}, skipping")
                continue
            
            # Calculate future returns
            returns_df = calculate_future_returns(price_history)
            
            if returns_df is None:
                continue
            
            # For each report year, extract the targets
            for year, report_date in years.items():
                try:
                    # Convert report date to datetime
                    report_datetime = datetime.strptime(report_date, '%Y%m%d')
                    
                    # Find the closest trading day on or after the report date
                    valid_dates = returns_df.index[returns_df.index >= report_datetime]
                    
                    if len(valid_dates) == 0:
                        logger.warning(f"No trading days found after report date for {stock_code}_{year}")
                        continue
                    
                    reference_date = valid_dates[0]
                    
                    # Extract the targets for this report
                    target_row = returns_df.loc[reference_date:reference_date].copy()
                    
                    if len(target_row) == 0:
                        continue
                    
                    # Add stock code and year
                    target_row['stock_code'] = stock_code
                    target_row['report_year'] = year
                    target_row['report_date'] = report_date
                    
                    # Reset index to include date as a column
                    target_row.reset_index(inplace=True)
                    
                    all_targets.append(target_row)
                
                except Exception as e:
                    logger.error(f"Error processing targets for {stock_code}_{year}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to process stock {stock_code}: {e}")
    
    if not all_targets:
        logger.error("No targets generated for any stock")
        return None
    
    # Combine all targets
    combined_targets = pd.concat(all_targets, ignore_index=True)
    
    # Save combined targets
    targets_path = os.path.join(output_dir, 'stock_targets.csv')
    combined_targets.to_csv(targets_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Generated targets for {len(combined_targets)} reports. Saved to {targets_path}")
    
    # Create summary visualizations
    try:
        create_target_visualizations(combined_targets, output_dir)
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
    
    return combined_targets

def create_target_visualizations(targets_df, output_dir):
    """
    Create visualizations summarizing the target variables
    
    Args:
        targets_df (pd.DataFrame): DataFrame with target variables
        output_dir (str): Directory to save visualizations
    """
    # Create a directory for visualizations
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create histograms of returns for different time windows
    plt.figure(figsize=(15, 10))
    
    return_columns = [col for col in targets_df.columns if col.startswith('future_return_')]
    
    for i, col in enumerate(return_columns):
        plt.subplot(2, 3, i+1)
        
        # Convert window name for title (e.g., future_return_20d -> 20-day)
        window = col.split('_')[-1]
        window = window.replace('d', '-day')
        
        sns.histplot(targets_df[col].dropna(), kde=True)
        plt.title(f'Distribution of {window} Returns')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        
        # Add vertical line at 0
        plt.axvline(x=0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'return_distributions.png'))
    plt.close()
    
    # Create bar charts of up/down ratios
    plt.figure(figsize=(12, 6))
    
    up_columns = [col for col in targets_df.columns if col.startswith('future_up_')]
    up_ratios = [targets_df[col].mean() * 100 for col in up_columns]
    windows = [col.split('_')[-1].replace('d', '') for col in up_columns]
    
    plt.bar(windows, up_ratios)
    plt.title('Percentage of Stocks with Positive Returns')
    plt.xlabel('Time Window (days)')
    plt.ylabel('Positive Return %')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line at 50%
    plt.axhline(y=50, color='red', linestyle='--')
    
    # Add value labels
    for i, v in enumerate(up_ratios):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'positive_return_ratios.png'))
    plt.close()
    
    # Create distribution by category
    plt.figure(figsize=(15, 10))
    
    cat_columns = [col for col in targets_df.columns if col.startswith('future_category_')]
    
    for i, col in enumerate(cat_columns):
        plt.subplot(2, 3, i+1)
        
        # Convert window name for title
        window = col.split('_')[-1]
        window = window.replace('d', '-day')
        
        category_counts = targets_df[col].value_counts().sort_index()
        
        plt.pie(
            category_counts, 
            labels=category_counts.index, 
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title(f'Distribution of {window} Return Categories')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'return_categories.png'))
    plt.close()

def main():
    """Main function to run the target generation process"""
    parser = argparse.ArgumentParser(description='Generate target variables for model training')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get directories
    reports_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
    output_dir = config.get('targets_directory', './data/processed/targets')
    
    # Get date range for price data
    price_start_date = config.get('price_start_date', None)
    price_end_date = config.get('price_end_date', None)
    
    # Generate targets
    generate_targets(
        reports_dir=reports_dir,
        output_dir=output_dir,
        price_start_date=price_start_date,
        price_end_date=price_end_date
    )

if __name__ == "__main__":
    main()
