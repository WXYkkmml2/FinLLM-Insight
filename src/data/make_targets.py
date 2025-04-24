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

def get_stock_price_history(ticker, start_date, end_date):
    """
    Get stock price history for US stocks using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        
    Returns:
        pd.DataFrame: DataFrame with price history
    """
    try:
        import yfinance as yf
        
        # Format dates for yfinance
        start_date_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        
        if end_date is None:
            from datetime import datetime
            end_date_fmt = datetime.now().strftime("%Y-%m-%d")
        else:
            end_date_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        logger.info(f"Getting price history for {ticker} from {start_date_fmt} to {end_date_fmt}")
        
        # Download price data
        df = yf.download(
            ticker,
            start=start_date_fmt,
            end=end_date_fmt,
            progress=False
        )
        
        # Ensure index is datetime type
        df.index = pd.to_datetime(df.index)
        
        # Check if data was retrieved
        if df.empty:
            logger.warning(f"No price data found for {ticker}")
            return None
        
        # Ensure columns follow a standardized format
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in expected_columns:
            if col not in df.columns:
                logger.warning(f"Missing expected column {col} for {ticker}")
        

        # Convert column names to strings and use lowercase for consistency
        if 'adj close' in df.columns:
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)        
        # Rename 'adj close' to 'adj_close' for easier access
        if 'adj close' in df.columns:
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)
        
        logger.info(f"Retrieved {len(df)} days of price data for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error getting price history for {ticker}: {e}")
        return None

def calculate_future_returns(price_df, windows=[1, 5, 20, 60, 120]):
    """计算不同时间窗口的未来收益率"""
    if price_df is None or len(price_df) == 0:
        return None
    
    # 创建副本以避免修改原始数据
    df = price_df.copy()
    
    # 检查列名并标准化 - 确保 df.columns 是字符串列表，而不是元组
    columns_list = list(df.columns)  # 转换为列表以确保可以处理
    
    # 查找合适的价格列
    price_col = None
    if 'close' in columns_list:
        price_col = 'close'
    elif 'Close' in columns_list:
        price_col = 'Close'
        df.columns = [str(col).lower() for col in columns_list]  # 确保每个列名都是字符串
        price_col = 'close'  # 直接使用小写名称
    else:
        # 尝试其他可能的价格列名
        for col_name in ['adj_close', 'adj close', 'Adj Close', 'adjusted_close', 'price']:
            if col_name in columns_list:
                price_col = col_name
                break
    
    # 如果仍然找不到价格列，则记录错误并使用第一个数值列
    if price_col is None:
        print(f"警告: 找不到价格列。可用列: {columns_list}")
        # 尝试查找在(,)中形式的列，这通常是多级索引转换为列名的情况
        for col in columns_list:
            if isinstance(col, tuple) and col[1] == 'Close':
                price_col = col
                print(f"使用 {col} 作为价格列")
                break
            elif isinstance(col, tuple) and 'close' in col[1].lower():
                price_col = col
                print(f"使用 {col} 作为价格列")
                break
        
        # 如果仍然找不到，尝试使用第一个看起来是数值列的列
        if price_col is None:
            for col in columns_list:
                if pd.api.types.is_numeric_dtype(df[col]):
                    price_col = col
                    print(f"使用 {col} 作为价格列")
                    break
    
    if price_col is None:
        return None
    
    # 计算各窗口的未来收益率
    for window in windows:
        try:
            # 计算未来价格
            future_price = df[price_col].shift(-window)
            
            # 计算未来收益率(%)
            df[f'future_return_{window}d'] = (future_price - df[price_col]) / df[price_col] * 100
            
            # 创建二值目标
            df[f'future_up_{window}d'] = (df[f'future_return_{window}d'] > 0).astype(int)
            
            # 创建分类目标
            try:
                df[f'future_category_{window}d'] = pd.qcut(
                    df[f'future_return_{window}d'],
                    q=4,
                    labels=['poor', 'below_avg', 'above_avg', 'excellent']
                )
            except Exception as e:
                print(f"无法创建分类目标: {e}")
                # 创建简单的二分类替代
                df[f'future_category_{window}d'] = df[f'future_up_{window}d'].map({0: 'down', 1: 'up'})
        except Exception as e:
            print(f"计算窗口 {window} 的未来收益时出错: {e}")
    
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
            stock_code_cols = ['ticker', 'stock_code', 'company_code']
            year_cols = ['year', 'report_year']
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
                            
                            # 使用当前时间减去3个月作为报告日期（避免未来日期问题）
                            # 这是一个变通方法，这样我们就不会有一个未来的报告日期
                            adjusted_date = datetime.now() - timedelta(days=90)
                            mapping[stock_code][year] = adjusted_date.strftime('%Y%m%d')
                            
                            # Debug log
                            logger.info(f"Adjusted report date for {stock_code}_{year}: {adjusted_date.strftime('%Y%m%d')}")
            
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
                    if file.endswith('.html') or file.endswith('.txt'):
                        # 从文件名提取股票代码（假设格式：stock_code_year_*.html/txt）
                        parts = file.split('_')
                        if len(parts) >= 2:
                            stock_code = parts[0]
                            
                            # 使用文件修改日期
                            file_path = os.path.join(year_dir, file)
                            
                            # 使用当前时间减去3个月作为报告日期（避免未来日期问题）
                            adjusted_date = datetime.now() - timedelta(days=90)
                            
                            if stock_code not in mapping:
                                mapping[stock_code] = {}
                            
                            mapping[stock_code][year_int] = adjusted_date.strftime('%Y%m%d')
                            
                            # Debug log
                            logger.info(f"Adjusted report date for {stock_code}_{year_int}: {adjusted_date.strftime('%Y%m%d')}")
        
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
                ticker=stock_code,
                start_date=price_start_date,
                end_date=price_end_date
            )
            
            if price_history is None or len(price_history) < 20:  # Need at least 20 days for basic analysis
                logger.warning(f"Insufficient price data for {stock_code}, skipping")
                continue
            
            # Calculate future returns
            returns_df = calculate_future_returns(price_history)
            
            if returns_df is None:
                continue
            
            # Add a debug log to show the available trading dates
            logger.debug(f"Trading dates for {stock_code}: {returns_df.index[0]} to {returns_df.index[-1]}")
            
            # For each report year, extract the targets
            for year, report_date in years.items():
                try:
                    # Convert report date to datetime
                    try:
                        report_datetime = datetime.strptime(report_date, '%Y%m%d')
                    except ValueError:
                        # Try alternative date formats if needed
                        formats = ['%Y%m%d', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
                        for fmt in formats:
                            try:
                                report_datetime = datetime.strptime(report_date, fmt)
                                break
                            except:
                                continue
                        else:
                            # If all format attempts fail, use a default date (3 months ago)
                            report_datetime = datetime.now() - timedelta(days=90)
                            logger.warning(f"Could not parse report date '{report_date}' for {stock_code}_{year}, using default")
                    
                    # Find the closest trading day on or after the report date
                    valid_dates = returns_df.index[returns_df.index >= report_datetime]
                    
                    if len(valid_dates) == 0:
                        # FALLBACK STRATEGY: If no future dates, use the most recent trading day
                        logger.warning(f"No trading days found after report date {report_datetime} for {stock_code}_{year}")
                        
                        # Find the closest trading day before the report date
                        prior_dates = returns_df.index[returns_df.index <= report_datetime]
                        
                        if len(prior_dates) == 0:
                            logger.warning(f"No trading days found at all for {stock_code}_{year}")
                            continue
                        
                        reference_date = prior_dates[-1]  # Use the most recent prior trading day
                        logger.info(f"Falling back to closest prior trading day: {reference_date} for {stock_code}_{year}")
                    else:
                        reference_date = valid_dates[0]
                        logger.info(f"Using trading day: {reference_date} for {stock_code}_{year}")
                    
                    # Extract the targets for this report
                    target_row = returns_df.loc[reference_date:reference_date].copy()
                    
                    if len(target_row) == 0:
                        logger.warning(f"Could not extract target row for {stock_code}_{year} at {reference_date}")
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
    try:
        # Create a directory for visualizations
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 确保列名是字符串类型，处理可能的元组列名
        targets_df.columns = [col if isinstance(col, str) else f"{col[0]}_{col[1]}" for col in targets_df.columns]
        
        # 获取所有以'future_return_'开头的列
        return_columns = [col for col in targets_df.columns if isinstance(col, str) and col.startswith('future_return_')]
        
        if return_columns:  # 只在有相关列时创建图表
            try:
                # Create histograms of returns for different time windows
                plt.figure(figsize=(15, 10))
                
                for i, col in enumerate(return_columns[:6]):  # 最多显示6个窗口
                    plt.subplot(2, 3, i+1)
                    
                    # Convert window name for title (e.g., future_return_20d -> 20-day)
                    window = col.split('_')[-1]
                    window = window.replace('d', '-day')
                    
                    # 删除NaN值
                    valid_data = targets_df[col].dropna()
                    
                    if len(valid_data) > 0:  # 确保有有效数据
                        sns.histplot(valid_data, kde=True)
                        plt.title(f'Distribution of {window} Returns')
                        plt.xlabel('Return (%)')
                        plt.ylabel('Frequency')
                        
                        # Add vertical line at 0
                        plt.axvline(x=0, color='red', linestyle='--')
                
                # 使用更安全的布局调整
                try:
                    plt.tight_layout()
                except:
                    plt.subplots_adjust(wspace=0.3, hspace=0.3)
                
                plt.savefig(os.path.join(viz_dir, 'return_distributions.png'))
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating return distributions visualization: {e}")
        
        # 获取所有以'future_up_'开头的列
        up_columns = [col for col in targets_df.columns if isinstance(col, str) and col.startswith('future_up_')]
        
        if up_columns:  # 只在有相关列时创建图表
            try:
                # Create bar charts of up/down ratios
                plt.figure(figsize=(12, 6))
                
                # 安全计算比率，处理NaN值
                up_ratios = []
                windows = []
                
                for col in up_columns:
                    valid_data = targets_df[col].dropna()
                    if len(valid_data) > 0:  # 确保有有效数据
                        up_ratios.append(valid_data.mean() * 100)
                        windows.append(col.split('_')[-1].replace('d', ''))
                
                if len(up_ratios) > 0:  # 确保有数据要绘制
                    plt.bar(windows, up_ratios)
                    plt.title('Percentage of Stocks with Positive Returns')
                    plt.xlabel('Time Window (days)')
                    plt.ylabel('Positive Return %')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add horizontal line at 50%
                    plt.axhline(y=50, color='red', linestyle='--')
                    
                    # Add value labels - 确保v不是NaN
                    for i, v in enumerate(up_ratios):
                        if not np.isnan(v):  # 检查是否为NaN
                            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
                    
                    # 使用更安全的布局调整
                    try:
                        plt.tight_layout()
                    except:
                        plt.subplots_adjust(wspace=0.3, hspace=0.3)
                    
                    plt.savefig(os.path.join(viz_dir, 'positive_return_ratios.png'))
                
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating up/down ratios visualization: {e}")
        
        # 获取所有以'future_category_'开头的列
        cat_columns = [col for col in targets_df.columns if isinstance(col, str) and col.startswith('future_category_')]
        
        if cat_columns:  # 只在有相关列时创建图表
            try:
                # Create distribution by category
                plt.figure(figsize=(15, 10))
                
                for i, col in enumerate(cat_columns[:6]):  # 最多显示6个窗口
                    plt.subplot(2, 3, i+1)
                    
                    # Convert window name for title
                    window = col.split('_')[-1]
                    window = window.replace('d', '-day')
                    
                    # 安全地获取分类计数
                    valid_data = targets_df[col].dropna()
                    if len(valid_data) > 0:  # 确保有有效数据
                        category_counts = valid_data.value_counts().sort_index()
                        
                        if len(category_counts) > 0:  # 确保有数据要绘制
                            plt.pie(
                                category_counts, 
                                labels=category_counts.index, 
                                autopct='%1.1f%%',
                                startangle=90
                            )
                            plt.title(f'Distribution of {window} Return Categories')
                
                # 使用更安全的布局调整
                try:
                    plt.tight_layout()
                except:
                    plt.subplots_adjust(wspace=0.3, hspace=0.3)
                
                plt.savefig(os.path.join(viz_dir, 'return_categories.png'))
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating category distribution visualization: {e}")
    
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
        # 继续执行程序，不要因为可视化失败而中断整个目标生成过程

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
