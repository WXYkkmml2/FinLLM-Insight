#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Target Generation Module for FinLLM-Insight
This module creates target variables for model training based on stock price data.
"""

iimport os
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
    """
    Get historical stock price data for a given stock
    
    Args:
        stock_code (str): Stock code (6 digits)
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        
    Returns:
        pd.DataFrame: DataFrame with stock price history
    """
    try:
        # Determine market based on stock code prefix
        # 6xxxxx: Shanghai, 0xxxxx or 3xxxxx: Shenzhen
        if stock_code.startswith('6'):
            market = 'sh'
        else:
            market = 'sz'
        
        # Fetch stock price history using AKShare
        # Format stock_code to meet AKShare's requirements (e.g., sh600000 or sz000001)
        symbol = f"{market}{stock_code}"
        
        # AKShare's stock_zh_a_hist function gets A-share historical data
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # Use qfq (front-adjusted) price
        )
        
        # Rename columns to standard English names
        column_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        
        # Check if columns exist before renaming
        existing_columns = {}
        for cn_col, en_col in column_map.items():
            if cn_col in df.columns:
                existing_columns[cn_col] = en_col
        
        df = df.rename(columns=existing_columns)
        
        # Ensure 'date' is datetime type
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to get stock price history for {stock_code}: {e}")
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
    Create mapping from stock code to report publication dates
    
    Args:
        reports_dir (str): Directory containing reports
        
    Returns:
        dict: Mapping from stock code to report dates
    """
    mapping = {}
    
    # Check if download results file exists
    results_path = os.path.join(reports_dir, 'download_results.csv')
    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path, encoding='utf-8-sig')
            
            # Group by stock code and year
            for _, row in results_df.iterrows():
                if 'stock_code' in row and 'year' in row and pd.notna(row['year']):
                    stock_code = row['stock_code']
                    year = int(row['year'])
                    
                    # Assume report date is the file creation date if no specific date is available
                    # This is a simplification; in production, you should extract the actual report date
                    if 'file_path' in row and pd.notna(row['file_path']) and os.path.exists(row['file_path']):
                        # Use file modification time as a proxy for report date
                        file_date = datetime.fromtimestamp(os.path.getmtime(row['file_path']))
                        
                        # Structure: {stock_code: {year: date}}
                        if stock_code not in mapping:
                            mapping[stock_code] = {}
                        
                        mapping[stock_code][year] = file_date.strftime('%Y%m%d')
            
            return mapping
        
        except Exception as e:
            logger.error(f"Failed to load report dates from results file: {e}")
    
    # If no results file or error, walk through directories
    try:
        for year in os.listdir(reports_dir):
            year_dir = os.path.join(reports_dir, year)
            if os.path.isdir(year_dir) and year.isdigit():
                year_int = int(year)
                
                for file in os.listdir(year_dir):
                    if file.endswith('.pdf') or file.endswith('.txt'):
                        # Extract stock code from filename (assuming format: stock_code_year_*.pdf/txt)
                        parts = file.split('_')
                        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].isdigit():
                            stock_code = parts[0]
                            
                            # Use file modification date
                            file_path = os.path.join(year_dir, file)
                            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            if stock_code not in mapping:
                                mapping[stock_code] = {}
                            
                            mapping[stock_code][year_int] = file_date.strftime('%Y%m%d')
        
        return mapping
    
    except Exception as e:
        logger.error(f"Failed to create report date mapping: {e}")
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
