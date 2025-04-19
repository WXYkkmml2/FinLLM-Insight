#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Acquisition Module for FinLLM-Insight
This module downloads annual reports of Chinese listed companies using AKShare.
"""

import os
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_reports.log"),
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

def get_stock_list(index_name):
    """
    Get stock list from specified index
    
    Args:
        index_name (str): Name of the index (e.g., 'CSI300' for 沪深300指数)
        
    Returns:
        pd.DataFrame: DataFrame containing stock symbols and names
    """
    try:
        logger.info(f"Retrieving stock list for index: {index_name}")
        
        if index_name == "CSI300":
            # Get CSI300 constituent stocks
            stock_list = ak.index_stock_cons_csindex(symbol="000300")
            # Extract stock codes and names
            stock_list = stock_list[['成分券代码', '成分券名称']]
            stock_list.columns = ['code', 'name']
            # Format code to ensure 6 digits with leading zeros
            stock_list['code'] = stock_list['code'].apply(lambda x: f"{x:06d}")
        elif index_name == "CSI50":
            # Custom processing method: Get the Shanghai and Shenzhen 300 and retain only the top 50 companies
            stock_list = ak.index_stock_cons_csindex(symbol="000300")
            # Extract stock codes and names
            stock_list = stock_list[['成分券代码', '成分券名称']]
            stock_list.columns = ['code', 'name']
            # Format code to ensure 6 digits with leading zeros
            stock_list['code'] = stock_list['code'].apply(lambda x: f"{x:06d}")
            # Only keep first 50 companies
            stock_list = stock_list.head(50)
            logger.info(f"Limited to first 50 companies from CSI300")
        else:
            # Default to get all A-share stocks
            stock_list = ak.stock_info_a_code_name()
            stock_list.columns = ['code', 'name']
            # If using default, still limit to first 50 companies
            stock_list = stock_list.head(50)
            logger.info(f"Limited to first 50 companies from A-shares")
        
        logger.info(f"Retrieved {len(stock_list)} stocks")
        return stock_list
    
    except Exception as e:
        logger.error(f"Failed to get stock list: {e}")
        raise

def download_annual_reports(stock_list, save_dir, min_year=2018, delay=2):
    """
    Download annual reports for the given stock list
    
    Args:
        stock_list (pd.DataFrame): DataFrame with stock codes and names
        save_dir (str): Directory to save downloaded reports
        min_year (int): Minimum year for reports to download
        delay (int): Delay between downloads in seconds
        
    Returns:
        pd.DataFrame: DataFrame with download results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    current_year = datetime.now().year
    results = []
    
    # Create subdirectories for each year
    for year in range(min_year, current_year + 1):
        year_dir = os.path.join(save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
    
    # Process each stock
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Downloading Reports"):
        stock_code = row['code']
        stock_name = row['name']
        
        try:
            # Get all announcements for the stock
            # AKShare's stock_annual_report_cninfo function fetches from CNINFO (巨潮资讯网)
            df_announcements = ak.stock_annual_report_cninfo(symbol=stock_code)
            
            if df_announcements is None or len(df_announcements) == 0:
                logger.warning(f"No reports found for {stock_code} - {stock_name}")
                continue
                
            # Filter annual reports only
            annual_reports = df_announcements[
                df_announcements['title'].str.contains('年度报告|年报', case=False, na=False)
            ]
            
            if len(annual_reports) == 0:
                logger.warning(f"No annual reports found for {stock_code} - {stock_name}")
                continue
            
            # Process each annual report
            for _, report in annual_reports.iterrows():
                # Extract year from the report title or publish date
                if 'pubDate' in report:
                    report_date = pd.to_datetime(report['pubDate'])
                    report_year = report_date.year
                else:
                    # Try to extract year from title
                    title = report['title']
                    # Find the first occurrence of a year pattern (e.g., 2020年)
                    import re
                    year_pattern = re.search(r'(20\d{2})年', title)
                    if year_pattern:
                        report_year = int(year_pattern.group(1))
                    else:
                        logger.warning(f"Could not extract year from report: {title}")
                        continue
                
                # Skip if report year is less than min_year
                if report_year < min_year:
                    continue
                    
                # Download the report
                try:
                    # Create filename: stock_code_year_report.pdf
                    filename = f"{stock_code}_{report_year}_annual_report.pdf"
                    save_path = os.path.join(save_dir, str(report_year), filename)
                    
                    # Check if file already exists
                    if os.path.exists(save_path):
                        logger.info(f"Report already exists: {save_path}")
                        results.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'year': report_year,
                            'file_path': save_path,
                            'status': 'existing'
                        })
                        continue
                    
                    # Get PDF URL
                    if 'announcementURL' in report:
                        pdf_url = report['announcementURL']
                    elif 'pdf_url' in report:
                        pdf_url = report['pdf_url']
                    else:
                        pdf_url = report['attachmentUrl']
                    
                    # Download the PDF
                    import requests
                    r = requests.get(pdf_url, stream=True)
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"Downloaded report: {save_path}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': report_year,
                        'file_path': save_path,
                        'status': 'downloaded'
                    })
                    
                    # Add delay between downloads
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Failed to download report for {stock_code} - {report_year}: {e}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': report_year,
                        'file_path': None,
                        'status': 'error',
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error processing stock {stock_code}: {e}")
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'year': None,
                'file_path': None,
                'status': 'error',
                'error': str(e)
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_path = os.path.join(save_dir, 'download_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Download complete. Results saved to {results_path}")
    return results_df

def main():
    """Main function to run the download process"""
    parser = argparse.ArgumentParser(description='Download annual reports for Chinese listed companies')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get stock list
    index_name = config.get('china_stock_index', 'CSI300')
    stock_list = get_stock_list(index_name)
    
    # Download annual reports
    save_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
    min_year = config.get('min_year', 2018)
    delay = config.get('download_delay', 2)
    
    download_annual_reports(
        stock_list=stock_list,
        save_dir=save_dir,
        min_year=min_year,
        delay=delay
    )

if __name__ == "__main__":
    main()
