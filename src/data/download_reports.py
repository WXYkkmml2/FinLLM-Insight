#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Acquisition Module for FinLLM-Insight
This module downloads annual reports (10-K filings) of US listed companies.
"""
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import time
import json
import argparse
import logging
import requests 
from datetime import datetime
from pathlib import Path
import re
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

def download_file(url, save_path, max_retries=3, initial_delay=2):
    """
    Download file with retry mechanism
    
    Args:
        url (str): URL to download
        save_path (str): Path to save the file
        max_retries (int): Maximum number of retry attempts
        initial_delay (int): Initial delay before first retry
        
    Returns:
        bool: Success status
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    for attempt in range(max_retries):
        try:
            # Stream the download to handle large files
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                
                # Get file size if available
                file_size = int(r.headers.get('Content-Length', 0))
                desc = f"Downloading {os.path.basename(save_path)}"
                
                # Save the file
                with open(save_path, 'wb') as f:
                    chunk_size = 8192
                    
                    if file_size > 0:
                        # Use tqdm for progress bar if file size is known
                        with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
                            for chunk in r.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # Simple download without progress bar
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
            
            # Verify file was downloaded successfully
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                logger.info(f"Successfully downloaded: {save_path}")
                return True
            else:
                logger.warning(f"Downloaded file is empty: {save_path}")
                if attempt < max_retries - 1:
                    continue
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt+1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download after {max_retries} attempts: {url}")
                return False
    
    return False

def get_stock_list(index_name="S&P500", max_count=500):
    """
    Get US stock list from specified index
    
    Args:
        index_name (str): Name of the index (e.g., 'S&P500')
        max_count (int): Maximum number of stocks to include
        
    Returns:
        pd.DataFrame: DataFrame containing stock symbols and names
    """
    logger.info(f"Retrieving stock list for {index_name}")
    
    try:
        if index_name == "S&P500":
            # Get S&P 500 components from Wikipedia
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            stock_list = sp500[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        elif index_name == "S&P400":
            # Get S&P 400 components from Wikipedia
            sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
            stock_list = sp400[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        elif index_name == "S&P600":
            # Get S&P 600 components from Wikipedia
            sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')
            stock_list = sp600[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        elif index_name == "ALL":
            # Combine S&P 500, 400, and 600 for a broader list
            logger.info("Getting combined stock list from S&P 500, 400, and 600")
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
            sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')
            
            sp500_list = sp500[0][['Symbol', 'Security']]
            sp400_list = sp400[0][['Symbol', 'Security']]
            sp600_list = sp600[0][['Symbol', 'Security']]
            
            stock_list = pd.concat([sp500_list, sp400_list, sp600_list], ignore_index=True)
            stock_list.columns = ['code', 'name']
            stock_list = stock_list.drop_duplicates(subset=['code'])
            
        else:
            # Default to S&P 500 if index name is not recognized
            logger.warning(f"Unknown index: {index_name}, defaulting to S&P500")
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            stock_list = sp500[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        
        # Clean ticker symbols (remove dot notation in favor of dash)
        stock_list['code'] = stock_list['code'].str.replace('.', '-')
        
        # Limit number of stocks if specified
        if max_count > 0 and max_count < len(stock_list):
            logger.info(f"Limiting to {max_count} stocks")
            stock_list = stock_list.head(max_count)
        
        logger.info(f"Successfully retrieved {len(stock_list)} stocks")
        return stock_list
    
    except Exception as e:
        logger.error(f"Failed to retrieve stock list: {e}")
        
        # Return a minimal default list to allow code to continue
        default_stocks = pd.DataFrame({
            'code': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'name': ['Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc.', 'Meta Platforms Inc.']
        })
        logger.info(f"Using default stock list with {len(default_stocks)} stocks")
        return default_stocks

def get_annual_reports(stock_code, api_key, years=None):
    """
    Get annual report information for a specific stock using Financial Modeling Prep API
    
    Args:
        stock_code (str): Stock symbol
        api_key (str): Financial Modeling Prep API key
        years (list): List of years to get reports for, or None for recent years
        
    Returns:
        pd.DataFrame: DataFrame with annual report information
    """
    logger.info(f"Getting 10-K reports for {stock_code}")
    
    # Set default year range (last 5 years) if not specified
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year-5, current_year+1))
    
    try:
        # Construct API URL
        fmp_url = f"https://financialmodelingprep.com/api/v3/sec_filings/{stock_code}?type=10-K&page=0&apikey={api_key}"
        
        # Make API request
        response = requests.get(fmp_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Process results
        reports = []
        
        for report in data:
            filing_type = report.get('type', '')
            
            # Only process 10-K filings
            if filing_type.lower() != '10-k':
                continue
            
            date_string = report.get('fillingDate', '')
            if not date_string:
                continue
                
            # Extract year from filing date
            date = date_string[:10]
            try:
                year = int(date_string[:4])
                
                # Skip if not in requested years
                if year not in years:
                    continue
                
                link = report.get('finalLink', '')
                if not link:
                    continue
                
                reports.append({
                    'stock_code': stock_code,
                    'year': year,
                    'title': f"{stock_code} {year} Annual Report (10-K)",
                    'url': link,
                    'filing_date': date
                })
            except (ValueError, IndexError):
                logger.warning(f"Invalid date format: {date_string}")
                continue
        
        # Create DataFrame
        if reports:
            df = pd.DataFrame(reports)
            logger.info(f"Found {len(df)} 10-K reports for {stock_code}")
            return df
        else:
            logger.warning(f"No 10-K reports found for {stock_code}")
            
            # Create dummy entry for each year
            dummy_reports = []
            for year in years:
                dummy_reports.append({
                    'stock_code': stock_code,
                    'year': year,
                    'title': f"{stock_code} {year} Annual Report (10-K)",
                    'url': '',
                    'filing_date': f"{year}-12-31"
                })
            
            return pd.DataFrame(dummy_reports)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {stock_code}: {e}")
        
        # Create dummy entries for all requested years
        dummy_reports = []
        for year in years:
            dummy_reports.append({
                'stock_code': stock_code,
                'year': year,
                'title': f"{stock_code} {year} Annual Report (10-K)",
                'url': '',
                'filing_date': f"{year}-12-31"
            })
        
        return pd.DataFrame(dummy_reports)
    
    except Exception as e:
        logger.error(f"Error getting annual reports for {stock_code}: {e}")
        
        # Create dummy entries for all requested years
        dummy_reports = []
        for year in years:
            dummy_reports.append({
                'stock_code': stock_code,
                'year': year,
                'title': f"{stock_code} {year} Annual Report (10-K)",
                'url': '',
                'filing_date': f"{year}-12-31"
            })
        
        return pd.DataFrame(dummy_reports)

def download_annual_reports(stock_list, save_dir, api_key, min_year=2018, max_year=None, delay=2, max_stocks=None):
    """
    Download annual reports for the given stock list
    
    Args:
        stock_list (pd.DataFrame): DataFrame with stock codes and names
        save_dir (str): Directory to save downloaded reports
        api_key (str): Financial Modeling Prep API key
        min_year (int): Minimum year for reports to download
        max_year (int): Maximum year for reports to download (default: current year)
        delay (int): Delay between downloads in seconds
        max_stocks (int): Maximum number of stocks to process
        
    Returns:
        pd.DataFrame: DataFrame with download results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set max_year to current year if not specified
    if max_year is None:
        max_year = datetime.now().year
    
    years = list(range(min_year, max_year + 1))
    
    # Create subdirectory for each year
    for year in years:
        year_dir = os.path.join(save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
    
    # Limit number of stocks if specified
    if max_stocks and max_stocks < len(stock_list):
        logger.info(f"Limiting to {max_stocks} stocks")
        stock_list = stock_list.head(max_stocks)
    
    results = []
    
    # Process each stock
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Downloading Annual Reports"):
        stock_code = row['code']
        stock_name = row['name']
        logger.info(f"Processing stock: {stock_code} - {stock_name}")
        
        try:
            # Get annual report information
            annual_reports = get_annual_reports(stock_code, api_key, years)
            
            # Download reports for each year
            for year in years:
                # Prepare file path
                filename = f"{stock_code}_{year}_annual_report.html"
                year_dir = os.path.join(save_dir, str(year))
                save_path = os.path.join(year_dir, filename)
                
                # Check if file already exists
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                    logger.info(f"File already exists: {save_path}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'existing'
                    })
                    continue
                
                # Find report for this year
                year_report = annual_reports[annual_reports['year'] == year]
                
                if year_report.empty or not year_report.iloc[0]['url']:
                    logger.warning(f"No report URL found for {stock_code} {year}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': '',
                        'status': 'no_url'
                    })
                    continue
                
                # Get URL
                url = year_report.iloc[0]['url']
                
                # Download report
                logger.info(f"Downloading report for {stock_code} {year}: {url}")
                success = download_file(url, save_path)
                
                if success:
                    logger.info(f"Successfully downloaded report for {stock_code} {year}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'downloaded'
                    })
                else:
                    logger.error(f"Failed to download report for {stock_code} {year}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'failed'
                    })
                
                # Add delay to avoid rate limiting
                time.sleep(delay)
        
        except Exception as e:
            logger.error(f"Error processing {stock_code}: {e}")
            
            # Add error entry for this stock
            for year in years:
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'year': year,
                    'file_path': '',
                    'status': 'error'
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    if not results_df.empty:
        results_path = os.path.join(save_dir, 'download_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        logger.info(f"Download results saved to: {results_path}")
        
        # Print summary
        status_counts = results_df['status'].value_counts()
        logger.info("Download summary:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
    
    return results_df

def main():
    """Main function to run the download process"""
    parser = argparse.ArgumentParser(description='Download annual reports for US listed companies')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--max_stocks', type=int, default=0, 
                        help='Maximum number of stocks to process (0 for all)')
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config(args.config_path)
        
        # Get API key
        api_key = config.get('financial_modelling_prep_api_key', '')
        if not api_key:
            logger.error("Financial Modeling Prep API key not found in config")
            return 1
        
        # Get stock list
        logger.info("Getting stock list")
        index_name = config.get('us_stock_index', 'S&P500')
        stock_list = get_stock_list(index_name)
        
        if len(stock_list) == 0:
            logger.error("Failed to get stock list")
            return 1
        
        # Set parameters
        save_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
        min_year = config.get('min_year', 2018)
        max_year = config.get('max_year', None)
        delay = config.get('download_delay', 2)
        max_stocks = args.max_stocks if args.max_stocks > 0 else None
        
        # Download annual reports
        logger.info(f"Starting download of annual reports for years {min_year}-{max_year or 'current'}")
        download_annual_reports(
            stock_list=stock_list,
            save_dir=save_dir,
            api_key=api_key,
            min_year=min_year,
            max_year=max_year,
            delay=delay,
            max_stocks=max_stocks
        )
        
        logger.info("Annual report download completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
