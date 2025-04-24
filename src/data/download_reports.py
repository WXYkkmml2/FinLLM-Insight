#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Acquisition Module for FinLLM-Insight
This module downloads annual reports (10-K filings) of US listed companies from SEC.
"""

import os
import sys
# Adjusting project root to be robust for Colab/Kaggle environments
# Assumes the script is in src/data relative to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    level=logging.INFO, # 可以暂时改为 logging.DEBUG 查看更详细的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_reports.log", mode='w'), # 使用 mode='w' 清空旧日志
        logging.StreamHandler(sys.stdout) # 确保日志输出到控制台
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
        # Adjust config path to be relative to the script location if needed
        # In Colab/Kaggle, the script might be run from different directories
        # Assume config is in a 'config' folder at the project root
        resolved_config_path = os.path.join(project_root, config_path)
        logger.info(f"Attempting to load config from: {resolved_config_path}")

        with open(resolved_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("Config loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {resolved_config_path}. Please ensure config/config.json exists.")
        raise # Re-raise the exception
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config JSON: {e}. Please check the config file syntax.")
        raise # Re-raise the exception
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def download_file(url, save_path, max_retries=3, initial_delay=10):
    """
    Download file with retry mechanism. Uses environment variables for proxy.

    Args:
        url (str): URL to download
        save_path (str): Path to save the file
        max_retries (int): Maximum number of retry attempts
        initial_delay (int): Initial delay before first retry (will be multiplied on subsequent retries)

    Returns:
        bool: Success status
    """
    # requests 库会自动检查并使用 os.environ['HTTP_PROXY'] 和 os.environ['HTTPS_PROXY']
    # 这些变量应该在 main 函数中根据 config 文件设置

    headers = {'User-Agent': 'Mozilla/5.0'} # Standard User-Agent

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{max_retries}: Downloading {url}")

            # requests 会自动使用环境变量中的代理
            response = requests.get(url, headers=headers, timeout=60) # Increased timeout just in case

            if response.status_code == 200:
                # Get the content of the file
                page_content = response.content

                # Create directory if it doesn't exist (should be created by download_annual_reports, but safety check)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Write the content to the local file
                with open(save_path, "wb") as file:
                    file.write(page_content)

                # Verify file exists and is not empty (or too small)
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024: # > 1KB check
                    logger.info(f"Successfully downloaded: {save_path} ({os.path.getsize(save_path)/1024:.2f} KB)")
                    return True
                else:
                    # File exists but is empty or too small, likely an error page content
                    logger.warning(f"Downloaded file is empty or too small: {save_path} ({os.path.getsize(save_path)} bytes)")
                    # Clean up the small file before retrying
                    if os.path.exists(save_path):
                         os.remove(save_path)
                    # Continue to retry or fail

            else:
                logger.error(f'Response not 200. Status: {response.status_code}')
                # Log response content for non-200 status to help debug
                # logger.debug(f"Response content (first 500 chars): {response.text[:500]}")


            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt) # Exponential backoff
                logger.info(f"Waiting {wait_time:.2f}s before retrying...")
                time.sleep(wait_time)

        except requests.exceptions.Timeout:
             logger.error(f"Download timed out after 60s.")
             if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.2f}s before retrying...")
                time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            logger.error(f"Download request error: {e}")
            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.2f}s before retrying...")
                time.sleep(wait_time)

        except Exception as e:
            logger.error(f"An unexpected error occurred during download: {e}")
            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.2f}s before retrying...")
                time.sleep(wait_time)


    logger.error(f"Failed to download {url} after {max_retries} attempts.")
    return False

def get_stock_list(index_name="S&P500", max_stocks=50):
    """
    Get US stock list from specified index. Uses environment variables for proxy.

    Args:
        index_name (str): Name of the index (e.g., 'S&P500', 'S&P400', 'S&P600', 'ALL')
        max_stocks (int): Maximum number of stocks to include (0 for all)

    Returns:
        pd.DataFrame: DataFrame containing stock symbols and names
    """
    logger.info(f"Retrieving stock list for {index_name}")

    try:
        # requests will automatically use the proxy environment variables here too
        if index_name == "S&P500":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        elif index_name == "S&P400":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        elif index_name == "S%26P600" or index_name == "S&P600": # Handle both %26 and &
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        elif index_name == "ALL":
             logger.info("Getting combined stock list from S&P 500, 400, and 600")
             # Need to fetch multiple pages and combine
             sp500_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0][['Symbol', 'Security']]
             sp400_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0][['Symbol', 'Security']]
             sp600_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0][['Symbol', 'Security']]

             stock_list = pd.concat([sp500_df, sp400_df, sp600_df], ignore_index=True)
             stock_list.columns = ['ticker', 'company_name']
             stock_list = stock_list.drop_duplicates(subset=['ticker'])

        else:
            logger.warning(f"Unknown index: {index_name}, defaulting to S&P500")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

        if index_name != "ALL":
             stock_list = pd.read_html(url)[0][['Symbol', 'Security']]
             stock_list.columns = ['ticker', 'company_name']


        # Clean ticker symbols (remove dot notation in favor of dash)
        # Use .loc to avoid SettingWithCopyWarning
        stock_list.loc[:, 'ticker'] = stock_list['ticker'].str.replace('.', '-', regex=False)


        # Limit number of stocks if specified
        if max_stocks > 0 and max_stocks < len(stock_list):
            logger.info(f"Limiting to {max_stocks} stocks")
            stock_list = stock_list.head(max_stocks)

        logger.info(f"Successfully retrieved {len(stock_list)} stocks")
        return stock_list

    except Exception as e:
        logger.error(f"Failed to retrieve stock list: {e}")

        # Return a minimal default list to allow code to continue
        default_stocks = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'company_name': ['Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc.', 'Meta Platforms Inc.']
        })
        logger.warning(f"Using default stock list with {len(default_stocks)} stocks due to error.")
        return default_stocks


def get_annual_reports(ticker, api_key, min_year=2018, max_year=None):
    """
    Get annual report information for a specific stock using Financial Modeling Prep API.
    Uses environment variables for proxy.

    Args:
        ticker (str): Stock symbol
        api_key (str): Financial Modeling Prep API key
        min_year (int): Minimum year to fetch reports for
        max_year (int): Maximum year to fetch reports for (None for current year)

    Returns:
        pd.DataFrame: DataFrame with annual report information
    """
    logger.info(f"Getting 10-K reports for {ticker}")

    # Set default max year to current year if not specified
    if max_year is None:
        max_year = datetime.now().year

    try:
        # Construct API URL
        fmp_url = f'https://financialmodelingprep.com/api/v3/sec_filings/{ticker}?type=10-K&page=0&apikey={api_key}'

        # Hide API key in logs
        logger.info(f"API URL: {fmp_url.replace(api_key, 'API_KEY_HIDDEN')}")

        # Make the request using requests (will use proxy env vars if set)
        response = requests.get(fmp_url, timeout=30) # Use timeout

        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}")
            # Log error response content if it might contain useful info
            # logger.debug(f"FMP API Error Response: {response.text[:500]}")
            return pd.DataFrame()

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
             logger.error(f"Failed to parse JSON response from FMP API.")
             # logger.debug(f"Raw FMP API response: {response.text[:500]}")
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to process FMP API response: {e}")
            return pd.DataFrame()


        if not data:
            logger.warning(f"No data returned for {ticker} from FMP API.")
            return pd.DataFrame()

        logger.info(f"FMP API returned {len(data)} filings for {ticker}.")

        # Process and filter results
        reports = []

        for report in data:
            filing_type = report.get('type', '')

            # Only process 10-K filings
            if not filing_type.lower() in ['10-k', '10k']:
                # logger.debug(f"Skipping filing type: {filing_type}") # Optional: log skipped types
                continue

            date_string = report.get('fillingDate', '')
            if not date_string or len(date_string) < 4:
                logger.warning(f"Skipping 10-K report with missing/invalid fillingDate for {ticker}: {date_string}")
                continue

            # Extract year from filing date
            try:
                 year = int(date_string[:4])
            except ValueError:
                 logger.warning(f"Skipping 10-K report with invalid year format for {ticker}: {date_string}")
                 continue


            # Skip if too old or too new
            if year < min_year or (max_year is not None and year > max_year):
                # logger.debug(f"Skipping report for {ticker} from year {year} (outside {min_year}-{max_year or 'current'})") # Optional: log skipped years
                continue

            link = report.get('finalLink', '')
            if not link:
                logger.warning(f"No finalLink found for 10-K report for {ticker} from year {year}.")
                continue

            # Check if the link is to the expected SEC domain
            if 'sec.gov/Archives/edgar/data/' not in link:
                 logger.warning(f"Final link does not appear to be a standard SEC EDGAR link for {ticker} year {year}: {link}")
                 # Decide whether to skip or try downloading anyway. Let's try anyway for now.
                 # continue # Uncomment this line to skip non-standard SEC links

            # If we reach here, the report matches criteria
            # logger.debug(f"Found eligible 10-K report for {ticker} year {year} with link: {link}") # Optional: debug found reports

            reports.append({
                'ticker': ticker,
                'year': year,
                'title': f"{ticker} {year} Annual Report (10-K)",
                'url': link,
                'filing_date': date_string[:10] # Store full date part
            })


        # Create DataFrame from collected reports
        if reports:
            df = pd.DataFrame(reports)
            # Optional: Sort by year
            df = df.sort_values(by='year').reset_index(drop=True)
            logger.info(f"Filtered down to {len(df)} eligible 10-K reports for {ticker} in years {min_year}-{max_year or 'current'}.")
            return df
        else:
            logger.warning(f"No eligible 10-K reports found for {ticker} in years {min_year}-{max_year or 'current'}.")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
         logger.error(f"Network error during FMP API request for {ticker}: {e}")
         return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred getting annual reports for {ticker}: {e}", exc_info=True)
        return pd.DataFrame()


def download_annual_reports(stock_list, save_dir, api_key, min_year=2018, max_year=None, delay=60, max_stocks=None):
    """
    Download annual reports for the given stock list.

    Args:
        stock_list (pd.DataFrame): DataFrame with stock codes and names
        save_dir (str): Directory to save downloaded reports
        api_key (str): Financial Modeling Prep API key
        min_year (int): Minimum year for reports to download
        max_year (int): Maximum year for reports to download (default: current year)
        delay (int): Delay in seconds to wait *between processing each report*.
                     This is the main delay to avoid hitting SEC too fast.
        max_stocks (int): Maximum number of stocks to process

    Returns:
        pd.DataFrame: DataFrame with download results
    """
    # Create save directory if it doesn't exist
    # Adjust save_dir to be relative to project_root
    resolved_save_dir = os.path.join(project_root, save_dir)
    os.makedirs(resolved_save_dir, exist_ok=True)
    logger.info(f"Saving reports to: {resolved_save_dir}")


    # Set max_year to current year if not specified
    if max_year is None:
        max_year = datetime.now().year

    # Create subdirectory for each year within the save_dir
    years_range = list(range(min_year, max_year + 1))
    for year in years_range:
        year_dir = os.path.join(resolved_save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)


    # Limit number of stocks if specified (handled in get_stock_list, but double-check here)
    # if max_stocks is handled in get_stock_list, this might not be needed here
    # if max_stocks and max_stocks < len(stock_list):
    #     logger.info(f"Limiting processing to {max_stocks} stocks")
    #     stock_list = stock_list.head(max_stocks)


    results = []
    download_count = 0
    failed_count = 0
    skipped_existing = 0

    # Process each stock
    # tqdm provides the progress bar
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Downloading Annual Reports"):
        ticker = row['ticker']
        company_name = row['company_name']
        logger.info(f"--- Processing Stock: {ticker} - {company_name} ---")

        try:
            # Get annual report information for this stock
            annual_reports = get_annual_reports(ticker, api_key, min_year, max_year)

            if annual_reports.empty:
                logger.warning(f"No eligible 10-K reports found for {ticker} in the specified years.")
                # Add 'no_reports_found' status for all years in range for this stock
                for year in years_range:
                     results.append({
                         'ticker': ticker, 'company_name': company_name, 'year': year,
                         'file_path': '', 'status': 'no_reports_found'
                     })
                # Still apply delay before next stock, helps pace API/SEC requests
                logger.info(f"Waiting {delay}s before processing next stock...")
                time.sleep(delay)
                continue # Move to the next stock


            # Download reports for each year found for this stock
            # Add delay BEFORE the first download attempt for this stock/report
            logger.info(f"Waiting {delay}s before first download attempt for {ticker}...")
            time.sleep(delay)


            reports_for_this_stock = annual_reports.to_dict('records') # Convert DataFrame to list of dicts

            for i, report in enumerate(reports_for_this_stock):
                year = report['year']
                url = report['url']
                filing_date = report['filing_date']

                # Prepare file path
                # Use resolved_save_dir
                filename = f"{ticker}_{year}_annual_report.html"
                year_dir = os.path.join(resolved_save_dir, str(year))
                save_path = os.path.join(year_dir, filename)

                # Check if file already exists and is not too small
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024: # > 1KB check
                    logger.info(f"File already exists and is likely complete: {save_path}")
                    results.append({
                        'ticker': ticker, 'company_name': company_name, 'year': year,
                        'file_path': save_path, 'status': 'existing', 'filing_date': filing_date
                    })
                    skipped_existing += 1
                    continue # Skip download, move to next report


                # Download report - download_file handles retries and uses proxy env vars
                # Pass a higher initial_delay to download_file for retries on the same URL
                # The main pacing is controlled by time.sleep(delay) AFTER each report attempt
                success = download_file(url, save_path, initial_delay=30) # Use a decent delay for retries

                if success:
                    logger.info(f"Successfully processed report for {ticker} {year}")
                    results.append({
                        'ticker': ticker, 'company_name': company_name, 'year': year,
                        'file_path': save_path, 'status': 'downloaded', 'filing_date': filing_date
                    })
                    download_count += 1
                else:
                    logger.error(f"Failed to process report for {ticker} {year}")
                    results.append({
                        'ticker': ticker, 'company_name': company_name, 'year': year,
                        'file_path': '', 'status': 'failed', 'filing_date': filing_date # No file path on failure
                    })
                    failed_count += 1

                # Add delay BETWEEN processing each report (including retries)
                # This is crucial for pacing requests to SEC/FMP
                # No delay needed AFTER the very last report of the very last stock
                if not (i == len(reports_for_this_stock) - 1 and _ == len(stock_list) - 1):
                     logger.info(f"Waiting {delay}s before processing next report/stock...")
                     time.sleep(delay)


            # After processing all found reports for this stock, check for years in range that were missed
            found_report_years_this_stock = set(annual_reports['year'])
            for year in years_range:
                 # Check if this year was in the desired range but no report was found/added
                 # Avoid adding 'no_report_for_year' if the status is already 'no_reports_found', 'downloaded', 'existing', or 'failed' for this year
                 if year not in found_report_years_this_stock:
                     if not any(r['ticker'] == ticker and r['year'] == year for r in results):
                         results.append({
                             'ticker': ticker, 'company_name': company_name, 'year': year,
                             'file_path': '', 'status': 'no_report_for_year', 'filing_date': None # No filing date if no report
                         })


        except Exception as e:
            logger.error(f"An unexpected error occurred while processing stock {ticker}: {e}", exc_info=True)

            # Add 'error' status for all years in range for this stock
            for year in years_range:
                results.append({
                    'ticker': ticker, 'company_name': company_name, 'year': year,
                    'file_path': '', 'status': 'error', 'filing_date': None
                })

            # Still apply delay before next stock even if an error occurred
            logger.info(f"Waiting {delay}s before processing next stock...")
            time.sleep(delay)


    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Ensure all desired years are present in results_df even if no reports were found for a stock
    # This makes the results CSV comprehensive for the requested range
    all_possible_combinations = pd.MultiIndex.from_product([stock_list['ticker'], years_range], names=['ticker', 'year']).to_frame(index=False)
    results_df = pd.merge(all_possible_combinations, results_df, on=['ticker', 'year'], how='left')
    # Fill missing info for combinations where no status was set (shouldn't happen if error handling/no_report_for_year logic is perfect, but good for robustness)
    results_df['company_name'] = results_df.apply(lambda row: stock_list[stock_list['ticker'] == row['ticker']]['company_name'].iloc[0] if pd.isna(row['company_name']) else row['company_name'], axis=1)
    results_df['status'] = results_df['status'].fillna('not_attempted') # Or 'no_report_for_year' if logic above misses something
    # Reorder columns for clarity
    results_df = results_df[['ticker', 'company_name', 'year', 'filing_date', 'file_path', 'status']]


    # Save results to CSV relative to project_root
    results_path = os.path.join(project_root, save_dir, 'download_results.csv')
    try:
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        logger.info(f"Download results saved to: {results_path}")
    except Exception as e:
         logger.error(f"Failed to save results CSV to {results_path}: {e}")


    # Print summary
    if not results_df.empty:
        logger.info("\n--- Download Summary ---")
        status_counts = results_df['status'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        logger.info("------------------------")

    logger.info("Annual report download process finished.")

    return results_df # Return the results DataFrame


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure config directory exists if not using default path and running from project root
    default_config_dir = os.path.join(project_root, 'config')
    os.makedirs(default_config_dir, exist_ok=True)
    default_config_path = os.path.join(default_config_dir, 'config.json')

    # Create a placeholder config if it doesn't exist, to guide the user
    if not os.path.exists(default_config_path):
        logger.warning(f"Default config file not found at {default_config_path}. Creating a placeholder.")
        placeholder_config = {
          "financial_modelling_prep_api_key": "YOUR_API_KEY_HERE",
          "us_stock_index": "S&P500",
          "max_stocks": 5,
          "annual_reports_html_save_directory": "data/raw/annual_reports", # Relative to project root
          "min_year": 2020,
          "max_year": 2025,
          "download_delay": 60, # Delay in seconds BETWEEN processing each report
          "http_proxy": "", # Add your proxy here, e.g., "http://user:pass@host:port"
          "https_proxy": "" # Add your proxy here, usually the same as http_proxy
        }
        try:
            with open(default_config_path, 'w', encoding='utf-8') as f:
                json.dump(placeholder_config, f, indent=4)
            logger.info(f"Placeholder config created at {default_config_path}. Please edit it with your API key and proxy.")
        except Exception as e:
             logger.error(f"Failed to create placeholder config file: {e}")


    # Run the main function
    sys.exit(main())
