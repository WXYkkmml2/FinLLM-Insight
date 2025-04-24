#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Format Conversion Module for FinLLM-Insight
This module converts HTML 10-K reports to text for further processing.
"""
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import json
import argparse
import logging
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# PDF extraction libraries
import PyPDF2
import pdfplumber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("convert_formats.log"),
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

def extract_text_from_html(html_path):
    """
    Extract text from HTML file (SEC 10-K report)
    
    Args:
        html_path (str): Path to the HTML file
        
    Returns:
        str: Extracted text
    """
    logger.info(f"Extracting text from HTML: {html_path}")
    
    try:
        with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
            html_content = f.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract text
        text = soup.get_text(separator=' ')
        
        # Clean up text
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    except Exception as e:
        logger.error(f"Failed to extract text from HTML: {e}")
        return None

def extract_text_from_pdf(pdf_path, method='pdfplumber'):
    """
    Extract text from PDF file (backup method)
    
    Args:
        pdf_path (str): Path to the PDF file
        method (str): Extraction method ('pdfplumber' or 'pypdf2')
        
    Returns:
        str: Extracted text
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    if method == 'pdfplumber':
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
            method = 'pypdf2'
    
    if method == 'pypdf2':
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                for i in range(num_pages):
                    page = reader.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text with PyPDF2: {e}")
            raise
    
    raise ValueError(f"Unsupported method: {method}")

def preprocess_10k_text(text):
    """
    Preprocess 10-K report text
    - Remove excessive whitespace
    - Clean up special characters
    - Extract meaningful sections
    
    Args:
        text (str): Original text
        
    Returns:
        str: Preprocessed text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that aren't useful
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}\"\'\$\%\&\@\!\?\/\\\|]', '', text)
    
    # Look for common 10-K sections and highlight them
    important_sections = [
        "Business",
        "Risk Factors", 
        "Management's Discussion and Analysis",
        "Financial Statements",
        "Controls and Procedures",
        "Executive Officers",
        "Management",
        "Corporate Governance",
        "Executive Compensation",
        "Security Ownership",
        "Related Party Transactions"
    ]
    
    for section in important_sections:
        # Look for section headers using regex
        pattern = r'(Item\s+\d+[A-Z]?\s*[.-]\s*' + section + r')'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        # Add section markers
        for match in matches:
            start_pos = match.start()
            if start_pos > 0:
                # Insert a section marker
                text = text[:start_pos] + "\n\n=== " + match.group(1).upper() + " ===\n\n" + text[start_pos + len(match.group(1)):]
    
    # Strip excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def process_report_files(input_dir, output_dir, years=None, overwrite=False, file_type='html'):
    """
    Process report files and save as text
    
    Args:
        input_dir (str): Directory containing report files
        output_dir (str): Directory to save processed text files
        years (list): List of years to process, or None for all
        overwrite (bool): Whether to overwrite existing files
        file_type (str): Type of files to process ('html' or 'pdf')
        
    Returns:
        pd.DataFrame: Results of the conversion process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all report files
    report_files = []
    
    if years:
        # Process specific years
        for year in years:
            year_dir = os.path.join(input_dir, str(year))
            if os.path.exists(year_dir):
                for file in os.listdir(year_dir):
                    if file.endswith(f'.{file_type}'):
                        report_files.append((os.path.join(year_dir, file), year))
    else:
        # Process all files
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(f'.{file_type}'):
                    # Extract year from directory name
                    year = os.path.basename(root)
                    if year.isdigit():
                        report_files.append((os.path.join(root, file), year))
    
    results = []
    
    # Process each report file
    for report_path, year in tqdm(report_files, desc=f"Processing {file_type} files"):
        try:
            # Create year directory in output
            year_output_dir = os.path.join(output_dir, year)
            os.makedirs(year_output_dir, exist_ok=True)
            
            # Get filename without extension
            file_base = os.path.basename(report_path)
            file_name = os.path.splitext(file_base)[0]
            
            # Output text file path
            text_file_path = os.path.join(year_output_dir, f"{file_name}.txt")
            
            # Skip if file exists and overwrite is False
            if os.path.exists(text_file_path) and not overwrite:
                logger.info(f"Skipping existing file: {text_file_path}")
                results.append({
                    'report_path': report_path,
                    'text_path': text_file_path,
                    'year': year,
                    'status': 'skipped'
                })
                continue
            
            # Extract text based on file type
            if file_type == 'html':
                raw_text = extract_text_from_html(report_path)
            elif file_type == 'pdf':
                raw_text = extract_text_from_pdf(report_path)
            else:
                logger.error(f"Unsupported file type: {file_type}")
                continue
            
            if not raw_text:
                logger.warning(f"Failed to extract text from {report_path}")
                results.append({
                    'report_path': report_path,
                    'text_path': None,
                    'year': year,
                    'status': 'extraction_failed'
                })
                continue
            
            # Preprocess text
            processed_text = preprocess_10k_text(raw_text)
            
            # Save processed text
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            logger.info(f"Processed: {report_path} -> {text_file_path}")
            results.append({
                'report_path': report_path,
                'text_path': text_file_path,
                'year': year,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing {report_path}: {e}")
            results.append({
                'report_path': report_path,
                'text_path': None,
                'year': year,
                'status': 'error',
                'error': str(e)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(output_dir, 'conversion_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Conversion complete. Results saved to {results_path}")
    return results_df

# ... (之前的导入和日志配置)

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

        # >>> 在这里插入读取代理配置并设置环境变量的代码块 <<<

        import os # 确保 os 库已经导入

        http_proxy_url = config.get('http_proxy')
        https_proxy_url = config.get('https_proxy')

        # 注意：代理认证（用户名和密码）应该直接包含在 config.json 中的代理 URL 字符串里，
        # 例如："http_proxy": "http://用户名:密码@代理地址:端口"
        # requests 库会自动解析这种格式。

        if http_proxy_url:
            os.environ['HTTP_PROXY'] = http_proxy_url
            logger.info(f"Setting HTTP_PROXY to {http_proxy_url}") # 日志中可能需要隐藏密码

        if https_proxy_url:
            os.environ['HTTPS_PROXY'] = https_proxy_url
            logger.info(f"Setting HTTPS_PROXY to {https_proxy_url}") # 日志中可能需要隐藏密码

        # >>> 插入的代码块结束 <<<

        # >>> 在设置 os.environ['HTTP_PROXY']/['HTTPS_PROXY'] 代码块之后插入以下测试代码 <<<
        
        logger.info("--- Testing Proxy Setting ---")
        try:
            # 使用一个会显示你的出口 IP 的服务，例如 httpbin.org
            test_url = 'https://httpbin.org/ip' 
            logger.info(f"Attempting to fetch external IP via: {test_url}")
        
            # 注意：这里不需要手动指定 proxies=...，因为 requests 会自动读取 os.environ
            test_response = requests.get(test_url, timeout=10) 
        
            if test_response.status_code == 200:
                external_ip_info = test_response.json()
                external_ip = external_ip_info.get('origin', 'N/A')
                logger.info(f"Request originated from IP: {external_ip}")
                logger.info("--- Proxy Test Complete ---")
        
                # 你可以通过这个 IP 地址，对比一下你的 Colab 默认 IP 
                # (可以在不设置代理的情况下运行 requests.get('https://httpbin.org/ip') 查看)
                # 如果这里的 IP 与 Colab 默认 IP 不同，说明代理很可能生效了。
        
            else:
                logger.error(f"Proxy test failed. Status code: {test_response.status_code}")
                logger.error(f"Proxy test response: {test_response.text}")
                logger.info("--- Proxy Test Complete (Failed) ---")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during proxy test request: {e}")
            logger.info("--- Proxy Test Complete (Error) ---")
        
        # >>> 插入的测试代码结束 <<<
        
        # ... (后续调用 download_annual_reports 的代码)

        
        # Get stock list
        logger.info("Getting stock list")
        # ... (获取股票列表的代码不变)
        index_name = config.get('us_stock_index', 'S&P500')
        max_stocks = args.max_stocks if args.max_stocks > 0 else config.get('max_stocks', 50)
        stock_list = get_stock_list(index_name, max_stocks)

        if len(stock_list) == 0:
            logger.error("Failed to get stock list")
            return 1

        # Set parameters
        save_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
        min_year = config.get('min_year', 2020)
        max_year = config.get('max_year', None)
        delay = config.get('download_delay', 2) # 这个 delay 依然控制处理不同报告之间的间隔

        # Download annual reports
        logger.info(f"Starting download of annual reports for years {min_year}-{max_year or 'current'}")
        download_annual_reports(
            stock_list=stock_list,
            save_dir=save_dir,
            api_key=api_key,
            min_year=min_year,
            max_year=max_year,
            delay=delay, # 传递处理报告之间的延迟
            max_stocks=max_stocks
            # 注意：这里不需要传递 proxy_config 参数了给 download_annual_reports 或 download_file
            # 因为 requests 会自动读取 os.environ 中的 HTTP_PROXY/HTTPS_PROXY
        )

        logger.info("Annual report download completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1

# ... (download_annual_reports 和 download_file 函数定义不变)
# 请确保 download_file 函数定义中没有 proxy_config 参数，让它保持简洁：
# def download_file(url, save_path, max_retries=3, initial_delay=10):
#     ... requests.get(url, headers=headers, timeout=30) # requests 会自动检查环境变量
if __name__ == "__main__":
    main()
