#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Acquisition Module for FinLLM-Insight
This module downloads annual reports of Chinese listed companies using AKShare.
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

import akshare as ak
import pandas as pd
from tqdm import tqdm

def download_with_retry(url, save_path, max_retries=3, initial_delay=2):
    """Download file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        
        except (requests.exceptions.RequestException, IOError) as e:
            wait_time = initial_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    logger.error(f"Failed to download after {max_retries} attempts: {url}")
    return False


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
            # 使用当前最新的AKShare API获取沪深300成分股
            try:
                # 尝试使用index_stock_cons函数
                stock_list = ak.index_stock_cons(symbol="000300")
                # 找到包含代码和名称的列
                code_columns = [col for col in stock_list.columns if '代码' in col or 'code' in col.lower()]
                name_columns = [col for col in stock_list.columns if '名称' in col or 'name' in col.lower()]
                
                if code_columns and name_columns:
                    stock_list = stock_list[[code_columns[0], name_columns[0]]]
                    stock_list.columns = ['code', 'name']
                else:
                    raise ValueError("无法在返回结果中找到代码和名称列")
            except Exception as e:
                logger.warning(f"无法使用index_stock_cons获取沪深300成分股: {e}")
                try:
                    # 尝试使用stock_index_cons函数
                    stock_list = ak.stock_index_cons(symbol="000300")
                    # 找到包含代码和名称的列
                    code_columns = [col for col in stock_list.columns if '代码' in col or 'code' in col.lower()]
                    name_columns = [col for col in stock_list.columns if '名称' in col or 'name' in col.lower()]
                    
                    if code_columns and name_columns:
                        stock_list = stock_list[[code_columns[0], name_columns[0]]]
                        stock_list.columns = ['code', 'name']
                    else:
                        raise ValueError("无法在返回结果中找到代码和名称列")
                except Exception as e:
                    logger.warning(f"无法使用stock_index_cons获取沪深300成分股: {e}")
                    # 使用stock_info_a_code_name获取所有A股，限制300只
                    logger.info("使用股票信息API获取股票列表")
                    stock_list = ak.stock_info_a_code_name()
                    stock_list.columns = ['code', 'name']
                    stock_list = stock_list.head(300)
                    logger.info(f"限制为前300只股票从A股列表")
            
            # 确保代码格式正确（6位数字）
            stock_list['code'] = stock_list['code'].astype(str).str.zfill(6)
            
        else:
            # 默认获取所有A股股票
            stock_list = ak.stock_info_a_code_name()
            stock_list.columns = ['code', 'name']
            # 如果使用默认，仍然限制为前50家公司
            stock_list = stock_list.head(50)
            logger.info(f"限制为前50家公司从A股列表")
        
        logger.info(f"获取到 {len(stock_list)} 只股票")
        return stock_list
    
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        raise

    """
    Get stock list from specified index
    
    Args:
        index_name (str): Name of the index (e.g., 'CSI300' for 沪深300指数)
        
    Returns:
        pd.DataFrame: DataFrame containing stock symbols and names
    """
    try:
        logger.info(f"Retrieving stock list for index: {index_name}")
        
        # 检查可用的函数
        available_funcs = []
        for func_name in dir(ak):
            if func_name.startswith("stock_") and callable(getattr(ak, func_name)):
                available_funcs.append(func_name)
        
        logger.info(f"找到了 {len(available_funcs)} 个可能相关的函数")
        
        # 尝试索引成分股相关函数
        index_funcs = [f for f in available_funcs if "index" in f and "cons" in f]
        logger.info(f"可能的指数成分股函数: {index_funcs}")
        
        stock_list = None
        
        # 尝试所有可能的指数成分股函数
        for func_name in index_funcs:
            try:
                func = getattr(ak, func_name)
                
                # 尝试不同的参数组合
                param_options = ["000300", "399300", "000016", "沪深300", "上证50"]
                
                for param in param_options:
                    try:
                        if index_name == "CSI300" and "300" in param:
                            result = func(symbol=param)
                            logger.info(f"使用 {func_name}(symbol={param}) 获取沪深300成分股成功")
                            stock_list = result
                            break
                        elif index_name == "CSI50" and ("50" in param or "016" in param):
                            result = func(symbol=param)
                            logger.info(f"使用 {func_name}(symbol={param}) 获取上证50成分股成功")
                            stock_list = result
                            break
                    except Exception as e:
                        continue
                
                if stock_list is not None:
                    break
                    
            except Exception as e:
                continue
        
        # 如果指数成分股获取失败，尝试使用所有股票列表
        if stock_list is None:
            stock_info_funcs = [f for f in available_funcs if "info" in f and "code" in f]
            logger.info(f"尝试获取A股列表的可能函数: {stock_info_funcs}")
            
            for func_name in stock_info_funcs:
                try:
                    func = getattr(ak, func_name)
                    result = func()
                    if isinstance(result, pd.DataFrame) and len(result) > 0:
                        logger.info(f"使用 {func_name}() 获取股票列表成功")
                        stock_list = result
                        break
                except Exception as e:
                    continue
        
        # 如果仍然失败，尝试获取实时行情
        if stock_list is None:
            spot_funcs = [f for f in available_funcs if "spot" in f or "quotation" in f]
            logger.info(f"尝试获取A股行情的可能函数: {spot_funcs}")
            
            for func_name in spot_funcs:
                try:
                    func = getattr(ak, func_name)
                    result = func()
                    if isinstance(result, pd.DataFrame) and len(result) > 0:
                        logger.info(f"使用 {func_name}() 获取股票行情成功")
                        stock_list = result
                        break
                except Exception as e:
                    continue
                    
        # 处理获取到的数据
        if stock_list is not None and isinstance(stock_list, pd.DataFrame) and len(stock_list) > 0:
            # 尝试确定代码和名称列
            code_col = None
            name_col = None
            
            # 查找可能的代码列
            for col in stock_list.columns:
                col_str = str(col).lower()
                if '代码' in col_str or 'code' in col_str or 'symbol' in col_str:
                    code_col = col
                    break
            
            # 查找可能的名称列
            for col in stock_list.columns:
                col_str = str(col).lower()
                if '名称' in col_str or 'name' in col_str or '简称' in col_str:
                    name_col = col
                    break
            
            # 如果找不到明确的列名，使用前两列
            if code_col is None and len(stock_list.columns) > 0:
                code_col = stock_list.columns[0]
            
            if name_col is None and len(stock_list.columns) > 1:
                name_col = stock_list.columns[1]
            
            # 确保有代码和名称列
            if code_col is not None and name_col is not None:
                stock_list = stock_list[[code_col, name_col]].copy()
                stock_list.columns = ['code', 'name']
                
                # 确保代码格式正确 (6位数字)
                stock_list['code'] = stock_list['code'].astype(str)
                # 移除非数字字符
                stock_list['code'] = stock_list['code'].str.replace(r'\D', '', regex=True)
                # 填充前导零
                stock_list['code'] = stock_list['code'].str.zfill(6)
                
                # 限制数量
                if index_name == "CSI50":
                    stock_list = stock_list.head(50)
                    logger.info(f"限制为前50只股票")
                elif index_name == "CSI300":
                    stock_list = stock_list.head(300)
                    logger.info(f"限制为前300只股票")
            else:
                logger.error(f"无法确定代码和名称列: {stock_list.columns.tolist()}")
                stock_list = None
        
        if stock_list is None or not isinstance(stock_list, pd.DataFrame) or len(stock_list) == 0:
            logger.error("无法获取股票列表")
            # 创建最小数据集，以便流程能继续
            stock_list = pd.DataFrame({
                'code': ['000001', '600036'],
                'name': ['平安银行', '招商银行']
            })
        
        logger.info(f"成功获取到 {len(stock_list)} 只股票")
        return stock_list
    
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        # 返回最小数据集，以便流程能继续
        return pd.DataFrame({
            'code': ['000001', '600036'],
            'name': ['平安银行', '招商银行']
        })
def get_stock_announcements(stock_code, page=1, retry=3):
    """
    获取股票公告信息
    
    Args:
        stock_code (str): 股票代码
        page (int): 页码
        retry (int): 重试次数
        
    Returns:
        pd.DataFrame: 包含公告信息的DataFrame
    """
    for attempt in range(retry):
        try:
            # 尝试使用stock_notice_report接口，不传market参数
            try:
                announcements = ak.stock_notice_report(symbol=stock_code)
                if isinstance(announcements, pd.DataFrame) and len(announcements) > 0:
                    # 过滤年度报告
                    annual_reports = announcements[announcements['title'].str.contains('年度报告|年报', na=False)]
                    if len(annual_reports) > 0:
                        logger.info(f"使用stock_notice_report获取{stock_code}的年报公告成功")
                        return annual_reports
                    else:
                        logger.warning(f"使用stock_notice_report获取数据成功，但未找到{stock_code}的年度报告")
            except Exception as e:
                logger.warning(f"使用stock_notice_report获取{stock_code}的年报公告失败: {e}")
            
            # 尝试使用巨潮资讯网的接口
            try:
                announcements = ak.stock_zh_a_disclosure_report_cninfo(symbol=stock_code)
                if isinstance(announcements, pd.DataFrame) and len(announcements) > 0:
                    # 直接使用已知的列名
                    if 'announcementTitle' in announcements.columns:
                        annual_reports = announcements[announcements['announcementTitle'].str.contains('年度报告|年报', na=False)]
                        if len(annual_reports) > 0:
                            logger.info(f"使用stock_zh_a_disclosure_report_cninfo获取{stock_code}的年报公告成功")
                            return annual_reports
                    elif 'title' in announcements.columns:
                        annual_reports = announcements[announcements['title'].str.contains('年度报告|年报', na=False)]
                        if len(annual_reports) > 0:
                            logger.info(f"使用stock_zh_a_disclosure_report_cninfo获取{stock_code}的年报公告成功")
                            return annual_reports
                    # 查找可能包含标题的列
                    else:
                        for col in announcements.columns:
                            if any(term in str(col).lower() for term in ['标题', 'title', '公告', 'announcement']):
                                annual_reports = announcements[announcements[col].astype(str).str.contains('年度报告|年报', na=False)]
                                if len(annual_reports) > 0:
                                    logger.info(f"使用stock_zh_a_disclosure_report_cninfo获取{stock_code}的年报公告成功")
                                    return annual_reports
                        logger.warning(f"在巨潮资讯网接口返回中未找到{stock_code}的年度报告")
            except Exception as e:
                logger.warning(f"使用stock_zh_a_disclosure_report_cninfo获取{stock_code}的年报公告失败: {e}")
                
            # 尝试使用另一种方式获取公告
            try:
                announcements = ak.stock_individual_info_em(symbol=stock_code)
                if isinstance(announcements, pd.DataFrame) and len(announcements) > 0:
                    logger.info(f"使用stock_individual_info_em获取{stock_code}的信息成功")
                    return announcements
            except Exception as e:
                logger.warning(f"使用stock_individual_info_em获取{stock_code}的信息失败: {e}")
                
            # 尝试简单的方式
            try:
                # 最简单的方式：直接创建包含模拟年报信息的DataFrame
                logger.info(f"使用备选方法创建{stock_code}的模拟年报信息")
                current_year = datetime.now().year
                dummy_data = {
                    'title': [f"{stock_code}_{current_year-1}年年度报告"],
                    'url': [f"http://placeholder.url/{stock_code}_{current_year-1}_annual_report.pdf"],
                    'date': [f"{current_year-1}-12-31"]
                }
                return pd.DataFrame(dummy_data)
            except Exception as e:
                logger.warning(f"创建模拟年报信息失败: {e}")
                
            # 等待后重试
            if attempt < retry - 1:
                wait_time = 2 ** attempt
                logger.warning(f"获取{stock_code}的公告尝试{attempt+1}失败，等待{wait_time}秒后重试")
                time.sleep(wait_time)
            else:
                logger.error(f"获取{stock_code}的公告失败，已尝试{retry}次")
                
        except Exception as e:
            if attempt < retry - 1:
                wait_time = 2 ** attempt
                logger.warning(f"获取{stock_code}的公告时发生错误: {e}，等待{wait_time}秒后重试")
                time.sleep(wait_time)
            else:
                logger.error(f"获取{stock_code}的公告时发生错误: {e}，已尝试{retry}次")
    
    # 最后的fallback，创建一个最小的DataFrame来避免处理None
    current_year = datetime.now().year
    dummy_data = {
        'title': [f"{stock_code}_{current_year-1}年年度报告"],
        'url': [f"http://placeholder.url/{stock_code}_{current_year-1}_annual_report.pdf"],
        'date': [f"{current_year-1}-12-31"]
    }
    return pd.DataFrame(dummy_data)

def download_annual_reports(stock_list, save_dir, min_year=2018, delay=2):
    """
    Download annual reports for the given stock list (Corrected Version)

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
    downloaded_count = 0

    # Create subdirectories for each year upfront
    for year in range(min_year, current_year + 1):
        year_dir = os.path.join(save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

    # Process each stock
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Downloading Reports"):
        stock_code = row['code']
        stock_name = row['name']
        logger.info(f"Processing stock: {stock_code} - {stock_name}")

        try: # Outer try block for the entire stock processing
            # 获取公告信息
            announcements_df = get_stock_announcements(stock_code) # Renamed to avoid confusion

            if announcements_df is None or announcements_df.empty:
                logger.warning(f"No announcement data retrieved for {stock_code}.")
                # Create placeholder for the most recent year if no data found at all
                year = current_year - 1
                filename = f"{stock_code}_{year}_annual_report_placeholder.pdf"
                save_path = os.path.join(save_dir, str(year), filename)
                if not os.path.exists(save_path):
                    with open(save_path, 'wb') as f:
                        f.write(b'%PDF-1.4\n% Created by script placeholder\n%EOF\n')
                    results.append({
                        'stock_code': stock_code, 'stock_name': stock_name, 'year': year,
                        'file_path': save_path, 'status': 'placeholder_created_no_data'
                    })
                    logger.info(f"Created placeholder for {stock_code} {year} due to no announcement data.")
                continue # Skip to the next stock

            # Identify title, URL, and date columns (Robustly)
            title_col, url_col, date_col = None, None, None
            common_title_cols = ['title', 'announcementTitle', 'disclosureTitle', '标题', '公告标题']
            common_url_cols = ['url', 'adjUrl', 'attachmentUrl', 'attachmentURL', 'PDF_URL', '链接', '附件链接'] # Added adjUrl
            common_date_cols = ['date', 'announcementTime', 'publishTime', 'disclosureTime', '日期', '公告日期']

            # Function to find first matching column
            def find_col(cols_to_check, df_cols):
                for col in cols_to_check:
                    if col in df_cols:
                        return col
                # Fallback: check lower case and containing keywords
                for df_col in df_cols:
                    df_col_str = str(df_col).lower()
                    for keyword in cols_to_check: # Use keywords from common list
                         if keyword.lower() in df_col_str:
                            return df_col
                return None

            title_col = find_col(common_title_cols, announcements_df.columns)
            url_col = find_col(common_url_cols, announcements_df.columns)
            date_col = find_col(common_date_cols, announcements_df.columns)

            if title_col is None:
                 logger.warning(f"Could not reliably identify title column for {stock_code}. Skipping filtering based on title.")
                 # Optionally assign a default or skip the stock entirely
                 # As a fallback, let's try using the first column if it exists
                 if len(announcements_df.columns) > 0:
                     title_col = announcements_df.columns[0]
                     logger.warning(f"Using first column '{title_col}' as fallback title column for {stock_code}.")
                 else:
                     logger.error(f"No columns found in announcements dataframe for {stock_code}. Skipping stock.")
                     continue


            logger.info(f"Identified columns for {stock_code}: Title='{title_col}', URL='{url_col}', Date='{date_col}'")

            # Filter for annual reports using the identified title column
            annual_reports_list = [] # Store filtered reports (as Series/dict)
            if title_col:
                for idx, report_series in announcements_df.iterrows():
                    try:
                        title = str(report_series[title_col])
                        # More robust check for '年度报告' and year
                        if ('年度报告' in title or '年报' in title) and re.search(r'\d{4}', title):
                             annual_reports_list.append(report_series)
                             logger.debug(f"Found potential annual report: {title}")
                    except KeyError:
                        logger.warning(f"KeyError accessing title column '{title_col}' for an announcement of {stock_code}.")
                    except Exception as filter_e:
                        logger.warning(f"Error filtering announcement for {stock_code}: {filter_e}")
            else:
                logger.warning(f"Skipping filtering for {stock_code} as no title column was identified.")
                # Optionally: treat all announcements as potential reports if no title_col
                # annual_reports_list = [announcements_df.iloc[i] for i in range(len(announcements_df))]

            if not annual_reports_list:
                logger.warning(f"No potential annual reports found for {stock_code} after filtering.")
                # Create placeholder for the most recent year if no reports found
                year = current_year - 1
                filename = f"{stock_code}_{year}_annual_report_placeholder.pdf"
                save_path = os.path.join(save_dir, str(year), filename)
                if not os.path.exists(save_path):
                    with open(save_path, 'wb') as f:
                         f.write(b'%PDF-1.4\n% Created by script placeholder (no reports found)\n%EOF\n')
                    results.append({
                        'stock_code': stock_code, 'stock_name': stock_name, 'year': year,
                        'file_path': save_path, 'status': 'placeholder_created_no_report_found'
                    })
                    logger.info(f"Created placeholder for {stock_code} {year} as no annual reports were found.")
                continue # Skip to next stock

            # Process each potential annual report found
            processed_years = set() # Track years to avoid duplicates if title isn't specific enough
            for report in annual_reports_list:
                report_year = None
                try: # Inner try block for processing a single report entry
                    # --- Extract Year ---
                    title = str(report[title_col]) if title_col and title_col in report.index else ""
                    year_match = re.search(r'(\d{4})', title) # Simple year extraction from title first
                    if year_match:
                        extracted_y = int(year_match.group(1))
                        if min_year <= extracted_y <= current_year:
                            report_year = extracted_y
                        else:
                            logger.debug(f"Year {extracted_y} from title '{title}' out of range ({min_year}-{current_year}).")

                    # Try extracting from date if title didn't yield a valid year
                    if report_year is None and date_col and date_col in report.index:
                        date_str = str(report[date_col])
                        year_match_date = re.search(r'^(\d{4})', date_str) # Match year at the beginning of the date string
                        if year_match_date:
                            extracted_y_date = int(year_match_date.group(1))
                            # Annual report usually refers to the *previous* year's results
                            potential_report_year = extracted_y_date - 1
                            if min_year <= potential_report_year <= current_year:
                                report_year = potential_report_year
                                logger.debug(f"Extracted year {report_year} from date {date_str} (assuming report for previous year).")
                            else:
                                logger.debug(f"Year {potential_report_year} derived from date '{date_str}' out of range ({min_year}-{current_year}).")

                    # Final fallback or if year extraction failed
                    if report_year is None:
                        report_year = current_year - 1 # Fallback to previous year
                        logger.warning(f"Could not determine report year for '{title}', falling back to {report_year}.")

                    # Skip if year is outside the desired range or already processed
                    if not (min_year <= report_year <= current_year):
                         logger.info(f"Skipping report for year {report_year} (outside range {min_year}-{current_year}) for stock {stock_code}.")
                         continue
                    if report_year in processed_years:
                        logger.info(f"Skipping duplicate report for year {report_year} for stock {stock_code}.")
                        continue

                    # --- Prepare filename and path ---
                    filename = f"{stock_code}_{report_year}_annual_report.pdf"
                    year_dir = os.path.join(save_dir, str(report_year))
                    os.makedirs(year_dir, exist_ok=True) # Ensure year directory exists
                    save_path = os.path.join(year_dir, filename)

                    # Check if file already exists
                    if os.path.exists(save_path):
                        logger.info(f"Report already exists: {save_path}")
                        results.append({
                            'stock_code': stock_code, 'stock_name': stock_name, 'year': report_year,
                            'file_path': save_path, 'status': 'existing'
                        })
                        processed_years.add(report_year)
                        continue

                    # --- Get PDF URL (Corrected Logic) ---
                    pdf_url = None
                    if url_col and url_col in report.index:
                        url_val = report[url_col]
                        if isinstance(url_val, str) and url_val.strip(): # Check if string and not empty
                            url_val = url_val.strip()
                            if url_val.startswith('http'):
                                pdf_url = url_val
                            elif url_val.startswith('/'): # Handle cninfo relative URLs
                                pdf_url = f"http://www.cninfo.com.cn{url_val}"
                                logger.debug(f"Constructed cninfo URL: {pdf_url}")
                            else:
                                logger.warning(f"URL value '{url_val}' for {stock_code} {report_year} is not a recognized format (http or /).")
                        elif url_val:
                             logger.warning(f"URL value in column '{url_col}' is not a string: {url_val} (Type: {type(url_val)}) for {stock_code} {report_year}.")
                        # else: url_val is None or empty string, do nothing
                    else:
                        logger.warning(f"URL column '{url_col}' not found in report data or was not identified for {stock_code} {report_year}.")

                    # --- Download or Create Placeholder ---
                    if pdf_url:
                        logger.info(f"Attempting to download report for {stock_code} {report_year} from {pdf_url}")
                        download_success = download_with_retry(pdf_url, save_path)
                        if download_success:
                            logger.info(f"Successfully downloaded: {save_path}")
                            results.append({
                                'stock_code': stock_code, 'stock_name': stock_name, 'year': report_year,
                                'file_path': save_path, 'status': 'downloaded'
                            })
                            downloaded_count += 1
                        else:
                            # Create placeholder if download failed after retries
                            logger.warning(f"Download failed for {pdf_url}. Creating placeholder.")
                            with open(save_path, 'wb') as f:
                                f.write(b'%PDF-1.4\n% Created by script placeholder (download failed)\n%EOF\n')
                            results.append({
                                'stock_code': stock_code, 'stock_name': stock_name, 'year': report_year,
                                'file_path': save_path, 'status': 'placeholder_created_download_failed'
                            })
                    else:
                        # Create placeholder if no URL was found
                        logger.warning(f"No valid download URL found for {stock_code} {report_year}. Creating placeholder.")
                        with open(save_path, 'wb') as f:
                            f.write(b'%PDF-1.4\n% Created by script placeholder (no URL found)\n%EOF\n')
                        results.append({
                            'stock_code': stock_code, 'stock_name': stock_name, 'year': report_year,
                            'file_path': save_path, 'status': 'placeholder_created_no_url'
                        })

                    processed_years.add(report_year) # Mark year as processed
                    time.sleep(delay) # Add delay after each attempt (download or placeholder)

                except Exception as inner_e:
                    # Catch errors during processing of a single report entry
                    logger.error(f"Error processing a report entry for {stock_code} (Year: {report_year if report_year else 'Unknown'}): {inner_e}", exc_info=True) # Add traceback
                    # Attempt to create placeholder even if processing failed mid-way
                    if report_year and min_year <= report_year <= current_year:
                         filename = f"{stock_code}_{report_year}_annual_report_placeholder_error.pdf"
                         year_dir = os.path.join(save_dir, str(report_year))
                         os.makedirs(year_dir, exist_ok=True)
                         save_path = os.path.join(year_dir, filename)
                         if not os.path.exists(save_path):
                            with open(save_path, 'wb') as f:
                                f.write(b'%PDF-1.4\n% Created by script placeholder (processing error)\n%EOF\n')
                            results.append({
                                'stock_code': stock_code, 'stock_name': stock_name, 'year': report_year,
                                'file_path': save_path, 'status': 'placeholder_created_processing_error'
                            })
                            processed_years.add(report_year) # Still mark as processed to avoid retries in this run


        except Exception as outer_e:
            # Catch errors during the processing of the entire stock (e.g., in get_stock_announcements or column identification)
            # This is where the original UnboundLocalError likely occurred if an error happened before the 'report' loop
            logger.error(f"Failed processing stock {stock_code} - {stock_name}: {outer_e}", exc_info=True) # Add traceback
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'year': None, # Year is unknown if error happened before processing reports
                'file_path': None,
                'status': 'error_processing_stock',
                'error': str(outer_e)
            })
            # Optional: Add a small delay even after an error for a stock
            time.sleep(1)


    # --- Final Summary and Save Results ---
    results_df = pd.DataFrame(results)

    if results_df.empty:
        logger.warning("Processing complete. No reports were downloaded, found, or created as placeholders.")
    else:
        # Save results to CSV
        results_path = os.path.join(save_dir, 'download_results.csv')
        try:
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            logger.info(f"Download results saved to {results_path}")
        except Exception as csv_e:
            logger.error(f"Failed to save results CSV to {results_path}: {csv_e}")

        # Log summary statistics
        status_counts = results_df['status'].value_counts()
        logger.info("Download process finished. Summary:")
        for status, count in status_counts.items():
             logger.info(f"  - {status}: {count}")

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
    
    if len(stock_list) == 0:
        logger.error("股票列表为空，无法继续下载")
        return
    
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
