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

def find_correct_column_name(df, possible_names):
    """
    Find the correct column name from possible options
    
    Args:
        df (pd.DataFrame): DataFrame to search in
        possible_names (list): List of possible column names
        
    Returns:
        str or None: Found column name or None
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None

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
            # Get all announcements for the stock with pagination
            all_announcements = []
            page = 1
            max_pages = 10  # 设置最大页数限制，可以根据需要调整
            
            while page <= max_pages:
                try:
                    # AKShare的stock_annual_report_cninfo函数获取巨潮资讯网数据
                    # 注意：ak.stock_annual_report_cninfo可能不接受page参数，
                    # 如果不支持分页，可以考虑使用其他函数或修改查询策略
                    page_data = ak.stock_annual_report_cninfo(symbol=stock_code)
                    
                    # 打印调试信息
                    if page == 1:
                        logger.info(f"获取数据的列名: {page_data.columns.tolist()}")
                        logger.info(f"首页数据样例:\n{page_data.head(2)}")
                    
                    if page_data is None or len(page_data) == 0:
                        logger.info(f"未获取到{stock_code}的公告数据或到达最后一页")
                        break
                    
                    all_announcements.append(page_data)
                    logger.info(f"获取{stock_code}的第{page}页公告，共{len(page_data)}条")
                    
                    # 如果数据量少于一页，说明不需要继续获取
                    if len(page_data) < 10:
                        break
                    
                    page += 1
                    time.sleep(delay)  # 添加延迟避免请求过快
                except Exception as e:
                    logger.error(f"获取{stock_code}的第{page}页公告时出错: {e}")
                    break
            
            # 合并所有页的数据
            if not all_announcements:
                logger.warning(f"没有获取到{stock_code}的任何公告数据")
                continue
                
            df_announcements = pd.concat(all_announcements, ignore_index=True)
            logger.info(f"获取到{stock_code}的公告总数: {len(df_announcements)}")
            
            # 确定列名（支持中英文）
            title_column = find_correct_column_name(
                df_announcements, 
                ['公告标题', 'title', 'announcementTitle', '标题']
            )
            url_column = find_correct_column_name(
                df_announcements,
                ['公告链接', '公告URL', 'announcementURL', 'pdf_url', 'attachmentUrl']
            )
            date_column = find_correct_column_name(
                df_announcements,
                ['公告时间', 'pubDate', 'announcementTime', '发布时间']
            )
            
            if not title_column or not url_column:
                logger.error(f"无法找到{stock_code}的公告标题或URL列，可用列: {df_announcements.columns.tolist()}")
                continue
            
            # 筛选年度报告（使用更广泛的匹配模式）
            annual_report_patterns = [
                '年度报告', '年报', '年度财务报告', '年度业绩报告',
                '年年度报告', '年年报',  # 例如"2023年年度报告"
                '年审计报告', '年财务报告'
            ]
            filter_pattern = '|'.join(annual_report_patterns)
            
            annual_reports = df_announcements[
                df_announcements[title_column].str.contains(filter_pattern, case=False, na=False)
            ]
            
            logger.info(f"找到{stock_code}的年度报告{len(annual_reports)}份")
            if len(annual_reports) == 0:
                # 打印部分标题，便于调试
                logger.warning(f"未找到{stock_code}的年度报告，部分公告标题示例: {df_announcements[title_column].head(5).tolist()}")
                continue
            
            # 处理每份年度报告
            for _, report in annual_reports.iterrows():
                try:
                    # 从报告标题或发布日期提取年份
                    report_year = None
                    report_date = None
                    
                    # 尝试从发布日期获取年份
                    if date_column and date_column in report and pd.notna(report[date_column]):
                        try:
                            report_date = pd.to_datetime(report[date_column])
                            report_year = report_date.year
                        except:
                            logger.warning(f"无法解析日期: {report[date_column]}")
                    
                    # 如果未能从日期获取年份，尝试从标题获取
                    if not report_year:
                        # 尝试从标题提取年份，例如"2023年年度报告"
                        import re
                        title = report[title_column]
                        year_pattern = re.search(r'(20\d{2})年', title)
                        if year_pattern:
                            report_year = int(year_pattern.group(1))
                        else:
                            # 如果无法提取年份，使用当前年份-1作为默认值
                            logger.warning(f"无法从标题提取年份: {title}")
                            report_year = current_year - 1
                    
                    # 如果报告年份小于最小年份，跳过
                    if report_year < min_year:
                        logger.info(f"跳过{report_year}年的报告，早于最小年份{min_year}")
                        continue
                    
                    # 下载报告PDF
                    filename = f"{stock_code}_{report_year}_annual_report.pdf"
                    save_path = os.path.join(save_dir, str(report_year), filename)
                    
                    # 检查文件是否已存在
                    if os.path.exists(save_path):
                        logger.info(f"报告已存在: {save_path}")
                        results.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'year': report_year,
                            'file_path': save_path,
                            'status': 'existing'
                        })
                        continue
                    
                    # 获取PDF URL
                    pdf_url = report[url_column]
                    if not pdf_url:
                        logger.warning(f"未找到PDF URL: {report[title_column]}")
                        continue
                    
                    # 下载PDF
                    logger.info(f"开始下载: {stock_code}_{report_year} - {report[title_column]}")
                    download_success = download_with_retry(pdf_url, save_path)
                    
                    if not download_success:
                        results.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'year': report_year,
                            'file_path': None,
                            'status': 'download_failed',
                            'error': "下载失败，多次尝试后仍未成功"
                        })
                        continue
                    
                    logger.info(f"下载成功: {save_path}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': report_year,
                        'file_path': save_path,
                        'status': 'downloaded'
                    })
                    
                    # 添加下载间隔
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"下载{stock_code}_{report_year}年报时出错: {e}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': report_year if report_year else 'unknown',
                        'file_path': None,
                        'status': 'error',
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"处理股票{stock_code}时出错: {e}")
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'year': None,
                'file_path': None,
                'status': 'error',
                'error': str(e)
            })
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果到CSV
    results_path = os.path.join(save_dir, 'download_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    # 打印下载统计
    if not results_df.empty:
        success_count = len(results_df[results_df['status'].isin(['downloaded', 'existing'])])
        failed_count = len(results_df) - success_count
        logger.info(f"下载完成。成功: {success_count}，失败: {failed_count}。结果保存至 {results_path}")
    else:
        logger.warning("未下载任何文件")
    
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
