#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块：下载中国或美国上市公司年度报告
支持多市场数据获取，中国使用AKShare，美国使用Financial Modeling Prep API
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

# 尝试导入AKShare (用于中国市场)
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: AKShare not available. China market data will be limited.")

# 配置日志
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
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise

def download_with_retry(url, save_path, headers=None, max_retries=3, initial_delay=2):
    """下载文件，支持重试机制"""
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            r.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        
        except (requests.exceptions.RequestException, IOError) as e:
            wait_time = initial_delay * (2 ** attempt)
            logger.warning(f"下载尝试 {attempt+1}/{max_retries} 失败: {e}. {wait_time}秒后重试...")
            time.sleep(wait_time)
    
    logger.error(f"{max_retries}次尝试后下载失败: {url}")
    return False

#########################################
# 中国市场相关函数
#########################################

def get_china_stock_list(index_name="CSI300", max_count=300):
    """获取指定指数的中国股票列表"""
    if not AKSHARE_AVAILABLE:
        logger.error("AKShare未安装，无法获取中国股票列表")
        return pd.DataFrame({
            'code': ['000001', '600036'],
            'name': ['平安银行', '招商银行']
        })
    
    logger.info(f"获取指数 {index_name} 的股票列表")
    
    try:
        # 沪深300成分股
        if index_name == "CSI300":
            try:
                stock_list = ak.index_stock_cons_csindex(symbol="000300")
                stock_list = stock_list[['成分券代码', '成分券名称']].copy()
                stock_list.columns = ['code', 'name']
            except Exception as e:
                logger.warning(f"使用index_stock_cons_csindex获取沪深300成分股失败: {e}")
                try:
                    stock_list = ak.index_stock_cons(symbol="000300")
                    # 找到代码和名称列
                    code_col = [col for col in stock_list.columns if '代码' in col or 'code' in col.lower()][0]
                    name_col = [col for col in stock_list.columns if '名称' in col or 'name' in col.lower()][0]
                    stock_list = stock_list[[code_col, name_col]].copy()
                    stock_list.columns = ['code', 'name']
                except Exception as e2:
                    logger.warning(f"使用index_stock_cons获取沪深300成分股失败: {e2}")
                    # 获取所有A股
                    stock_list = ak.stock_info_a_code_name()
                    stock_list.columns = ['code', 'name']
                    stock_list = stock_list.head(max_count)
                    logger.info(f"使用所有A股列表，限制为前{max_count}只")
        elif index_name == "CSI500":
            # 中证500成分股
            try:
                stock_list = ak.index_stock_cons_csindex(symbol="000905")
                stock_list = stock_list[['成分券代码', '成分券名称']].copy()
                stock_list.columns = ['code', 'name']
            except Exception as e:
                logger.warning(f"获取中证500成分股失败: {e}")
                # 获取所有A股
                stock_list = ak.stock_info_a_code_name()
                stock_list.columns = ['code', 'name']
                stock_list = stock_list.head(max_count)
                logger.info(f"使用所有A股列表，限制为前{max_count}只")
        else:
            # 获取所有A股
            stock_list = ak.stock_info_a_code_name()
            stock_list.columns = ['code', 'name']
            if max_count > 0:
                stock_list = stock_list.head(max_count)
            
        # 确保代码格式正确
        stock_list['code'] = stock_list['code'].astype(str).str.zfill(6)
        
        logger.info(f"成功获取 {len(stock_list)} 只股票")
        return stock_list
        
    except Exception as e:
        logger.error(f"获取中国股票列表失败: {e}")
        # 返回简化的列表以便程序继续运行
        return pd.DataFrame({
            'code': ['000001', '600036'],
            'name': ['平安银行', '招商银行']
        })

def get_china_annual_reports(stock_code, years=None):
    """获取指定中国股票的年报信息"""
    if not AKSHARE_AVAILABLE:
        logger.error("AKShare未安装，无法获取中国年报信息")
        return pd.DataFrame()
    
    logger.info(f"获取 {stock_code} 的年报信息")
    
    # 设置默认年份范围(最近5年)
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year-5, current_year+1))
    
    all_reports = []
    
    try:
        # 使用AKShare获取公告信息
        for year in years:
            try:
                # 设置日期范围
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                # 获取公告信息
                announcement_df = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=stock_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if isinstance(announcement_df, pd.DataFrame) and not announcement_df.empty:
                    # 查找标题列
                    title_col = None
                    for col in announcement_df.columns:
                        if '标题' in col or 'title' in str(col).lower():
                            title_col = col
                            break
                    
                    if title_col:
                        # 筛选年报公告
                        year_reports = announcement_df[
                            announcement_df[title_col].str.contains(
                                f"{year}.*年度报告|{year}.*年报", 
                                regex=True, 
                                na=False
                            )
                        ]
                        
                        if not year_reports.empty:
                            # 提取URL列
                            url_col = None
                            for col in announcement_df.columns:
                                if '链接' in col or 'url' in str(col).lower() or 'link' in str(col).lower():
                                    url_col = col
                                    break
                            
                            if url_col:
                                for _, row in year_reports.iterrows():
                                    # 提取详情页链接
                                    detail_url = row[url_col]
                                    if not detail_url.lower().endswith('.pdf'):
                                        # 尝试构造PDF链接
                                        if 'announcementId' in row:
                                            pdf_url = f"http://static.cninfo.com.cn/finalpage/{row['announcementId']}.PDF"
                                        elif 'link_id' in row:
                                            pdf_url = f"http://static.cninfo.com.cn/finalpage/{row['link_id']}.PDF"
                                        else:
                                            pdf_url = detail_url
                                    else:
                                        pdf_url = detail_url
                                    
                                    all_reports.append({
                                        'stock_code': stock_code,
                                        'year': year,
                                        'title': row[title_col],
                                        'url': pdf_url,
                                        'detail_url': detail_url
                                    })
            except Exception as e:
                logger.warning(f"获取 {stock_code} {year}年公告时出错: {e}")
                
                # 使用备用查询URL
                search_url = f"http://www.cninfo.com.cn/new/fulltextSearch/full?searchkey={stock_code}+{year}年度报告&sdate=&edate=&isfulltext=false&sortName=pubdate&sortType=desc"
                all_reports.append({
                    'stock_code': stock_code,
                    'year': year,
                    'title': f"{stock_code}_{year}年年度报告(备用链接)",
                    'url': search_url,
                    'detail_url': search_url
                })
                logger.info(f"使用备用链接 {stock_code} {year}年")
    
    except Exception as e:
        logger.warning(f"获取年报信息时出错: {e}")
    
    # 如果没有找到报告，创建占位信息
    if not all_reports:
        for year in years:
            search_url = f"http://www.cninfo.com.cn/new/fulltextSearch/full?searchkey={stock_code}+{year}年度报告&sdate=&edate=&isfulltext=false&sortName=pubdate&sortType=desc"
            all_reports.append({
                'stock_code': stock_code,
                'year': year,
                'title': f"{stock_code}_{year}年年度报告(占位)",
                'url': search_url,
                'detail_url': search_url
            })
    
    return pd.DataFrame(all_reports)

def download_china_annual_reports(stock_list, save_dir, min_year=2018, max_year=None, delay=1, max_stocks=None):
    """下载中国公司年报"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置年份范围
    if max_year is None:
        max_year = datetime.now().year
    
    years = list(range(min_year, max_year + 1))
    
    # 为每年创建子目录
    for year in years:
        year_dir = os.path.join(save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
    
    # 限制股票数量
    if max_stocks and max_stocks < len(stock_list):
        logger.info(f"限制处理股票数量为: {max_stocks}")
        stock_list = stock_list.head(max_stocks)
    
    results = []
    
    # 处理每只股票
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载中国公司年报"):
        stock_code = row['code']
        stock_name = row['name']
        logger.info(f"处理股票: {stock_code} - {stock_name}")
        
        try:
            # 获取年报信息
            annual_reports = get_china_annual_reports(stock_code, years)
            
            # 下载每年的年报
            for year in years:
                # 准备文件路径
                filename = f"{stock_code}_{year}_annual_report.pdf"
                year_dir = os.path.join(save_dir, str(year))
                save_path = os.path.join(year_dir, filename)
                
                # 检查文件是否已经存在
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                    logger.info(f"文件已存在: {save_path}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'existing'
                    })
                    continue
                
                # 查找对应年份的年报
                year_report = annual_reports[annual_reports['year'] == year]
                
                if year_report.empty:
                    logger.warning(f"未找到 {stock_code} {year}年的年报信息")
                    continue
                
                # 获取URL
                url = year_report.iloc[0]['url']
                
                # 下载PDF
                logger.info(f"开始下载 {stock_code} {year}年年报: {url}")
                download_success = download_with_retry(url, save_path)
                
                if download_success:
                    logger.info(f"成功下载 {stock_code} {year}年年报")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'downloaded'
                    })
                else:
                    # 尝试使用detail_url
                    detail_url = year_report.iloc[0]['detail_url']
                    if detail_url != url:
                        logger.info(f"尝试使用备用链接下载: {detail_url}")
                        download_success = download_with_retry(detail_url, save_path)
                        
                        if download_success:
                            logger.info(f"成功通过备用链接下载 {stock_code} {year}年年报")
                            results.append({
                                'stock_code': stock_code,
                                'stock_name': stock_name,
                                'year': year,
                                'file_path': save_path,
                                'status': 'downloaded'
                            })
                        else:
                            logger.error(f"下载 {stock_code} {year}年年报失败")
                            results.append({
                                'stock_code': stock_code,
                                'stock_name': stock_name,
                                'year': year,
                                'file_path': save_path,
                                'status': 'failed'
                            })
                    else:
                        logger.error(f"下载 {stock_code} {year}年年报失败")
                        results.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'year': year,
                            'file_path': save_path,
                            'status': 'failed'
                        })
                
                # 添加延迟避免请求过快
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 时出错: {e}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_path = os.path.join(save_dir, 'download_results_china.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        logger.info(f"下载结果已保存到: {results_path}")
    
    return results_df

#########################################
# 美国市场相关函数
#########################################

def get_us_stock_list(index_name="S&P500", max_count=500):
    """获取美国股票列表"""
    logger.info(f"获取美国 {index_name} 的股票列表")
    
    try:
        if index_name == "S&P500":
            # 从维基百科获取标普500成分股
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            stock_list = sp500[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        elif index_name == "S&P400":
            # 获取标普中盘400成分股
            sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
            stock_list = sp400[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        elif index_name == "S&P600":
            # 获取标普小盘600成分股
            sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')
            stock_list = sp600[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        else:
            # 如果非标准指数，获取标普500作为默认
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            stock_list = sp500[0][['Symbol', 'Security']]
            stock_list.columns = ['code', 'name']
        
        # 限制股票数量
        if max_count and max_count < len(stock_list):
            stock_list = stock_list.head(max_count)
        
        # 清理代码中的特殊字符
        stock_list['code'] = stock_list['code'].str.replace('.', '-')
        
        logger.info(f"成功获取 {len(stock_list)} 只美国股票")
        return stock_list
    
    except Exception as e:
        logger.error(f"获取美国股票列表失败: {e}")
        # 返回简化的列表以便程序继续运行
        return pd.DataFrame({
            'code': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'name': ['Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc.', 'Meta Platforms Inc.']
        })

def get_us_annual_reports(stock_code, api_key, years=None):
    """
    使用Financial Modeling Prep API获取美国公司10-K报告
    
    Args:
        stock_code: 股票代码
        api_key: Financial Modeling Prep API密钥
        years: 年份列表，None则获取最近5年
        
    Returns:
        DataFrame: 包含10-K报告信息
    """
    logger.info(f"获取 {stock_code} 的10-K报告信息")
    
    # 设置默认年份范围(最近5年)
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year-5, current_year+1))
    
    # 构建API请求URL
    url = f"https://financialmodelingprep.com/api/v3/sec_filings/{stock_code}?type=10-K&page=0&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        all_reports = []
        
        # 筛选10-K报告
        for item in data:
            filing_type = item.get('type', '')
            if filing_type.lower() not in ['10-k', '10k']:
                continue
                
            date_string = item.get('fillingDate', '')
            if not date_string:
                continue
                
            date = date_string[:10]
            year = int(date_string[:4])
            
            # 检查年份是否在指定范围内
            if year not in years:
                continue
                
            # 获取报告链接
            report_url = item.get('finalLink', '')
            if not report_url:
                continue
                
            all_reports.append({
                'stock_code': stock_code,
                'year': year,
                'title': f"{stock_code} {year} Annual Report (10-K)",
                'url': report_url,
                'detail_url': report_url,
                'filing_date': date
            })
        
        # 如果没有找到报告，创建占位信息
        if not all_reports:
            for year in years:
                all_reports.append({
                    'stock_code': stock_code,
                    'year': year,
                    'title': f"{stock_code} {year} Annual Report (Not Found)",
                    'url': '',
                    'detail_url': '',
                    'filing_date': f"{year}-12-31"
                })
        
        return pd.DataFrame(all_reports)
    
    except Exception as e:
        logger.error(f"获取 {stock_code} 的10-K报告失败: {e}")
        
        # 返回占位数据
        all_reports = []
        for year in years:
            all_reports.append({
                'stock_code': stock_code,
                'year': year,
                'title': f"{stock_code} {year} Annual Report (Error)",
                'url': '',
                'detail_url': '',
                'filing_date': f"{year}-12-31"
            })
        
        return pd.DataFrame(all_reports)

def download_us_annual_reports(stock_list, save_dir, api_key, min_year=2018, max_year=None, delay=1, max_stocks=None):
    """
    下载美国公司10-K报告
    
    Args:
        stock_list: 股票列表DataFrame
        save_dir: 保存目录
        api_key: Financial Modeling Prep API密钥
        min_year: 最早年份
        max_year: 最晚年份(默认当前年份)
        delay: 请求延迟时间(秒)
        max_stocks: 最多处理的股票数量
        
    Returns:
        DataFrame: 下载结果
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置年份范围
    if max_year is None:
        max_year = datetime.now().year
    
    years = list(range(min_year, max_year + 1))
    
    # 为每年创建子目录
    for year in years:
        year_dir = os.path.join(save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
    
    # 限制股票数量
    if max_stocks and max_stocks < len(stock_list):
        logger.info(f"限制处理股票数量为: {max_stocks}")
        stock_list = stock_list.head(max_stocks)
    
    results = []
    
    # 处理每只股票
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载美国公司年报"):
        stock_code = row['code']
        stock_name = row['name']
        logger.info(f"处理股票: {stock_code} - {stock_name}")
        
        try:
            # 获取10-K报告信息
            annual_reports = get_us_annual_reports(stock_code, api_key, years)
            
            # 下载每年的报告
            for year in years:
                # 准备文件路径
                filename = f"{stock_code}_{year}_annual_report.html"
                year_dir = os.path.join(save_dir, str(year))
                save_path = os.path.join(year_dir, filename)
                
                # 检查文件是否已经存在
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                    logger.info(f"文件已存在: {save_path}")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'existing'
                    })
                    continue
                
                # 查找对应年份的报告
                year_report = annual_reports[annual_reports['year'] == year]
                
                if year_report.empty or not year_report.iloc[0]['url']:
                    logger.warning(f"未找到 {stock_code} {year}年的10-K报告")
                    continue
                
                # 获取URL
                url = year_report.iloc[0]['url']
                
                # 下载报告
                logger.info(f"开始下载 {stock_code} {year}年10-K报告: {url}")
                download_success = download_with_retry(url, save_path)
                
                if download_success:
                    logger.info(f"成功下载 {stock_code} {year}年10-K报告")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'downloaded'
                    })
                else:
                    logger.error(f"下载 {stock_code} {year}年10-K报告失败")
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'year': year,
                        'file_path': save_path,
                        'status': 'failed'
                    })
                
                # 添加延迟避免请求过快
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 时出错: {e}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_path = os.path.join(save_dir, 'download_results_us.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        logger.info(f"下载结
