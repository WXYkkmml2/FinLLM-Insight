#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Filtering Script for FinLLM-Insight
This script filters existing data to keep only specified companies and years.
"""

import os
import shutil
import argparse
import logging
import json
import pandas as pd
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("filter_data.log"),
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
        logger.error(f"加载配置文件失败: {e}")
        raise

def get_companies_to_keep(top_n=50, input_file=None):
    """获取要保留的公司列表"""
    if input_file and os.path.exists(input_file):
        try:
            # 从文件加载公司列表
            df = pd.read_csv(input_file)
            companies = df['code'].astype(str).tolist()
            return companies[:top_n]  # 只保留前N家
        except Exception as e:
            logger.error(f"从文件加载公司列表失败: {e}")
    
    logger.info("没有指定公司列表文件或文件不存在，将保留遇到的前50家公司")
    return []  # 返回空列表，稍后动态填充

def filter_annual_reports(reports_dir, output_dir, companies_to_keep, min_year, max_year):
    """筛选年报文件，只保留指定公司和年份范围内的报告"""
    
    if not os.path.exists(reports_dir):
        logger.error(f"报告目录不存在: {reports_dir}")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 动态收集公司列表（如果没有预先指定）
    dynamic_companies = []
    
    # 处理每个年份目录
    for year in range(min_year, max_year + 1):
        year_dir = os.path.join(reports_dir, str(year))
        output_year_dir = os.path.join(output_dir, str(year))
        
        if not os.path.exists(year_dir):
            logger.warning(f"年份目录不存在: {year_dir}")
            continue
        
        os.makedirs(output_year_dir, exist_ok=True)
        
        # 处理该年份下的所有文件
        for filename in os.listdir(year_dir):
            if not (filename.endswith('.pdf') or filename.endswith('.txt')):
                continue
                
            # 从文件名中提取公司代码
            parts = filename.split('_')
            if len(parts) < 2 or not parts[0].isdigit():
                continue
                
            company_code = parts[0]
            
            # 如果公司列表为空，则动态收集
            if not companies_to_keep and len(dynamic_companies) < 50:
                if company_code not in dynamic_companies:
                    dynamic_companies.append(company_code)
            
            # 检查是否保留该文件
            if not companies_to_keep or company_code in companies_to_keep or company_code in dynamic_companies:
                # 复制文件到输出目录
                source_file = os.path.join(year_dir, filename)
                target_file = os.path.join(output_year_dir, filename)
                shutil.copy2(source_file, target_file)
                logger.info(f"已复制: {source_file} -> {target_file}")
    
    # 如果使用了动态公司列表，保存它
    if dynamic_companies and not companies_to_keep:
        with open(os.path.join(output_dir, 'filtered_companies.txt'), 'w') as f:
            for company in dynamic_companies:
                f.write(f"{company}\n")
        logger.info(f"已动态选择并保存 {len(dynamic_companies)} 家公司")
    
    return True

def filter_processed_data(source_dir, output_dir, companies_to_keep, min_year, max_year):
    """筛选处理后的数据，只保留指定公司和年份的数据"""
    if not os.path.exists(source_dir):
        logger.warning(f"源目录不存在: {source_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个年份目录
    for year in range(min_year, max_year + 1):
        year_dir = os.path.join(source_dir, str(year))
        output_year_dir = os.path.join(output_dir, str(year))
        
        if os.path.exists(year_dir):
            os.makedirs(output_year_dir, exist_ok=True)
            
            for filename in os.listdir(year_dir):
                parts = filename.split('_')
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                    
                company_code = parts[0]
                if not companies_to_keep or company_code in companies_to_keep:
                    source_file = os.path.join(year_dir, filename)
                    target_file = os.path.join(output_year_dir, filename)
                    shutil.copy2(source_file, target_file)
                    logger.info(f"已复制: {source_file} -> {target_file}")
    
    # 复制结果文件（如果存在）
    for result_file in ['conversion_results.csv', 'companies_metadata.json']:
        source_file = os.path.join(source_dir, result_file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(output_dir, result_file))
            logger.info(f"已复制结果文件: {result_file}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='筛选FinLLM-Insight项目数据，只保留指定公司和年份')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='配置文件路径')
    parser.add_argument('--company_list', type=str, default=None,
                        help='要保留的公司列表文件路径')
    parser.add_argument('--top_n', type=int, default=50,
                        help='要保留的公司数量')
    parser.add_argument('--min_year', type=int, default=None,
                        help='最小年份（覆盖配置文件）')
    parser.add_argument('--max_year', type=int, default=None,
                        help='最大年份（覆盖配置文件）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config_path)
    
    # Get a list of companies to keep
    companies_to_keep = get_companies_to_keep(args.top_n, args.company_list)
    
    # Get the year range
    min_year = args.min_year if args.min_year is not None else config.get('min_year', 2021)
    max_year = args.max_year if args.max_year is not None else 2022  # 默认保留两年
    
    # Filter original annual report data
    annual_reports_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
    filtered_annual_reports_dir = annual_reports_dir + '_filtered'
    filter_annual_reports(annual_reports_dir, filtered_annual_reports_dir, companies_to_keep, min_year, max_year)
    
    # Filter the processed text data
    text_reports_dir = config.get('processed_reports_text_directory', './data/processed/text_reports')
    filtered_text_reports_dir = text_reports_dir + '_filtered'
    filter_processed_data(text_reports_dir, filtered_text_reports_dir, companies_to_keep, min_year, max_year)
    
    # Update configuration information 
    logger.info(f"""
    Data filtering is completed! To use filtered data, update the following settings in the configuration file:
    
    "annual_reports_html_save_directory": "{filtered_annual_reports_dir}",
    "processed_reports_text_directory": "{filtered_text_reports_dir}",
    "min_year": {min_year},
    
    Or move the filtered directory contents to the original directory to replace the original data.
    """)

if __name__ == "__main__":
    main()
