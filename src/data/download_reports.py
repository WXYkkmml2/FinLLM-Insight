
import requests
import json
import pandas as pd
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_report(url, path):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    file_extension = url.split('.')[-1]
    path = path + '.' + file_extension
    
    logger.info(f"Downloading: {url} to {path}")
    logger.info(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        # Get the content of the file
        page_content = response.content

        # Write the content to the local file
        with open(path, "wb") as file:
            file.write(page_content)
        logger.info(f"Download successful, file size: {len(page_content)} bytes")
        return True
    else:
        logger.error(f'Response not 200. Failed for: {url}')
        return False

def get_all_tickers():
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        ticker_list_500 = sp500[0].Symbol.to_list()
        
        # 只获取前5个进行测试
        ticker_list = ticker_list_500[:5]
        logger.info(f"Retrieved {len(ticker_list)} tickers")
        return ticker_list
    except Exception as e:
        logger.error(f"Failed to get tickers: {e}")
        return ["AAPL", "MSFT"]  # 默认测试股票

def main():
    # 加载配置
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        api_key = config.get('financial_modelling_prep_api_key', '')
        if not api_key:
            logger.error("API key not found in config")
            return
        
        save_dir = './data/raw/annual_reports'
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Config loading error: {e}")
        return

    # 获取股票列表
    ticker_list = get_all_tickers()
    
    # 逐个处理股票
    for i, ticker in enumerate(ticker_list):
        logger.info(f"Processing {i+1}/{len(ticker_list)}: {ticker}")
        
        # 获取10-K URL
        fmp_10k_url = f'https://financialmodelingprep.com/api/v3/sec_filings/{ticker}?type=10-K&page=0&apikey={api_key}'
        
        try:
            response = requests.get(fmp_10k_url)
            logger.info(f"API Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code}")
                continue
                
            data = json.loads(response.content)
            
            if not data:
                logger.warning(f"No data returned for {ticker}")
                continue
                
            logger.info(f"Retrieved {len(data)} filings for {ticker}")
            
            success_count = 0
            
            for d in data:
                filing_type = d.get('type', '')
                if not ((filing_type.lower() == '10-k') or (filing_type.lower() == '10k')):
                    continue
                    
                date_string = d.get('fillingDate', '')
                if not date_string:
                    continue
                    
                date = date_string[:10]
                year = date_string[:4]
                
                if int(year) < 2018:
                    continue
                    
                link = d.get('finalLink', '')
                if not link:
                    continue
                
                # 准备保存路径
                year_dir = os.path.join(save_dir, year)
                os.makedirs(year_dir, exist_ok=True)
                save_path = os.path.join(year_dir, f"{ticker}_{year}_annual_report")
                
                # 下载报告
                success = download_report(link, save_path)
                if success:
                    success_count += 1
                
                # 添加延迟避免API限制
                time.sleep(2)
            
            logger.info(f"Successfully downloaded {success_count} reports for {ticker}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            
    logger.info("Download process completed")

if __name__ == "__main__":
    main()
