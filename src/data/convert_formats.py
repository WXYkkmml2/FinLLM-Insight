#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Format Conversion Module for FinLLM-Insight
This module converts PDF reports to text and performs Chinese text preprocessing.
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

# PDF extraction libraries
import PyPDF2
import pdfplumber

# For Chinese text processing
from hanziconv import HanziConv

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

def extract_text_from_pdf(pdf_path, method='pdfplumber'):
    """
    Extract text from PDF file using specified method
    
    Args:
        pdf_path (str): Path to the PDF file
        method (str): Extraction method ('pdfplumber' or 'pypdf2')
        
    Returns:
        str: Extracted text
    """
    logger.info(f"Extracting text from: {pdf_path}")
    
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

def preprocess_chinese_text(text):
    """
    Preprocess Chinese text
    - Convert traditional to simplified Chinese
    - Remove redundant whitespace
    - Normalize punctuation
    
    Args:
        text (str): Original text
        
    Returns:
        str: Preprocessed text
    """
    # Convert traditional to simplified Chinese
    text = HanziConv.toSimplified(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize various whitespace characters
    text = re.sub(r'[\u3000\u00A0\u2002-\u200A\u202F\u205F]', ' ', text)
    
    # Remove special characters that are not useful
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2000-\u206f\u0000-\u007f]', '', text)
    
    # Normalize punctuation
    punctuation_map = {
        '：': ': ',
        '；': '; ',
        '，': ', ',
        '。': '. ',
        '！': '! ',
        '？': '? ',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>',
    }
    
    for cn_punct, en_punct in punctuation_map.items():
        text = text.replace(cn_punct, en_punct)
    
    return text.strip()

def process_pdf_files(input_dir, output_dir, years=None, overwrite=False):
    """
    Process all PDF files in the input directory and save as text
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save processed text files
        years (list): List of years to process, or None for all
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        pd.DataFrame: Results of the conversion process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = []
    
    if years:
        # Process specific years
        for year in years:
            year_dir = os.path.join(input_dir, str(year))
            if os.path.exists(year_dir):
                for file in os.listdir(year_dir):
                    if file.endswith('.pdf'):
                        pdf_files.append((os.path.join(year_dir, file), year))
    else:
        # Process all files
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.pdf'):
                    # Extract year from directory name
                    year = os.path.basename(root)
                    if year.isdigit():
                        pdf_files.append((os.path.join(root, file), year))
    
    results = []
    
    # Process each PDF file
    for pdf_path, year in tqdm(pdf_files, desc="Processing PDF files"):
        try:
            # Create year directory in output
            year_output_dir = os.path.join(output_dir, year)
            os.makedirs(year_output_dir, exist_ok=True)
            
            # Get filename without extension
            file_base = os.path.basename(pdf_path)
            file_name = os.path.splitext(file_base)[0]
            
            # Output text file path
            text_file_path = os.path.join(year_output_dir, f"{file_name}.txt")
            
            # Skip if file exists and overwrite is False
            if os.path.exists(text_file_path) and not overwrite:
                logger.info(f"Skipping existing file: {text_file_path}")
                results.append({
                    'pdf_path': pdf_path,
                    'text_path': text_file_path,
                    'year': year,
                    'status': 'skipped'
                })
                continue
            
            # Extract text from PDF
            raw_text = extract_text_from_pdf(pdf_path)
            
            # Preprocess text
            processed_text = preprocess_chinese_text(raw_text)
            
            # Save processed text
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            logger.info(f"Processed: {pdf_path} -> {text_file_path}")
            results.append({
                'pdf_path': pdf_path,
                'text_path': text_file_path,
                'year': year,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            results.append({
                'pdf_path': pdf_path,
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

def main():
    """Main function to run the conversion process"""
    parser = argparse.ArgumentParser(description='Convert annual reports from PDF to text')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing files')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get directories
    input_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
    output_dir = config.get('processed_reports_text_directory', './data/processed/text_reports')
    
    # Get years to process
    min_year = config.get('min_year', 2018)
    years = list(range(min_year, 2026))  # Processing reports up to 2025
    
    # Process PDF files
    process_pdf_files(
        input_dir=input_dir,
        output_dir=output_dir,
        years=years,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()
