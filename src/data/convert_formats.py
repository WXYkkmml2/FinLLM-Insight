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

def main():
    """Main function to run the conversion process"""
    parser = argparse.ArgumentParser(description='Convert annual reports to text format')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing files')
    parser.add_argument('--file_type', type=str, default='html', choices=['html', 'pdf'],
                        help='Type of files to process')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get directories
    input_dir = config.get('annual_reports_html_save_directory', './data/raw/annual_reports')
    output_dir = config.get('processed_reports_text_directory', './data/processed/text_reports')
    
    # Get years to process
    min_year = config.get('min_year', 2018)
    max_year = config.get('max_year', None)
    
    if max_year is None:
        max_year = datetime.now().year
        
    years = list(range(min_year, max_year + 1))
    
    # Process report files
    process_report_files(
        input_dir=input_dir,
        output_dir=output_dir,
        years=years,
        overwrite=args.overwrite,
        file_type=args.file_type
    )

if __name__ == "__main__":
    from datetime import datetime
    main()
