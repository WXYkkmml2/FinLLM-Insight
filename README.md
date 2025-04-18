# FinLLM-Insight

Financial Report Analysis and Stock Prediction System Powered by Large Language Models

## Project Overview

FinLLM-Insight is a system that leverages Large Language Models (LLMs) to analyze company annual reports and predict stock investment value. This project integrates natural language processing, Retrieval-Augmented Generation (RAG), and machine learning technologies to provide data-driven decision support for investors.

## Key Features

- **Automated Data Acquisition**: Automatically download annual reports of listed companies from the Chinese securities market
- **Intelligent Text Analysis**: Deep analysis of financial report content using large language models
- **Feature Engineering**: Generate structured features based on LLM analysis
- **Investment Prediction**: Predict future stock performance and provide investment recommendations
- **Interactive Query**: Implement a RAG-based intelligent Q&A system supporting interactive exploration of financial report content

## System Architecture

![System Architecture](path_to_architecture_image.png)

## Quick Start

### Requirements

- Python 3.8+
- Dependencies: See `requirements.txt`

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FinLLM-Insight.git
cd FinLLM-Insight

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Configure your parameters in `config/config.json`:

```json
{
    "annual_reports_html_save_directory": "./data/raw/annual_reports",
    "china_stock_index": "CSI300",
    "min_year": 2018,
    "download_delay": 2
}
```

### Usage

#### Method 1: Run Individual Modules

```bash
# Download annual reports
python src/data/download_reports.py --config_path config/config.json

# Generate embeddings
python src/features/embeddings.py --config_path config/config.json

# More steps...
```

#### Method 2: Jupyter Notebook

Open the Jupyter notebooks in the `notebooks/` directory and follow the instructions to execute the complete workflow.

## Module Description

### Data Acquisition and Preprocessing

Use AKShare API to obtain financial reports of Chinese listed companies, and perform format conversion and preprocessing.

### Text Embedding and Feature Generation

Convert financial report text into vector representations and use large language models to generate structured features.

### Model Training and Evaluation

Train prediction models based on generated features, evaluate model performance, and generate investment recommendations.

### RAG Q&A System

Implement a Retrieval-Augmented Generation (RAG) system to support natural language queries about financial report content.

## Experimental Results

Brief presentation of system performance and analysis results...

## Contributing

Contributions through Pull Requests or Issues are welcome!

## Acknowledgements

This project is inspired by [GPT-InvestAR](https://github.com/UditGupta10/GPT-InvestAR) and includes multiple improvements and innovations based on it.

Thanks to the support from CSC6052/5051/4100/DDA6307/MDS5110 NLP course.

## License

[MIT License](LICENSE)
