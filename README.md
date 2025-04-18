# GitHub Project Organization Guide for FinLLM-Insight

## Directory Structure

Here's the directory structure for  FinLLM-Insight project:

```
FinLLM-Insight/
├── data/                  # Data storage directory
│   ├── raw/               # Raw data (downloaded annual reports)
│   └── processed/         # Processed data
├── notebooks/             # Jupyter notebooks
│   ├── data_exploration.ipynb        # Data exploration
│   ├── model_training.ipynb          # Model training
│   └── results_visualization.ipynb   # Results visualization
├── src/                   # Source code
│   ├── data/              # Data processing related code
│   │   ├── download_reports.py       # Download annual reports
│   │   ├── convert_formats.py        # Format conversion
│   │   └── make_targets.py           # Generate target variables
│   ├── features/          # Feature engineering related code
│   │   ├── embeddings.py             # Generate embeddings
│   │   └── llm_features.py           # Generate features using LLM
│   ├── models/            # Model related code
│   │   ├── train.py                  # Model training
│   │   └── predict.py                # Model prediction
│   └── rag/               # RAG component
│       └── rag_component.py          # RAG implementation
├── config/                # Configuration files
│   └── config.json        # Main configuration file
├── requirements.txt       # Project dependencies
├── setup.py               # Installation script
├── LICENSE                # License file
└── README.md              # Project documentation
```
