#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embeddings Generation Module for FinLLM-Insight
This module creates embeddings for annual report text and stores them in a vector database.
"""

import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio

# Text processing and chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector database
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embeddings.log"),
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

def load_report_text(file_path):
    """
    Load report text from file
    
    Args:
        file_path (str): Path to text file
        
    Returns:
        str: Report text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        logger.error(f"Failed to load text from {file_path}: {e}")
        return None

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Failed to split text: {e}")
        return []

async def create_chroma_client(persist_directory):
    """
    Create a ChromaDB async client
    
    Args:
        persist_directory (str): Directory to persist the database
        
    Returns:
        chromadb.AsyncClient: ChromaDB async client
    """
    try:
        client = chromadb.AsyncClient(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        return client
    
    except Exception as e:
        logger.error(f"Failed to create ChromaDB async client: {e}")
        raise

def create_embedding_function(embedding_model):
    """
    Create embedding function based on specified model
    
    Args:
        embedding_model (str): Name of the embedding model
        
    Returns:
        embedding_function: Function to generate embeddings
    """
    try:
        if embedding_model.lower() == "openai":
            # Use OpenAI's embeddings (requires API key)
            import openai
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
            return openai_ef
        else:
            # Use Hugging Face models (offline, no API key needed)
            huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=os.environ.get("HUGGINGFACE_API_KEY", None),
                model_name=embedding_model
            )
            return huggingface_ef
    except Exception as e:
        logger.error(f"Failed to create embedding function: {e}")
        raise

async def process_reports(text_dir, output_dir, embedding_model, chunk_size=1000, chunk_overlap=200):
    """
    Process all reports and generate embeddings using async methods
    
    Args:
        text_dir (str): Directory containing text reports
        output_dir (str): Directory to save embeddings
        embedding_model (str): Name of the embedding model
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        dict: Processing statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create vector database client
    db_path = os.path.join(output_dir, "chroma_db")
    client = await create_chroma_client(db_path)
    
    # Create embedding function based on specified model
    embedding_func = create_embedding_function(embedding_model)
    
    # Create a collection for each company
    companies = {}
    stats = {
        "total_files": 0,
        "total_chunks": 0,
        "processed_files": 0,
        "error_files": 0
    }
    
    # Get all text files
    text_files = []
    for root, _, files in os.walk(text_dir):
        for file in files:
            if file.endswith('.txt'):
                text_files.append(os.path.join(root, file))
    
    stats["total_files"] = len(text_files)
    
    # Process each text file
    for file_path in tqdm(text_files, desc="Processing reports"):
        try:
            # Extract company code and year from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            if len(parts) < 2:
                logger.warning(f"Invalid filename format: {filename}")
                stats["error_files"] += 1
                continue
            
            company_code = parts[0]
            year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "unknown"
            
            # Load report text
            text = load_report_text(file_path)
            
            if text is None or len(text) < 100:  # Skip if text is too short
                logger.warning(f"Empty or too short text in {file_path}")
                stats["error_files"] += 1
                continue
            
            # Split text into chunks
            chunks = split_text_into_chunks(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                stats["error_files"] += 1
                continue
            
            # Update statistics
            stats["total_chunks"] += len(chunks)
            
            # Create or get collection for this company
            collection_name = f"company_{company_code}"
            
            try:
                collection = await client.get_collection(name=collection_name, embedding_function=embedding_func)
                logger.info(f"Using existing collection for {company_code}")
            except Exception:
                collection = await client.create_collection(name=collection_name, embedding_function=embedding_func)
                logger.info(f"Created new collection for {company_code}")
            
            # Prepare chunks for insertion
            ids = [f"{company_code}_{year}_{i}" for i in range(len(chunks))]
            metadatas = [{
                "company_code": company_code,
                "year": year,
                "chunk_index": i,
                "source_file": file_path,
            } for i in range(len(chunks))]
            
            # Add chunks to collection
            await collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            # Add to companies dict
            if company_code not in companies:
                companies[company_code] = []
            
            companies[company_code].append({
                "year": year,
                "file_path": file_path,
                "chunk_count": len(chunks)
            })
            
            stats["processed_files"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            stats["error_files"] += 1
    
    # Persist the database
    await client.persist()
    
    # Save companies metadata
    companies_path = os.path.join(output_dir, "companies_metadata.json")
    with open(companies_path, 'w', encoding='utf-8') as f:
        json.dump(companies, f, ensure_ascii=False, indent=2)
    
    # Save statistics
    stats_path = os.path.join(output_dir, "embedding_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Embedding generation complete. Stats: {stats}")
    return stats

def main():
    """Main function to run the embedding generation process"""
    parser = argparse.ArgumentParser(description='Generate embeddings for annual reports')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get parameters
    text_dir = config.get('processed_reports_text_directory', './data/processed/text_reports')
    output_dir = config.get('embeddings_directory', './data/processed/embeddings')
    embedding_model = config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
    chunk_size = config.get('chunk_size', 1000)
    chunk_overlap = config.get('chunk_overlap', 200)
    
    # Process reports using asyncio
    asyncio.run(process_reports(
        text_dir=text_dir,
        output_dir=output_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ))

if __name__ == "__main__":
    main()
