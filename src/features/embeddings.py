#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embeddings Generation Module for FinLLM-Insight
This module creates embeddings for annual report text and stores them in a vector database.
"""
# 确保导入了 chromadb 库
import chromadb
# 确保导入了 Settings 类，虽然在新方式中创建客户端时可能不再直接用它作为参数，
# 但脚本其他地方可能还需要 Settings
from chromadb.config import Settings
import logging # 确保 logging 库被导入
# 确保 logger 对象在脚本的日志配置部分被正确定义了
# logger = logging.getLogger(__name__)

import asyncio
from chromadb.utils import embedding_functions
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import concurrent.futures
import json
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
#import asyncio

# Text processing and chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

async def create_chroma_client(db_path: str):
    """
    Creates and returns a ChromaDB async client for version 1.0.6.

    Args:
        db_path: The path to the ChromaDB persistent storage.

    Returns:
        A ChromaDB AsyncClient instance (based on likely v1.0.6 API).
    """
    logger.info(f"Creating ChromaDB client at: {db_path}")

    # >>> 尝试导入 Async Client 类，适配 v1.0.6 API 可能性 <<<
    ChromaAsyncClientClass = None # Initialize
    import_success = False
    tried_paths = []

    # 尝试 1: 尝试从 chromadb.api.async_client 导入 AsyncClient
    try:
        from chromadb.api.async_client import AsyncClient as ChromaAsyncClientClass_Attempt1
        ChromaAsyncClientClass = ChromaAsyncClientClass_Attempt1
        logger.info("Attempt 1: Successfully imported AsyncClient from chromadb.api.async_client")
        import_success = True
    except ImportError:
        tried_paths.append("chromadb.api.async_client.AsyncClient")
        logger.warning("Attempt 1 failed: AsyncClient not found in chromadb.api.async_client, trying other paths.")

    if not import_success:
        # 尝试 2: 尝试从 chromadb 顶层导入 AsyncClient (虽然第一错误日志曾提示没有，但代码可能混淆了)
        try:
            from chromadb import AsyncClient as ChromaAsyncClientClass_Attempt2
            ChromaAsyncClientClass = ChromaAsyncClientClass_Attempt2
            logger.info("Attempt 2: Successfully imported AsyncClient from chromadb (top level)")
            import_success = True
        except ImportError:
            tried_paths.append("chromadb.AsyncClient (top level)")
            logger.warning("Attempt 2 failed: AsyncClient not found directly in chromadb, trying other paths.")

    if not import_success:
         # 尝试 3: 尝试从 chromadb.api 导入 AsyncClient
        try:
            from chromadb.api import AsyncClient as ChromaAsyncClientClass_Attempt3
            ChromaAsyncClientClass = ChromaAsyncClientClass_Attempt3
            logger.info("Attempt 3: Successfully imported AsyncClient from chromadb.api")
            import_success = True
        except ImportError:
            tried_paths.append("chromadb.api.AsyncClient")
            # 没有更多常见的异步客户端路径了

    if not import_success or ChromaAsyncClientClass is None:
         logger.error(f"AsyncClient (or equivalent) class not found in common locations for chromadb 1.0.6 after trying: {', '.join(tried_paths)}")
         logger.error("Please check the official chromadb documentation for version 1.0.6 (docs.trychroma.com) regarding async client initialization.")
         # 报告最终的导入错误
         raise ImportError(f"Could not find a suitable AsyncClient class in chromadb version 1.0.6 after trying: {', '.join(tried_paths)}")


    # >>> 导入尝试结束 <<<

    try:
        # 使用 Settings 配置客户端
        settings = Settings(
            persist_directory=db_path,
            is_persistent=True
        )
        client = ChromaAsyncClientClass(settings=settings)
        logger.info("ChromaDB client created successfully using Settings")
        return client

    except Exception as e:
        # 如果 AsyncClient 不接受 path 参数，或者创建时发生其他错误，会捕获到这里
        logger.error(f"Failed to create ChromaDB client instance with path={db_path}: {e}", exc_info=True)
        logger.error("It's possible AsyncClient in version 1.0.6 does not take a 'path' argument for local persistence.")
        logger.error("Please verify the correct client creation method for local persistence in chromadb 1.0.6 documentation.")
        raise # Re-raise the exception
  

def create_embedding_function(embedding_model):
    """
    Create embedding function based on specified model
    
    Args:
        embedding_model (str): Name of the embedding model
        
    Returns:
        embedding_function: Function to generate embeddings
    """
    try:
        # Use ChromaDB's built-in SentenceTransformerEmbeddingFunction
        from chromadb.utils import embedding_functions
        
        logger.info(f"Creating embedding function for model {embedding_model}")
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        logger.info("Embedding function created successfully")
        
        return embedding_func
        
    except Exception as e:
        logger.error(f"Failed to create embedding function: {e}")
        raise

def process_reports_parallel(text_dir, output_dir, embedding_model, chunk_size=1000, chunk_overlap=200, max_workers=4, batch_size=100):
    """使用多线程并行处理报告"""
    import concurrent.futures
    
    # Delete existing database directory
    db_path = os.path.join(output_dir, "chroma_db")
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        logger.info(f"Deleted existing database directory at {db_path}")
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Create embedding function
    embedding_func = create_embedding_function(embedding_model)
    
    # Get all report files
    report_files = []
    for root, _, files in os.walk(text_dir):
        for file in files:
            if file.endswith('.txt'):
                report_files.append(os.path.join(root, file))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_report, file_path, client, embedding_func, chunk_size, chunk_overlap, batch_size): file_path 
                         for file_path in report_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(report_files)):
            file_path = future_to_file[future]
            try:
                result = future.result()
                # 处理结果...
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
    
    # Calculate statistics
    total_chunks = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    logger.info(f"Processed {len(results)} reports with {total_chunks} successful chunks and {error_count} errors")
    
    return results

def process_single_report(file_path, client, embedding_func, chunk_size, chunk_overlap, batch_size=100):
    """Process a single report file"""
    try:
        # Extract company code and year from filename
        file_name = os.path.basename(file_path)
        company_code = file_name.split('_')[0]
        year = file_name.split('_')[1]
        
        # Create collection name
        collection_name = f"company_{company_code}"
        
        # Check if collection exists and delete it
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection {collection_name}")
        except Exception as e:
            logger.debug(f"Collection {collection_name} does not exist or could not be deleted: {e}")
        
        # Create new collection
        collection = client.create_collection(name=collection_name, embedding_function=embedding_func)
        
        # Load and split text
        text = load_report_text(file_path)
        if not text:
            raise Exception("Failed to load text")
        
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_metadatas = [{"year": year, "company_code": company_code} for _ in batch_chunks]
            batch_ids = [f"{company_code}_{year}_{j}" for j in range(i, i + len(batch_chunks))]
            
            collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        return {"status": "success", "company_code": company_code, "year": year, "chunks": len(chunks)}
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {"status": "error", "file": file_path, "error": str(e)}

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
    embedding_model = config.get('embedding_model', 'BAAI/bge-small-zh-v1.5')
    chunk_size = config.get('chunk_size', 2000)
    chunk_overlap = config.get('chunk_overlap', 100)
    max_workers = config.get('max_workers', 4)
    batch_size = config.get('batch_size', 100)
    
    # Get API key from config
    api_key = config.get('huggingface_api_key')
    if api_key:
        os.environ["HUGGINGFACE_API_KEY"] = api_key
        os.environ["CHROMA_HUGGINGFACE_API_KEY"] = api_key
    elif not os.environ.get("HUGGINGFACE_API_KEY"):
        logger.error("HUGGINGFACE_API_KEY not found in config or environment variables")
        sys.exit(1)
    
    # Process reports using synchronous method
    try:
        stats = process_reports_parallel(
            text_dir=text_dir,
            output_dir=output_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        logger.info(f"Processing completed with stats: {stats}")
        
        if stats['error_reports'] > 0:
            logger.warning(f"Failed to process {stats['error_reports']} files")
            logger.warning("Please check the logs for detailed error messages")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

def optimize_memory_usage(text):
    """优化内存使用"""
    # 使用生成器而不是一次性加载所有文本
    def text_generator(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

if __name__ == "__main__":
    main()
