#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG (Retrieval-Augmented Generation) Component for FinLLM-Insight
This module provides interactive query capabilities for annual report content.
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
import pandas as pd
from tqdm import tqdm
import time

# For LLM API calls
import openai
from openai import OpenAI

# For vector database interaction
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_component.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinancialReportRAG:
    """
    RAG system for financial report analysis
    """
    
    def __init__(self, config_path='config/config.json'):
        """
        Initialize the RAG system
        
        Args:
            config_path (str): Path to configuration file
            
        Raises:
            RuntimeError: If critical components initialization fails
        """
        try:
            self.config = self._load_config(config_path)
            self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo-16k')
            self.embedding_model = self.config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
            self.embeddings_dir = self.config.get('embeddings_directory', './data/processed/embeddings')
            self.max_tokens = self.config.get('max_tokens_per_call', 12000)
            
            # Initialize vector DB - critical component
            try:
                self.db_client, self.embedding_func = self._connect_to_vector_db()
            except Exception as e:
                logger.error(f"Failed to connect to vector database: {e}")
                raise RuntimeError(f"Vector database connection failed: {e}") from e
            
            # Load company metadata - critical component
            try:
                self.companies = self._load_company_metadata()
                if not self.companies:
                    logger.error("No company metadata could be loaded")
                    raise RuntimeError("No company data available in the system")
            except Exception as e:
                logger.error(f"Failed to load company metadata: {e}")
                raise RuntimeError(f"Company metadata loading failed: {e}") from e
            
            # Load LLM client - critical component
            try:
                self.llm_client = self._initialize_llm_client()
                if not self.llm_client:
                    raise RuntimeError("LLM client initialization returned None")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise RuntimeError(f"LLM client initialization failed: {e}") from e
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.critical(f"RAG system initialization failed: {e}")
            raise

    def _load_config(self, config_path):
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
    
    def _connect_to_vector_db(self):
        """
        Connect to vector database
        
        Returns:
            tuple: (ChromaDB client, embedding function)
            
        Raises:
            ValueError: If embedding model configuration is invalid
            ConnectionError: If database connection fails
        """
        # Verify embedding model configuration
        if not self.embedding_model:
            raise ValueError("Embedding model not specified in configuration")
        
        # Create embedding function
        try:
            if self.embedding_model.lower() == "openai":
                # Verify API key is available
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                    
                # Use OpenAI's embeddings
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-ada-002"
                )
                embedding_func = openai_ef
            else:
                # Use Hugging Face models
                huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
                    api_key=os.environ.get("HUGGINGFACE_API_KEY", None),
                    model_name=self.embedding_model
                )
                embedding_func = huggingface_ef
        except Exception as e:
            logger.error(f"Failed to create embedding function: {e}")
            raise ValueError(f"Failed to create embedding function: {e}") from e
        
        # Verify database path
        db_path = os.path.join(self.embeddings_dir, "chroma_db")
        if not os.path.exists(self.embeddings_dir):
            logger.warning(f"Embeddings directory does not exist: {self.embeddings_dir}")
            os.makedirs(self.embeddings_dir, exist_ok=True)
            logger.info(f"Created embeddings directory: {self.embeddings_dir}")
        
        # Create client
        try:
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path
            ))
            
            # Verify connection by testing a simple operation
            collections = client.list_collections()
            logger.info(f"Connected to vector database with {len(collections)} collections")
            return client, embedding_func
        
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}")
            raise ConnectionError(f"Vector database connection failed: {e}") from e

    def _load_company_metadata(self):
        """
        Load company metadata
        
        Returns:
            dict: Company metadata
        """
        try:
            metadata_path = os.path.join(self.embeddings_dir, "companies_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    companies = json.load(f)
                logger.info(f"Loaded metadata for {len(companies)} companies")
                return companies
            else:
                # If metadata file doesn't exist, try to gather from collections
                companies = {}
                collections = self.db_client.list_collections()
                
                for collection in collections:
                    if collection.name.startswith("company_"):
                        company_code = collection.name.replace("company_", "")
                        companies[company_code] = {"years": []}
                        
                        # Try to get years from collection
                        collection_obj = self.db_client.get_collection(
                            name=collection.name, 
                            embedding_function=self.embedding_func
                        )
                        
                        results = collection_obj.get()
                        if results and 'metadatas' in results and results['metadatas']:
                            years = set()
                            for metadata in results['metadatas']:
                                if 'year' in metadata:
                                    years.add(metadata['year'])
                            
                            for year in years:
                                companies[company_code]["years"].append({"year": year})
                
                logger.info(f"Generated metadata for {len(companies)} companies")
                return companies
        
        except Exception as e:
            logger.error(f"Failed to load company metadata: {e}")
            return {}
    
    def _initialize_llm_client(self):
        """
        Initialize LLM client
        
        Returns:
            client: LLM client
        """
        if "gpt" in self.llm_model.lower() or "openai" in self.llm_model.lower():
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            # Add support for other LLM models as needed
            logger.warning(f"Unsupported LLM model: {self.llm_model}, using OpenAI as fallback")
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def call_llm_with_retry(self, model, messages, max_retries=3, initial_delay=1):
        """
        Call LLM API with retry mechanism
        
        Args:
            model (str): Model name
            messages (list): List of message dicts
            max_retries (int): Maximum number of retry attempts
            initial_delay (int): Initial delay before first retry
            
        Returns:
            response: API response
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000
                )
                return response
            except Exception as e:
                wait_time = initial_delay * (2 ** attempt)
                logger.warning(f"LLM API call attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        logger.error(f"LLM API call failed after {max_retries} attempts")
        raise Exception(f"Failed to call LLM API after {max_retries} attempts")
    
    def get_available_companies(self):
        """
        Get list of available companies
        
        Returns:
            list: List of company codes
        """
        return list(self.companies.keys())
    
    def get_company_years(self, company_code):
        """
        Get available years for a company
        
        Args:
            company_code (str): Company code
            
        Returns:
            list: List of available years
        """
        if company_code not in self.companies:
            return []
        
        company_data = self.companies[company_code]
        
        if isinstance(company_data, dict) and "years" in company_data:
            return [y["year"] for y in company_data["years"]]
        elif isinstance(company_data, list):
            return [item["year"] for item in company_data]
        else:
            return []
    
    def query(self, question, company_code=None, year=None, max_results=5):
        """
        Query the system with a question
        
        Args:
            question (str): Question to ask
            company_code (str): Company code to focus on, or None for all
            year (str): Report year to focus on, or None for all
            max_results (int): Maximum number of results to return
            
        Returns:
            dict: Query results
        """

        try:
            # 定义元数据字段映射
            metadata_mappings = {
                "company_code": ["company_code", "公司代码", "股票代码", "代码"],
                "year": ["year", "报告年份", "年份"],
                "chunk_index": ["chunk_index", "索引", "index"],
                "source_file": ["source_file", "文件路径", "file_path", "path"]
            }
            
            # 处理查询结果
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    raw_metadata = {}
                    if results['metadatas'] and results['metadatas'][0]:
                        raw_metadata = results['metadatas'][0][i]
                    
                    # 构建块信息
                    chunk_info = {"content": doc}
                    
                    # 处理元数据字段
                    for standard_key, possible_keys in metadata_mappings.items():
                        # 从元数据中查找值
                        value = None
                        for key in possible_keys:
                            if key in raw_metadata and raw_metadata[key] is not None:
                                value = raw_metadata[key]
                                break
                        
                        # 设置默认值
                        if value is None:
                            if standard_key == "company_code" and company_code:
                                value = company_code
                            elif standard_key == "year" and year:
                                value = year
                            elif standard_key == "year":
                                value = "unknown"
                            elif standard_key == "chunk_index":
                                value = i
                            elif standard_key == "source_file":
                                value = "unknown"
                        
                        chunk_info[standard_key] = value
                    
                    relevant_chunks.append(chunk_info)
              
                except Exception as e:
                    logger.error(f"Error querying collection {collection_name}: {e}")

            # If no company specified or no results found, query all collections
            if not company_code or not relevant_chunks:
                collections = self.db_client.list_collections()
                
                if not collections:
                    return {
                        "answer": "No company data found in the system. Please ensure the system has been initialized with annual report data.",
                        "sources": []
                    }
                
                for collection_info in collections:
                    if collection_info.name.startswith("company_"):
                        
                        try:
                            collection = self.db_client.get_collection(name=collection_info.name, embedding_function=self.embedding_func)
                            
                            # Query collection
                            results = collection.query(
                                query_texts=[question],
                                n_results=3  # Fewer results per company when querying all
                            )
                            
                            if results and results['documents'] and results['documents'][0]:
                                for i, doc in enumerate(results['documents'][0]):
                                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                                    
                                    current_company = collection_info.name.replace("company_", "")
                                    
                                    chunk_info = {
                                        "content": doc,
                                        "company_code": metadata.get("company_code", current_company),
                                        "year": metadata.get("year", "unknown"),
                                        "chunk_index": metadata.get("chunk_index", i),
                                        "source_file": metadata.get("source_file", "unknown")
                                    }
                                    
                                    relevant_chunks.append(chunk_info)
                        
                        except Exception as e:
                            logger.error(f"Error querying collection {collection_info.name}: {e}")
            
            # Keep only top results across all collections
            relevant_chunks = sorted(relevant_chunks, key=lambda x: x.get("score", 0), reverse=True)[:max_results]
            

            if not relevant_chunks:
                return {
                    "answer": "No relevant information found for your question. Please try rephrasing your question or specify a company code.",
                    "sources": []
                }
            
            # Format relevant chunks for the prompt
            context_text = ""
            sources = []
            
            for i, chunk in enumerate(relevant_chunks):
                context_text += f"\n\nDocument Chunk {i+1} [Source: Company {chunk['company_code']}, Annual Report {chunk['year']}]:\n{chunk['content']}"
                
                source_info = {
                    "company_code": chunk["company_code"],
                    "year": chunk["year"],
                    "file": os.path.basename(chunk["source_file"]) if "source_file" in chunk else "unknown"
                }
                sources.append(source_info)
            
            # Generate answer using LLM
            system_prompt = """You are a professional financial analyst specializing in analyzing annual reports of listed companies and answering investor questions.
Your answers should be based on the provided annual report excerpts, accurate, objective, and comprehensive.
If the information in the provided excerpts is insufficient to fully answer the question, clearly indicate these limitations.
Data and facts cited in your response must come from the provided annual report excerpts; do not add your own information.
Your response format should be professional and well-structured, suitable for investor reference.
"""
            
            user_prompt = f"""Please answer the following question about company annual reports:

Question: {question}

Here are relevant annual report excerpts:
{context_text}

Please answer the question based on these annual report excerpts. If the information is insufficient, please clearly indicate this."""
            

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ] 
                # Request failed with status code 429
                if not self.llm_client:
                    raise ValueError("LLM client not initialized")
                    
                #Request failed with status code 429
                response = self.call_llm_with_retry(self.llm_model, messages)
                
                answer = response.choices[0].message.content
                
                return {
                    "answer": answer,
                    "sources": sources
                }
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error generating the answer. Error details: {str(e)}"
                logger.error(f"Error generating answer: {e}")
                return {
                    "answer": error_message,
                    "sources": sources
                }
        
        except Exception as e:
           
            logger.error(f"Error in query: {e}", exc_info=True)
            return {
                "answer": "I'm sorry, but I encountered a technical problem while processing your question. Please try again later or contact support if the problem persists.",
                "sources": []
            }            
           

def interactive_cli(rag_system):
    """
    Run interactive command-line interface
    
    Args:
        rag_system (FinancialReportRAG): Initialized RAG system
    """
    print("\nWelcome to FinLLM-Insight Financial Report Analysis System")
    print("="*50)
    print("You can ask questions in natural language, and the system will answer based on annual report content")
    print("Enter 'exit' or 'quit' to exit the program")
    print("Enter 'companies' to see available company codes")
    print("="*50)
    
    while True:
        print("\nPlease enter your question (format: [company_code] [year] question):")
        user_input = input("> ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the system. Goodbye!")
            break
        
        if user_input.lower() == "companies":
            companies = rag_system.get_available_companies()
            print(f"\nThe system contains {len(companies)} companies:")
            for i, company in enumerate(companies):
                print(f"{company}", end="\t")
                if (i+1) % 10 == 0:
                    print("")
            print("\n")
            continue
        
        # Parse input for company code and year
        parts = user_input.split()
        company_code = None
        year = None
        question = user_input
        
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 6:
            company_code = parts[0]
            question = " ".join(parts[1:])
            
            # Check if second part is a year
            if len(parts) >= 3 and parts[1].isdigit() and 2000 <= int(parts[1]) <= 2030:
                year = parts[1]
                question = " ".join(parts[2:])
        
        print("\nProcessing query, please wait...\n")
        
        if company_code:
            print(f"Company code: {company_code}")
        if year:
            print(f"Year: {year}")
        print(f"Question: {question}")
        print("-"*50)
        
        # Query RAG system
        start_time = time.time()
        result = rag_system.query(question, company_code, year)
        end_time = time.time()
        
        # Display results
        print("\nAnswer:")
        print(result["answer"])
        print("\nInformation sources:")
        for source in result["sources"]:
            print(f"- Company {source['company_code']}, Annual Report {source['year']}")
        
        print(f"\nQuery time: {end_time - start_time:.2f} seconds")


def main():
    """Main function to run the RAG component"""
    parser = argparse.ArgumentParser(description='RAG component for financial report analysis')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_system = FinancialReportRAG(config_path=args.config_path)
        
        # Run in interactive mode if specified
        if args.interactive:
            interactive_cli(rag_system)
        else:
            # Print usage information
            print("RAG system initialized successfully.")
            print("Use this module by importing the FinancialReportRAG class.")
            print("For interactive mode, run with --interactive flag.")
        
        return 0
    except Exception as e:
        logger.critical(f"Failed to initialize or run RAG system: {e}", exc_info=True)
        print(f"ERROR: {str(e)}")
        print("Check the log file for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
