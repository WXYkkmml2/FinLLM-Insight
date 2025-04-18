#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG (Retrieval-Augmented Generation) Component for FinLLM-Insight
This module provides interactive query capabilities for annual report content.
"""

import os
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
        """
        self.config = self._load_config(config_path)
        self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo-16k')
        self.embedding_model = self.config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
        self.embeddings_dir = self.config.get('embeddings_directory', './data/processed/embeddings')
        self.max_tokens = self.config.get('max_tokens_per_call', 12000)
        
        # Initialize vector DB
        self.db_client, self.embedding_func = self._connect_to_vector_db()
        
        # Load company metadata
        self.companies = self._load_company_metadata()
        
        # Load LLM client
        self.llm_client = self._initialize_llm_client()
        
        logger.info("RAG system initialized successfully")
    
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
        """
        try:
            # Create embedding function
            if self.embedding_model.lower() == "openai":
                # Use OpenAI's embeddings
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ.get("OPENAI_API_KEY"),
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
            
            # Create client
            db_path = os.path.join(self.embeddings_dir, "chroma_db")
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path
            ))
            
            logger.info("Connected to vector database")
            return client, embedding_func
        
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}")
            raise
    
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
            # Validate inputs
            if company_code and company_code not in self.companies:
                return {
                    "answer": f"公司代码 {company_code} 不在系统中。可用的公司代码: {', '.join(self.get_available_companies()[:10])}...",
                    "sources": []
                }
            
            if company_code and year and year not in self.get_company_years(company_code):
                available_years = self.get_company_years(company_code)
                return {
                    "answer": f"公司 {company_code} 在系统中没有 {year} 年的年报。可用年份: {', '.join(available_years)}",
                    "sources": []
                }
            
            # Prepare for query
            relevant_chunks = []
            
            # If company_code is specified, query only that collection
            if company_code:
                collection_name = f"company_{company_code}"
                try:
                    collection = self.db_client.get_collection(name=collection_name, embedding_function=self.embedding_func)
                    
                    # Query constraints
                    where_clause = None
                    if year:
                        where_clause = {"year": year}
                    
                    # Query collection
                    results = collection.query(
                        query_texts=[question],
                        where=where_clause,
                        n_results=max_results
                    )
                    
                    if results and results['documents'] and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                            
                            chunk_info = {
                                "content": doc,
                                "company_code": metadata.get("company_code", company_code),
                                "year": metadata.get("year", year if year else "未知"),
                                "chunk_index": metadata.get("chunk_index", i),
                                "source_file": metadata.get("source_file", "未知")
                            }
                            
                            relevant_chunks.append(chunk_info)
                
                except Exception as e:
                    logger.error(f"Error querying collection {collection_name}: {e}")
            
            # If no company specified or no results found, query all collections
            if not company_code or not relevant_chunks:
                collections = self.db_client.list_collections()
                
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
                                        "year": metadata.get("year", "未知"),
                                        "chunk_index": metadata.get("chunk_index", i),
                                        "source_file": metadata.get("source_file", "未知")
                                    }
                                    
                                    relevant_chunks.append(chunk_info)
                        
                        except Exception as e:
                            logger.error(f"Error querying collection {collection_info.name}: {e}")
            
            # Keep only top results across all collections
            relevant_chunks = sorted(relevant_chunks, key=lambda x: x.get("score", 0), reverse=True)[:max_results]
            
            if not relevant_chunks:
                return {
                    "answer": "没有找到与问题相关的信息。请尝试重新表述您的问题，或指定具体的公司代码。",
                    "sources": []
                }
            
            # Format relevant chunks for the prompt
            context_text = ""
            sources = []
            
            for i, chunk in enumerate(relevant_chunks):
                context_text += f"\n\n文档片段 {i+1} [来源: {chunk['company_code']}公司 {chunk['year']}年报]:\n{chunk['content']}"
                
                source_info = {
                    "company_code": chunk["company_code"],
                    "year": chunk["year"],
                    "file": os.path.basename(chunk["source_file"]) if "source_file" in chunk else "未知"
                }
                sources.append(source_info)
            
            # Generate answer using LLM
            system_prompt = """你是一位专业的金融分析师，擅长分析上市公司年度报告并回答投资者的问题。
你的回答应该基于提供的年报片段，准确、客观、全面。
如果年报片段中的信息不足以完整回答问题，请明确指出信息的局限性。
回答中引用的数据和事实必须来自提供的年报片段，不要添加你自己的信息。
回答格式应专业、结构清晰，适合投资者参考。
"""
            
            user_prompt = f"""请回答以下关于公司年报的问题：

问题：{question}

以下是相关年报片段：
{context_text}

请基于上述年报片段回答问题，如果信息不足，请明确指出。"""
            
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                answer = response.choices[0].message.content
                
                return {
                    "answer": answer,
                    "sources": sources
                }
            
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                return {
                    "answer": f"生成回答时出错: {str(e)}",
                    "sources": sources
                }
        
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "answer": f"查询处理出错: {str(e)}",
                "sources": []
            }

def interactive_cli(rag_system):
    """
    Run interactive command-line interface
    
    Args:
        rag_system (FinancialReportRAG): Initialized RAG system
    """
    print("\n欢迎使用 FinLLM-Insight 财报分析系统")
    print("="*50)
    print("您可以通过自然语言提问，系统会基于公司年报内容为您解答")
    print("输入 'exit' 或 'quit' 退出程序")
    print("输入 'companies' 查看可用的公司代码")
    print("="*50)
    
    while True:
        print("\n请输入您的问题 (格式: [公司代码] [年份] 问题内容):")
        user_input = input("> ").strip()
        
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("感谢使用，再见！")
            break
        
        if user_input.lower() == "companies":
            companies = rag_system.get_available_companies()
            print(f"\n系统中共有 {len(companies)} 家公司:")
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
        
        print("\n正在查询，请稍候...\n")
        
        if company_code:
            print(f"公司代码: {company_code}")
        if year:
            print(f"年份: {year}")
        print(f"问题: {question}")
        print("-"*50)
        
        # Query RAG system
        start_time = time.time()
        result = rag_system.query(question, company_code, year)
        end_time = time.time()
        
        # Display results
        print("\n回答:")
        print(result["answer"])
        print("\n信息来源:")
        for source in result["sources"]:
            print(f"- {source['company_code']}公司 {source['year']}年 年报")
        
        print(f"\n查询耗时: {end_time - start_time:.2f} 秒")

def main():
    """Main function to run the RAG component"""
    parser = argparse.ArgumentParser(description='RAG component for financial report analysis')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = FinancialReportRAG(config_path=args.config_path)
    
    # Run in interactive mode if specified
    if args.interactive:
        interactive_cli(rag_system)
    else:
        # Print usage information
        print("RAG system initialized.")
        print("Use this module by importing the FinancialReportRAG class.")
        print("For interactive mode, run with --interactive flag.")

if __name__ == "__main__":
    main()
