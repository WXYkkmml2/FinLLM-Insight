#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Feature Generation Module for FinLLM-Insight
This module generates structured features from annual reports using LLM analysis.
"""

import os
import json
import argparse
import logging
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime

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
        logging.FileHandler("llm_features.log"),
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

def load_questions(questions_path='./questions.json'):
    """
    Load questions for LLM analysis
    
    Args:
        questions_path (str): Path to questions JSON file
        
    Returns:
        dict: Questions for LLM analysis
    """
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        
        # Default questions if file not found
        default_questions = {
            "financial_health": {
                "question": "基于年报内容，评估公司的财务健康状况。考虑收入增长、利润率、债务水平、现金流等因素。分析公司的财务状况是否稳健，是否存在潜在风险。给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "business_model": {
                "question": "基于年报内容，分析公司的商业模式和竞争优势。公司的核心业务是什么？它在行业中的竞争地位如何？公司是否有持续的竞争优势？给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "future_growth": {
                "question": "基于年报内容，评估公司的未来增长潜力。公司在哪些领域投资？市场前景如何？有哪些增长驱动因素？给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "management_quality": {
                "question": "基于年报内容，评价公司管理层的质量。管理层的战略决策是否合理？执行能力如何？是否有良好的公司治理？给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "risk_assessment": {
                "question": "基于年报内容，评估公司面临的主要风险。包括市场风险、经营风险、财务风险、法律和合规风险等。这些风险的严重程度如何？公司应对风险的措施是否充分？给出1-10的评分（1分表示风险极高，10分表示风险极低）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "industry_outlook": {
                "question": "基于年报内容，分析公司所处行业的前景。行业发展趋势如何？有哪些机遇和挑战？公司在行业中的定位如何？给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "esg_performance": {
                "question": "基于年报内容，评估公司在环境、社会和治理(ESG)方面的表现。公司是否有可持续发展战略？社会责任表现如何？公司治理结构是否健全？给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "innovation_capability": {
                "question": "基于年报内容，评价公司的创新能力。公司在研发方面的投入如何？有哪些创新成果？创新对公司业务的影响如何？给出1-10的评分（10分为最佳）并详细解释理由。",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "investment_recommendation": {
                "question": "基于年报内容和以上分析，给出对该公司股票的投资建议。考虑公司基本面、估值水平、增长前景和风险因素。你的建议是强烈买入、买入、持有、卖出还是强烈卖出？请详细解释理由。",
                "type": "categorical",
                "categories": ["强烈卖出", "卖出", "持有", "买入", "强烈买入"]
            }
        }
        
        return default_questions

def connect_to_vector_db(embeddings_dir, embedding_model):
    """
    Connect to vector database
    
    Args:
        embeddings_dir (str): Directory containing vector database
        embedding_model (str): Name of the embedding model
        
    Returns:
        chromadb.Client: ChromaDB client
    """
    try:
        # Create embedding function
        if embedding_model.lower() == "openai":
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
                model_name=embedding_model
            )
            embedding_func = huggingface_ef
        
        # Create client
        db_path = os.path.join(embeddings_dir, "chroma_db")
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path
        ))
        
        return client, embedding_func
    
    except Exception as e:
        logger.error(f"Failed to connect to vector database: {e}")
        raise

def get_company_collections(client):
    """
    Get all company collections from vector database
    
    Args:
        client (chromadb.Client): ChromaDB client
        
    Returns:
        list: List of collection names
    """
    try:
        collections = client.list_collections()
        company_collections = [c.name for c in collections if c.name.startswith("company_")]
        return company_collections
    
    except Exception as e:
        logger.error(f"Failed to get company collections: {e}")
        return []

def extract_score_from_response(response, score_range=None):
    """
    Extract numeric score from LLM response
    
    Args:
        response (str): LLM response text
        score_range (list): Valid score range [min, max]
        
    Returns:
        float: Extracted score
    """
    # Default score range
    if score_range is None:
        score_range = [1, 10]
    
    min_score, max_score = score_range
    
    # Pattern to match scores like "Score: 7" or "评分：7" or just "7/10"
    patterns = [
        r'(?:Score|评分|得分|分数)[:：]?\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*[/／]\s*10',
        r'(\d+(?:\.\d+)?)\s*分'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                score = float(matches[0])
                # Ensure score is within range
                score = max(min_score, min(score, max_score))
                return score
            except (ValueError, IndexError):
                continue
    
    # If no score found, try to infer from text
    positive_words = ['excellent', 'good', 'strong', 'positive', '优秀', '良好', '强劲', '积极']
    negative_words = ['poor', 'weak', 'negative', 'bad', '差', '弱', '消极', '不良']
    
    positive_count = sum(response.lower().count(word) for word in positive_words)
    negative_count = sum(response.lower().count(word) for word in negative_words)
    
    if positive_count > negative_count:
        return (max_score + min_score) * 0.75  # Above average
    elif negative_count > positive_count:
        return (max_score + min_score) * 0.25  # Below average
    else:
        return (max_score + min_score) / 2  # Average
    
def extract_category_from_response(response, categories):
    """
    Extract categorical value from LLM response
    
    Args:
        response (str): LLM response text
        categories (list): Valid categories
        
    Returns:
        str: Extracted category
    """
    # Count occurrences of each category in the response
    counts = {category: response.count(category) for category in categories}
    
    # Return category with highest count
    max_category = max(counts.items(), key=lambda x: x[1])
    
    # If no category found or all have zero count, try to infer
    if max_category[1] == 0:
        positive_words = ['buy', 'recommend', 'positive', 'opportunity', '买入', '推荐', '机会']
        negative_words = ['sell', 'avoid', 'negative', 'risk', '卖出', '避免', '风险']
        neutral_words = ['hold', 'neutral', 'wait', '持有', '中性', '等待']
        
        positive_count = sum(response.lower().count(word) for word in positive_words)
        negative_count = sum(response.lower().count(word) for word in negative_words)
        neutral_count = sum(response.lower().count(word) for word in neutral_words)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return "买入"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "卖出"
        else:
            return "持有"
    
    return max_category[0]

def query_llm(client, embedding_func, company_code, year, question, llm_model, max_tokens_per_call=12000):
    """
    Query LLM with company report data
    
    Args:
        client (chromadb.Client): ChromaDB client
        embedding_func: Embedding function
        company_code (str): Company code
        year (str): Report year
        question (str): Question for LLM
        llm_model (str): LLM model name
        max_tokens_per_call (int): Maximum tokens per LLM call
        
    Returns:
        str: LLM response
    """
    try:
        # Get company collection
        collection_name = f"company_{company_code}"
        collection = client.get_collection(name=collection_name, embedding_function=embedding_func)
        
        # Query for relevant chunks
        results = collection.query(
            query_texts=[question],
            where={"year": year},
            n_results=10
        )
        
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            logger.warning(f"No documents found for {company_code} {year}")
            return f"没有找到{company_code} {year}年的报告数据。"
        
        # Combine relevant chunks
        relevant_text = "\n\n".join(results['documents'][0])
        
        # Limit text length to avoid exceeding token limits
        if len(relevant_text) > max_tokens_per_call * 2:  # Rough character to token ratio
            relevant_text = relevant_text[:max_tokens_per_call * 2]
        
        # Prepare prompt
        system_prompt = f"""你是一位专业的财务分析师，擅长分析上市公司年度报告。
现在，你需要基于提供的年度报告片段（来自{company_code}公司{year}年的年报），回答一个具体问题。
请只基于提供的年报信息进行分析，不要使用你可能知道的其他信息。
如果年报中没有足够的信息来回答问题，请明确指出，而不是猜测或使用外部知识。
请确保分析全面、客观，注意同时考虑正面和负面因素。
"""
        
        user_prompt = f"""以下是{company_code}公司{year}年年度报告的相关片段：

{relevant_text}

问题：{question}

请基于以上年报内容回答问题。要求：
1. 分析要全面、客观、具体，避免泛泛而谈
2. 如果是评分题，必须给出明确的分数，并详细解释评分理由
3. 如果是分类题，必须从给定选项中选择一个，并详细解释选择理由
4. 只基于提供的年报信息，不引入外部信息
"""
        
        # Call LLM API
        if "gpt" in llm_model.lower() or "openai" in llm_model.lower():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system",
