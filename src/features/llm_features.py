#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Feature Generation Module for FinLLM-Insight
This module generates structured features from annual reports using LLM analysis.
"""

import os
# 设置环境变量以解决 tokenizers 的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import logging
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import concurrent.futures
from datetime import datetime
import hashlib

# For LLM API calls
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

# Initialize LLM response cache
class LLMResponseCache:
    """Cache for LLM responses to avoid repeat API calls"""
    
    def __init__(self, cache_dir="./cache/llm_responses"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, prompt, model):
        """Generate a unique cache key based on prompt and model"""
        key = f"{prompt}_{model}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_cached_response(self, prompt, model):
        """Retrieve cached response if available"""
        cache_key = self.get_cache_key(prompt, model)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def cache_response(self, prompt, model, response):
        """Cache a response"""
        cache_key = self.get_cache_key(prompt, model)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

# Initialize cache
response_cache = LLMResponseCache()

def call_llm_with_retry(client, model, messages, max_retries=3, initial_delay=1):
    """Call LLM API with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )
            return response
        except Exception as e:
            wait_time = initial_delay * (2 ** attempt)
            logger.warning(f"LLM API call attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    logger.error(f"LLM API call failed after {max_retries} attempts")
    raise Exception(f"Failed to call LLM API after {max_retries} attempts")

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

def load_questions(questions_path='config/questions.json'):
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
                "question": "Assess the company's financial health based on the content of the annual report. Consider factors such as revenue growth, profit margins, debt level, cash flow, etc. Analyze whether the company's financial situation is stable and whether there are potential risks. Give a score of 1-10 (10 points are the best) and explain the reasons in detail.",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "business_model": {
                "question": "Based on the content of the annual report, analyze the company's business model and competitive advantages. What is the company's core business? What is its competitive position in the industry? Does the company have a continuous competitive advantage? Give a rating of 1-10 (10 points are the best) and explain the reasons in detail.",
                "type": "numeric",
                "score_range": [1, 10]
            },
            "future_growth": {
                "question": "Based on the content of the annual report, evaluate the company's future growth potential. In which areas does the company invest? What are the market prospects? What are the growth drivers? Give a 1-10 rating (10 points are the best) and explain the reasons in detail.",
                "type": "numeric",
                "score_range": [1, 10]
            }
        }
        
        return default_questions

def connect_to_vector_db(embeddings_dir, embedding_model):
    """
    Connect to vector database and create embedding function
    
    Args:
        embeddings_dir (str): Directory containing embeddings
        embedding_model (str): Name of embedding model to use
        
    Returns:
        tuple: (client, embedding_function)
    """
    try:
        # Load config
        config = load_config('config/config.json')
        
        # Use SentenceTransformerEmbeddingFunction
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Create client with new API
        db_path = os.path.join(embeddings_dir, "chroma_db")
        client = chromadb.PersistentClient(path=db_path)
        
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
    
    # Updated patterns to match scores in both English and Chinese formats
    patterns = [
        r'(?:Score|Rating|评分|分数)[:：]?\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*[/／]\s*10',
        r'(\d+(?:\.\d+)?)\s*分',
        r'(?:give|rate|assign|assess).*?(\d+(?:\.\d+)?)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
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
        # Generate a random score instead of using fixed 5.5
        import random
        return min_score + random.random() * (max_score - min_score)
        
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
        positive_words = ['buy', 'recommend', 'positive', 'opportunity']
        negative_words = ['sell', 'avoid', 'negative', 'risk']
        neutral_words = ['hold', 'neutral', 'wait']
        
        positive_count = sum(response.lower().count(word) for word in positive_words)
        negative_count = sum(response.lower().count(word) for word in negative_words)
        neutral_count = sum(response.lower().count(word) for word in neutral_words)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return "Buy"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "Sell"
        else:
            return "Hold"
    
    return max_category[0]

def query_llm_batch(client, embedding_func, company_code, year, questions, llm_model, max_tokens_per_call=12000):
    """
    Query LLM with multiple questions in a single call
    
    Args:
        client (chromadb.Client): ChromaDB client
        embedding_func: Embedding function
        company_code (str): Company code
        year (str): Report year
        questions (dict): Dictionary of questions
        llm_model (str): LLM model name
        max_tokens_per_call (int): Maximum tokens per LLM call
        
    Returns:
        dict: Dictionary of question keys to responses
    """
    try:
        # Get company collection
        collection_name = f"company_{company_code}"
        collection = client.get_collection(name=collection_name, embedding_function=embedding_func)
        
        # Combine all questions
        combined_question = "Please analyze the following aspects of the company: "
        for key, question_data in questions.items():
            combined_question += f"\n\n{key.upper()}: {question_data['question']}"
        
        # Query for relevant chunks
        results = collection.query(
            query_texts=[combined_question],
            where={"year": year},
            n_results=10
        )
        
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            logger.warning(f"No documents found for {company_code} {year}")
            empty_results = {key: f"No report data found for company {company_code} year {year}." 
                            for key in questions.keys()}
            return empty_results
        
        # Smart selection of relevant chunks
        relevant_chunks = []
        
        for i, doc in enumerate(results['documents'][0]):
            # Calculate relevance score
            if 'distances' in results and results['distances'][0]:
                relevance = 1.0 - min(results['distances'][0][i], 1.0)  # Convert distance to relevance
            else:
                relevance = 1.0  # Default relevance
            
            metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
            
            chunk_info = {
                "content": doc,
                "relevance": relevance,
                "company_code": metadata.get("company_code", company_code),
                "year": metadata.get("year", year),
                "chunk_index": metadata.get("chunk_index", i)
            }
            
            relevant_chunks.append(chunk_info)
        
        # Sort chunks by relevance
        relevant_chunks = sorted(relevant_chunks, key=lambda x: x["relevance"], reverse=True)
        
        # Dynamically adjust text amount
        max_chars = max_tokens_per_call // 2
        current_chars = 0
        selected_chunks = []
        
        for chunk in relevant_chunks:
            if current_chars + len(chunk["content"]) <= max_chars:
                selected_chunks.append(chunk)
                current_chars += len(chunk["content"])
            elif not selected_chunks:  # At least include one chunk
                selected_chunks.append(chunk)
                break
            else:
                break
        
        # Combine relevant chunks
        relevant_text = "\n\n".join(chunk["content"] for chunk in selected_chunks)
        
        # Prepare the prompt
        system_prompt = """You are a professional financial analyst. You will be given multiple questions about a company's annual report.
For each question, provide your answer in a clearly marked section. 
Use the following format for each answer:
QUESTION_KEY: [Your detailed answer]
SCORE: [Numeric score if requested]
CATEGORY: [Category if requested]

Be thorough but concise in your analysis.
"""
        
        user_prompt = f"""Based on the company {company_code}'s {year} annual report sections below:

{relevant_text}

Please answer each of the following questions:

{combined_question}

For each question, follow the format specified in the instructions. Always include SCORE or CATEGORY when applicable.
"""
        
        # Check cache
        cache_key = f"{system_prompt}\n{user_prompt}"
        cached_response = response_cache.get_cached_response(cache_key, llm_model)
        
        if cached_response:
            logger.info(f"Using cached batch response for {company_code} {year}")
            
            # Parse cached response
            parsed_responses = {}
            for question_key in questions.keys():
                pattern = rf"{question_key.upper()}:\s(.*?)(?=\n[A-Z_]+:|$)"
                match = re.search(pattern, cached_response, re.DOTALL)
                if match:
                    parsed_responses[question_key] = match.group(1).strip()
                else:
                    parsed_responses[question_key] = "Failed to extract response for this question."
            
            return parsed_responses
        
        # Call LLM API if not in cache
        if "gpt" in llm_model.lower() or "openai" in llm_model.lower():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            
            # Cache the full response
            response_cache.cache_response(cache_key, llm_model, response_text)
            
            # Parse responses for each question
            parsed_responses = {}
            for question_key in questions.keys():
                pattern = rf"{question_key.upper()}:\s(.*?)(?=\n[A-Z_]+:|$)"
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    parsed_responses[question_key] = match.group(1).strip()
                else:
                    parsed_responses[question_key] = "Failed to extract response for this question."
            
            return parsed_responses
        
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")
    
    except Exception as e:
        logger.error(f"Error in batch query for {company_code} {year}: {e}")
        return {key: f"Error: {str(e)}" for key in questions.keys()}

def process_company(company_code, year, report_path, config):
    """
    Process a single company's report for a specific year
    
    Args:
        company_code (str): Company code
        year (int): Report year
        report_path (str): Path to the report file
        config (dict): Configuration parameters
        
    Returns:
        dict: Features for the company and year
    """
    try:
        # 初始化特征字典
        features = {
            "company_code": company_code,
            "report_year": year,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 加载问题
        questions = load_questions(config.get('questions_path', 'config/questions.json'))
        
        # 连接向量数据库
        embeddings_dir = config.get('embeddings_directory', './data/processed/embeddings')
        llm_model = config.get('llm_model', 'gpt-3.5-turbo-16k')
        embedding_model = config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
        max_tokens_per_call = config.get('max_tokens_per_call', 12000)
        
        client, embedding_func = connect_to_vector_db(embeddings_dir, embedding_model)
        
        # 查询LLM
        responses = query_llm_batch(
            client=client,
            embedding_func=embedding_func,
            company_code=company_code,
            year=str(year),
            questions=questions,
            llm_model=llm_model,
            max_tokens_per_call=max_tokens_per_call
        )
        
        # 处理响应
        for question_key, response in responses.items():
            try:
                # 获取问题详情
                question_data = questions[question_key]
                question_type = question_data.get("type", "numeric")
                
                # 提取结构化数据
                if question_type == "numeric":
                    score_range = question_data.get("score_range", [1, 10])
                    score = extract_score_from_response(response, score_range)
                    features[f"{question_key}_score"] = score
                
                elif question_type == "categorical":
                    categories = question_data.get("categories", [])
                    category = extract_category_from_response(response, categories)
                    features[f"{question_key}_category"] = category
                
            except Exception as e:
                logger.error(f"Error processing response for question {question_key} for {company_code} {year}: {e}")
                continue
        
        return features
    
    except Exception as e:
        logger.error(f"Error processing company {company_code} for year {year}: {e}")
        return None

def generate_features(reports_dir, output_dir, config_path=None):
    """
    Generate features from annual reports using LLM analysis
    
    Args:
        reports_dir (str): Directory containing annual reports
        output_dir (str): Directory to save generated features
        config_path (str): Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证报告目录是否存在
    if not os.path.exists(reports_dir):
        logger.error(f"Reports directory not found: {reports_dir}")
        return None
    
    logger.info(f"Scanning reports directory: {reports_dir}")
    
    # 连接向量数据库
    client, embedding_func = connect_to_vector_db(reports_dir, config.get('embedding_model', 'BAAI/bge-large-zh-v1.5'))
    
    # 获取所有公司集合
    company_collections = get_company_collections(client)
    
    if not company_collections:
        logger.error("No company collections found in vector database")
        return None
    
    # 处理每个公司
    all_features = []
    for collection_name in tqdm(company_collections, desc="Processing companies"):
        try:
            company_code = collection_name.replace("company_", "")
            collection = client.get_collection(name=collection_name, embedding_function=embedding_func)
            
            # 获取所有年份
            years = set()
            for metadata in collection.get()['metadatas']:
                if 'year' in metadata:
                    years.add(int(metadata['year']))
            
            for year in years:
                try:
                    # 生成特征
                    features = process_company(company_code, year, None, config)
                    if features is not None:
                        all_features.append(features)
                
                except Exception as e:
                    logger.error(f"Error processing {company_code} {year}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to process company {company_code}: {e}")
    
    if not all_features:
        logger.error("No features generated for any company")
        return None
    
    # Combine all features
    combined_features = pd.DataFrame(all_features)
    
    # Save combined features
    features_path = os.path.join(output_dir, 'llm_features.csv')
    combined_features.to_csv(features_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Generated features for {len(combined_features)} reports. Saved to {features_path}")
    
    return combined_features

def generate_features_parallel(embeddings_dir, output_dir, llm_model, embedding_model, questions_path=None, max_tokens_per_call=12000, max_workers=4, incremental=True, config_path='config/config.json'):
    """
    Generate features for all companies using LLM analysis with parallel processing
    
    Args:
        embeddings_dir (str): Directory containing vector database
        output_dir (str): Directory to save features
        llm_model (str): LLM model name
        embedding_model (str): Embedding model name
        questions_path (str): Path to questions file
        max_tokens_per_call (int): Maximum tokens per LLM call
        max_workers (int): Maximum number of worker threads
        incremental (bool): Whether to skip already processed entries
        config_path (str): Path to configuration file

    Returns:
        pd.DataFrame: Combined features for all companies
    """
    # Load configuration to get year range
    config = load_config(config_path)
    min_year = config.get('min_year', 2020) # Default to 2020 if not found
    max_year = config.get('max_year', datetime.now().year) # Default to current year if not found
    years_to_process = list(range(min_year, max_year + 1)) # Create list of years based on config

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing features for incremental processing
    existing_features = {}
    features_path = os.path.join(output_dir, "llm_features.csv")
    
    if incremental and os.path.exists(features_path):
        try:
            existing_df = pd.read_csv(features_path, encoding='utf-8-sig')
            # Create index of company and year pairs
            for _, row in existing_df.iterrows():
                 if 'company_code' in row and 'report_year' in row:
                    # Ensure year is stored as string for consistent key format if needed, but usually int is fine
                    key = f"{row['company_code']}_{int(row['report_year'])}"
                    existing_features[key] = True
            logger.info(f"Found {len(existing_features)} existing feature entries for incremental processing")
        except Exception as e:
            logger.error(f"Error loading existing features: {e}")
    
    # Load questions
    questions = load_questions(questions_path)
    
    # Connect to vector database (might be needed by workers)
    # Consider initializing client/embedding_func within the worker if needed, or passing them
    client, embedding_func = connect_to_vector_db(embeddings_dir, embedding_model)

    # Get company collections
    company_collections = get_company_collections(client)

    if not company_collections:
        logger.error("No company collections found in vector database")
        return None

    all_features = []

    def process_company(collection_name):
        """Process a single company and its reports based on config years"""
        company_features = []
        company_code = collection_name.replace("company_", "")
        # Note: Reconnecting to DB within worker might be safer depending on library thread-safety
        local_client, local_embedding_func = connect_to_vector_db(embeddings_dir, embedding_model)

        try:
            # Get collection (still needed for querying)
            try:
                 collection = local_client.get_collection(name=collection_name, embedding_function=local_embedding_func)
            except Exception as e:
                 logger.error(f"Failed to get collection {collection_name} in worker: {e}")
                 return []

            # Process each year based on the config range
            for year in years_to_process:
                # Check if already processed (incremental mode)
                key = f"{company_code}_{year}"
                if incremental and key in existing_features:
                    # Log skipping only once per worker start if desired, or keep as is
                    # logger.info(f"Worker skipping already processed {company_code} {year}")
                    continue

                features = {
                    "company_code": company_code,
                    "report_year": year, # Store the year being processed
                    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                responses = {}

                # Batch process all questions
                batch_questions = {k: v for k, v in questions.items()}

                # Pass year as string to query_llm_batch as it expects string for ChromaDB metadata query
                batch_responses = query_llm_batch(
                    client=local_client,
                    embedding_func=local_embedding_func,
                    company_code=company_code,
                    year=str(year), # Convert year to string for ChromaDB query
                    questions=batch_questions,
                    llm_model=llm_model,
                    max_tokens_per_call=max_tokens_per_call
                )

                # Process responses for each question
                for question_key, response in batch_responses.items():
                    try:
                        # Get question details
                        question_data = questions[question_key]
                        question_type = question_data.get("type", "numeric")

                        # Store full response
                        responses[question_key] = response

                        # Extract structured data
                        if question_type == "numeric":
                            score_range = question_data.get("score_range", [1, 10])
                            score = extract_score_from_response(response, score_range)
                            features[f"{question_key}_score"] = score

                        elif question_type == "categorical":
                            categories = question_data.get("categories", [])
                            category = extract_category_from_response(response, categories)
                            features[f"{question_key}_category"] = category

                    except Exception as e:
                        logger.error(f"Error processing question {question_key} for {company_code} {year}: {e}")

                # Store responses in a separate file
                responses_dir = os.path.join(output_dir, "responses")
                os.makedirs(responses_dir, exist_ok=True)

                responses_file = os.path.join(responses_dir, f"{company_code}_{year}_responses.json")
                with open(responses_file, 'w', encoding='utf-8') as f:
                    json.dump(responses, f, ensure_ascii=False, indent=2)

                # Add to features list
                company_features.append(features)

        except Exception as e:
            logger.error(f"Error processing company {company_code} in worker: {e}")

        return company_features

    # Use thread pool to process companies in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_company = {executor.submit(process_company, collection_name): collection_name 
                            for collection_name in company_collections}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_company), total=len(company_collections), desc="Processing companies"):
            company_name = future_to_company[future]
            try:
                company_features = future.result()
                all_features.extend(company_features)
            except Exception as e:
                logger.error(f"Error getting results for company {company_name}: {e}")
    
    # Handle existing features in incremental mode
    if incremental and os.path.exists(features_path):
        try:
            # Load existing features
            existing_df = pd.read_csv(features_path, encoding='utf-8-sig')
            
            # Only create new DataFrame if we have new features
            if all_features:
                # Create DataFrame with new features
                new_features_df = pd.DataFrame(all_features)
                
                # Combine with existing features
                features_df = pd.concat([existing_df, new_features_df], ignore_index=True)
                
                logger.info(f"Combined {len(existing_df)} existing and {len(new_features_df)} new feature entries")
            else:
                features_df = existing_df
                logger.info("No new features to add, using existing features")
                
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            if all_features:
                features_df = pd.DataFrame(all_features)
            else:
                return None
    else:
        # No existing features or not in incremental mode
        if not all_features:
            logger.error("No features generated for any company")
            return None
        
        features_df = pd.DataFrame(all_features)
    
    # Save features
    features_df.to_csv(features_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Generated features for {len(features_df)} reports. Saved to {features_path}")
    return features_df
    

        
def main():
    """Main function to run the feature generation process"""
    parser = argparse.ArgumentParser(description='Generate features from annual reports using LLM')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--questions_path', type=str, default='config/questions.json',
                        help='Path to questions file')
    parser.add_argument('--parallel', action='store_true', 
                        help='Use parallel processing')
    parser.add_argument('--incremental', action='store_true', 
                        help='Use incremental processing (skip existing entries)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for parallel processing')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Get parameters
    embeddings_dir = config.get('embeddings_directory', './data/processed/embeddings')
    output_dir = config.get('features_directory', './data/processed/features')
    llm_model = config.get('llm_model', 'gpt-3.5-turbo-16k')
    embedding_model = config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
    max_tokens_per_call = config.get('max_tokens_per_call', 12000)
    
    # Generate features
    if args.parallel:
        generate_features_parallel(
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            llm_model=llm_model,
            embedding_model=embedding_model,
            questions_path=args.questions_path,
            max_tokens_per_call=max_tokens_per_call,
            max_workers=args.max_workers,
            incremental=args.incremental,
            config_path=args.config_path
        )
    else:
        generate_features(
            reports_dir=embeddings_dir,
            output_dir=output_dir,
            config_path=args.config_path
        )

if __name__ == "__main__":
    main()
