#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Processing Pipeline for FinLLM-Insight
This module integrates all components of the system for end-to-end execution.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
import concurrent.futures
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
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

def run_module(module_path, args=None, check_result=True):
    """
    Run a Python module as a subprocess
    
    Args:
        module_path (str): Path to Python module
        args (list): Command line arguments
        check_result (bool): Whether to check return code
        
    Returns:
        bool: Success status
    """
    import subprocess
    
    if args is None:
        args = []
    
    cmd = [sys.executable, module_path] + args
    cmd_str = " ".join(cmd)
    
    logger.info(f"Running: {cmd_str}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=check_result)
        end_time = time.time()
        
        logger.info(f"Command completed in {end_time - start_time:.2f} seconds")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def ensure_directories(config):
    """
    Ensure all required directories exist
    
    Args:
        config (dict): Configuration dictionary
    """
    directories = [
        config.get('annual_reports_html_save_directory', './data/raw/annual_reports'),
        config.get('processed_reports_text_directory', './data/processed/text_reports'),
        config.get('targets_directory', './data/processed/targets'),
        config.get('embeddings_directory', './data/processed/embeddings'),
        config.get('features_directory', './data/processed/features'),
        config.get('models_directory', './models'),
        config.get('predictions_directory', './predictions')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def run_data_download(config_path, parallel=False):
    """
    Run the data download step
    
    Args:
        config_path (str): Path to config file
        parallel (bool): Whether to run in parallel mode
        
    Returns:
        bool: Success status
    """
    logger.info("STEP 1: Downloading annual reports")
    
    # Get module path
    module_path = os.path.join("src", "data", "download_reports.py")
    
    # Prepare arguments
    args = ["--config_path", config_path]
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("Annual reports download completed successfully")
    else:
        logger.error("Annual reports download failed")
    
    return success

def run_data_conversion(config_path, parallel=False):
    """
    Run the data conversion step
    
    Args:
        config_path (str): Path to config file
        parallel (bool): Whether to run in parallel mode
        
    Returns:
        bool: Success status
    """
    logger.info("STEP 2: Converting reports to text")
    
    # Get module path
    module_path = os.path.join("src", "data", "convert_formats.py")
    
    # Prepare arguments
    args = ["--config_path", config_path]
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("Report conversion completed successfully")
    else:
        logger.error("Report conversion failed")
    
    return success

def run_target_generation(config_path, parallel=False):
    """
    Run the target generation step
    
    Args:
        config_path (str): Path to config file
        parallel (bool): Whether to run in parallel mode
        
    Returns:
        bool: Success status
    """
    logger.info("STEP 3: Generating target variables")
    
    # Get module path
    module_path = os.path.join("src", "data", "make_targets.py")
    
    # Prepare arguments
    args = ["--config_path", config_path]
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("Target generation completed successfully")
    else:
        logger.error("Target generation failed")
    
    return success

def run_embeddings_generation(config_path, parallel=False):
    """
    Run the embeddings generation step
    
    Args:
        config_path (str): Path to config file
        parallel (bool): Whether to run in parallel mode
        
    Returns:
        bool: Success status
    """
    logger.info("STEP 4: Generating embeddings")
    
    # Get module path
    module_path = os.path.join("src", "features", "embeddings.py")
    
    # Prepare arguments
    args = ["--config_path", config_path]
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("Embeddings generation completed successfully")
    else:
        logger.error("Embeddings generation failed")
    
    return success

def run_feature_generation(config_path, parallel=False):
    """
    Run the LLM feature generation step
    
    Args:
        config_path (str): Path to config file
        parallel (bool): Whether to run in parallel mode
        
    Returns:
        bool: Success status
    """
    logger.info("STEP 5: Generating LLM features")
    
    # Get module path
    module_path = os.path.join("src", "features", "llm_features.py")
    
    # Prepare arguments
    args = ["--config_path", config_path]
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("LLM feature generation completed successfully")
    else:
        logger.error("LLM feature generation failed")
    
    return success

def run_model_training(config_path, model_type="regression", model_name="random_forest", target_window=None):
    """
    Run the model training step
    
    Args:
        config_path (str): Path to config file
        model_type (str): Type of model to train
        model_name (str): Name of model algorithm
        target_window (int): Target prediction window
        
    Returns:
        bool: Success status
    """
    logger.info(f"STEP 6: Training {model_type} model with {model_name}")
    
    # Get module path
    module_path = os.path.join("src", "models", "train.py")
    
    # Prepare arguments
    args = [
        "--config_path", config_path,
        "--model_type", model_type,
        "--model_name", model_name
    ]
    
    if target_window is not None:
        args.extend(["--target_window", str(target_window)])
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")
    
    return success

def run_prediction(config_path, model_type="regression", target_window=None):
    """
    Run the prediction step
    
    Args:
        config_path (str): Path to config file
        model_type (str): Type of model to use
        target_window (int): Target prediction window
        
    Returns:
        bool: Success status
    """
    logger.info(f"STEP 7: Making predictions with {model_type} model")
    
    # Get module path
    module_path = os.path.join("src", "models", "predict.py")
    
    # Prepare arguments
    args = [
        "--config_path", config_path,
        "--model_type", model_type
    ]
    
    if target_window is not None:
        args.extend(["--target_window", str(target_window)])
    
    # Execute module
    success = run_module(module_path, args)
    
    if success:
        logger.info("Prediction completed successfully")
    else:
        logger.error("Prediction failed")
    
    return success

def run_complete_pipeline(config_path, skip_steps=None, parallel=False):
    """
    Run the complete processing pipeline
    
    Args:
        config_path (str): Path to config file
        skip_steps (list): List of step numbers to skip
        parallel (bool): Whether to run steps in parallel when possible
        
    Returns:
        bool: Overall success status
    """
    if skip_steps is None:
        skip_steps = []
    
    logger.info("Starting FinLLM-Insight complete pipeline")
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Get parameters from config
    target_window = config.get('target_window', 60)
    model_type = config.get('model_type', 'regression')
    model_name = config.get('model_name', 'random_forest')
    
    # Track step results
    results = {}
    
    # Step 1: Download annual reports
    if 1 not in skip_steps:
        results[1] = run_data_download(config_path, parallel)
    else:
        logger.info("STEP 1: Skipped annual reports download")
        results[1] = True
    
    # Step 2: Convert reports to text
    if 2 not in skip_steps and results.get(1, True):
        results[2] = run_data_conversion(config_path, parallel)
    else:
        if 2 in skip_steps:
            logger.info("STEP 2: Skipped report conversion")
            results[2] = True
        else:
            logger.warning("STEP 2: Skipped due to previous step failure")
            results[2] = False
    
    # Step 3: Generate target variables
    if 3 not in skip_steps:
        # This step can run in parallel with steps 2 and 4
        if parallel and (2 not in skip_steps):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_target_generation, config_path, parallel)
                
                # Continue with step 2 while step 3 runs
                if not results.get(2, False):
                    results[2] = run_data_conversion(config_path, parallel)
                
                # Wait for step 3 to complete
                results[3] = future.result()
        else:
            results[3] = run_target_generation(config_path, parallel)
    else:
        logger.info("STEP 3: Skipped target generation")
        results[3] = True
    
    # Step 4: Generate embeddings
    if 4 not in skip_steps and results.get(2, True):
        results[4] = run_embeddings_generation(config_path, parallel)
    else:
        if 4 in skip_steps:
            logger.info("STEP 4: Skipped embeddings generation")
            results[4] = True
        else:
            logger.warning("STEP 4: Skipped due to previous step failure")
            results[4] = False
    
    # Step 5: Generate LLM features
    if 5 not in skip_steps and results.get(4, True):
        results[5] = run_feature_generation(config_path, parallel)
    else:
        if 5 in skip_steps:
            logger.info("STEP 5: Skipped LLM feature generation")
            results[5] = True
        else:
            logger.warning("STEP 5: Skipped due to previous step failure")
            results[5] = False
    
    # Step 6: Train model
    if 6 not in skip_steps and results.get(3, True) and results.get(5, True):
        results[6] = run_model_training(config_path, model_type, model_name, target_window)
    else:
        if 6 in skip_steps:
            logger.info("STEP 6: Skipped model training")
            results[6] = True
        else:
            logger.warning("STEP 6: Skipped due to previous step failure")
            results[6] = False
    
    # Step 7: Make predictions
    if 7 not in skip_steps and results.get(5, True) and results.get(6, True):
        results[7] = run_prediction(config_path, model_type, target_window)
    else:
        if 7 in skip_steps:
            logger.info("STEP 7: Skipped prediction")
            results[7] = True
        else:
            logger.warning("STEP 7: Skipped due to previous step failure")
            results[7] = False
    
    # Create step descriptions for summary
    step_descriptions = {
        1: "下载年报数据",
        2: "转换报告格式",
        3: "生成目标变量",
        4: "生成文本嵌入",
        5: "生成LLM特征",
        6: "训练预测模型",
        7: "生成预测结果"
    }
    
    # Calculate overall success
    all_success = all(results.values())
    completed_steps = sum(1 for step, success in results.items() if success)
    total_steps = len(results)
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
    
    # Create execution summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "overall_success": all_success,
        "elapsed_time": elapsed_time,
        "elapsed_time_str": time_str,
        "step_results": {step_descriptions[step]: "成功" if success else "失败" 
                         for step, success in results.items()}
    }
    
    # Print summary
    logger.info("Pipeline execution summary:")
    logger.info(f"Timestamp: {summary['timestamp']}")
    logger.info(f"Completed {completed_steps}/{total_steps} steps")
    logger.info(f"Overall status: {'成功' if all_success else '失败'}")
    logger.info(f"Elapsed time: {time_str}")
    logger.info("Step results:")
    
    for step_desc, status in summary['step_results'].items():
        logger.info(f"  {step_desc}: {status}")
    
    # Save summary to file
    summary_dir = "logs"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")
    
    return all_success

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Run the complete FinLLM-Insight pipeline')
    parser.add_argument('--config_path', type=str, default='config/config.json', 
                        help='Path to configuration file')
    parser.add_argument('--skip_steps', type=str, default='',
                        help='Comma-separated list of step numbers to skip (e.g., "1,3,5")')
    parser.add_argument('--parallel', action='store_true',
                        help='Run steps in parallel when possible')
    parser.add_argument('--step', type=int, default=0,
                        help='Run a specific step (0 for all steps)')
    args = parser.parse_args()
    
    # Parse skip steps
    skip_steps = []
    if args.skip_steps:
        try:
            skip_steps = [int(s.strip()) for s in args.skip_steps.split(',')]
        except ValueError:
            logger.error(f"Invalid --skip_steps format: {args.skip_steps}. Use comma-separated numbers.")
            return 1
    
    try:
        if args.step == 0:
            # Run the complete pipeline
            success = run_complete_pipeline(args.config_path, skip_steps, args.parallel)
        else:
            # Run a specific step
            if args.step == 1:
                success = run_data_download(args.config_path, args.parallel)
            elif args.step == 2:
                success = run_data_conversion(args.config_path, args.parallel)
            elif args.step == 3:
                success = run_target_generation(args.config_path, args.parallel)
            elif args.step == 4:
                success = run_embeddings_generation(args.config_path, args.parallel)
            elif args.step == 5:
                success = run_feature_generation(args.config_path, args.parallel)
            elif args.step == 6:
                # Load configuration for model parameters
                config = load_config(args.config_path)
                model_type = config.get('model_type', 'regression')
                model_name = config.get('model_name', 'random_forest')
                target_window = config.get('target_window', 60)
                success = run_model_training(args.config_path, model_type, model_name, target_window)
            elif args.step == 7:
                # Load configuration for model parameters
                config = load_config(args.config_path)
                model_type = config.get('model_type', 'regression')
                target_window = config.get('target_window', 60)
                success = run_prediction(args.config_path, model_type, target_window)
            else:
                logger.error(f"Invalid step number: {args.step}. Must be between 0 and 7.")
                return 1
        
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Pipeline execution failed with error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
