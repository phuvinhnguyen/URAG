#!/usr/bin/env python3
"""
Command-line interface for URAG evaluation system.

Usage:
    python cli.py --system simplellm --dataset datasets/example.json
    python cli.py --system simplellm --dataset datasets/example.json --output results/ --alpha 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from mc_evaluation import ConformalEvaluationPipeline
from systems import get_system


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load and validate dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate dataset format
    if not isinstance(data, dict):
        raise ValueError("Dataset must be a JSON object with 'calibration' and 'test' keys")
    
    if 'calibration' not in data or 'test' not in data:
        raise ValueError("Dataset must contain 'calibration' and 'test' keys")
    
    cal_data = data['calibration']
    test_data = data['test']
    
    if not isinstance(cal_data, list) or not isinstance(test_data, list):
        raise ValueError("Both 'calibration' and 'test' must be arrays")
    
    # Validate sample format
    required_fields = ['id', 'question', 'options', 'correct_answer']
    for split_name, split_data in [('calibration', cal_data), ('test', test_data)]:
        for i, sample in enumerate(split_data):
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Missing required field '{field}' in {split_name} sample {i}")
    
    logger.info(f"Loaded dataset: {len(cal_data)} calibration, {len(test_data)} test samples")
    return data


def save_temp_files(data: Dict[str, Any], output_dir: str) -> tuple[str, str]:
    """Save calibration and test data to temporary files."""
    os.makedirs(output_dir, exist_ok=True)
    
    cal_path = os.path.join(output_dir, "temp_calibration.json")
    test_path = os.path.join(output_dir, "temp_test.json")
    
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump(data['calibration'], f, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(data['test'], f, indent=2)
    
    return cal_path, test_path


def cleanup_temp_files(cal_path: str, test_path: str):
    """Remove temporary files."""
    try:
        if os.path.exists(cal_path):
            os.remove(cal_path)
        if os.path.exists(test_path):
            os.remove(test_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files: {e}")


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way."""
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"LAC Coverage: {metrics['lac_coverage']:.4f}")
    print(f"APS Coverage: {metrics['aps_coverage']:.4f}")
    print(f"LAC Avg Set Size: {metrics['lac_avg_set_size']:.4f}")
    print(f"APS Avg Set Size: {metrics['aps_avg_set_size']:.4f}")
    
    print(f"\nThresholds:")
    print(f"  LAC Threshold: {metrics['thresholds']['lac_threshold']:.4f}")
    print(f"  APS Threshold: {metrics['thresholds']['aps_threshold']:.4f}")
    
    print(f"\nOutput Files:")
    for file_type, file_path in results['output_files'].items():
        print(f"  {file_type.title()}: {file_path}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG systems with conformal prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --system simplellm --dataset datasets/example.json
  python cli.py --system simplellm --dataset datasets/example.json --output results/ --alpha 0.05
  python cli.py --system simplellm --dataset datasets/example.json --model microsoft/DialoGPT-small
        """
    )
    
    parser.add_argument(
        '--system', 
        type=str, 
        required=True,
        help='System name to evaluate (e.g., simplellm, simplerag)'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to dataset JSON file'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--alpha', 
        type=float, 
        default=0.1,
        help='Conformal prediction error rate (default: 0.1 for 90%% coverage)'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='microsoft/DialoGPT-small',
        help='Model name for LLM-based systems (default: microsoft/DialoGPT-small)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use: auto, cpu, cuda (default: auto)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}")
        data = load_dataset(args.dataset)
        
        # Initialize system
        logger.info(f"Initializing system: {args.system}")
        system = get_system(args.system, model_name=args.model, device=args.device)
        
        # Create pipeline
        pipeline = ConformalEvaluationPipeline(system)
        
        # Save temporary files
        cal_path, test_path = save_temp_files(data, args.output)
        
        try:
            # Run evaluation
            logger.info("Starting evaluation...")
            results = pipeline.run_evaluation(
                calibration_data_path=cal_path,
                test_data_path=test_path,
                alpha=args.alpha,
                output_dir=args.output
            )
            
            # Print results
            print_results(results)
            
            logger.info("Evaluation completed successfully!")
            
        finally:
            # Cleanup temporary files
            cleanup_temp_files(cal_path, test_path)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
