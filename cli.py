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
import yaml
import torch
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


def auto_detect_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Auto-detected device: {device} (GPU: {torch.cuda.get_device_name()})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info(f"Auto-detected device: {device} (Apple Silicon)")
    else:
        device = "cpu"
        logger.info(f"Auto-detected device: {device}")
    
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['system', 'dataset']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file")
    
    # Validate system config
    if 'name' not in config['system']:
        raise ValueError("Missing 'name' field in system configuration")
    
    logger.info(f"Loaded configuration for system: {config['system']['name']}")
    return config


def run_from_config(config_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Run evaluation from YAML configuration file."""
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Extract system configuration
        system_config = config['system']
        system_name = system_config['name']
        system_args = system_config.get('args', {})
        
        # Auto-detect device if not specified
        if 'device' not in system_args:
            system_args['device'] = auto_detect_device()
        
        # Load dataset
        dataset_path = config['dataset']
        logger.info(f"Loading dataset from {dataset_path}")
        data = load_dataset(dataset_path)
        
        # Initialize system
        logger.info(f"Initializing system: {system_name}")
        system = get_system(system_name, **system_args)
        
        # Create pipeline
        pipeline = ConformalEvaluationPipeline(system)
        
        # Get output directory
        output_dir = config.get('output', 'results')
        
        # Save temporary files
        cal_path, test_path = save_temp_files(data, output_dir)
        
        try:
            # Run evaluation
            logger.info("Starting evaluation...")
            
            # Extract alpha from system args or use default
            alpha = system_config.get('alpha', 0.1)
            
            results = pipeline.run_evaluation(
                calibration_data_path=cal_path,
                test_data_path=test_path,
                alpha=alpha,
                output_dir=output_dir
            )
            
            # Print results
            print_results(results)
            
            logger.info("Evaluation completed successfully!")
            return results
            
        finally:
            # Cleanup temporary files
            cleanup_temp_files(cal_path, test_path)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


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
  # Direct command line usage
  python cli.py --system simplellm --dataset datasets/example.json
  python cli.py --system simplellm --dataset datasets/example.json --output results/ --alpha 0.05
  python cli.py --system simplellm --dataset datasets/example.json --model microsoft/DialoGPT-small
  
  # Config file usage
  python cli.py --config config.yaml
  python cli.py --config config.yaml --verbose
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--system', 
        type=str,
        help='System name to evaluate (e.g., simplellm, simplerag)'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str,
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
    
    # Check if using config file or direct CLI
    if args.config:
        # Use config file mode
        run_from_config(args.config, args.verbose)
    else:
        # Use direct CLI mode
        if not args.system or not args.dataset:
            parser.error("--system and --dataset are required when not using --config")
        
        # Configure logging
        if args.verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
        
        # Auto-detect device if set to auto
        device = auto_detect_device() if args.device == 'auto' else args.device
        
        try:
            # Load dataset
            logger.info(f"Loading dataset from {args.dataset}")
            data = load_dataset(args.dataset)
            
            # Initialize system
            logger.info(f"Initializing system: {args.system}")
            system = get_system(args.system, model_name=args.model, device=device)
            
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
