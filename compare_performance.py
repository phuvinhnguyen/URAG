#!/usr/bin/env python3
"""
Performance Comparison Script

This script compares performance metrics between two evaluation results and computes
relative differences (ratios and absolute differences) for key metrics.

Usage:
    python compare_performance.py result1.json result2.json output.json
    python compare_performance.py --help

Metrics compared:
- accuracy: Classification accuracy
- lac_avg_set_size: LAC average set size 
- aps_avg_set_size: APS average set size
- lac_coverage: LAC coverage rate
- aps_coverage: APS coverage rate
- Combined metrics: averages of LAC/APS metrics
"""

import json
import sys
import argparse
import numpy as np
from typing import Dict, Any
from pathlib import Path


def load_metrics(file_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")


def extract_scores(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant scores from metrics dictionary."""
    scores = {}
    
    # Basic metrics
    scores['accuracy'] = metrics.get('accuracy', 0.0)
    scores['lac_avg_set_size'] = metrics.get('lac_avg_set_size', 0.0)
    scores['aps_avg_set_size'] = metrics.get('aps_avg_set_size', 0.0)
    scores['lac_coverage'] = metrics.get('lac_coverage', 0.0)
    scores['aps_coverage'] = metrics.get('aps_coverage', 0.0)
    
    # Combined metrics
    scores['avg_set_size'] = (scores['lac_avg_set_size'] + scores['aps_avg_set_size']) / 2
    scores['avg_coverage'] = (scores['lac_coverage'] + scores['aps_coverage']) / 2

    scores['lac_set_sizes'] = metrics.get('lac_set_sizes', [])
    scores['aps_set_sizes'] = metrics.get('aps_set_sizes', [])
    scores['set_size'] = ((np.array(scores['lac_set_sizes']) + np.array(scores['aps_set_sizes'])) / 2).tolist()

    return scores


def compute_comparisons(scores1: Dict[str, float], scores2: Dict[str, float]) -> Dict[str, Any]:
    """Compute ratios and differences between two score sets."""
    comparison = {
        'score_1': scores1,
        'score_2': scores2,
        'ratios': {},  # score_1 / score_2
        'differences': {}  # score_1 - score_2
    }
    
    for metric in scores1.keys():
        val1 = scores1[metric]
        val2 = scores2[metric]

        if val1 == None or val2 == None:
            comparison['ratios'][metric] = None
            comparison['differences'][metric] = None
            continue
        
        if isinstance(val1, list) and isinstance(val2, list):
            ratio = (np.array(val1) / np.array(val2)).mean().item()
            difference = (np.array(val1) - np.array(val2)).mean().item()
            comparison['ratios'][metric] = ratio
            comparison['differences'][metric] = difference
        else:
            # Compute ratio (handle division by zero)
            if val2 != 0:
                ratio = val1 / val2
            else:
                ratio = float('inf') if val1 > 0 else 0.0 if val1 == 0 else float('-inf')
            
            # Compute difference
            difference = val1 - val2
            
            comparison['ratios'][metric] = ratio
            comparison['differences'][metric] = difference
    
    return comparison


def format_output(comparison: Dict[str, Any], file1: str, file2: str) -> Dict[str, Any]:
    """Format the final output with metadata."""
    output = {
        'metadata': {
            'description': 'Performance comparison between two evaluation results',
            'file_1': file1,
            'file_2': file2,
            'metrics_compared': list(comparison['score_1'].keys())
        },
        'comparison': comparison
    }
    
    return output


def save_results(output: Dict[str, Any], output_file: str):
    """Save comparison results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison results saved to: {output_file}")


def print_summary(comparison: Dict[str, Any]):
    """Print a human-readable summary."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    scores1 = comparison['score_1']
    scores2 = comparison['score_2']
    ratios = comparison['ratios']
    differences = comparison['differences']
    
    print(f"{'Metric':<20} {'System 1':<12} {'System 2':<12} {'Ratio':<12} {'Difference':<12}")
    print("-" * 70)
    
    for metric in scores1.keys():
        val1 = scores1[metric]
        val2 = scores2[metric]
        ratio = ratios[metric]
        diff = differences[metric]
        
        # Format ratio display
        if ratio == float('inf'):
            ratio_str = "∞"
        elif ratio == float('-inf'):
            ratio_str = "-∞"
        else:
            ratio_str = f"{ratio:.4f}"

        if isinstance(val1, list) and isinstance(val2, list):
            val1 = np.array(val1).mean().item()
            val2 = np.array(val2).mean().item()
        
        print(f"{metric:<20} {val1:<12.4f} {val2:<12.4f} {ratio_str:<12} {diff:<12.4f}")
    
    # Print interpretation if available
    if 'interpretation' in comparison:
        interp = comparison['interpretation']
        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)
        
        for key, value in interp['summary'].items():
            print(f"• {key.title()}: {value}")
        
        print("\nRecommendations:")
        for rec in interp['recommendations']:
            print(f"• {rec}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance metrics between two evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_performance.py results1/metrics.json results2/metrics.json comparison.json
  python compare_performance.py system1_metrics.json system2_metrics.json output.json
        """
    )
    
    parser.add_argument('file1', help='Path to first metrics JSON file')
    parser.add_argument('file2', help='Path to second metrics JSON file')
    parser.add_argument('output', help='Path to output comparison JSON file')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Only save to file, do not print summary')
    
    args = parser.parse_args()
    
    try:
        # Load metrics from both files
        print(f"Loading metrics from {args.file1}")
        metrics1 = load_metrics(args.file1)
        
        print(f"Loading metrics from {args.file2}")
        metrics2 = load_metrics(args.file2)
        
        # Extract scores
        scores1 = extract_scores(metrics1)
        scores2 = extract_scores(metrics2)
        
        # Compute comparisons
        print("Computing comparisons...")
        comparison = compute_comparisons(scores1, scores2)
                
        # Format output
        output = format_output(comparison, args.file1, args.file2)
        
        # Save results
        save_results(output, args.output)
        
        # Print summary unless quiet mode
        if not args.quiet:
            print_summary(comparison)
        
        print(f"\n✓ Comparison completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()