#!/usr/bin/env python3
"""
Compute LLM-only (simplellm) metrics for SciFact benchmark:
- Set size (correct and incorrect) from results_full_recomputed (no recomputation)
- Accuracy (correct and incorrect) from results_full
"""

import json
import os
from pathlib import Path
import numpy as np

# Base directories
BASE_DIR_RECOMPUTED = Path('/media/volume/LLMRag/URAG/results_full_recomputed')
BASE_DIR_FULL = Path('/media/volume/LLMRag/URAG/results_full')

def load_evaluation_metrics(eval_dir):
    """Load evaluation metrics from a directory."""
    metrics_file = eval_dir / 'evaluation_metrics.json'
    if not metrics_file.exists():
        return None
    with open(metrics_file, 'r') as f:
        return json.load(f)

def load_test_results(eval_dir):
    """Load test results from a directory."""
    test_file = eval_dir / 'test_results.json'
    if not test_file.exists():
        return None
    with open(test_file, 'r') as f:
        return json.load(f)

def compute_accuracy(test_results):
    """Compute accuracy for correct and incorrect cases."""
    if test_results is None or len(test_results) == 0:
        return None
    
    correct_samples = []
    incorrect_samples = []
    
    for result in test_results:
        predicted = result.get('predicted_answer', '')
        correct = result.get('correct_answer', '')
        is_correct = (predicted == correct)
        
        if is_correct:
            correct_samples.append(1)
        else:
            incorrect_samples.append(1)
    
    total = len(test_results)
    correct_count = len(correct_samples)
    incorrect_count = len(incorrect_samples)
    
    accuracy_overall = correct_count / total if total > 0 else 0.0
    
    return {
        'total_samples': total,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'accuracy_overall': accuracy_overall,
        'accuracy_correct': 1.0,  # By definition, if prediction is correct, accuracy is 1.0
        'accuracy_incorrect': 0.0  # By definition, if prediction is incorrect, accuracy is 0.0
    }

def compute_setsize_stats(metrics, test_results):
    """Compute set size statistics for correct and incorrect samples (no recomputation)."""
    if metrics is None or test_results is None:
        return None
    
    lac_set_sizes = metrics.get('lac_set_sizes', [])
    aps_set_sizes = metrics.get('aps_set_sizes', [])
    
    if len(lac_set_sizes) != len(aps_set_sizes) or len(lac_set_sizes) != len(test_results):
        print(f"Warning: Mismatch in array lengths. LAC: {len(lac_set_sizes)}, APS: {len(aps_set_sizes)}, Test results: {len(test_results)}")
        return None
    
    # Compute setsize for each sample (average of aps and lac)
    setsizes = []
    is_correct = []
    
    for i, result in enumerate(test_results):
        predicted = result.get('predicted_answer', '')
        correct = result.get('correct_answer', '')
        
        # Compute setsize as average of aps and lac
        setsize = (aps_set_sizes[i] + lac_set_sizes[i]) / 2.0
        setsizes.append(setsize)
        is_correct.append(predicted == correct)
    
    # Separate correct and incorrect samples
    correct_setsizes = [setsizes[i] for i in range(len(setsizes)) if is_correct[i]]
    incorrect_setsizes = [setsizes[i] for i in range(len(setsizes)) if not is_correct[i]]
    
    stats = {
        'total_samples': len(setsizes),
        'correct_samples': len(correct_setsizes),
        'incorrect_samples': len(incorrect_setsizes),
        'avg_setsize_all': np.mean(setsizes) if setsizes else 0.0,
        'avg_setsize_correct': np.mean(correct_setsizes) if correct_setsizes else 0.0,
        'avg_setsize_incorrect': np.mean(incorrect_setsizes) if incorrect_setsizes else 0.0,
    }
    
    return stats

def main():
    """Main function to compute LLM-only metrics for SciFact."""
    
    # Find all SciFact directories for simplellm
    scifact_dirs_recomputed = []
    scifact_dirs_full = []
    
    # From results_full_recomputed
    simplellm_recomputed = BASE_DIR_RECOMPUTED / 'simplellm' / 'normal'
    if simplellm_recomputed.exists():
        for llm_dir in simplellm_recomputed.iterdir():
            if llm_dir.is_dir():
                for dataset_dir in llm_dir.iterdir():
                    if dataset_dir.is_dir() and 'scifact' in dataset_dir.name.lower():
                        scifact_dirs_recomputed.append((llm_dir.name, dataset_dir))
    
    # From results_full
    simplellm_full = BASE_DIR_FULL / 'simplellm' / 'normal'
    if simplellm_full.exists():
        for llm_dir in simplellm_full.iterdir():
            if llm_dir.is_dir():
                for dataset_dir in llm_dir.iterdir():
                    if dataset_dir.is_dir() and 'scifact' in dataset_dir.name.lower():
                        scifact_dirs_full.append((llm_dir.name, dataset_dir))
    
    print("=" * 80)
    print("LLM-Only (simplellm) Metrics for SciFact Benchmark")
    print("=" * 80)
    print()
    
    # Collect results
    all_results = []
    
    for llm_name, eval_dir_recomputed in scifact_dirs_recomputed:
        # Find corresponding full directory
        eval_dir_full = None
        for llm_name_full, dataset_dir_full in scifact_dirs_full:
            if llm_name_full == llm_name:
                eval_dir_full = dataset_dir_full
                break
        
        print(f"Processing: {llm_name} / {eval_dir_recomputed.name}")
        
        # Load data from recomputed (for set size)
        metrics = load_evaluation_metrics(eval_dir_recomputed)
        test_results_recomputed = load_test_results(eval_dir_recomputed)
        
        # Load data from full (for accuracy)
        test_results_full = None
        if eval_dir_full:
            test_results_full = load_test_results(eval_dir_full)
        
        if metrics is None or test_results_recomputed is None:
            print(f"  Warning: Missing data in {eval_dir_recomputed}")
            continue
        
        # Compute set size stats (from recomputed)
        setsize_stats = compute_setsize_stats(metrics, test_results_recomputed)
        
        # Compute accuracy stats (from full)
        accuracy_stats = None
        if test_results_full:
            accuracy_stats = compute_accuracy(test_results_full)
        elif test_results_recomputed:
            # Fallback to recomputed if full not available
            accuracy_stats = compute_accuracy(test_results_recomputed)
        
        if setsize_stats and accuracy_stats:
            result = {
                'llm': llm_name,
                'dataset': eval_dir_recomputed.name,
                'setsize_correct': setsize_stats['avg_setsize_correct'],
                'setsize_incorrect': setsize_stats['avg_setsize_incorrect'],
                'setsize_all': setsize_stats['avg_setsize_all'],
                'accuracy_overall': accuracy_stats['accuracy_overall'],
                'correct_samples': setsize_stats['correct_samples'],
                'incorrect_samples': setsize_stats['incorrect_samples'],
                'total_samples': setsize_stats['total_samples']
            }
            all_results.append(result)
            
            print(f"  Set Size - Correct: {result['setsize_correct']:.4f}")
            print(f"  Set Size - Incorrect: {result['setsize_incorrect']:.4f}")
            print(f"  Set Size - All: {result['setsize_all']:.4f}")
            print(f"  Accuracy - Overall: {result['accuracy_overall']:.4f}")
            print(f"  Samples - Correct: {result['correct_samples']}, Incorrect: {result['incorrect_samples']}, Total: {result['total_samples']}")
            print()
    
    # Compute average across LLMs
    if all_results:
        print("=" * 80)
        print("Average across all LLMs:")
        print("=" * 80)
        
        avg_setsize_correct = np.mean([r['setsize_correct'] for r in all_results])
        avg_setsize_incorrect = np.mean([r['setsize_incorrect'] for r in all_results])
        avg_setsize_all = np.mean([r['setsize_all'] for r in all_results])
        avg_accuracy = np.mean([r['accuracy_overall'] for r in all_results])
        
        print(f"Set Size - Correct: {avg_setsize_correct:.4f}")
        print(f"Set Size - Incorrect: {avg_setsize_incorrect:.4f}")
        print(f"Set Size - All: {avg_setsize_all:.4f}")
        print(f"Accuracy - Overall: {avg_accuracy:.4f}")
        
        # Save results
        output = {
            'benchmark': 'scifact',
            'method': 'simplellm',
            'per_llm': all_results,
            'average': {
                'setsize_correct': avg_setsize_correct,
                'setsize_incorrect': avg_setsize_incorrect,
                'setsize_all': avg_setsize_all,
                'accuracy_overall': avg_accuracy
            }
        }
        
        output_file = Path('/media/volume/LLMRag/URAG/scifact_llm_only_results.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
