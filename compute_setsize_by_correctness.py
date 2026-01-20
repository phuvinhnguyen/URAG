#!/usr/bin/env python3
"""
Compute setsize (average between aps and lac) on correct and incorrect samples
for each RAG method and simplellm, only from 'normal' sets.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys

# Add parent directory to path to import metrics
sys.path.insert(0, str(Path(__file__).parent))
from metrics import ConformalMetrics

# RAG methods and simplellm
RAG_METHODS = [
    'fidrag',
    'fusionrag',
    'hyderag',
    'raptorrag',
    'ratrag',
    'replugrag',
    'selfrag',
    'simplerag',
    'simplellm'
]

def find_all_evaluation_dirs(base_dir):
    """Find all directories containing evaluation_metrics.json files, only from 'normal' sets."""
    eval_dirs = []
    base_path = Path(base_dir)
    
    for rag_method in RAG_METHODS:
        method_path = base_path / rag_method
        if not method_path.exists():
            continue
        
        # Only process 'normal' subdirectories
        normal_path = method_path / 'normal'
        if not normal_path.exists():
            continue
            
        # Find all evaluation_metrics.json files under 'normal'
        for eval_file in normal_path.rglob('evaluation_metrics.json'):
            eval_dirs.append(eval_file.parent)
    
    return eval_dirs

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

def load_calibration_results(eval_dir):
    """Load calibration results from a directory."""
    cal_file = eval_dir / 'calibration_results.json'
    if not cal_file.exists():
        return None
    
    with open(cal_file, 'r') as f:
        return json.load(f)

def compute_setsize_stats(metrics, test_results):
    """Compute setsize statistics for correct and incorrect samples."""
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
        'std_setsize_correct': np.std(correct_setsizes) if correct_setsizes else 0.0,
        'std_setsize_incorrect': np.std(incorrect_setsizes) if incorrect_setsizes else 0.0,
    }
    
    return stats

def compute_thresholds_from_calibration(calibration_results, alpha=0.1):
    """
    Compute thresholds from calibration results, filtering by correctness.
    Returns thresholds for correct and incorrect cases.
    """
    if calibration_results is None:
        return None, None, None, None
    
    metrics = ConformalMetrics()
    
    # Filter calibration set by correctness
    cal_correct = []
    cal_incorrect = []
    
    for result in calibration_results:
        if 'conformal_probabilities' not in result or not result.get('conformal_probabilities'):
            continue
        
        predicted = result.get('predicted_answer', '')
        correct = result.get('correct_answer', '')
        
        if predicted == correct:
            cal_correct.append(result)
        else:
            cal_incorrect.append(result)
    
    def compute_thresholds(cal_data):
        if not cal_data:
            return None, None
        
        lac_scores = []
        aps_scores = []
        
        for result in cal_data:
            probabilities = result['conformal_probabilities']
            correct_answer = result.get('correct_answer', '')
            
            lac_score = metrics.compute_lac_score(probabilities, correct_answer)
            aps_score = metrics.compute_aps_score(probabilities, correct_answer)
            
            lac_scores.append(lac_score)
            aps_scores.append(aps_score)
        
        if not lac_scores:
            return None, None
        
        n = len(lac_scores)
        q_level = np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1)
        
        lac_threshold = np.quantile(lac_scores, q_level, method='higher')
        aps_threshold = np.quantile(aps_scores, q_level, method='higher')
        
        return lac_threshold, aps_threshold
    
    # Compute thresholds for correct and incorrect cases
    lac_thresh_correct, aps_thresh_correct = compute_thresholds(cal_correct)
    lac_thresh_incorrect, aps_thresh_incorrect = compute_thresholds(cal_incorrect)
    
    return lac_thresh_correct, aps_thresh_correct, lac_thresh_incorrect, aps_thresh_incorrect

def compute_recomputed_setsize_stats(test_results, lac_thresh_correct, aps_thresh_correct, 
                                     lac_thresh_incorrect, aps_thresh_incorrect):
    """
    Recompute conformal prediction on filtered correct/incorrect cases using provided thresholds.
    Filter test sets by correctness, then compute set sizes using the thresholds.
    """
    if test_results is None:
        return None
    
    if lac_thresh_correct is None or aps_thresh_correct is None or \
       lac_thresh_incorrect is None or aps_thresh_incorrect is None:
        return None
    
    metrics = ConformalMetrics()
    
    # Filter test set by correctness
    test_correct = []
    test_incorrect = []
    
    for result in test_results:
        if 'conformal_probabilities' not in result or not result.get('conformal_probabilities'):
            continue
        
        predicted = result.get('predicted_answer', '')
        correct = result.get('correct_answer', '')
        
        if predicted == correct:
            test_correct.append(result)
        else:
            test_incorrect.append(result)
    
    # Compute set sizes and coverage on filtered test sets
    def compute_set_sizes_and_coverage(test_data, lac_thresh, aps_thresh):
        if not test_data or lac_thresh is None or aps_thresh is None:
            return [], [], 0.0, 0.0
        
        lac_set_sizes = []
        aps_set_sizes = []
        lac_coverage_count = 0
        aps_coverage_count = 0
        
        for result in test_data:
            probabilities = result['conformal_probabilities']
            correct_answer = result.get('correct_answer', '')
            
            lac_pred_set = metrics.compute_prediction_set_lac(probabilities, lac_thresh)
            aps_pred_set = metrics.compute_prediction_set_aps(probabilities, aps_thresh)
            
            lac_set_sizes.append(len(lac_pred_set))
            aps_set_sizes.append(len(aps_pred_set))
            
            # Check coverage: if correct answer is in prediction set
            if correct_answer in lac_pred_set:
                lac_coverage_count += 1
            if correct_answer in aps_pred_set:
                aps_coverage_count += 1
        
        total_samples = len(test_data)
        lac_coverage = lac_coverage_count / total_samples if total_samples > 0 else 0.0
        aps_coverage = aps_coverage_count / total_samples if total_samples > 0 else 0.0
        
        return lac_set_sizes, aps_set_sizes, lac_coverage, aps_coverage
    
    # Compute set sizes and coverage for correct and incorrect cases
    lac_sizes_correct, aps_sizes_correct, lac_cov_correct, aps_cov_correct = compute_set_sizes_and_coverage(
        test_correct, lac_thresh_correct, aps_thresh_correct)
    lac_sizes_incorrect, aps_sizes_incorrect, lac_cov_incorrect, aps_cov_incorrect = compute_set_sizes_and_coverage(
        test_incorrect, lac_thresh_incorrect, aps_thresh_incorrect)
    
    # Compute average setsize (mean of APS and LAC)
    setsizes_correct = [(aps_sizes_correct[i] + lac_sizes_correct[i]) / 2.0 
                       for i in range(len(aps_sizes_correct))] if aps_sizes_correct else []
    setsizes_incorrect = [(aps_sizes_incorrect[i] + lac_sizes_incorrect[i]) / 2.0 
                         for i in range(len(aps_sizes_incorrect))] if aps_sizes_incorrect else []
    
    # Compute average coverage (mean of APS and LAC)
    avg_coverage_correct = (lac_cov_correct + aps_cov_correct) / 2.0
    avg_coverage_incorrect = (lac_cov_incorrect + aps_cov_incorrect) / 2.0
    
    return {
        'avg_setsize_correct_recomputed': np.mean(setsizes_correct) if setsizes_correct else 0.0,
        'avg_setsize_incorrect_recomputed': np.mean(setsizes_incorrect) if setsizes_incorrect else 0.0,
        'coverage_correct_recomputed': avg_coverage_correct,
        'coverage_incorrect_recomputed': avg_coverage_incorrect,
        'lac_coverage_correct_recomputed': lac_cov_correct,
        'aps_coverage_correct_recomputed': aps_cov_correct,
        'lac_coverage_incorrect_recomputed': lac_cov_incorrect,
        'aps_coverage_incorrect_recomputed': aps_cov_incorrect,
        'num_correct_samples_recomputed': len(setsizes_correct),
        'num_incorrect_samples_recomputed': len(setsizes_incorrect),
    }

def main():
    base_dir = '/media/volume/LLMRag/URAG/results_full_recomputed'
    
    # Find all evaluation directories (only from 'normal' sets)
    print("Finding all evaluation directories (normal sets only)...")
    eval_dirs = find_all_evaluation_dirs(base_dir)
    print(f"Found {len(eval_dirs)} evaluation directories from 'normal' sets")
    
    # Aggregate results by RAG method
    method_stats = defaultdict(list)
    
    # Process each evaluation directory
    for eval_dir in eval_dirs:
        # Extract RAG method from path
        parts = eval_dir.parts
        rag_method_idx = -1
        for i, part in enumerate(parts):
            if part in RAG_METHODS:
                rag_method_idx = i
                break
        
        if rag_method_idx == -1:
            continue
        
        rag_method = parts[rag_method_idx]
        
        # Load data
        metrics = load_evaluation_metrics(eval_dir)
        test_results = load_test_results(eval_dir)
        calibration_results = load_calibration_results(eval_dir)
        
        if metrics is None or test_results is None:
            continue
        
        # Compute stats (original method)
        stats = compute_setsize_stats(metrics, test_results)
        if stats is None:
            continue
        
        # Store calibration and test results for later processing
        stats['calibration_results'] = calibration_results
        stats['test_results'] = test_results
        
        # Extract benchmark identifier (llm and dataset)
        # Path structure: .../normal/{llm}/{dataset}/
        benchmark_id = None
        if 'normal' in parts:
            normal_idx = parts.index('normal')
            if normal_idx + 2 < len(parts):
                llm = parts[normal_idx + 1]
                dataset = parts[normal_idx + 2]
                benchmark_id = (llm, dataset)
        
        stats['benchmark_id'] = benchmark_id
        stats['rag_method'] = rag_method
        stats['path'] = str(eval_dir)
        method_stats[rag_method].append(stats)
    
    # Group experiments by benchmark
    # For each benchmark, compute simplerag thresholds and apply to all methods
    print("\nComputing simplerag thresholds per benchmark and applying to all methods...")
    
    # Get all unique benchmarks
    all_benchmarks = set()
    for rag_method in method_stats.keys():
        for stats in method_stats[rag_method]:
            if stats.get('benchmark_id'):
                all_benchmarks.add(stats['benchmark_id'])
    
    print(f"Found {len(all_benchmarks)} unique benchmarks")
    
    # Process each benchmark
    for benchmark_id in sorted(all_benchmarks):
        llm, dataset = benchmark_id
        print(f"\nProcessing benchmark: {llm}/{dataset}")
        
        # Find simplerag calibration data for this benchmark
        simplerag_cal = None
        for rag_method in method_stats.keys():
            for stats in method_stats[rag_method]:
                if stats.get('benchmark_id') == benchmark_id and stats.get('rag_method') == 'simplerag':
                    simplerag_cal = stats.get('calibration_results')
                    break
            if simplerag_cal is not None:
                break
        
        if simplerag_cal is None:
            print(f"  Warning: No simplerag calibration data found for {llm}/{dataset}, skipping")
            continue
        
        # Compute simplerag thresholds for this benchmark
        simplerag_lac_thresh_correct, simplerag_aps_thresh_correct, \
        simplerag_lac_thresh_incorrect, simplerag_aps_thresh_incorrect = \
            compute_thresholds_from_calibration(simplerag_cal)
        
        if simplerag_lac_thresh_correct is None or simplerag_aps_thresh_correct is None or \
           simplerag_lac_thresh_incorrect is None or simplerag_aps_thresh_incorrect is None:
            print(f"  Warning: Could not compute thresholds for {llm}/{dataset}, skipping")
            continue
        
        print(f"  Thresholds - Correct: LAC={simplerag_lac_thresh_correct:.4f}, APS={simplerag_aps_thresh_correct:.4f}")
        print(f"  Thresholds - Incorrect: LAC={simplerag_lac_thresh_incorrect:.4f}, APS={simplerag_aps_thresh_incorrect:.4f}")
        
        # Apply these thresholds to all methods for this benchmark
        for rag_method in method_stats.keys():
            for stats in method_stats[rag_method]:
                if stats.get('benchmark_id') != benchmark_id:
                    continue
                
                test_results = stats.get('test_results')
                if test_results is None:
                    continue
                
                # Use simplerag thresholds for all methods in this benchmark
                recomputed_stats = compute_recomputed_setsize_stats(
                    test_results,
                    simplerag_lac_thresh_correct,
                    simplerag_aps_thresh_correct,
                    simplerag_lac_thresh_incorrect,
                    simplerag_aps_thresh_incorrect
                )
                
                if recomputed_stats:
                    stats.update(recomputed_stats)
    
    # Clean up stored data
    for rag_method in method_stats.keys():
        for stats in method_stats[rag_method]:
            if 'calibration_results' in stats:
                del stats['calibration_results']
            if 'test_results' in stats:
                del stats['test_results']
            if 'benchmark_id' in stats:
                del stats['benchmark_id']
    
    # Aggregate statistics per RAG method
    print("\n" + "="*80)
    print("SETSIZE STATISTICS BY RAG METHOD")
    print("="*80)
    
    final_results = {}
    
    for rag_method in RAG_METHODS:
        if rag_method not in method_stats:
            continue
        
        all_stats = method_stats[rag_method]
        
        # Aggregate across all experiments for this method
        total_samples = sum(s['total_samples'] for s in all_stats)
        total_correct = sum(s['correct_samples'] for s in all_stats)
        total_incorrect = sum(s['incorrect_samples'] for s in all_stats)
        
        # Weighted average setsize (original method)
        weighted_all = sum(s['avg_setsize_all'] * s['total_samples'] 
                          for s in all_stats if s['total_samples'] > 0)
        weighted_correct = sum(s['avg_setsize_correct'] * s['correct_samples'] 
                              for s in all_stats if s['correct_samples'] > 0)
        weighted_incorrect = sum(s['avg_setsize_incorrect'] * s['incorrect_samples'] 
                                for s in all_stats if s['incorrect_samples'] > 0)
        
        avg_setsize_all = weighted_all / total_samples if total_samples > 0 else 0.0
        avg_setsize_correct = weighted_correct / total_correct if total_correct > 0 else 0.0
        avg_setsize_incorrect = weighted_incorrect / total_incorrect if total_incorrect > 0 else 0.0
        
        # Average recomputed setsize and coverage (across all benchmarks)
        recomputed_stats_list = [s for s in all_stats if 'avg_setsize_correct_recomputed' in s]
        if recomputed_stats_list:
            # Simple average across benchmarks (not weighted by sample count)
            avg_setsize_correct_recomputed = np.mean([s['avg_setsize_correct_recomputed'] 
                                                     for s in recomputed_stats_list 
                                                     if s['avg_setsize_correct_recomputed'] > 0])
            avg_setsize_incorrect_recomputed = np.mean([s['avg_setsize_incorrect_recomputed'] 
                                                       for s in recomputed_stats_list 
                                                       if s['avg_setsize_incorrect_recomputed'] > 0])
            coverage_correct_recomputed = np.mean([s['coverage_correct_recomputed'] 
                                                  for s in recomputed_stats_list 
                                                  if 'coverage_correct_recomputed' in s])
            coverage_incorrect_recomputed = np.mean([s['coverage_incorrect_recomputed'] 
                                                     for s in recomputed_stats_list 
                                                     if 'coverage_incorrect_recomputed' in s])
        else:
            avg_setsize_correct_recomputed = 0.0
            avg_setsize_incorrect_recomputed = 0.0
            coverage_correct_recomputed = 0.0
            coverage_incorrect_recomputed = 0.0
        
        final_results[rag_method] = {
            'avg_setsize_all': avg_setsize_all,
            'avg_setsize_correct': avg_setsize_correct,
            'avg_setsize_incorrect': avg_setsize_incorrect,
            'avg_setsize_correct_recomputed': avg_setsize_correct_recomputed,
            'avg_setsize_incorrect_recomputed': avg_setsize_incorrect_recomputed,
            'coverage_correct_recomputed': coverage_correct_recomputed,
            'coverage_incorrect_recomputed': coverage_incorrect_recomputed,
            'total_samples': total_samples,
            'total_correct_samples': total_correct,
            'total_incorrect_samples': total_incorrect,
            'num_experiments': len(all_stats)
        }
        
        print(f"\n{rag_method.upper()}:")
        print(f"  Average Setsize (All):                    {avg_setsize_all:.4f}")
        print(f"  Average Setsize (Correct):                {avg_setsize_correct:.4f}")
        print(f"  Average Setsize (Incorrect):              {avg_setsize_incorrect:.4f}")
        print(f"  Average Setsize (Correct, Recomputed):   {avg_setsize_correct_recomputed:.4f}")
        print(f"  Average Setsize (Incorrect, Recomputed):  {avg_setsize_incorrect_recomputed:.4f}")
        print(f"  Coverage (Correct, Recomputed):           {coverage_correct_recomputed:.4f}")
        print(f"  Coverage (Incorrect, Recomputed):         {coverage_incorrect_recomputed:.4f}")
        print(f"  Total Samples:                            {total_samples}")
        print(f"  Total Correct Samples:                    {total_correct}")
        print(f"  Total Incorrect Samples:                  {total_incorrect}")
        print(f"  Number of Experiments:                    {len(all_stats)}")
    
    # Save results to JSON
    output_file = '/media/volume/LLMRag/URAG/setsize_by_correctness.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'RAG Method':<20} {'Avg Setsize (All)':<20} {'Avg Setsize (Correct)':<25} {'Avg Setsize (Incorrect)':<25}")
    print("-"*100)
    for rag_method in sorted(final_results.keys()):
        result = final_results[rag_method]
        print(f"{rag_method:<20} {result['avg_setsize_all']:<20.4f} {result['avg_setsize_correct']:<25.4f} {result['avg_setsize_incorrect']:<25.4f}")
    
    # Print recomputed summary table
    print("\n" + "="*80)
    print("RECOMPUTED SUMMARY TABLE (Filtered Conformal Prediction)")
    print("="*80)
    print(f"{'RAG Method':<20} {'Avg Setsize (Correct)':<25} {'Avg Setsize (Incorrect)':<25} {'Coverage (Correct)':<20} {'Coverage (Incorrect)':<20}")
    print("-"*120)
    for rag_method in sorted(final_results.keys()):
        result = final_results[rag_method]
        print(f"{rag_method:<20} {result['avg_setsize_correct_recomputed']:<25.4f} {result['avg_setsize_incorrect_recomputed']:<25.4f} {result['coverage_correct_recomputed']:<20.4f} {result['coverage_incorrect_recomputed']:<20.4f}")

if __name__ == '__main__':
    main()
