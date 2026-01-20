#!/usr/bin/env python3
"""
Generate detailed reports for each benchmark with conformal prediction metrics.
Computes set sizes, coverage rates, and accuracy for each RAG method per benchmark.
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

def find_benchmark_dirs(base_dir_recomputed, base_dir_full):
    """Find all benchmark directories from both recomputed and full results."""
    benchmarks = defaultdict(lambda: {'recomputed': {}, 'full': {}})
    
    # Process recomputed results (for conformal prediction metrics)
    base_path_recomputed = Path(base_dir_recomputed)
    for rag_method in RAG_METHODS:
        method_path = base_path_recomputed / rag_method / 'normal'
        if not method_path.exists():
            continue
        
        for eval_file in method_path.rglob('evaluation_metrics.json'):
            eval_dir = eval_file.parent
            parts = eval_dir.parts
            
            # Extract benchmark: .../normal/{llm}/{dataset}/
            if 'normal' in parts:
                normal_idx = parts.index('normal')
                if normal_idx + 2 < len(parts):
                    llm = parts[normal_idx + 1]
                    dataset = parts[normal_idx + 2]
                    benchmark_id = (llm, dataset)
                    benchmarks[benchmark_id]['recomputed'][rag_method] = eval_dir
    
    # Process full results (for accuracy)
    base_path_full = Path(base_dir_full)
    for rag_method in RAG_METHODS:
        method_path = base_path_full / rag_method / 'normal'
        if not method_path.exists():
            continue
        
        for eval_file in method_path.rglob('evaluation_metrics.json'):
            eval_dir = eval_file.parent
            parts = eval_dir.parts
            
            # Extract benchmark: .../normal/{llm}/{dataset}/
            if 'normal' in parts:
                normal_idx = parts.index('normal')
                if normal_idx + 2 < len(parts):
                    llm = parts[normal_idx + 1]
                    dataset = parts[normal_idx + 2]
                    benchmark_id = (llm, dataset)
                    benchmarks[benchmark_id]['full'][rag_method] = eval_dir
    
    return benchmarks

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
    """Compute accuracy from test results."""
    if not test_results:
        return 0.0, 0, 0
    
    correct_count = 0
    total_count = 0
    
    for result in test_results:
        predicted = result.get('predicted_answer', '')
        correct = result.get('correct_answer', '')
        
        if predicted == correct:
            correct_count += 1
        total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy, correct_count, total_count

def compute_conformal_metrics(metrics, test_results):
    """
    Compute conformal prediction metrics: set sizes and coverage for all, correct, and incorrect cases.
    Compares with evaluation_metrics.json values.
    """
    if metrics is None or test_results is None:
        return None
    
    lac_set_sizes = metrics.get('lac_set_sizes', [])
    aps_set_sizes = metrics.get('aps_set_sizes', [])
    
    if len(lac_set_sizes) != len(aps_set_sizes) or len(lac_set_sizes) != len(test_results):
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
    
    # Separate by correctness
    correct_setsizes = [setsizes[i] for i in range(len(setsizes)) if is_correct[i]]
    incorrect_setsizes = [setsizes[i] for i in range(len(setsizes)) if not is_correct[i]]
    
    # Compute coverage (from evaluation_metrics.json)
    lac_coverage = metrics.get('lac_coverage', 0.0)
    aps_coverage = metrics.get('aps_coverage', 0.0)
    avg_coverage = (lac_coverage + aps_coverage) / 2.0
    
    # Compute coverage for correct and incorrect cases separately
    # We need to check if correct answer is in prediction set
    metrics_obj = ConformalMetrics()
    lac_threshold = metrics.get('thresholds', {}).get('lac_threshold', None)
    aps_threshold = metrics.get('thresholds', {}).get('aps_threshold', None)
    
    lac_cov_correct = 0
    aps_cov_correct = 0
    lac_cov_incorrect = 0
    aps_cov_incorrect = 0
    correct_count = 0
    incorrect_count = 0
    
    if lac_threshold is not None and aps_threshold is not None:
        for i, result in enumerate(test_results):
            if 'conformal_probabilities' not in result or not result.get('conformal_probabilities'):
                continue
            
            probabilities = result['conformal_probabilities']
            correct_answer = result.get('correct_answer', '')
            predicted = result.get('predicted_answer', '')
            
            lac_pred_set = metrics_obj.compute_prediction_set_lac(probabilities, lac_threshold)
            aps_pred_set = metrics_obj.compute_prediction_set_aps(probabilities, aps_threshold)
            
            if predicted == correct_answer:
                correct_count += 1
                if correct_answer in lac_pred_set:
                    lac_cov_correct += 1
                if correct_answer in aps_pred_set:
                    aps_cov_correct += 1
            else:
                incorrect_count += 1
                if correct_answer in lac_pred_set:
                    lac_cov_incorrect += 1
                if correct_answer in aps_pred_set:
                    aps_cov_incorrect += 1
    
    coverage_correct = (lac_cov_correct + aps_cov_correct) / (2.0 * correct_count) if correct_count > 0 else 0.0
    coverage_incorrect = (lac_cov_incorrect + aps_cov_incorrect) / (2.0 * incorrect_count) if incorrect_count > 0 else 0.0
    
    # Get values from evaluation_metrics.json for comparison
    eval_avg_setsize_all = (metrics.get('lac_avg_set_size', 0.0) + metrics.get('aps_avg_set_size', 0.0)) / 2.0
    
    # Compare computed setsize with evaluation_metrics.json
    setsize_diff = abs(np.mean(setsizes) - eval_avg_setsize_all) if setsizes else 0.0
    
    return {
        'setsize_all': np.mean(setsizes) if setsizes else 0.0,
        'setsize_correct': np.mean(correct_setsizes) if correct_setsizes else 0.0,
        'setsize_incorrect': np.mean(incorrect_setsizes) if incorrect_setsizes else 0.0,
        'coverage_all': avg_coverage,
        'coverage_correct': coverage_correct,
        'coverage_incorrect': coverage_incorrect,
        'eval_setsize_all': eval_avg_setsize_all,  # From evaluation_metrics.json
        'setsize_diff': setsize_diff,  # Difference between computed and eval
        'num_samples': len(setsizes),
        'num_correct': len(correct_setsizes),
        'num_incorrect': len(incorrect_setsizes),
        'lac_coverage': lac_coverage,
        'aps_coverage': aps_coverage,
        'lac_avg_set_size': metrics.get('lac_avg_set_size', 0.0),
        'aps_avg_set_size': metrics.get('aps_avg_set_size', 0.0),
    }

def generate_benchmark_report(benchmark_id, benchmark_data, output_dir):
    """Generate a detailed report for a single benchmark."""
    llm, dataset = benchmark_id
    
    report = {
        'benchmark': {
            'llm': llm,
            'dataset': dataset
        },
        'methods': {}
    }
    
    # Process each method
    for rag_method in RAG_METHODS:
        recomputed_dir = benchmark_data['recomputed'].get(rag_method)
        full_dir = benchmark_data['full'].get(rag_method)
        
        if recomputed_dir is None:
            continue
        
        method_data = {
            'method': rag_method,
            'accuracy': None,
            'conformal_metrics': None,
            'has_data': False
        }
        
        # Load data from recomputed (for conformal metrics)
        metrics = load_evaluation_metrics(recomputed_dir)
        test_results_recomputed = load_test_results(recomputed_dir)
        
        # Load data from full (for accuracy)
        test_results_full = None
        if full_dir is not None:
            test_results_full = load_test_results(full_dir)
        
        if metrics is not None and test_results_recomputed is not None:
            # Compute conformal metrics
            conformal_metrics = compute_conformal_metrics(metrics, test_results_recomputed)
            method_data['conformal_metrics'] = conformal_metrics
            method_data['has_data'] = True
            
            # Compute accuracy (use full results if available, otherwise recomputed)
            if test_results_full is not None:
                accuracy, correct_count, total_count = compute_accuracy(test_results_full)
                method_data['accuracy'] = {
                    'value': accuracy,
                    'correct_count': correct_count,
                    'total_count': total_count
                }
            else:
                # Fallback to recomputed if full not available
                accuracy, correct_count, total_count = compute_accuracy(test_results_recomputed)
                method_data['accuracy'] = {
                    'value': accuracy,
                    'correct_count': correct_count,
                    'total_count': total_count
                }
        
        report['methods'][rag_method] = method_data
    
    # Save report
    output_file = output_dir / f"{llm}_{dataset}_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {llm} / {dataset}")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Accuracy':<12} {'Setsize (All)':<15} {'Setsize (Cor)':<15} {'Setsize (Inc)':<15} "
          f"{'Cov (All)':<12} {'Cov (Cor)':<12} {'Cov (Inc)':<12} {'Eval Setsize':<15} {'Diff':<10}")
    print("-"*140)
    
    for rag_method in sorted(report['methods'].keys()):
        method_data = report['methods'][rag_method]
        if not method_data['has_data']:
            continue
        
        acc = method_data['accuracy']['value'] if method_data['accuracy'] else 0.0
        conf = method_data['conformal_metrics']
        
        if conf:
            print(f"{rag_method:<15} {acc:<12.4f} {conf['setsize_all']:<15.4f} {conf['setsize_correct']:<15.4f} "
                  f"{conf['setsize_incorrect']:<15.4f} {conf['coverage_all']:<12.4f} "
                  f"{conf['coverage_correct']:<12.4f} {conf['coverage_incorrect']:<12.4f} "
                  f"{conf['eval_setsize_all']:<15.4f} {conf['setsize_diff']:<10.4f}")
    
    return report

def main():
    base_dir_recomputed = '/media/volume/LLMRag/URAG/results_full_recomputed'
    base_dir_full = '/media/volume/LLMRag/URAG/results_full'
    output_dir = Path('/media/volume/LLMRag/URAG/benchmark_reports')
    output_dir.mkdir(exist_ok=True)
    
    print("Finding all benchmarks...")
    benchmarks = find_benchmark_dirs(base_dir_recomputed, base_dir_full)
    print(f"Found {len(benchmarks)} unique benchmarks")
    
    all_reports = {}
    
    # Generate report for each benchmark
    for benchmark_id in sorted(benchmarks.keys()):
        llm, dataset = benchmark_id
        benchmark_data = benchmarks[benchmark_id]
        
        report = generate_benchmark_report(benchmark_id, benchmark_data, output_dir)
        all_reports[f"{llm}/{dataset}"] = report
    
    # Save summary report
    summary_file = output_dir / 'all_benchmarks_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    # Print per-benchmark summary table (organized by benchmark)
    print(f"\n\n{'='*180}")
    print("PER-BENCHMARK SUMMARY: Method Performance on Each Benchmark")
    print(f"{'='*180}")
    
    # Group by benchmark and show all methods
    for benchmark_key in sorted(all_reports.keys()):
        report = all_reports[benchmark_key]
        llm = report['benchmark']['llm']
        dataset = report['benchmark']['dataset']
        
        print(f"\n{'='*180}")
        print(f"BENCHMARK: {llm} / {dataset}")
        print(f"{'='*180}")
        print(f"{'Method':<15} {'Accuracy':<12} {'Setsize (Cor)':<18} {'Setsize (Inc)':<18} "
              f"{'Cov (Cor)':<15} {'Cov (Inc)':<15} {'Setsize (All)':<18} {'Cov (All)':<15}")
        print("-"*180)
        
        # Show all methods for this benchmark
        for rag_method in sorted(RAG_METHODS):
            method_data = report['methods'].get(rag_method)
            
            if method_data is None or not method_data.get('has_data'):
                print(f"{rag_method:<15} {'N/A':<12} {'N/A':<18} {'N/A':<18} {'N/A':<15} {'N/A':<15} {'N/A':<18} {'N/A':<15}")
                continue
            
            conf = method_data.get('conformal_metrics')
            acc = method_data.get('accuracy', {}).get('value', 0.0)
            
            if conf:
                print(f"{rag_method:<15} {acc:<12.4f} {conf['setsize_correct']:<18.4f} {conf['setsize_incorrect']:<18.4f} "
                      f"{conf['coverage_correct']:<15.4f} {conf['coverage_incorrect']:<15.4f} "
                      f"{conf['setsize_all']:<18.4f} {conf['coverage_all']:<15.4f}")
    
    # Create CSV files organized by benchmark
    csv_file = output_dir / 'per_benchmark_summary.csv'
    with open(csv_file, 'w') as f:
        # Header
        f.write("Benchmark,Method,Accuracy,Setsize_Correct,Setsize_Incorrect,Coverage_Correct,Coverage_Incorrect,Setsize_All,Coverage_All\n")
        
        for benchmark_key in sorted(all_reports.keys()):
            report = all_reports[benchmark_key]
            benchmark_name = f"{report['benchmark']['llm']}/{report['benchmark']['dataset']}"
            
            for rag_method in sorted(RAG_METHODS):
                method_data = report['methods'].get(rag_method)
                
                if method_data is None or not method_data.get('has_data'):
                    f.write(f"{benchmark_name},{rag_method},N/A,N/A,N/A,N/A,N/A,N/A,N/A\n")
                    continue
                
                conf = method_data.get('conformal_metrics')
                acc = method_data.get('accuracy', {}).get('value', 0.0)
                
                if conf:
                    f.write(f"{benchmark_name},{rag_method},{acc:.4f},"
                           f"{conf['setsize_correct']:.4f},{conf['setsize_incorrect']:.4f},"
                           f"{conf['coverage_correct']:.4f},{conf['coverage_incorrect']:.4f},"
                           f"{conf['setsize_all']:.4f},{conf['coverage_all']:.4f}\n")
    
    # Also create a pivot-style CSV (benchmarks as rows, methods as columns)
    pivot_csv_file = output_dir / 'benchmark_pivot_tables.csv'
    
    # Create separate pivot tables for each metric
    metrics_to_pivot = ['Accuracy', 'Setsize_Correct', 'Setsize_Incorrect', 'Coverage_Correct', 'Coverage_Incorrect', 'Setsize_All', 'Coverage_All']
    
    with open(pivot_csv_file, 'w') as f:
        for metric in metrics_to_pivot:
            f.write(f"\n=== {metric} ===\n")
            f.write("Benchmark," + ",".join(sorted(RAG_METHODS)) + "\n")
            
            for benchmark_key in sorted(all_reports.keys()):
                report = all_reports[benchmark_key]
                benchmark_name = f"{report['benchmark']['llm']}/{report['benchmark']['dataset']}"
                
                values = []
                for rag_method in sorted(RAG_METHODS):
                    method_data = report['methods'].get(rag_method)
                    
                    if method_data is None or not method_data.get('has_data'):
                        values.append("N/A")
                        continue
                    
                    conf = method_data.get('conformal_metrics')
                    acc = method_data.get('accuracy', {}).get('value', 0.0)
                    
                    if metric == 'Accuracy':
                        values.append(f"{acc:.4f}")
                    elif metric == 'Setsize_Correct':
                        values.append(f"{conf['setsize_correct']:.4f}" if conf else "N/A")
                    elif metric == 'Setsize_Incorrect':
                        values.append(f"{conf['setsize_incorrect']:.4f}" if conf else "N/A")
                    elif metric == 'Coverage_Correct':
                        values.append(f"{conf['coverage_correct']:.4f}" if conf else "N/A")
                    elif metric == 'Coverage_Incorrect':
                        values.append(f"{conf['coverage_incorrect']:.4f}" if conf else "N/A")
                    elif metric == 'Setsize_All':
                        values.append(f"{conf['setsize_all']:.4f}" if conf else "N/A")
                    elif metric == 'Coverage_All':
                        values.append(f"{conf['coverage_all']:.4f}" if conf else "N/A")
                
                f.write(f"{benchmark_name}," + ",".join(values) + "\n")
            f.write("\n")
    
    print(f"\n\nAll reports saved to: {output_dir}")
    print(f"Summary JSON saved to: {summary_file}")
    print(f"CSV summary saved to: {csv_file}")

if __name__ == '__main__':
    main()
