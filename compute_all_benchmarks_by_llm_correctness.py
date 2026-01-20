#!/usr/bin/env python3
"""
Compute RAG metrics for all benchmarks, filtered by LLM-only correctness.
- Accuracy from results_full
- Conformal prediction set size from results_full_recomputed (recomputed from scratch)
- Filter samples based on whether LLM (simplellm) got them correct or incorrect
- Generate LaTeX table with set size data
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

# RAG methods (excluding fidrag)
RAG_METHODS = [
    'fusionrag',
    'hyderag',
    'raptorrag',
    'ratrag',
    'replugrag',
    'selfrag',
    'simplerag',
    'simplellm'  # For identifying LLM correctness
]

# Base directories
BASE_DIR_FULL = Path('/media/volume/LLMRag/URAG/results_full')
BASE_DIR_FULL_RECOMPUTED = Path('/media/volume/LLMRag/URAG/results_full_recomputed')

# Method name mapping
METHOD_NAME_MAP = {
    'fusionrag': 'Fusion',
    'hyderag': 'HyDE',
    'raptorrag': 'RAPTOR',
    'ratrag': 'RAT',
    'replugrag': 'REPLUG',
    'selfrag': 'Self-RAG',
    'simplerag': 'Naive',
    'simplellm': 'LLM-Only'
}

# Benchmark name mapping for LaTeX
BENCHMARK_NAME_MAP = {
    'healthver_mcqa_0.1': 'Healthver',
    'scifact_mcqa_0.1': 'SciFact',
    'odex_0.1': 'Odex',
    'commit_message_qa_0.1': 'LCA',
    'crag_task_1_and_2_mcqa_0.1': 'CRAG',
    'multinewsum_mcqa_0.1': 'NewsSum',
    'dialfact_0.1': 'DialFact',
    'olympiadbench_0.1': 'Olympiad',
    'dialfact_wrong_context_0.1': 'DialFact-WC',
    'odex_wrong_context_0.1': 'Odex-WC',
    'crag_task_1_and_2_mcqa_example_0.1': 'CRAG-Ex',
    'crag_task_1_and_2_mcqa_tiny_0.1': 'CRAG-Tiny'
}

# Benchmark categories and order for the table
BENCHMARK_CATEGORIES = {
    'Healthcare': ['healthver_mcqa_0.1'],
    'Code': ['odex_0.1', 'commit_message_qa_0.1'],
    'Research': ['scifact_mcqa_0.1'],
    'Math': ['olympiadbench_0.1'],
    'General Text': ['crag_task_1_and_2_mcqa_0.1', 'multinewsum_mcqa_0.1', 'dialfact_0.1']
}

# Ordered list of benchmarks as they appear in the table
TABLE_BENCHMARK_ORDER = [
    'healthver_mcqa_0.1',  # Healthcare
    'odex_0.1',            # Code
    'commit_message_qa_0.1',  # Code (LCA)
    'scifact_mcqa_0.1',    # Research
    'olympiadbench_0.1',   # Math
    'crag_task_1_and_2_mcqa_0.1',  # General Text (CRAG)
    'multinewsum_mcqa_0.1',  # General Text (NewsSum)
    'dialfact_0.1'         # General Text (DialFact)
]

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

def compute_thresholds_from_calibration(calibration_results, alpha=0.1):
    """
    Compute thresholds from calibration results.
    Returns thresholds for all calibration data (not split by correctness).
    """
    if calibration_results is None or len(calibration_results) == 0:
        return None, None
    
    metrics = ConformalMetrics()
    
    lac_scores = []
    aps_scores = []
    
    for result in calibration_results:
        if 'conformal_probabilities' not in result or not result.get('conformal_probabilities'):
            continue
        
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

def recompute_setsize_by_llm_correctness(test_results_recomputed, calibration_results_recomputed, llm_correctness_mask):
    """
    Recompute conformal prediction set size from scratch, filtered by LLM correctness.
    
    Args:
        test_results_recomputed: Test results from results_full_recomputed
        calibration_results_recomputed: Calibration results from results_full_recomputed
        llm_correctness_mask: List of booleans indicating LLM correctness for each sample
    
    Returns:
        dict with setsize_correct, setsize_incorrect, setsize_all
    """
    if test_results_recomputed is None or len(test_results_recomputed) == 0:
        return None
    
    if len(test_results_recomputed) != len(llm_correctness_mask):
        return None
    
    # Compute thresholds from calibration data
    lac_threshold, aps_threshold = compute_thresholds_from_calibration(calibration_results_recomputed)
    
    if lac_threshold is None or aps_threshold is None:
        return None
    
    metrics = ConformalMetrics()
    
    # Filter test results by LLM correctness
    test_correct = []
    test_incorrect = []
    
    for i, result in enumerate(test_results_recomputed):
        if 'conformal_probabilities' not in result or not result.get('conformal_probabilities'):
            continue
        
        if llm_correctness_mask[i]:
            test_correct.append(result)
        else:
            test_incorrect.append(result)
    
    def compute_set_sizes(test_data):
        """Compute set sizes for a set of test data."""
        if not test_data:
            return []
        
        set_sizes = []
        for result in test_data:
            probabilities = result['conformal_probabilities']
            
            lac_pred_set = metrics.compute_prediction_set_lac(probabilities, lac_threshold)
            aps_pred_set = metrics.compute_prediction_set_aps(probabilities, aps_threshold)
            
            # Average set size (mean of APS and LAC)
            avg_set_size = (len(lac_pred_set) + len(aps_pred_set)) / 2.0
            set_sizes.append(avg_set_size)
        
        return set_sizes
    
    # Compute set sizes for correct and incorrect cases
    setsizes_correct = compute_set_sizes(test_correct)
    setsizes_incorrect = compute_set_sizes(test_incorrect)
    
    return {
        'setsize_correct': np.mean(setsizes_correct) if setsizes_correct else None,
        'setsize_incorrect': np.mean(setsizes_incorrect) if setsizes_incorrect else None,
        'num_correct': len(setsizes_correct),
        'num_incorrect': len(setsizes_incorrect)
    }

def main():
    """Main function to compute metrics for all benchmarks filtered by LLM correctness."""
    
    # Find all benchmarks
    all_benchmarks = defaultdict(lambda: defaultdict(dict))  # {llm: {benchmark: {method: path}}}
    
    # Find directories in results_full
    simplellm_full = BASE_DIR_FULL / 'simplellm' / 'normal'
    if simplellm_full.exists():
        for llm_dir in simplellm_full.iterdir():
            if llm_dir.is_dir():
                llm_name = llm_dir.name
                for dataset_dir in llm_dir.iterdir():
                    if dataset_dir.is_dir():
                        benchmark_name = dataset_dir.name
                        all_benchmarks[llm_name][benchmark_name]['simplellm'] = {
                            'full': dataset_dir,
                            'recomputed': None
                        }
                        
                        # Find other RAG methods for same LLM and benchmark
                        for method in RAG_METHODS:
                            if method == 'simplellm':
                                continue
                            method_path_full = BASE_DIR_FULL / method / 'normal' / llm_name / benchmark_name
                            if method_path_full.exists():
                                if method not in all_benchmarks[llm_name][benchmark_name]:
                                    all_benchmarks[llm_name][benchmark_name][method] = {}
                                all_benchmarks[llm_name][benchmark_name][method]['full'] = method_path_full
    
    # Find directories in results_full_recomputed
    simplellm_recomputed = BASE_DIR_FULL_RECOMPUTED / 'simplellm' / 'normal'
    if simplellm_recomputed.exists():
        for llm_dir in simplellm_recomputed.iterdir():
            if llm_dir.is_dir():
                llm_name = llm_dir.name
                for dataset_dir in llm_dir.iterdir():
                    if dataset_dir.is_dir():
                        benchmark_name = dataset_dir.name
                        if llm_name in all_benchmarks and benchmark_name in all_benchmarks[llm_name]:
                            if 'simplellm' in all_benchmarks[llm_name][benchmark_name]:
                                all_benchmarks[llm_name][benchmark_name]['simplellm']['recomputed'] = dataset_dir
                        
                        # Find other RAG methods
                        for method in RAG_METHODS:
                            if method == 'simplellm':
                                continue
                            method_path_recomputed = BASE_DIR_FULL_RECOMPUTED / method / 'normal' / llm_name / benchmark_name
                            if method_path_recomputed.exists():
                                if llm_name in all_benchmarks and benchmark_name in all_benchmarks[llm_name]:
                                    if method in all_benchmarks[llm_name][benchmark_name]:
                                        all_benchmarks[llm_name][benchmark_name][method]['recomputed'] = method_path_recomputed
    
    print("=" * 80)
    print("Computing Set Size for All Benchmarks (Filtered by LLM Correctness)")
    print("=" * 80)
    print()
    
    # Process only 8b model (llama_3.1_8b_instruct)
    target_llm = 'llama_3.1_8b_instruct'
    
    if target_llm not in all_benchmarks:
        print(f"Error: {target_llm} not found")
        return
    
    # Collect results: {benchmark: {method: {setsize_correct, setsize_incorrect}}}
    results = defaultdict(dict)
    
    for benchmark_name in sorted(all_benchmarks[target_llm].keys()):
        benchmark_data = all_benchmarks[target_llm][benchmark_name]
        
        # Get LLM-only test results to determine correctness mask
        if 'simplellm' not in benchmark_data:
            continue
        
        llm_data = benchmark_data['simplellm']
        if not llm_data.get('full') or not llm_data.get('recomputed'):
            continue
        
        llm_test_results_full = load_test_results(llm_data['full'])
        llm_test_results_recomputed = load_test_results(llm_data['recomputed'])
        
        if llm_test_results_full is None or llm_test_results_recomputed is None:
            continue
        
        # Create LLM correctness mask
        llm_correctness_mask = []
        for result in llm_test_results_full:
            predicted = result.get('predicted_answer', '')
            correct = result.get('correct_answer', '')
            llm_correctness_mask.append(predicted == correct)
        
        print(f"Processing: {benchmark_name} (LLM Correct: {sum(llm_correctness_mask)}, Incorrect: {len(llm_correctness_mask) - sum(llm_correctness_mask)})")
        
        # Process each RAG method (excluding simplellm and fidrag)
        for method in RAG_METHODS:
            if method == 'simplellm' or method == 'fidrag':
                continue
            
            if method not in benchmark_data:
                continue
            
            method_data = benchmark_data[method]
            if not method_data.get('recomputed'):
                continue
            
            method_test_results_recomputed = load_test_results(method_data['recomputed'])
            method_calibration_recomputed = load_calibration_results(method_data['recomputed'])
            
            if method_test_results_recomputed is None:
                continue
            
            # Compute set size
            setsize_stats = recompute_setsize_by_llm_correctness(
                method_test_results_recomputed, method_calibration_recomputed, llm_correctness_mask)
            
            if setsize_stats and setsize_stats['setsize_correct'] is not None:
                method_display = METHOD_NAME_MAP.get(method, method)
                if benchmark_name not in results:
                    results[benchmark_name] = {}
                results[benchmark_name][method_display] = {
                    'setsize_correct': setsize_stats['setsize_correct'],
                    'setsize_incorrect': setsize_stats['setsize_incorrect']
                }
        
        # Also compute LLM-only set size
        llm_calibration_recomputed = load_calibration_results(llm_data['recomputed'])
        llm_setsize_stats = recompute_setsize_by_llm_correctness(
            llm_test_results_recomputed, llm_calibration_recomputed, llm_correctness_mask)
        
        if llm_setsize_stats and llm_setsize_stats['setsize_correct'] is not None:
            if benchmark_name not in results:
                results[benchmark_name] = {}
            results[benchmark_name]['LLM-Only'] = {
                'setsize_correct': llm_setsize_stats['setsize_correct'],
                'setsize_incorrect': llm_setsize_stats['setsize_incorrect']
            }
    
    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("Generating LaTeX Table")
    print("=" * 80)
    
    # Get all methods (excluding LLM-Only for main table)
    methods = ['Fusion', 'HyDE', 'RAPTOR', 'RAT', 'REPLUG', 'Self-RAG', 'Naive']
    
    # Use the ordered benchmark list for the table
    benchmark_order = [b for b in TABLE_BENCHMARK_ORDER if b in results]
    
    # Generate LaTeX table
    latex_output = []
    latex_output.append("\\begin{table*}[t]")
    latex_output.append("\\centering")
    latex_output.append("\\footnotesize")
    latex_output.append("\\setlength{\\tabcolsep}{4pt}")
    latex_output.append("\\caption{\\textbf{RAG Performance on 8B Model for LLM-Correct vs LLM-Incorrect Cases}.")
    latex_output.append("\\cmark: LLM originally correct, \\xmark: LLM originally incorrect.}")
    latex_output.append("\\label{tab:rag_correct_incorrect_8b_setsize}")
    latex_output.append("\\begin{adjustbox}{center, max width=\\textwidth}")
    latex_output.append("\\begin{tabularx}{\\textwidth}{c l *{" + str(len(benchmark_order)) + "}{c}}")
    latex_output.append("\\toprule")
    
    # Header row 1: Categories
    header1 = "& & \\textbf{Healthcare} & \\multicolumn{2}{c}{\\textbf{Code}} & \\textbf{Research} & \\textbf{Math} & \\multicolumn{3}{c}{\\textbf{General Text}} \\\\"
    latex_output.append(header1)
    
    # Header row 2: Category underlines
    cmidrules = "\\cmidrule(lr){3-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-10}"
    latex_output.append(cmidrules)
    
    # Header row 3: Benchmark names
    header3 = "\\textbf{LLM} & \\textbf{RAG} &"
    header3 += " \\textbf{Healthver} &"
    header3 += " \\textbf{Odex} &"
    header3 += " \\textbf{LCA} &"
    header3 += " \\textbf{SciFact} &"
    header3 += " \\textbf{Olympiad} &"
    header3 += " \\textbf{CRAG} &"
    header3 += " \\textbf{NewsSum} &"
    header3 += " \\textbf{DialFact} \\\\"
    latex_output.append(header3)
    latex_output.append("\\midrule")
    
    # Section header for Set Size
    latex_output.append("\\multicolumn{" + str(len(benchmark_order) + 2) + "}{c}{\\textit{Uncertainty} — \\textbf{Set Size (SS) $\\downarrow$}} \\\\")
    latex_output.append("\\cdashline{1-" + str(len(benchmark_order) + 2) + "}[2pt/4pt]")
    
    # Data rows: For each method, create two rows (correct and incorrect)
    for method in methods:
        # Correct row
        row_correct = "\\cmark & \\textbf{" + method + "}"
        for benchmark in benchmark_order:
            if benchmark in results and method in results[benchmark]:
                data = results[benchmark][method]
                correct = data.get('setsize_correct', 0.0)
                row_correct += f" & {correct:.2f}"
            else:
                row_correct += " & N/A"
        row_correct += " \\\\"
        latex_output.append(row_correct)
        
        # Incorrect row
        row_incorrect = "\\xmark & \\textbf{" + method + "}"
        for benchmark in benchmark_order:
            if benchmark in results and method in results[benchmark]:
                data = results[benchmark][method]
                incorrect = data.get('setsize_incorrect', 0.0)
                row_incorrect += f" & {incorrect:.2f}"
            else:
                row_incorrect += " & N/A"
        row_incorrect += " \\\\"
        latex_output.append(row_incorrect)
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabularx}")
    latex_output.append("\\end{adjustbox}")
    latex_output.append("\\end{table*}")
    
    # Save LaTeX table
    latex_file = Path('/media/volume/LLMRag/URAG/docs/setsize_correct_incorrect_table.tex')
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_output))
    
    print(f"\nLaTeX table saved to: {latex_file}")
    
    # Also save JSON results
    json_file = Path('/media/volume/LLMRag/URAG/all_benchmarks_setsize_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {json_file}")
    
    # Print summary
    print("\nSummary:")
    for benchmark in benchmark_order:
        print(f"\n{benchmark}:")
        if benchmark in results:
            for method in methods + ['LLM-Only']:
                if method in results[benchmark]:
                    data = results[benchmark][method]
                    print(f"  {method}: Correct={data['setsize_correct']:.4f}, Incorrect={data['setsize_incorrect']:.4f}")

if __name__ == '__main__':
    main()
