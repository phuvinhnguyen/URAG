#!/usr/bin/env python3
"""
Recompute conformal probabilities from generated responses.

This script:
1. Automatically finds all directories containing evaluation_metrics.json in a parent directory
2. Takes generated_response strings from calibration and test results
3. Forward passes through the model to extract probabilities for Answer|A, Answer|B, etc.
4. Updates conformal_probabilities for all items
5. Recomputes calibration thresholds and evaluation metrics

Usage:
    python recompute_probabilities.py <parent_dir> [--alpha 0.1] [--output_dir <dir>] [--model <model_name>]

Examples:
    # Override input directories (default behavior)
    python recompute_probabilities.py /media/volume/LLMRag2/URAG/results_old
    
    # Save to new parent directory (reconstructs structure)
    python recompute_probabilities.py /media/volume/LLMRag2/URAG/results_old --output_dir /media/volume/LLMRag2/URAG/results_new
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

# Import metrics from URAG
sys.path.insert(0, '/media/volume/LLMRag2/URAG')
from metrics import ConformalMetrics
from llmasjudge import correct_prediction

# Model name mapping: path_segment -> HuggingFace model name
MODEL_MAP = {
    "llama_3.1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # "llama_3.2_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    # "flan_t5_base": "google/flan-t5-base",
}

# Answer format options
ANSWER_FORMATS = ["Answer|A", "Answer|B", "Answer|C", "Answer|D", "Answer|E", "Answer|F"]


def load_model(model_name: str):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


def extract_answer_probabilities(tokenizer, model, generated_response: str, options: List[str]) -> Dict[str, float]:
    """
    Extract probabilities for answer options from a generated response.
    
    This function finds where "Answer|" appears in the generated response,
    then extracts logits for all possible option letters at that position.
    
    Args:
        tokenizer: Model tokenizer
        model: Model
        generated_response: The full generated response string
        options: List of option letters (e.g., ["A", "B", "C", "D"])
    
    Returns:
        Dictionary mapping option letters to probabilities
    """
    # Find the position where "Answer|" appears (get the last occurrence)
    answer_prefix = "Answer|"
    answer_pos = generated_response.rfind(answer_prefix)
    
    if answer_pos == -1:
        # If "Answer|" not found, use the last token position
        inputs = tokenizer(generated_response, return_tensors="pt", truncation=True)
        for k in inputs:
            if torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :].squeeze(0)  # Last token logits
    else:
        # Get text up to and including "Answer|" (without the actual answer letter)
        prefix_text = generated_response[:answer_pos + len(answer_prefix)]
        
        # Tokenize the prefix
        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", truncation=True)
        for k in prefix_inputs:
            if torch.is_tensor(prefix_inputs[k]):
                prefix_inputs[k] = prefix_inputs[k].to("cuda")
        
        # Forward pass to get logits at the position right after "Answer|"
        with torch.no_grad():
            outputs = model(**prefix_inputs)
        logits = outputs.logits[:, -1, :].squeeze(0)  # Logits at position after "Answer|"
    
    # Get token IDs for each option letter
    # For Llama tokenizer, single letters are typically tokenized as single tokens
    option_logits_list = []
    
    for opt in options:
        # Encode just the letter to get its token ID
        token_ids = tokenizer.encode(opt, add_special_tokens=False)
        if len(token_ids) > 0:
            token_id = token_ids[-1]  # Use the last token (in case of multi-token encoding)
            if token_id < logits.shape[0]:  # Check bounds
                option_logits_list.append((opt, logits[token_id].item()))
            else:
                option_logits_list.append((opt, -100.0))  # Out of bounds, use small logit
        else:
            # Fallback: try with space
            token_ids = tokenizer.encode(f" {opt}", add_special_tokens=False)
            if len(token_ids) > 0:
                token_id = token_ids[-1]
                if token_id < logits.shape[0]:
                    option_logits_list.append((opt, logits[token_id].item()))
                else:
                    option_logits_list.append((opt, -100.0))
            else:
                option_logits_list.append((opt, -100.0))
    
    # Extract logit values
    logit_values = torch.tensor([logit for _, logit in option_logits_list], dtype=torch.float32)
    
    # Compute probabilities using softmax
    probs = F.softmax(logit_values, dim=0)
    
    # Create probability dictionary
    option_probs = {}
    for i, (opt, _) in enumerate(option_logits_list):
        option_probs[opt] = float(probs[i].item())
    
    return option_probs


def recompute_probabilities_for_results(tokenizer, model, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Recompute conformal_probabilities for all results.
    
    Args:
        tokenizer: Model tokenizer
        model: Model
        results: List of result dictionaries
    
    Returns:
        Updated results with new conformal_probabilities
    """
    updated_results = []
    
    for result in tqdm(results, desc="Recomputing probabilities"):
        updated_result = result.copy()
        
        if 'generated_response' not in result:
            print(f"Warning: No generated_response for ID {result.get('id', 'unknown')}")
            updated_results.append(updated_result)
            continue
        
        generated_response = result['generated_response']
        options = result.get('options', ['A', 'B', 'C', 'D'])
        
        try:
            # Extract probabilities
            probabilities = extract_answer_probabilities(
                tokenizer, model, generated_response, options
            )
            updated_result['conformal_probabilities'] = probabilities
            
            # Keep original predicted_answer unchanged to preserve accuracy
            # Only probabilities are recomputed for coverage and set size calculations
        except Exception as e:
            print(f"Error processing ID {result.get('id', 'unknown')}: {e}")
            # Keep original probabilities if recomputation fails
            pass
        
        updated_results.append(updated_result)
    
    return updated_results


def compute_calibration_thresholds(calibration_results: List[Dict[str, Any]], 
                                   alpha: float = 0.1) -> Tuple[float, float]:
    """
    Compute calibration thresholds for LAC and APS.
    
    Args:
        calibration_results: Results from calibration set
        alpha: Desired error rate (e.g., 0.1 for 90% coverage)
    
    Returns:
        Tuple of (lac_threshold, aps_threshold)
    """
    metrics = ConformalMetrics()
    lac_scores = []
    aps_scores = []
    
    for result in calibration_results:
        if 'conformal_probabilities' not in result or not result['conformal_probabilities']:
            continue
        
        correct_answer = result.get('correct_answer', result.get('answer'))
        probabilities = result['conformal_probabilities']
        
        lac_score = metrics.compute_lac_score(probabilities, correct_answer)
        aps_score = metrics.compute_aps_score(probabilities, correct_answer)
        
        lac_scores.append(lac_score)
        aps_scores.append(aps_score)
    
    if not lac_scores:
        print("Warning: No valid calibration scores found, using default thresholds")
        return 0.5, 0.5
    
    n = len(lac_scores)
    q_level = np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1)
    
    lac_threshold = np.quantile(lac_scores, q_level, method='higher')
    aps_threshold = np.quantile(aps_scores, q_level, method='higher')
    
    print(f"Calibration thresholds: LAC={lac_threshold:.4f}, APS={aps_threshold:.4f}")
    
    return lac_threshold, aps_threshold


def evaluate_with_conformal_prediction(test_results: List[Dict[str, Any]], 
                                       lac_threshold: float, aps_threshold: float) -> Dict[str, Any]:
    """
    Evaluate test results using conformal prediction.
    
    Args:
        test_results: Results from test set
        lac_threshold: LAC threshold from calibration
        aps_threshold: APS threshold from calibration
    
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = ConformalMetrics()
    total_samples = len(test_results)
    correct_predictions = 0
    lac_coverage = 0
    aps_coverage = 0
    lac_set_sizes = []
    aps_set_sizes = []
    
    for result in test_results:
        if 'conformal_probabilities' not in result or not result['conformal_probabilities']:
            continue
        
        correct_answer = result.get('correct_answer', result.get('answer'))
        probabilities = result['conformal_probabilities']
        predicted_answer = result.get('predicted_answer', '')
        
        # Get correct set for coverage metrics (handles multiple correct answers)
        correctness, correct_set = correct_prediction(
            list(probabilities.keys()), correct_answer, result.get('question', '')
        )
        
        # Accuracy: direct comparison (predicted_answer has highest prob, so should match original)
        if isinstance(correct_answer, list):
            if predicted_answer in correct_answer:
                correct_predictions += 1
        else:
            if str(predicted_answer).strip().lower() == str(correct_answer).strip().lower():
                correct_predictions += 1
        
        # LAC prediction set and coverage
        lac_pred_set = metrics.compute_prediction_set_lac(probabilities, lac_threshold)
        if not correct_set.isdisjoint(set(lac_pred_set)):
            lac_coverage += 1
        lac_set_sizes.append(len(lac_pred_set))
        
        # APS prediction set and coverage
        aps_pred_set = metrics.compute_prediction_set_aps(probabilities, aps_threshold)
        if not correct_set.isdisjoint(set(aps_pred_set)):
            aps_coverage += 1
        aps_set_sizes.append(len(aps_pred_set))
    
    results = {
        'total_samples': total_samples,
        'accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0,
        'lac_coverage': lac_coverage / total_samples if total_samples > 0 else 0.0,
        'aps_coverage': aps_coverage / total_samples if total_samples > 0 else 0.0,
        'lac_avg_set_size': float(np.mean(lac_set_sizes)) if lac_set_sizes else 0.0,
        'aps_avg_set_size': float(np.mean(aps_set_sizes)) if aps_set_sizes else 0.0,
        'lac_set_sizes': lac_set_sizes,
        'aps_set_sizes': aps_set_sizes,
        'thresholds': {
            'lac_threshold': lac_threshold,
            'aps_threshold': aps_threshold
        }
    }
    
    return results


def extract_model_from_path(path: str) -> Optional[str]:
    """Extract model name from path and return HuggingFace model name, or None if not found."""
    parts = path.split(os.sep)
    for part in parts:
        if part in MODEL_MAP:
            return MODEL_MAP[part]
    return None


def find_evaluation_directories(parent_dir: str) -> List[str]:
    """
    Find all directories containing evaluation_metrics.json recursively.
    
    Args:
        parent_dir: Parent directory to search in
    
    Returns:
        List of directory paths containing evaluation_metrics.json
    """
    evaluation_dirs = []
    
    for root, dirs, files in os.walk(parent_dir):
        if 'evaluation_metrics.json' in files:
            evaluation_dirs.append(root)
    
    return sorted(evaluation_dirs)


def process_single_directory(results_dir: str, output_dir: str, tokenizer, model, alpha: float):
    """
    Process a single directory: recompute probabilities and metrics.
    
    Args:
        results_dir: Input directory containing calibration_results.json and test_results.json
        output_dir: Output directory (can be same as results_dir to override)
        tokenizer: Model tokenizer
        model: Model
        alpha: Error rate for conformal prediction
    """
    # Check if output already exists and skip if complete
    output_metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    output_calibration_path = os.path.join(output_dir, 'calibration_results.json')
    output_test_path = os.path.join(output_dir, 'test_results.json')
    
    if (os.path.exists(output_metrics_path) and 
        os.path.exists(output_calibration_path) and 
        os.path.exists(output_test_path)):
        print(f"\nSkipping {results_dir}: Output already exists in {output_dir}")
        return
    
    # Load data
    calibration_path = os.path.join(results_dir, 'calibration_results.json')
    test_path = os.path.join(results_dir, 'test_results.json')
    original_metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
    
    if not os.path.exists(calibration_path):
        print(f"Warning: {calibration_path} not found, skipping...")
        return
    
    if not os.path.exists(test_path):
        print(f"Warning: {test_path} not found, skipping...")
        return
    
    # Load original accuracy if available (to preserve it)
    original_accuracy = None
    if os.path.exists(original_metrics_path):
        try:
            with open(original_metrics_path, 'r') as f:
                original_metrics = json.load(f)
                original_accuracy = original_metrics.get('accuracy')
                if original_accuracy is not None:
                    print(f"Found original accuracy: {original_accuracy:.4f} (will be preserved)")
        except Exception as e:
            print(f"Warning: Could not load original metrics: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {results_dir}")
    print(f"{'='*60}")
    
    print(f"Loading calibration results from: {calibration_path}")
    with open(calibration_path, 'r') as f:
        calibration_results = json.load(f)
    
    print(f"Loading test results from: {test_path}")
    with open(test_path, 'r') as f:
        test_results = json.load(f)
    
    print(f"Loaded {len(calibration_results)} calibration samples and {len(test_results)} test samples")
    
    # Recompute probabilities for calibration set
    print("\nRecomputing probabilities for calibration set...")
    calibration_results = recompute_probabilities_for_results(tokenizer, model, calibration_results)
    
    # Recompute probabilities for test set
    print("\nRecomputing probabilities for test set...")
    test_results = recompute_probabilities_for_results(tokenizer, model, test_results)
    
    # Compute calibration thresholds
    print("\nComputing calibration thresholds...")
    lac_threshold, aps_threshold = compute_calibration_thresholds(calibration_results, alpha)
    
    # Evaluate test set
    print("\nEvaluating test set...")
    evaluation_metrics = evaluate_with_conformal_prediction(
        test_results, lac_threshold, aps_threshold
    )
    
    # Preserve original accuracy if available
    if original_accuracy is not None:
        print(f"Preserving original accuracy: {evaluation_metrics['accuracy']:.4f} -> {original_accuracy:.4f}")
        evaluation_metrics['accuracy'] = original_accuracy
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    calibration_output = os.path.join(output_dir, 'calibration_results.json')
    test_output = os.path.join(output_dir, 'test_results.json')
    metrics_output = os.path.join(output_dir, 'evaluation_metrics.json')
    
    print(f"\nSaving results to {output_dir}...")
    with open(calibration_output, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    with open(test_output, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    with open(metrics_output, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
    print(f"LAC Coverage: {evaluation_metrics['lac_coverage']:.4f}")
    print(f"APS Coverage: {evaluation_metrics['aps_coverage']:.4f}")
    print(f"LAC Avg Set Size: {evaluation_metrics['lac_avg_set_size']:.4f}")
    print(f"APS Avg Set Size: {evaluation_metrics['aps_avg_set_size']:.4f}")
    print(f"LAC Threshold: {lac_threshold:.4f}")
    print(f"APS Threshold: {aps_threshold:.4f}")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recompute conformal probabilities from generated responses. "
                    "Automatically finds all directories containing evaluation_metrics.json."
    )
    parser.add_argument(
        'parent_dir',
        type=str,
        help='Parent directory to search for evaluation_metrics.json files'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Desired error rate for conformal prediction (default: 0.1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output parent directory (default: None, overrides input directories). '
             'If provided, reconstructs the directory structure relative to parent_dir.'
    )
    
    args = parser.parse_args()
    
    # Find all directories containing evaluation_metrics.json
    print(f"Searching for evaluation directories in: {args.parent_dir}")
    evaluation_dirs = find_evaluation_directories(args.parent_dir)
    
    if not evaluation_dirs:
        print(f"No directories containing evaluation_metrics.json found in {args.parent_dir}")
        return
    
    # Filter directories by model name and group by model
    model_dirs = {}
    for dir_path in evaluation_dirs:
        model_name = extract_model_from_path(dir_path)
        if model_name:
            if model_name not in model_dirs:
                model_dirs[model_name] = []
            model_dirs[model_name].append(dir_path)
        else:
            print(f"Skipping {dir_path}: model name not found in path")
    
    if not model_dirs:
        print("No directories with recognized model names found")
        return
    
    # Process each model group
    model_cache = {}
    for model_name, dirs in model_dirs.items():
        if model_name not in model_cache:
            print(f"\nLoading model: {model_name}")
            model_cache[model_name] = load_model(model_name)
    
        tokenizer, model = model_cache[model_name]
        
        for results_dir in dirs:
            if args.output_dir is None:
                output_dir = results_dir
            else:
                rel_path = os.path.relpath(results_dir, args.parent_dir)
                output_dir = os.path.join(args.output_dir, rel_path)
        
            try:
                process_single_directory(results_dir, output_dir, tokenizer, model, args.alpha)
            except Exception as e:
                print(f"\nError processing {results_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
    print(f"\n{'='*60}")
    print(f"Completed processing {sum(len(dirs) for dirs in model_dirs.values())} directories")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

