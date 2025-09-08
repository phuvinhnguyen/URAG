"""
Multiple Choice Evaluation with Conformal Prediction

This module evaluates RAG systems on multiple choice questions using conformal prediction
techniques including LAC (Least Ambiguous Classifier) and APS (Adaptive Prediction Sets).

Data format:
[
    {
        "id": "1",
        "question": "What is the capital of France?\nA. Paris\nB. London\nC. Berlin\nD. Rome\n",
        "answer": "A",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A",
        ... # Extra information for RAG system
    },
    ...
]
"""
import json
import orjson
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

from loguru import logger
from systems.simplellm import SimpleLLMSystem
from systems.abstract import AbstractRAGSystem
from metrics import ConformalMetrics
from sklearn.metrics import roc_auc_score
from llmasjudge import correct_prediction

class SystemEvaluator:
    """
    Evaluator that works with any AbstractRAGSystem for conformal prediction.
    
    This class handles the evaluation logic while delegating the actual
    system processing to the provided AbstractRAGSystem implementation.
    """
    
    def __init__(self, rag_system: AbstractRAGSystem):
        """
        Initialize the evaluator with a RAG system.
        
        Args:
            rag_system: Any implementation of AbstractRAGSystem
        """
        self.rag_system = rag_system
        logger.info(f"Initialized evaluator with system: {type(rag_system).__name__}")
    
    def evaluate_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a list of samples using the RAG system.
        
        Args:
            samples: List of input samples
            
        Returns:
            List of evaluation results with conformal probabilities
        """
        logger.info(f"Evaluating {len(samples)} samples")
        
        raw_results = []
        # Process samples through the RAG system
        for index in tqdm(range(0, len(samples), self.rag_system.get_batch_size()), desc="Processing samples"):
            batch = samples[index:index+self.rag_system.get_batch_size()]
            raw_result = self.rag_system.batch_process_samples(batch)
            raw_results.extend(raw_result)
        
        # Ensure all results have the required format for conformal prediction
        processed_results = []
        for i, result in enumerate(raw_results):
            # Copy original sample data
            sample = samples[i] if i < len(samples) else {}
            
            # Update with system results
            processed_result = {
                'generated_response': result.get('generated_response', ''),
                'predicted_answer': result.get('predicted_answer', 'Unknown'),
                'conformal_probabilities': result.get('conformal_probabilities', {}),
                'id': sample.get('id', 'unknown'),
                'question': sample.get('question', ''),
                'correct_answer': sample.get('correct_answer', ''),
                'options': sample.get('options', []),
            }
            
            # Add any additional fields from the system
            for key, value in result.items():
                if key not in processed_result:
                    processed_result[key] = value
            
            processed_results.append(processed_result)
        
        return processed_results


class ConformalEvaluationPipeline:
    """Main evaluation pipeline for conformal prediction on multiple choice questions."""
    
    def __init__(self, rag_system: AbstractRAGSystem):
        """
        Initialize the evaluation pipeline with a RAG system.
        
        Args:
            rag_system: Any implementation of AbstractRAGSystem
        """
        self.evaluator = SystemEvaluator(rag_system)
        self.metrics = ConformalMetrics()
        self.rag_system = rag_system
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'rb') as f:
            data = orjson.loads(f.read())
        return data if isinstance(data, list) else [data]
    
    def compute_calibration_thresholds(self, calibration_results: List[Dict[str, Any]], 
                                     alpha: float = 0.1) -> Tuple[float, float]:
        """
        Compute calibration thresholds for LAC and APS.
        
        Args:
            calibration_results: Results from calibration set
            alpha: Desired error rate (e.g., 0.1 for 90% coverage)
            
        Returns:
            Tuple of (lac_threshold, aps_threshold)
        """
        lac_scores = []
        aps_scores = []
        
        for result in calibration_results:
            if 'conformal_probabilities' not in result or not result['conformal_probabilities']:
                continue
                
            correct_answer = result.get('correct_answer', result.get('answer'))
            probabilities = result['conformal_probabilities']
            
            lac_score = self.metrics.compute_lac_score(probabilities, correct_answer)
            aps_score = self.metrics.compute_aps_score(probabilities, correct_answer)
            
            lac_scores.append(lac_score)
            aps_scores.append(aps_score)
        
        if not lac_scores:
            logger.warning("No valid calibration scores found, using default thresholds")
            return 0.5, 0.5
        
        n = len(lac_scores)
        q_level = np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1)
        
        lac_threshold = np.quantile(lac_scores, q_level, method='higher')
        aps_threshold = np.quantile(aps_scores, q_level, method='higher')
        
        logger.info(f"Calibration thresholds: LAC={lac_threshold:.4f}, APS={aps_threshold:.4f}")
        
        return lac_threshold, aps_threshold

    def evaluate_with_conformal_prediction(self, test_results: List[Dict[str, Any]], 
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
        total_samples = len(test_results)
        correct_predictions = 0
        lac_coverage = 0
        aps_coverage = 0
        lac_set_sizes = []
        aps_set_sizes = []
        auc_label = []
        auc_predict = []
        
        for result in test_results:
            if 'conformal_probabilities' not in result or not result['conformal_probabilities']:
                continue
                
            correct_answer = result.get('correct_answer', result.get('answer'))
            probabilities = result['conformal_probabilities']
            predicted_answer = result.get('predicted_answer', '')

            correctness, correct_set = correct_prediction(list(probabilities.keys()), correct_answer, result.get('question', ''))
            auc_label += correctness
            auc_predict += list(probabilities.values())
            
            # Accuracy
            if predicted_answer in correct_set:
                correct_predictions += 1
            
            # LAC prediction set and coverage
            lac_pred_set = self.metrics.compute_prediction_set_lac(probabilities, lac_threshold)
            if not correct_set.isdisjoint(set(lac_pred_set)):
                lac_coverage += 1
            lac_set_sizes.append(len(lac_pred_set))
            
            # APS prediction set and coverage
            aps_pred_set = self.metrics.compute_prediction_set_aps(probabilities, aps_threshold)
            if not correct_set.isdisjoint(set(aps_pred_set)):
                aps_coverage += 1
            aps_set_sizes.append(len(aps_pred_set))       

        try:
            auc_label = np.array(auc_label)
            auc_predict = np.array(auc_predict)
            auroc = roc_auc_score(auc_label, auc_predict)
            order = np.argsort(-auc_predict)
            label_sorted = auc_label[order]
            n = len(label_sorted)
            r_grid = np.linspace(0, 1, 1001)
            acc_curve = []
            for r in r_grid:
                keep = max(1, int(np.round((1 - r) * n)))
                acc_curve.append(label_sorted[:keep].mean())
            aurac = np.trapz(acc_curve, r_grid)
        except Exception as e:
            auroc = None
            aurac = None
            print(e)
        
        results = {
            'total_samples': total_samples,
            'accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0,
            'lac_coverage': lac_coverage / total_samples if total_samples > 0 else 0.0,
            'aps_coverage': aps_coverage / total_samples if total_samples > 0 else 0.0,
            'lac_avg_set_size': float(np.mean(lac_set_sizes)) if lac_set_sizes else 0.0,
            'aps_avg_set_size': float(np.mean(aps_set_sizes)) if aps_set_sizes else 0.0,
            'lac_set_sizes': lac_set_sizes,
            'aps_set_sizes': aps_set_sizes,
            'auroc': float(auroc) if auroc is not None else None,
            'aurac': float(aurac) if aurac is not None else None,
            'thresholds': {
                'lac_threshold': lac_threshold,
                'aps_threshold': aps_threshold
            }
        }
        
        return results
    
    def add_custom_metric(self, metric_name: str, metric_func: callable):
        """
        Add custom metric function to the evaluation pipeline.
        
        Args:
            metric_name: Name of the metric
            metric_func: Function that takes (probabilities, correct_answer) and returns a score
        """
        setattr(self.metrics, f"compute_{metric_name}_score", staticmethod(metric_func))
        logger.info(f"Added custom metric: {metric_name}")
    
    def run_evaluation(self, calibration_data_path: str, test_data_path: str, 
                      alpha: float = 0.1, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            calibration_data_path: Path to calibration data
            test_data_path: Path to test data
            alpha: Desired error rate for conformal prediction
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        logger.info("Loading calibration and test data")
        calibration_data = self.load_data(calibration_data_path)
        test_data = self.load_data(test_data_path)
        
        # Evaluate calibration set
        logger.info("Evaluating calibration set")
        calibration_results = self.evaluator.evaluate_samples(calibration_data)
        
        # Compute thresholds
        logger.info("Computing calibration thresholds")
        lac_threshold, aps_threshold = self.compute_calibration_thresholds(calibration_results, alpha)
        
        # Evaluate test set
        logger.info("Evaluating test set")
        test_results = self.evaluator.evaluate_samples(test_data)
        
        # Compute final metrics
        logger.info("Computing final metrics")
        final_metrics = self.evaluate_with_conformal_prediction(
            test_results, lac_threshold, aps_threshold
        )
        
        # Save detailed results
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        calibration_output = os.path.join(output_dir, f"calibration_results.json")
        test_output = os.path.join(output_dir, f"test_results.json")
        metrics_output = os.path.join(output_dir, f"evaluation_metrics.json")
        
        logger.info("Saving results")
        with open(calibration_output, 'w') as f:
            json.dump(calibration_results, f, indent=4)
        
        with open(test_output, 'w') as f:
            json.dump(test_results, f, indent=4)
        
        with open(metrics_output, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        # Log summary
        logger.info("Evaluation Summary:")
        logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"LAC Coverage: {final_metrics['lac_coverage']:.4f}")
        logger.info(f"APS Coverage: {final_metrics['aps_coverage']:.4f}")
        logger.info(f"LAC Avg Set Size: {final_metrics['lac_avg_set_size']:.4f}")
        logger.info(f"APS Avg Set Size: {final_metrics['aps_avg_set_size']:.4f}")
        
        return {
            'calibration_results': calibration_results,
            'test_results': test_results,
            'metrics': final_metrics,
            'output_files': {
                'calibration': calibration_output,
                'test': test_output,
                'metrics': metrics_output
            }
        }
