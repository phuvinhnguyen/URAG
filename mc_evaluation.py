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
import os
import re
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from tqdm.auto import tqdm
from systems.simplellm import SimpleLLMSystem
from systems.abstract import AbstractRAGSystem
from metrics import ConformalMetrics
from sklearn.metrics import roc_auc_score


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
        
        # Process samples through the RAG system
        raw_results = self.rag_system.batch_process_samples(samples)
        
        # Ensure all results have the required format for conformal prediction
        processed_results = []
        for i, result in enumerate(raw_results):
            # Copy original sample data
            sample = samples[i] if i < len(samples) else {}
            processed_result = sample.copy()
            
            # Update with system results
            processed_result.update({
                'generated_response': result.get('generated_response', ''),
                'predicted_answer': result.get('predicted_answer', 'Unknown'),
                'conformal_probabilities': result.get('option_probabilities', {}),
                'available_options': sample.get('options', [])
            })
            
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
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
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
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        
        lac_threshold = np.quantile(lac_scores, q_level)
        aps_threshold = np.quantile(aps_scores, q_level)
        
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

            for l, p in probabilities.items():
                auc_label.append(1 if l == correct_answer else 0)
                auc_predict.append(p)
            
            # Accuracy
            if predicted_answer == correct_answer:
                correct_predictions += 1
            
            # LAC prediction set and coverage
            lac_pred_set = self.metrics.compute_prediction_set_lac(probabilities, lac_threshold)
            if correct_answer in lac_pred_set:
                lac_coverage += 1
            lac_set_sizes.append(len(lac_pred_set))
            
            # APS prediction set and coverage
            aps_pred_set = self.metrics.compute_prediction_set_aps(probabilities, aps_threshold)
            if correct_answer in aps_pred_set:
                aps_coverage += 1
            aps_set_sizes.append(len(aps_pred_set))       

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
        
        results = {
            'total_samples': total_samples,
            'accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0,
            'lac_coverage': lac_coverage / total_samples if total_samples > 0 else 0.0,
            'aps_coverage': aps_coverage / total_samples if total_samples > 0 else 0.0,
            'lac_avg_set_size': np.mean(lac_set_sizes) if lac_set_sizes else 0.0,
            'aps_avg_set_size': np.mean(aps_set_sizes) if aps_set_sizes else 0.0,
            'lac_set_sizes': lac_set_sizes,
            'aps_set_sizes': aps_set_sizes,
            'auroc': auroc,
            'aurac': aurac,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        calibration_output = os.path.join(output_dir, f"calibration_results_{timestamp}.json")
        test_output = os.path.join(output_dir, f"test_results_{timestamp}.json")
        metrics_output = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.json")
        
        with open(calibration_output, 'w') as f:
            json.dump(calibration_results, f, indent=2)
        
        with open(test_output, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        with open(metrics_output, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
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


# Example usage and testing
def get_example_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate example calibration and test data."""
    calibration_data = [
        {
            "id": "cal_1",
            "question": "What is the capital of France?\nA. Paris\nB. London\nC. Berlin\nD. Rome",
            "answer": "A",
            "correct_answer": "A",
            "options": ["A", "B", "C", "D"],
            "technique": "direct"
        },
        {
            "id": "cal_2",
            "question": "Which programming language is known for data science?\nA. COBOL\nB. Python\nC. Assembly\nD. FORTRAN",
            "answer": "B",
            "correct_answer": "B",
            "options": ["A", "B", "C", "D"],
            "technique": "direct"
        }
    ]
    
    test_data = [
        {
            "id": "test_1",
            "question": "What is 2 + 2?\n1) 3\n2) 4\n3) 5\n4) 6",
            "answer": "2",
            "correct_answer": "2",
            "options": ["1", "2", "3", "4"],
            "technique": "direct"
        }
    ]
    
    return calibration_data, test_data


# Example RAG System Implementations

class SimpleRAGSystem(AbstractRAGSystem):
    """
    Example RAG system that performs simple retrieval and augmentation.
    
    This demonstrates how to implement a traditional RAG system.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        """Initialize the RAG system with an LLM and simple retrieval."""
        # Initialize the LLM component
        self.llm_system = SimpleLLMSystem(model_name, device)
        
        # Simple knowledge base (in practice, this would be a vector database)
        self.knowledge_base = {
            "france": "France is a country in Europe. Its capital is Paris.",
            "python": "Python is a popular programming language widely used in data science.",
            "jupiter": "Jupiter is the largest planet in our solar system.",
            "shakespeare": "William Shakespeare was an English playwright who wrote Romeo and Juliet.",
            "math": "Mathematics involves calculations and problem-solving."
        }
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def _retrieve_context(self, question: str) -> str:
        """Simple keyword-based retrieval."""
        question_lower = question.lower()
        retrieved_docs = []
        
        for keyword, doc in self.knowledge_base.items():
            if keyword in question_lower:
                retrieved_docs.append(doc)
        
        return " ".join(retrieved_docs) if retrieved_docs else ""
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with RAG enhancement."""
        # Retrieve relevant context
        question = sample.get('question', '')
        retrieved_context = self._retrieve_context(question)
        
        # Augment sample with retrieved context
        augmented_sample = sample.copy()
        if retrieved_context:
            existing_context = sample.get('search_results', sample.get('context', ''))
            combined_context = f"{existing_context}\n{retrieved_context}".strip()
            augmented_sample['search_results'] = combined_context
            augmented_sample['technique'] = 'rag'
        
        # Process through LLM
        result = self.llm_system.process_sample(augmented_sample)
        
        # Add RAG-specific information
        result.update({
            'retrieved_docs': retrieved_context,
            'rag_enhanced': bool(retrieved_context)
        })
        
        return result


class ComplexReasoningSystem(AbstractRAGSystem):
    """
    Example of a complex multi-step reasoning system.
    
    This demonstrates how complex systems with multiple components
    can be integrated into the evaluation framework.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        """Initialize the complex reasoning system."""
        self.llm_system = SimpleLLMSystem(model_name, device)
        self.reasoning_steps = []
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def _extract_key_concepts(self, question: str) -> List[str]:
        """Extract key concepts from the question."""
        # Simple concept extraction (in practice, use NER or other methods)
        concepts = []
        question_lower = question.lower()
        
        concept_keywords = {
            'geography': ['capital', 'country', 'continent', 'city'],
            'science': ['planet', 'chemical', 'formula', 'element'],
            'literature': ['wrote', 'author', 'book', 'novel'],
            'math': ['+', '-', '*', '/', 'calculate', 'solve'],
            'programming': ['language', 'programming', 'code', 'python']
        }
        
        for domain, keywords in concept_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                concepts.append(domain)
        
        return concepts
    
    def _multi_step_reasoning(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-step reasoning."""
        question = sample.get('question', '')
        reasoning_steps = []
        
        # Step 1: Concept extraction
        concepts = self._extract_key_concepts(question)
        reasoning_steps.append(f"Identified concepts: {concepts}")
        
        # Step 2: Generate reasoning prompt
        if concepts:
            reasoning_prompt = f"This question involves {', '.join(concepts)}. Let me think through this step by step.\n\n{question}\n\nStep-by-step reasoning:"
        else:
            reasoning_prompt = f"Let me analyze this question step by step.\n\n{question}\n\nStep-by-step reasoning:"
        
        reasoning_steps.append("Generated reasoning prompt")
        
        # Step 3: Get reasoning response
        reasoning_sample = sample.copy()
        reasoning_sample['question'] = reasoning_prompt
        reasoning_result = self.llm_system.process_sample(reasoning_sample)
        
        reasoning_steps.append("Generated reasoning response")
        
        # Step 4: Extract final answer with explicit reasoning
        final_prompt = f"{reasoning_result['generated_response']}\n\nBased on the above reasoning, what is the final answer? Please provide it in the format <answer>X</answer>."
        
        final_sample = sample.copy()
        final_sample['question'] = final_prompt
        final_result = self.llm_system.process_sample(final_sample)
        
        reasoning_steps.append("Generated final answer")
        
        return {
            'reasoning_steps': reasoning_steps,
            'intermediate_response': reasoning_result['generated_response'],
            'final_result': final_result,
            'concepts_identified': concepts
        }
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with complex reasoning."""
        # Perform multi-step reasoning
        reasoning_data = self._multi_step_reasoning(sample)
        
        # Extract final result
        final_result = reasoning_data['final_result']
        
        # Return comprehensive result
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': final_result['generated_response'],
            'predicted_answer': final_result['predicted_answer'],
            'option_probabilities': final_result['option_probabilities'],
            'reasoning_steps': reasoning_data['reasoning_steps'],
            'intermediate_reasoning': reasoning_data['intermediate_response'],
            'concepts_identified': reasoning_data['concepts_identified'],
            'system_type': 'complex_reasoning'
        }


if __name__ == "__main__":
    # Example usage with different system types
    
    # For testing with example data
    cal_data, test_data = get_example_data()
    
    # Save example data
    with open("example_calibration.json", "w") as f:
        json.dump(cal_data, f, indent=2)
    
    with open("example_test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("Testing different RAG system implementations...")
    
    # Test 1: Simple LLM System
    print("\n1. Testing Simple LLM System")
    simple_system = SimpleLLMSystem("microsoft/DialoGPT-small", "auto")
    pipeline1 = ConformalEvaluationPipeline(simple_system)
    
    results1 = pipeline1.run_evaluation(
        calibration_data_path="example_calibration.json",
        test_data_path="example_test.json",
        alpha=0.1,
        output_dir="simple_llm_results"
    )
    print(f"Simple LLM - Accuracy: {results1['metrics']['accuracy']:.4f}")
    
    # Test 2: Simple RAG System
    print("\n2. Testing Simple RAG System")
    rag_system = SimpleRAGSystem("microsoft/DialoGPT-small", "auto")
    pipeline2 = ConformalEvaluationPipeline(rag_system)
    
    results2 = pipeline2.run_evaluation(
        calibration_data_path="example_calibration.json",
        test_data_path="example_test.json",
        alpha=0.1,
        output_dir="simple_rag_results"
    )
    print(f"Simple RAG - Accuracy: {results2['metrics']['accuracy']:.4f}")
    
    # Test 3: Complex Reasoning System
    print("\n3. Testing Complex Reasoning System")
    complex_system = ComplexReasoningSystem("microsoft/DialoGPT-small", "auto")
    pipeline3 = ConformalEvaluationPipeline(complex_system)
    
    results3 = pipeline3.run_evaluation(
        calibration_data_path="example_calibration.json",
        test_data_path="example_test.json",
        alpha=0.1,
        output_dir="complex_reasoning_results"
    )
    print(f"Complex Reasoning - Accuracy: {results3['metrics']['accuracy']:.4f}")
    
    print("\nAll evaluations completed successfully!")
    print("Check the output directories for detailed results from different systems.")

