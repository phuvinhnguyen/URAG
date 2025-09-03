#!/usr/bin/env python3
"""
Test script for Self-RAG system implementation.
This script tests the Self-RAG system with various scenarios including:
- Retrieval need evaluation
- Document relevance filtering
- Iterative refinement
- Support evaluation
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from loguru import logger

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from systems.selfrag import SelfRAGSystem
    from systems.selfllm import SelfLLMSystem
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class SelfRAGTester:
    """Test class for Self-RAG system functionality."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the tester with specified model."""
        self.model_name = model_name
        logger.info(f"Initializing Self-RAG tester with model: {model_name}")
        
    def create_test_samples(self) -> List[Dict[str, Any]]:
        """Create test samples for different scenarios."""
        return [
            {
                "id": "test_1",
                "question": "What is the capital of France?",
                "search_results": [
                    {
                        "page_snippet": "Paris is the capital and most populous city of France.",
                        "page_result": "Paris, the capital of France, is located in the north-central part of the country. It is the most populous city in France with over 2 million inhabitants.",
                        "page_url": "https://example.com/paris"
                    }
                ],
                "options": ["A", "B", "C", "D"],
                "option_texts": {
                    "A": "Paris",
                    "B": "London", 
                    "C": "Berlin",
                    "D": "Madrid"
                },
                "expected_answer": "A"
            },
            {
                "id": "test_2", 
                "question": "What is 2 + 2?",
                "search_results": [],  # No search results to test direct processing
                "options": ["A", "B", "C", "D"],
                "option_texts": {
                    "A": "3",
                    "B": "4",
                    "C": "5", 
                    "D": "6"
                },
                "expected_answer": "B"
            },
            {
                "id": "test_3",
                "question": "What are the main benefits of renewable energy?",
                "search_results": [
                    {
                        "page_snippet": "Renewable energy sources like solar and wind are sustainable.",
                        "page_result": "Renewable energy offers many benefits including reduced greenhouse gas emissions, energy independence, and long-term cost savings. Solar and wind power are among the most popular renewable sources.",
                        "page_url": "https://example.com/renewable"
                    },
                    {
                        "page_snippet": "Clean energy reduces pollution and creates jobs.",
                        "page_result": "The renewable energy sector has created millions of jobs worldwide while helping to reduce air pollution and combat climate change.",
                        "page_url": "https://example.com/clean-energy"
                    }
                ],
                "options": [],  # Empty options to test generation mode
                "expected_answer": None  # Will be generated
            },
            {
                "id": "test_4",
                "question": "How does photosynthesis work in plants?",
                "search_results": [
                    {
                        "page_snippet": "Photosynthesis is the process by which plants make food.",
                        "page_result": "Photosynthesis is a complex biological process where plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. This process occurs mainly in the chloroplasts of plant cells.",
                        "page_url": "https://example.com/photosynthesis"
                    }
                ],
                "options": ["A", "B", "C", "D"],
                "option_texts": {
                    "A": "Plants absorb sunlight and convert CO2 and water into glucose",
                    "B": "Plants only need water to survive",
                    "C": "Plants get energy from soil minerals",
                    "D": "Plants don't need sunlight"
                },
                "expected_answer": "A"
            }
        ]
    
    def test_selfrag_system(self) -> Dict[str, Any]:
        """Test the complete Self-RAG system."""
        logger.info("Testing Self-RAG System")
        logger.info("=" * 50)
        
        try:
            # Initialize Self-RAG system
            selfrag_system = SelfRAGSystem(
                model_name=self.model_name,
                device="auto",
                max_iterations=2,  # Reduced for faster testing
                relevance_threshold=0.7
            )
            
            test_samples = self.create_test_samples()
            results = []
            
            for i, sample in enumerate(test_samples):
                logger.info(f"\nProcessing Test Sample {i+1}/{len(test_samples)}")
                logger.info(f"Question: {sample['question']}")
                logger.info(f"Has search results: {len(sample.get('search_results', []))}")
                
                start_time = time.time()
                
                try:
                    result = selfrag_system.process_sample(sample)
                    processing_time = time.time() - start_time
                    
                    # Add test metadata
                    result.update({
                        'test_id': sample['id'],
                        'processing_time': processing_time,
                        'expected_answer': sample.get('expected_answer'),
                        'test_passed': self._evaluate_result(result, sample)
                    })
                    
                    results.append(result)
                    
                    # Log key results
                    logger.info(f"Predicted Answer: {result.get('predicted_answer', 'N/A')}")
                    logger.info(f"Expected Answer: {sample.get('expected_answer', 'N/A')}")
                    logger.info(f"Retrieval Performed: {result.get('retrieval_performed', False)}")
                    logger.info(f"Relevant Docs: {result.get('num_relevant_docs', 0)}")
                    logger.info(f"Support Score: {result.get('final_support_score', 'N/A')}")
                    logger.info(f"Processing Time: {processing_time:.2f}s")
                    logger.info(f"Test Passed: {result['test_passed']}")
                    
                except Exception as e:
                    logger.error(f"Error processing sample {sample['id']}: {e}")
                    results.append({
                        'test_id': sample['id'],
                        'error': str(e),
                        'test_passed': False
                    })
            
            return {
                'system_type': 'self_rag',
                'model_name': self.model_name,
                'total_samples': len(test_samples),
                'successful_tests': sum(1 for r in results if r.get('test_passed', False)),
                'results': results,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
            
        except Exception as e:
            logger.error(f"Error initializing Self-RAG system: {e}")
            return {
                'error': str(e),
                'system_type': 'self_rag',
                'model_name': self.model_name
            }
    
    def test_selfllm_components(self) -> Dict[str, Any]:
        """Test individual Self-LLM components."""
        logger.info("\nTesting Self-LLM Components")
        logger.info("=" * 50)
        
        try:
            selfllm_system = SelfLLMSystem(
                model_name=self.model_name,
                device="auto",
                technique="self"
            )
            
            # Test retrieval need evaluation
            questions = [
                "What is 2 + 2?",  # Should not need retrieval
                "What is the current population of Tokyo?",  # Should need retrieval
                "What are the latest developments in AI research?"  # Should need retrieval
            ]
            
            retrieval_results = []
            for question in questions:
                try:
                    needs_retrieval = selfllm_system.evaluate_retrieval_need(question)
                    retrieval_results.append({
                        'question': question,
                        'needs_retrieval': needs_retrieval
                    })
                    logger.info(f"Question: {question}")
                    logger.info(f"Needs Retrieval: {needs_retrieval}")
                except Exception as e:
                    logger.error(f"Error evaluating retrieval need: {e}")
                    retrieval_results.append({
                        'question': question,
                        'error': str(e)
                    })
            
            # Test relevance evaluation
            test_question = "What is the capital of France?"
            test_contexts = [
                "Paris is the capital of France and its largest city.",  # Relevant
                "The weather in London is cloudy today.",  # Irrelevant
                "France is a country in Europe with rich history."  # Somewhat relevant
            ]
            
            relevance_results = []
            for context in test_contexts:
                try:
                    is_relevant = selfllm_system.evaluate_relevance(test_question, context)
                    relevance_results.append({
                        'context': context,
                        'is_relevant': is_relevant
                    })
                    logger.info(f"Context: {context[:50]}...")
                    logger.info(f"Is Relevant: {is_relevant}")
                except Exception as e:
                    logger.error(f"Error evaluating relevance: {e}")
                    relevance_results.append({
                        'context': context,
                        'error': str(e)
                    })
            
            return {
                'component_type': 'self_llm',
                'model_name': self.model_name,
                'retrieval_evaluation': retrieval_results,
                'relevance_evaluation': relevance_results,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
            
        except Exception as e:
            logger.error(f"Error testing Self-LLM components: {e}")
            return {
                'error': str(e),
                'component_type': 'self_llm',
                'model_name': self.model_name
            }
    
    def _evaluate_result(self, result: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        """Evaluate if the test result is acceptable."""
        # Check if we got a predicted answer
        if not result.get('predicted_answer'):
            return False
            
        # If we have an expected answer, check if it matches
        expected = sample.get('expected_answer')
        if expected:
            predicted = result.get('predicted_answer', '').strip().upper()
            expected = expected.strip().upper()
            return predicted == expected
        
        # If no expected answer (generation mode), consider it passed if we got an answer
        return result.get('predicted_answer') != "Unknown"
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive tests for Self-RAG system."""
        logger.info("Starting Comprehensive Self-RAG Test Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Test Self-RAG system
        selfrag_results = self.test_selfrag_system()
        
        # Test Self-LLM components
        selfllm_results = self.test_selfllm_components()
        
        total_time = time.time() - start_time
        
        comprehensive_results = {
            'test_suite': 'comprehensive_selfrag',
            'model_name': self.model_name,
            'total_test_time': total_time,
            'selfrag_system_test': selfrag_results,
            'selfllm_components_test': selfllm_results,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        # Save results to file
        results_file = f"selfrag_test_results_{comprehensive_results['timestamp']}.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Test results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        # Print summary
        logger.info(f"\nTest Summary:")
        logger.info(f"Total Test Time: {total_time:.2f}s")
        if 'successful_tests' in selfrag_results:
            logger.info(f"Self-RAG Tests Passed: {selfrag_results['successful_tests']}/{selfrag_results['total_samples']}")
        
        return comprehensive_results


def main():
    """Main function to run Self-RAG tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Self-RAG System")
    parser.add_argument("--model", default="gpt2", help="Model name to use for testing")
    parser.add_argument("--component-only", action="store_true", help="Test only Self-LLM components")
    parser.add_argument("--system-only", action="store_true", help="Test only Self-RAG system")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    tester = SelfRAGTester(model_name=args.model)
    
    try:
        if args.component_only:
            results = tester.test_selfllm_components()
        elif args.system_only:
            results = tester.test_selfrag_system()
        else:
            results = tester.run_comprehensive_test()
        
        logger.info("Testing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
