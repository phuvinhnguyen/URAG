#!/usr/bin/env python3
"""
Test script for HyDE and Fusion RAG systems
Provides detailed logging and step-by-step execution tracking
"""

import sys
import os
import json
from datetime import datetime
from loguru import logger
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from systems.hyderag import HyDERAGSystem
from systems.fusionrag import FusionRAGSystem

# Configure detailed logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

class RAGSystemTester:
    """Comprehensive tester for RAG systems with detailed logging"""
    
    def __init__(self):
        self.test_results = {}
        self.test_samples = self._create_test_samples()
        
    def _create_test_samples(self) -> List[Dict[str, Any]]:
        """Create diverse test samples to evaluate RAG systems"""
        return [
            {
                "id": "test_001",
                "question": "What is the capital of France?",
                "options": ["A", "B", "C", "D"],
                "option_texts": {"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"},
                "correct_answer": "B",
                "category": "geography",
                "search_results": []  # Empty to test built-in knowledge base
            },
            {
                "id": "test_002", 
                "question": "Who wrote Pride and Prejudice?",
                "options": ["A", "B", "C", "D"],
                "option_texts": {"A": "Charles Dickens", "B": "Jane Austen", "C": "Charlotte Bronte", "D": "George Eliot"},
                "correct_answer": "B",
                "category": "literature",
                "search_results": []
            },
            {
                "id": "test_003",
                "question": "What is the largest planet in our solar system?",
                "options": ["A", "B", "C", "D"],
                "option_texts": {"A": "Earth", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
                "correct_answer": "C", 
                "category": "astronomy",
                "search_results": []
            },
            {
                "id": "test_004",
                "question": "What programming language is known for its use in data science?",
                "options": ["A", "B", "C", "D"],
                "option_texts": {"A": "Java", "B": "Python", "C": "C++", "D": "JavaScript"},
                "correct_answer": "B",
                "category": "technology",
                "search_results": []
            },
            {
                "id": "test_005",
                "question": "What is the main component of Earth's atmosphere?",
                "options": ["A", "B", "C", "D"],
                "option_texts": {"A": "Oxygen", "B": "Carbon Dioxide", "C": "Nitrogen", "D": "Hydrogen"},
                "correct_answer": "C",
                "category": "science",
                "search_results": [
                    {
                        "page_snippet": "Earth's atmosphere is composed primarily of nitrogen (78%) and oxygen (21%), with small amounts of other gases.",
                        "page_result": "The atmosphere of Earth is the layer of gases, commonly known as air, retained by Earth's gravity that surrounds the planet and forms its planetary atmosphere. The atmosphere protects life on Earth by creating pressure allowing for liquid water to exist on the surface, absorbing ultraviolet solar radiation, warming the surface through heat retention (greenhouse effect), and reducing temperature extremes between day and night."
                    }
                ]
            }
        ]
    
    def print_separator(self, title: str, char: str = "=", length: int = 80):
        """Print a formatted separator with title"""
        padding = (length - len(title) - 2) // 2
        separator = char * padding + f" {title} " + char * padding
        if len(separator) < length:
            separator += char
        print(f"\n{separator}")
    
    def print_step(self, step_num: int, description: str):
        """Print a formatted step description"""
        print(f"\n🔹 Step {step_num}: {description}")
        logger.info(f"STEP {step_num}: {description}")
    
    def test_hyde_system(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test HyDE RAG system with detailed logging"""
        self.print_separator(f"TESTING HYDE RAG - {sample['id']}")
        
        print(f"📝 Question: {sample['question']}")
        print(f"📚 Category: {sample['category']}")
        if 'option_texts' in sample:
            print(f"📋 Options:")
            for option, text in sample['option_texts'].items():
                marker = "✅" if option == sample['correct_answer'] else "  "
                print(f"  {marker} {option}: {text}")
        else:
            print(f"✅ Correct Answer: {sample['correct_answer']}")
        
        try:
            # Initialize HyDE system
            self.print_step(1, "Initializing HyDE RAG System")
            hyde_system = HyDERAGSystem(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                device="auto"
            )
            
            # Process sample
            self.print_step(2, "Processing sample through HyDE RAG")
            result = hyde_system.process_sample(sample)
            
            # Display results
            self.print_step(3, "Analyzing HyDE Results")
            self._display_results("HyDE", result, sample)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in HyDE system test: {str(e)}")
            return {"error": str(e), "system": "hyde"}
    
    def test_fusion_system(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test Fusion RAG system with detailed logging"""
        self.print_separator(f"TESTING FUSION RAG - {sample['id']}")
        
        print(f"📝 Question: {sample['question']}")
        print(f"📚 Category: {sample['category']}")
        if 'option_texts' in sample:
            print(f"📋 Options:")
            for option, text in sample['option_texts'].items():
                marker = "✅" if option == sample['correct_answer'] else "  "
                print(f"  {marker} {option}: {text}")
        else:
            print(f"✅ Correct Answer: {sample['correct_answer']}")
        
        try:
            # Initialize Fusion system
            self.print_step(1, "Initializing Fusion RAG System")
            fusion_system = FusionRAGSystem(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                device="auto",
                num_queries=3,
                k=60
            )
            
            # Process sample
            self.print_step(2, "Processing sample through Fusion RAG")
            result = fusion_system.process_sample(sample)
            
            # Display results
            self.print_step(3, "Analyzing Fusion Results")
            self._display_results("Fusion", result, sample)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Fusion system test: {str(e)}")
            return {"error": str(e), "system": "fusion"}
    
    def _display_results(self, system_name: str, result: Dict[str, Any], original_sample: Dict[str, Any]):
        """Display detailed results from RAG system"""
        print(f"\n📊 {system_name} RAG Results:")
        print("-" * 50)
        
        # Basic info
        print(f"🆔 Sample ID: {result.get('id', 'N/A')}")
        print(f"🤖 System Type: {result.get('system_type', 'N/A')}")
        print(f"🔧 Technique: {result.get('technique', 'N/A')}")
        
        # Generated response
        generated_response = result.get('generated_response', 'N/A')
        print(f"💬 Generated Response: {generated_response}")
        
        # Predicted answer
        predicted_answer = result.get('predicted_answer', 'N/A')
        print(f"🎯 Predicted Answer: {predicted_answer}")
        
        # Correctness check
        correct_answer = original_sample.get('correct_answer', '')
        is_correct = predicted_answer.strip() == correct_answer.strip()
        
        # Display expected answer text if available
        expected_text = correct_answer
        if 'option_texts' in original_sample and correct_answer in original_sample['option_texts']:
            expected_text = f"{correct_answer} ({original_sample['option_texts'][correct_answer]})"
        
        print(f"✅ Correct: {'YES' if is_correct else 'NO'} (Expected: {expected_text})")
        
        # Option probabilities
        if 'option_probabilities' in result:
            print(f"\n📈 Option Probabilities:")
            for option, prob in result['option_probabilities'].items():
                marker = "👆" if option == predicted_answer else "  "
                option_display = option
                if 'option_texts' in original_sample and option in original_sample['option_texts']:
                    option_display = f"{option} ({original_sample['option_texts'][option]})"
                print(f"{marker} {option_display}: {prob:.4f}")
        
        # System-specific information
        if system_name == "HyDE":
            self._display_hyde_specific(result)
        elif system_name == "Fusion":
            self._display_fusion_specific(result)
        
        # Method and additional info
        method = result.get('method', 'N/A')
        print(f"\n🔍 Method Used: {method}")
        
        if 'num_samples_generated' in result:
            print(f"🔄 Samples Generated: {result['num_samples_generated']}")
    
    def _display_hyde_specific(self, result: Dict[str, Any]):
        """Display HyDE-specific information"""
        print(f"\n🔬 HyDE-Specific Information:")
        
        if 'hypothetical_document' in result:
            hyp_doc = result['hypothetical_document'][:200] + "..." if len(result['hypothetical_document']) > 200 else result['hypothetical_document']
            print(f"📄 Hypothetical Document: {hyp_doc}")
        
        if 'retrieved_docs' in result:
            print(f"📚 Retrieved Documents: {result.get('num_retrieved_docs', 0)}")
            
        if 'document_source' in result:
            print(f"📍 Document Source: {result['document_source']}")
            
        if 'hyde_enhanced' in result:
            print(f"🚀 HyDE Enhanced: {result['hyde_enhanced']}")
    
    def _display_fusion_specific(self, result: Dict[str, Any]):
        """Display Fusion-specific information"""
        print(f"\n🔀 Fusion-Specific Information:")
        
        if 'diverse_queries' in result:
            print(f"🎯 Diverse Queries Generated:")
            for i, query in enumerate(result['diverse_queries'], 1):
                print(f"   {i}. {query}")
        
        if 'retrieved_docs' in result:
            print(f"📚 Retrieved Documents: {result.get('num_retrieved_docs', 0)}")
            
        if 'document_source' in result:
            print(f"📍 Document Source: {result['document_source']}")
            
        if 'rrf_k_parameter' in result:
            print(f"⚙️ RRF K Parameter: {result['rrf_k_parameter']}")
            
        if 'fusion_enhanced' in result:
            print(f"🚀 Fusion Enhanced: {result['fusion_enhanced']}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test of both RAG systems"""
        self.print_separator("RAG SYSTEMS COMPREHENSIVE TEST", "=", 100)
        print(f"🚀 Starting comprehensive test at {datetime.now()}")
        print(f"📊 Testing {len(self.test_samples)} samples on 2 RAG systems")
        
        all_results = {}
        
        for i, sample in enumerate(self.test_samples, 1):
            print(f"\n{'='*20} SAMPLE {i}/{len(self.test_samples)} {'='*20}")
            
            # Test HyDE
            hyde_result = self.test_hyde_system(sample.copy())
            
            # Test Fusion
            fusion_result = self.test_fusion_system(sample.copy())
            
            # Store results
            sample_id = sample['id']
            all_results[sample_id] = {
                'sample': sample,
                'hyde_result': hyde_result,
                'fusion_result': fusion_result
            }
        
        # Summary
        self._print_summary(all_results)
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _print_summary(self, all_results: Dict[str, Any]):
        """Print summary of test results"""
        self.print_separator("TEST SUMMARY", "=", 100)
        
        total_samples = len(all_results)
        hyde_correct = 0
        fusion_correct = 0
        
        print(f"📊 Total Samples Tested: {total_samples}")
        print(f"\n📈 Results by Sample:")
        print("-" * 80)
        
        for sample_id, results in all_results.items():
            sample = results['sample']
            correct_answer = sample['correct_answer'].strip()
            
            hyde_pred = results['hyde_result'].get('predicted_answer', '').strip()
            fusion_pred = results['fusion_result'].get('predicted_answer', '').strip()
            
            hyde_success = hyde_pred == correct_answer
            fusion_success = fusion_pred == correct_answer
            
            if hyde_success:
                hyde_correct += 1
            if fusion_success:
                fusion_correct += 1
            
            # Display expected answer with text if available
            expected_display = correct_answer
            if 'option_texts' in sample and correct_answer in sample['option_texts']:
                expected_display = f"{correct_answer} ({sample['option_texts'][correct_answer]})"
            
            print(f"{sample_id}: {sample['question'][:50]}...")
            print(f"  Expected: {expected_display}")
            print(f"  HyDE:     {results['hyde_result'].get('predicted_answer', 'N/A')} {'✅' if hyde_success else '❌'}")
            print(f"  Fusion:   {results['fusion_result'].get('predicted_answer', 'N/A')} {'✅' if fusion_success else '❌'}")
            print()
        
        # Overall accuracy
        hyde_accuracy = (hyde_correct / total_samples) * 100
        fusion_accuracy = (fusion_correct / total_samples) * 100
        
        print("🎯 ACCURACY SUMMARY:")
        print(f"  HyDE RAG:    {hyde_correct}/{total_samples} ({hyde_accuracy:.1f}%)")
        print(f"  Fusion RAG:  {fusion_correct}/{total_samples} ({fusion_accuracy:.1f}%)")
        
        if hyde_accuracy > fusion_accuracy:
            print(f"🏆 Winner: HyDE RAG (+{hyde_accuracy - fusion_accuracy:.1f}%)")
        elif fusion_accuracy > hyde_accuracy:
            print(f"🏆 Winner: Fusion RAG (+{fusion_accuracy - hyde_accuracy:.1f}%)")
        else:
            print("🤝 Tie!")
    
    def _save_results(self, all_results: Dict[str, Any]):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_test_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        serializable_results = {}
        for sample_id, results in all_results.items():
            serializable_results[sample_id] = {
                'sample': results['sample'],
                'hyde_result': results['hyde_result'],
                'fusion_result': results['fusion_result'],
                'timestamp': timestamp
            }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

def main():
    """Main function to run the RAG system tests"""
    print("🚀 RAG Systems Test Suite")
    print("Testing HyDE and Fusion RAG implementations")
    
    tester = RAGSystemTester()
    
    try:
        results = tester.run_comprehensive_test()
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        print(f"\n❌ Test suite failed: {str(e)}")

if __name__ == "__main__":
    main()
