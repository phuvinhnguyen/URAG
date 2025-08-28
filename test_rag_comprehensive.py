#!/usr/bin/env python3
"""
Comprehensive test suite for HyDERAG and FusionRAG systems.

Tests:
1. HyDE hypothetical document generation
2. Fusion multi-query generation 
3. Document extraction from page_result
4. Web crawling fallback when page_result missing
5. End-to-end processing comparison
"""

import json
import sys
import os
from typing import Dict, Any, List
from loguru import logger

# Add the project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Test import availability before running tests
def check_imports():
    """Check if all required modules can be imported."""
    missing_modules = []
    
    try:
        from utils.clean import clean_web_content
        from utils.get_html import get_web_content
        from utils.vectordb import QdrantVectorDB
    except ImportError as e:
        print(f"⚠️ Utils import warning: {e}")
        missing_modules.append("utils")
    
    try:
        from systems.hyderag import HyDERAGSystem
    except ImportError as e:
        print(f"⚠️ HyDERAG import warning: {e}")
        missing_modules.append("hyderag")
    
    try:
        from systems.fusionrag import FusionRAGSystem
    except ImportError as e:
        print(f"⚠️ FusionRAG import warning: {e}")
        missing_modules.append("fusionrag")
    
    if missing_modules:
        print(f"❌ Some modules cannot be imported: {missing_modules}")
        print("Make sure you're running from the project root directory")
        return False
    else:
        print("✅ All required modules can be imported successfully")
        return True

# Configure logger for testing
logger.remove()
logger.add(lambda msg: print(f"[LOG] {msg}", end=""), colorize=True, level="DEBUG")

class RAGTester:
    """Comprehensive tester for RAG systems."""
    
    def __init__(self):
        self.test_results = {
            'hyde': {},
            'fusion': {},
            'comparison': {}
        }
        
    def create_test_samples(self) -> Dict[str, Dict[str, Any]]:
        """Create various test samples for different scenarios."""
        
        return {
            # Scenario 1: Full CRAG sample with page_result
            'full_crag_sample': {
                'id': 'test_001',
                'question': 'What is artificial intelligence and how does machine learning relate to it?',
                'search_results': [
                    {
                        'page_snippet': 'AI is intelligence demonstrated by machines',
                        'page_result': '''
                        <html>
                        <head><title>Artificial Intelligence Overview</title></head>
                        <body>
                            <h1>Artificial Intelligence (AI)</h1>
                            <p>Artificial intelligence (AI) is intelligence demonstrated by machines, 
                            in contrast to the natural intelligence displayed by humans and animals. 
                            Leading AI textbooks define the field as the study of "intelligent agents": 
                            any device that perceives its environment and takes actions that maximize 
                            its chance of successfully achieving its goals.</p>
                            
                            <h2>Machine Learning</h2>
                            <p>Machine learning (ML) is a subset of artificial intelligence that 
                            focuses on algorithms that can learn from and make predictions or 
                            decisions based on data. Rather than being explicitly programmed 
                            to perform a task, ML systems improve their performance through experience.</p>
                            
                            <h2>Types of AI</h2>
                            <ul>
                                <li>Narrow AI: AI that is designed to perform a specific task</li>
                                <li>General AI: AI that can perform any intellectual task</li>
                                <li>Superintelligence: AI that surpasses human intelligence</li>
                            </ul>
                        </body>
                        </html>
                        ''',
                        'page_url': 'https://example.com/ai-overview'
                    },
                    {
                        'page_snippet': 'Deep learning uses neural networks',
                        'page_result': '''
                        <html>
                        <body>
                            <h1>Deep Learning</h1>
                            <p>Deep learning is a subset of machine learning that uses artificial 
                            neural networks with multiple layers (hence "deep") to model and 
                            understand complex patterns in data. These networks are inspired 
                            by the structure and function of the human brain.</p>
                            
                            <p>Deep learning has revolutionized many fields including computer vision, 
                            natural language processing, and speech recognition. Popular applications 
                            include image classification, language translation, and autonomous vehicles.</p>
                        </body>
                        </html>
                        ''',
                        'page_url': 'https://example.com/deep-learning'
                    }
                ],
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'A'
            },
            
            # Scenario 2: Sample with page_url but no page_result (needs crawling)
            'crawling_sample': {
                'id': 'test_002', 
                'question': 'What are the main programming languages used in data science?',
                'search_results': [
                    {
                        'page_snippet': 'Python and R are popular for data science',
                        'page_url': 'https://httpbin.org/html',  # Test URL that returns HTML
                        # No page_result - should trigger crawling
                    },
                    {
                        'page_snippet': 'SQL is essential for databases',
                        'page_url': 'https://httpbin.org/robots.txt',  # Simple text URL
                    }
                ],
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'B'
            },
            
            # Scenario 3: Legacy format sample (no page_result, no page_url)
            'legacy_sample': {
                'id': 'test_003',
                'question': 'Who painted the Mona Lisa?',
                'search_results': [
                    {
                        'content': 'Leonardo da Vinci painted the Mona Lisa between 1503-1519. It is housed in the Louvre Museum.',
                        'title': 'Mona Lisa Painting'
                    },
                    {
                        'snippet': 'The Mona Lisa is famous for its enigmatic smile and innovative painting techniques.',
                        'body': 'Da Vinci used sfumato technique to create the mysterious atmosphere of the painting.'
                    }
                ],
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'C'
            },
            
            # Scenario 4: Empty/minimal sample
            'minimal_sample': {
                'id': 'test_004',
                'question': 'What is the capital of France?',
                'search_results': [],  # No search results
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'A'
            }
        }

    def test_hyde_hypothetical_generation(self):
        """Test HyDE hypothetical document generation."""
        print("\n" + "="*60)
        print("🧪 TESTING HYDE HYPOTHETICAL DOCUMENT GENERATION")
        print("="*60)
        
        try:
            from systems.hyderag import HyDERAGSystem
            
            # Initialize HyDE system
            hyde_system = HyDERAGSystem()
            
            test_questions = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What are the benefits of renewable energy?",
                "Who invented the telephone?",
                "What causes climate change?"
            ]
            
            for i, question in enumerate(test_questions, 1):
                print(f"\n📝 Test {i}: {question}")
                
                try:
                    # Generate hypothetical document
                    hypothetical_doc = hyde_system.llm_system.generate_hypothetical_document(question)
                    
                    print(f"✅ Generated hypothetical document ({len(hypothetical_doc)} chars):")
                    print(f"   {hypothetical_doc[:200]}{'...' if len(hypothetical_doc) > 200 else ''}")
                    
                    # Validate hypothetical document
                    assert len(hypothetical_doc) > 50, "Hypothetical document should be substantial"
                    assert question.lower().split()[0] in hypothetical_doc.lower(), "Should relate to question"
                    
                    self.test_results['hyde'][f'hypothetical_test_{i}'] = {
                        'status': 'passed',
                        'question': question,
                        'doc_length': len(hypothetical_doc),
                        'doc_preview': hypothetical_doc[:100]
                    }
                    
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    self.test_results['hyde'][f'hypothetical_test_{i}'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            print(f"\n✅ HyDE hypothetical generation test completed")
            
        except Exception as e:
            print(f"❌ HyDE system initialization failed: {e}")
            self.test_results['hyde']['system_init'] = {'status': 'failed', 'error': str(e)}

    def test_fusion_multi_query_generation(self):
        """Test Fusion multi-query generation."""
        print("\n" + "="*60)
        print("🧪 TESTING FUSION MULTI-QUERY GENERATION")
        print("="*60)
        
        try:
            from systems.fusionrag import FusionRAGSystem
            
            # Initialize Fusion system
            fusion_system = FusionRAGSystem(num_queries=4)
            
            test_questions = [
                "What is artificial intelligence?",
                "How does renewable energy help the environment?", 
                "What are the causes of World War II?",
                "How do neural networks work?",
                "What is the theory of relativity?"
            ]
            
            for i, question in enumerate(test_questions, 1):
                print(f"\n📝 Test {i}: {question}")
                
                try:
                    # Generate diverse queries
                    diverse_queries = fusion_system.llm_system.generate_diverse_queries(question)
                    
                    print(f"✅ Generated {len(diverse_queries)} diverse queries:")
                    for j, query in enumerate(diverse_queries, 1):
                        print(f"   {j}. {query}")
                    
                    # Validate diverse queries
                    assert len(diverse_queries) >= 2, "Should generate multiple queries"
                    assert all(len(q.strip()) > 10 for q in diverse_queries), "Queries should be substantial"
                    
                    # Check diversity (queries should be different)
                    unique_queries = set(q.lower().strip() for q in diverse_queries)
                    assert len(unique_queries) >= len(diverse_queries) * 0.7, "Queries should be diverse"
                    
                    self.test_results['fusion'][f'multi_query_test_{i}'] = {
                        'status': 'passed',
                        'question': question,
                        'num_queries': len(diverse_queries),
                        'queries': diverse_queries,
                        'diversity_score': len(unique_queries) / len(diverse_queries)
                    }
                    
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    self.test_results['fusion'][f'multi_query_test_{i}'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            print(f"\n✅ Fusion multi-query generation test completed")
            
        except Exception as e:
            print(f"❌ Fusion system initialization failed: {e}")
            self.test_results['fusion']['system_init'] = {'status': 'failed', 'error': str(e)}

    def test_document_extraction(self):
        """Test document extraction from different sources."""
        print("\n" + "="*60)
        print("🧪 TESTING DOCUMENT EXTRACTION")
        print("="*60)
        
        test_samples = self.create_test_samples()
        
        for scenario_name, sample in test_samples.items():
            print(f"\n📄 Testing scenario: {scenario_name}")
            
            # Test HyDE extraction
            try:
                from systems.hyderag import HyDERAGSystem
                hyde_system = HyDERAGSystem()
                hyde_docs = hyde_system._extract_existing_documents(sample)
                
                print(f"   HyDE extracted {len(hyde_docs)} documents")
                for i, doc in enumerate(hyde_docs):
                    print(f"     Doc {i+1}: {len(doc)} chars - {doc[:80]}...")
                    
                self.test_results['hyde'][f'extraction_{scenario_name}'] = {
                    'status': 'passed',
                    'num_docs': len(hyde_docs),
                    'doc_lengths': [len(doc) for doc in hyde_docs]
                }
                
            except Exception as e:
                print(f"   ❌ HyDE extraction failed: {e}")
                self.test_results['hyde'][f'extraction_{scenario_name}'] = {
                    'status': 'failed', 
                    'error': str(e)
                }
            
            # Test Fusion extraction
            try:
                from systems.fusionrag import FusionRAGSystem
                fusion_system = FusionRAGSystem()
                fusion_docs = fusion_system._extract_existing_documents(sample)
                
                print(f"   Fusion extracted {len(fusion_docs)} documents")
                for i, doc in enumerate(fusion_docs):
                    print(f"     Doc {i+1}: {len(doc)} chars - {doc[:80]}...")
                    
                self.test_results['fusion'][f'extraction_{scenario_name}'] = {
                    'status': 'passed',
                    'num_docs': len(fusion_docs),
                    'doc_lengths': [len(doc) for doc in fusion_docs]
                }
                
            except Exception as e:
                print(f"   ❌ Fusion extraction failed: {e}")
                self.test_results['fusion'][f'extraction_{scenario_name}'] = {
                    'status': 'failed',
                    'error': str(e)
                }

    def test_web_crawling_fallback(self):
        """Test web crawling when page_result is missing."""
        print("\n" + "="*60)
        print("🧪 TESTING WEB CRAWLING FALLBACK")
        print("="*60)
        
        # Create sample that requires crawling
        crawling_sample = {
            'question': 'Test crawling',
            'search_results': [
                {
                    'page_snippet': 'Test snippet',
                    'page_url': 'https://httpbin.org/html'  # Returns simple HTML
                    # No page_result
                }
            ]
        }
        
        print("🌐 Testing with real URL (httpbin.org/html)...")
        
        # Test HyDE crawling
        try:
            from systems.hyderag import HyDERAGSystem
            hyde_system = HyDERAGSystem()
            hyde_docs = hyde_system._extract_existing_documents(crawling_sample)
            
            if hyde_docs and len(hyde_docs[0]) > 100:
                print(f"✅ HyDE successfully crawled content ({len(hyde_docs[0])} chars)")
                print(f"   Preview: {hyde_docs[0][:150]}...")
                self.test_results['hyde']['crawling_test'] = {
                    'status': 'passed',
                    'crawled_length': len(hyde_docs[0])
                }
            else:
                print("⚠️ HyDE crawling returned minimal content")
                self.test_results['hyde']['crawling_test'] = {
                    'status': 'warning',
                    'note': 'Minimal content returned'
                }
                
        except Exception as e:
            print(f"❌ HyDE crawling failed: {e}")
            self.test_results['hyde']['crawling_test'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Test Fusion crawling
        try:
            from systems.fusionrag import FusionRAGSystem
            fusion_system = FusionRAGSystem()
            fusion_docs = fusion_system._extract_existing_documents(crawling_sample)
            
            if fusion_docs and len(fusion_docs[0]) > 100:
                print(f"✅ Fusion successfully crawled content ({len(fusion_docs[0])} chars)")
                print(f"   Preview: {fusion_docs[0][:150]}...")
                self.test_results['fusion']['crawling_test'] = {
                    'status': 'passed',
                    'crawled_length': len(fusion_docs[0])
                }
            else:
                print("⚠️ Fusion crawling returned minimal content")
                self.test_results['fusion']['crawling_test'] = {
                    'status': 'warning',
                    'note': 'Minimal content returned'
                }
                
        except Exception as e:
            print(f"❌ Fusion crawling failed: {e}")
            self.test_results['fusion']['crawling_test'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_end_to_end_processing(self):
        """Test complete end-to-end processing."""
        print("\n" + "="*60)
        print("🧪 TESTING END-TO-END PROCESSING")
        print("="*60)
        
        # Use a simple test sample
        test_sample = {
            'id': 'e2e_test',
            'question': 'What is machine learning?',
            'search_results': [
                {
                    'page_snippet': 'ML is a subset of AI',
                    'page_result': '<html><body><h1>Machine Learning</h1><p>Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data.</p></body></html>'
                }
            ],
            'options': ['A', 'B', 'C', 'D'],
            'correct_answer': 'A'
        }
        
        # Test HyDE end-to-end
        print("\n🔄 Testing HyDE end-to-end processing...")
        try:
            from systems.hyderag import HyDERAGSystem
            hyde_system = HyDERAGSystem()
            
            # Process sample
            hyde_result = hyde_system.process_sample(test_sample)
            
            print(f"✅ HyDE processing completed")
            print(f"   Generated response: {hyde_result.get('generated_response', '')[:100]}...")
            print(f"   Predicted answer: {hyde_result.get('predicted_answer', 'N/A')}")
            print(f"   System type: {hyde_result.get('system_type', 'N/A')}")
            print(f"   Retrieved docs: {hyde_result.get('num_retrieved_docs', 0)}")
            print(f"   HyDE enhanced: {hyde_result.get('hyde_enhanced', False)}")
            
            self.test_results['hyde']['e2e_test'] = {
                'status': 'passed',
                'has_response': bool(hyde_result.get('generated_response')),
                'has_prediction': bool(hyde_result.get('predicted_answer')),
                'num_retrieved': hyde_result.get('num_retrieved_docs', 0),
                'hyde_enhanced': hyde_result.get('hyde_enhanced', False)
            }
            
        except Exception as e:
            print(f"❌ HyDE end-to-end failed: {e}")
            self.test_results['hyde']['e2e_test'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Test Fusion end-to-end
        print("\n🔄 Testing Fusion end-to-end processing...")
        try:
            from systems.fusionrag import FusionRAGSystem
            fusion_system = FusionRAGSystem()
            
            # Process sample
            fusion_result = fusion_system.process_sample(test_sample)
            
            print(f"✅ Fusion processing completed")
            print(f"   Generated response: {fusion_result.get('generated_response', '')[:100]}...")
            print(f"   Predicted answer: {fusion_result.get('predicted_answer', 'N/A')}")
            print(f"   System type: {fusion_result.get('system_type', 'N/A')}")
            print(f"   Retrieved docs: {fusion_result.get('num_retrieved_docs', 0)}")
            print(f"   Fusion enhanced: {fusion_result.get('fusion_enhanced', False)}")
            print(f"   Diverse queries: {fusion_result.get('diverse_queries', [])}")
            
            self.test_results['fusion']['e2e_test'] = {
                'status': 'passed',
                'has_response': bool(fusion_result.get('generated_response')),
                'has_prediction': bool(fusion_result.get('predicted_answer')),
                'num_retrieved': fusion_result.get('num_retrieved_docs', 0),
                'fusion_enhanced': fusion_result.get('fusion_enhanced', False),
                'num_queries': len(fusion_result.get('diverse_queries', []))
            }
            
        except Exception as e:
            print(f"❌ Fusion end-to-end failed: {e}")
            self.test_results['fusion']['e2e_test'] = {
                'status': 'failed',
                'error': str(e)
            }

    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # HyDE Results
        print("\n🔍 HyDE RAG Test Results:")
        hyde_passed = sum(1 for test in self.test_results['hyde'].values() 
                         if isinstance(test, dict) and test.get('status') == 'passed')
        hyde_total = len(self.test_results['hyde'])
        print(f"   Overall: {hyde_passed}/{hyde_total} tests passed")
        
        for test_name, result in self.test_results['hyde'].items():
            if isinstance(result, dict):
                status_emoji = "✅" if result.get('status') == 'passed' else "❌"
                print(f"   {status_emoji} {test_name}: {result.get('status', 'unknown')}")
                if result.get('status') == 'failed':
                    print(f"      Error: {result.get('error', 'Unknown error')}")
        
        # Fusion Results  
        print("\n🔀 Fusion RAG Test Results:")
        fusion_passed = sum(1 for test in self.test_results['fusion'].values()
                           if isinstance(test, dict) and test.get('status') == 'passed')
        fusion_total = len(self.test_results['fusion'])
        print(f"   Overall: {fusion_passed}/{fusion_total} tests passed")
        
        for test_name, result in self.test_results['fusion'].items():
            if isinstance(result, dict):
                status_emoji = "✅" if result.get('status') == 'passed' else "❌"
                print(f"   {status_emoji} {test_name}: {result.get('status', 'unknown')}")
                if result.get('status') == 'failed':
                    print(f"      Error: {result.get('error', 'Unknown error')}")
        
        # Summary
        total_passed = hyde_passed + fusion_passed
        total_tests = hyde_total + fusion_total
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("   🎉 EXCELLENT! Both systems are working well.")
        elif success_rate >= 60:
            print("   👍 GOOD! Most features are working correctly.")
        else:
            print("   ⚠️ NEEDS ATTENTION! Some critical issues found.")
        
        # Save detailed results
        with open('rag_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Detailed results saved to: rag_test_results.json")

    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting Comprehensive RAG System Testing...")
        print("🎯 Testing: HyDE hypothetical docs, Fusion multi-queries, document extraction, web crawling")
        
        # Check imports first
        print("\n🔍 Checking module imports...")
        if not check_imports():
            print("❌ Cannot proceed with tests due to import issues")
            print("💡 Make sure you're running from the URAG project root directory")
            print("💡 Try: cd C:\\Users\\ad\\Desktop\\URAG && python test_rag_comprehensive.py")
            return
        
        # Run all test suites
        self.test_hyde_hypothetical_generation()
        self.test_fusion_multi_query_generation() 
        self.test_document_extraction()
        self.test_web_crawling_fallback()
        self.test_end_to_end_processing()
        
        # Generate final report
        self.generate_test_report()


if __name__ == "__main__":
    tester = RAGTester()
    tester.run_all_tests()
