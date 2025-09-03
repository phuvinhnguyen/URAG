#!/usr/bin/env python3
"""
Simple test suite for HyDERAG and FusionRAG systems.
Handles import issues gracefully and provides clear feedback.
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    results = {
        'utils_available': False,
        'hyderag_available': False, 
        'fusionrag_available': False,
        'errors': []
    }
    
    # Test utils imports
    try:
        from utils.clean import clean_web_content
        from utils.get_html import get_web_content
        print("   ✅ utils.clean and utils.get_html imported successfully")
        results['utils_available'] = True
    except ImportError as e:
        print(f"   ❌ Utils import failed: {e}")
        results['errors'].append(f"Utils: {e}")
    
    # Test HyDERAG import
    try:
        from systems.hyderag import HyDERAGSystem
        print("   ✅ HyDERAG imported successfully")
        results['hyderag_available'] = True
    except ImportError as e:
        print(f"   ❌ HyDERAG import failed: {e}")
        results['errors'].append(f"HyDERAG: {e}")
    
    # Test FusionRAG import
    try:
        from systems.fusionrag import FusionRAGSystem
        print("   ✅ FusionRAG imported successfully")
        results['fusionrag_available'] = True
    except ImportError as e:
        print(f"   ❌ FusionRAG import failed: {e}")
        results['errors'].append(f"FusionRAG: {e}")
    
    return results

def test_document_extraction():
    """Test document extraction with page_result."""
    print("\n📄 Testing document extraction...")
    
    # Test data with page_result
    test_sample = {
        'question': 'What is AI?',
        'search_results': [
            {
                'page_snippet': 'AI is artificial intelligence',
                'page_result': '''
                <html>
                <body>
                    <h1>Artificial Intelligence</h1>
                    <p>Artificial intelligence (AI) is intelligence demonstrated by machines, 
                    in contrast to natural intelligence displayed by humans and animals.</p>
                    <p>AI research has been highly successful in developing effective 
                    techniques for solving a wide range of problems.</p>
                </body>
                </html>
                '''
            }
        ]
    }
    
    results = {'hyde': None, 'fusion': None}
    
    # Test HyDERAG extraction
    try:
        from systems.hyderag import HyDERAGSystem
        hyde_system = HyDERAGSystem()
        hyde_docs = hyde_system._extract_existing_documents(test_sample)
        
        if hyde_docs:
            doc_length = len(hyde_docs[0])
            has_html_content = 'artificial intelligence' in hyde_docs[0].lower()
            print(f"   ✅ HyDERAG: Extracted {len(hyde_docs)} docs, length={doc_length}, has_content={has_html_content}")
            results['hyde'] = {'success': True, 'length': doc_length, 'has_content': has_html_content}
        else:
            print("   ⚠️ HyDERAG: No documents extracted")
            results['hyde'] = {'success': False, 'reason': 'No docs extracted'}
            
    except Exception as e:
        print(f"   ❌ HyDERAG extraction failed: {e}")
        results['hyde'] = {'success': False, 'error': str(e)}
    
    # Test FusionRAG extraction
    try:
        from systems.fusionrag import FusionRAGSystem
        fusion_system = FusionRAGSystem()
        fusion_docs = fusion_system._extract_existing_documents(test_sample)
        
        if fusion_docs:
            doc_length = len(fusion_docs[0])
            has_html_content = 'artificial intelligence' in fusion_docs[0].lower()
            print(f"   ✅ FusionRAG: Extracted {len(fusion_docs)} docs, length={doc_length}, has_content={has_html_content}")
            results['fusion'] = {'success': True, 'length': doc_length, 'has_content': has_html_content}
        else:
            print("   ⚠️ FusionRAG: No documents extracted")
            results['fusion'] = {'success': False, 'reason': 'No docs extracted'}
            
    except Exception as e:
        print(f"   ❌ FusionRAG extraction failed: {e}")
        results['fusion'] = {'success': False, 'error': str(e)}
    
    return results

def test_hypothetical_generation():
    """Test HyDE hypothetical document generation."""
    print("\n🧠 Testing HyDE hypothetical document generation...")
    
    test_questions = [
        "What is machine learning?",
        "How do neural networks work?"
    ]
    
    results = []
    
    try:
        from systems.hyderag import HyDERAGSystem
        hyde_system = HyDERAGSystem()
        
        for question in test_questions:
            try:
                hypothetical_doc = hyde_system.llm_system.generate_hypothetical_document(question)
                doc_length = len(hypothetical_doc)
                print(f"   ✅ '{question}' -> {doc_length} chars")
                print(f"      Preview: {hypothetical_doc[:100]}...")
                results.append({'question': question, 'success': True, 'length': doc_length})
            except Exception as e:
                print(f"   ❌ '{question}' failed: {e}")
                results.append({'question': question, 'success': False, 'error': str(e)})
                
    except Exception as e:
        print(f"   ❌ HyDE system initialization failed: {e}")
        return [{'success': False, 'error': str(e)}]
    
    return results

def test_multi_query_generation():
    """Test Fusion multi-query generation."""
    print("\n🔀 Testing Fusion multi-query generation...")
    
    test_questions = [
        "What causes climate change?",
        "How do computers work?"
    ]
    
    results = []
    
    try:
        from systems.fusionrag import FusionRAGSystem
        fusion_system = FusionRAGSystem(num_queries=3)
        
        for question in test_questions:
            try:
                diverse_queries = fusion_system.llm_system.generate_diverse_queries(question)
                num_queries = len(diverse_queries)
                print(f"   ✅ '{question}' -> {num_queries} queries:")
                for i, query in enumerate(diverse_queries, 1):
                    print(f"      {i}. {query}")
                results.append({'question': question, 'success': True, 'num_queries': num_queries, 'queries': diverse_queries})
            except Exception as e:
                print(f"   ❌ '{question}' failed: {e}")
                results.append({'question': question, 'success': False, 'error': str(e)})
                
    except Exception as e:
        print(f"   ❌ Fusion system initialization failed: {e}")
        return [{'success': False, 'error': str(e)}]
    
    return results

def test_web_crawling():
    """Test web crawling fallback."""
    print("\n🌐 Testing web crawling fallback...")
    
    # Test sample without page_result (needs crawling)
    crawling_sample = {
        'question': 'Test crawling',
        'search_results': [
            {
                'page_snippet': 'Test snippet',
                'page_url': 'https://httpbin.org/html'  # Simple test URL
                # No page_result - should trigger crawling
            }
        ]
    }
    
    results = {'hyde': None, 'fusion': None}
    
    # Test HyDE crawling
    try:
        from systems.hyderag import HyDERAGSystem
        hyde_system = HyDERAGSystem()
        hyde_docs = hyde_system._extract_existing_documents(crawling_sample)
        
        if hyde_docs and len(hyde_docs[0]) > 50:
            print(f"   ✅ HyDE crawling: {len(hyde_docs[0])} chars extracted")
            print(f"      Preview: {hyde_docs[0][:80]}...")
            results['hyde'] = {'success': True, 'length': len(hyde_docs[0])}
        else:
            print(f"   ⚠️ HyDE crawling: Minimal content ({len(hyde_docs[0]) if hyde_docs else 0} chars)")
            results['hyde'] = {'success': False, 'reason': 'Minimal content'}
            
    except Exception as e:
        print(f"   ❌ HyDE crawling failed: {e}")
        results['hyde'] = {'success': False, 'error': str(e)}
    
    # Test Fusion crawling
    try:
        from systems.fusionrag import FusionRAGSystem
        fusion_system = FusionRAGSystem()
        fusion_docs = fusion_system._extract_existing_documents(crawling_sample)
        
        if fusion_docs and len(fusion_docs[0]) > 50:
            print(f"   ✅ Fusion crawling: {len(fusion_docs[0])} chars extracted")
            print(f"      Preview: {fusion_docs[0][:80]}...")
            results['fusion'] = {'success': True, 'length': len(fusion_docs[0])}
        else:
            print(f"   ⚠️ Fusion crawling: Minimal content ({len(fusion_docs[0]) if fusion_docs else 0} chars)")
            results['fusion'] = {'success': False, 'reason': 'Minimal content'}
            
    except Exception as e:
        print(f"   ❌ Fusion crawling failed: {e}")
        results['fusion'] = {'success': False, 'error': str(e)}
    
    return results

def main():
    """Run all tests and generate report."""
    print("=" * 60)
    print("🚀 RAG Systems Test Suite")
    print("=" * 60)
    print("Testing: HyDERAG and FusionRAG capabilities")
    print("Features: Hypothetical docs, Multi-queries, Document extraction, Web crawling")
    
    # Run all tests
    import_results = test_imports()
    extraction_results = test_document_extraction()
    hypothetical_results = test_hypothetical_generation()
    multiquery_results = test_multi_query_generation()
    crawling_results = test_web_crawling()
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    # Import status
    print(f"📦 Imports:")
    print(f"   Utils: {'✅' if import_results['utils_available'] else '❌'}")
    print(f"   HyDERAG: {'✅' if import_results['hyderag_available'] else '❌'}")
    print(f"   FusionRAG: {'✅' if import_results['fusionrag_available'] else '❌'}")
    
    if import_results['errors']:
        print(f"   Errors: {len(import_results['errors'])}")
        for error in import_results['errors']:
            print(f"     - {error}")
    
    # Document extraction
    print(f"📄 Document Extraction:")
    if extraction_results['hyde']:
        hyde_status = "✅" if extraction_results['hyde']['success'] else "❌"
        print(f"   HyDE: {hyde_status}")
        if extraction_results['hyde']['success']:
            print(f"     Length: {extraction_results['hyde']['length']} chars")
            print(f"     Has content: {extraction_results['hyde']['has_content']}")
    
    if extraction_results['fusion']:
        fusion_status = "✅" if extraction_results['fusion']['success'] else "❌"
        print(f"   Fusion: {fusion_status}")
        if extraction_results['fusion']['success']:
            print(f"     Length: {extraction_results['fusion']['length']} chars")
            print(f"     Has content: {extraction_results['fusion']['has_content']}")
    
    # Hypothetical generation
    print(f"🧠 HyDE Hypothetical Generation:")
    if hypothetical_results:
        success_count = sum(1 for r in hypothetical_results if r.get('success'))
        print(f"   Success: {success_count}/{len(hypothetical_results)}")
    
    # Multi-query generation
    print(f"🔀 Fusion Multi-Query Generation:")
    if multiquery_results:
        success_count = sum(1 for r in multiquery_results if r.get('success'))
        print(f"   Success: {success_count}/{len(multiquery_results)}")
    
    # Web crawling
    print(f"🌐 Web Crawling:")
    if crawling_results['hyde']:
        hyde_status = "✅" if crawling_results['hyde']['success'] else "⚠️"
        print(f"   HyDE: {hyde_status}")
    if crawling_results['fusion']:
        fusion_status = "✅" if crawling_results['fusion']['success'] else "⚠️"
        print(f"   Fusion: {fusion_status}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if import_results['hyderag_available'] and import_results['fusionrag_available']:
        print("   ✅ Both systems are available and functional")
        
        # Check key features
        features_working = []
        if extraction_results['hyde'] and extraction_results['hyde']['success']:
            features_working.append("HyDE extraction")
        if extraction_results['fusion'] and extraction_results['fusion']['success']:
            features_working.append("Fusion extraction")
        if hypothetical_results and any(r.get('success') for r in hypothetical_results):
            features_working.append("HyDE hypothetical docs")
        if multiquery_results and any(r.get('success') for r in multiquery_results):
            features_working.append("Fusion multi-queries")
        
        print(f"   ✅ Working features: {', '.join(features_working)}")
        
        if len(features_working) >= 3:
            print("   🎉 EXCELLENT! Most features working correctly")
        elif len(features_working) >= 2:
            print("   👍 GOOD! Core features working")
        else:
            print("   ⚠️ LIMITED! Some features need attention")
    else:
        print("   ❌ Import issues prevent full testing")
        print("   💡 Make sure you're running from URAG project root")
        print("   💡 Try: cd C:\\Users\\ad\\Desktop\\URAG && python test_rag_simple.py")

if __name__ == "__main__":
    main()
