# !curl -L -o dev_data.jsonl.bz2 https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2?download=
# !bunzip2 -c dev_data.jsonl.bz2 > ./dev_data.jsonl
#!/usr/bin/env python3
"""
RAG System for Question Answering using Gemini API and HuggingFace Models
Compatible with Kaggle environment - Modified for wrong answer generation with token rotation
"""

from typing import List, Dict
import warnings
import json
import time
warnings.filterwarnings("ignore")

# Install required packages (run this in Kaggle cell first)
"""
!pip install transformers torch sentence-transformers google-generativeai beautifulsoup4 numpy scikit-learn
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import random
from sklearn.metrics.pairwise import cosine_similarity

class MultiTokenWebRAGSystem:
    def __init__(self, gemini_api_keys: List[str] = None):
        """
        Initialize RAG system with multiple Gemini API tokens and local HuggingFace model support
        
        Args:
            gemini_api_keys: List of Google Gemini API keys for rotation
        """
        self.gemini_api_keys = gemini_api_keys or []
        self.current_key_index = 0
        self.gemini_model = None
        
        self._setup_gemini()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _setup_gemini(self):
        """Setup Gemini with current API key"""
        if self.gemini_api_keys and self.current_key_index < len(self.gemini_api_keys):
            try:
                current_key = self.gemini_api_keys[self.current_key_index]
                genai.configure(api_key=current_key)
                model = random.choice(['gemini-2.0-flash-exp', 'gemini-2.0-flash', 'gemini-2.5-flash-exp', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemma-3-27b-it'])
                self.gemini_model = genai.GenerativeModel(model)
                print(f"Using Gemini API key {self.current_key_index + 1}/{len(self.gemini_api_keys)}")
            except Exception as e:
                print(f"Error setting up Gemini with key {self.current_key_index}: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
    
    def _rotate_api_key(self):
        """Rotate to next API key"""
        if self.gemini_api_keys:
            self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
            print(f"Rotating to API key {self.current_key_index + 1}/{len(self.gemini_api_keys)}")
            self._setup_gemini()
            time.sleep(5)  # Brief pause between key rotations
    
    def chunk_text(self, text: str, chunk_size: int = 30, overlap: int = 10) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def find_relevant_chunks(self, question: str, chunks: List[str], top_k: int = 15) -> List[str]:
        """
        Find most relevant chunks for the question using embeddings
        
        Args:
            question: Input question
            chunks: List of text chunks
            top_k: Number of top chunks to return
            
        Returns:
            Most relevant chunks
        """
        results = []
        if not chunks or not self.use_local_model:
            results = chunks[:top_k]
        else:
            # Generate embeddings
            question_embedding = self.embedding_model.encode([question])
            chunk_embeddings = self.embedding_model.encode(chunks)
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        
            results = [chunks[i] for i in top_indices]
        
        # Sample subset for manageable context
        sample_size = min(top_k//3, len(results))
        return random.sample(results, sample_size) if results else []
    
    def answer_with_gemini(self, question: str, context: str, correct_answer: str, max_retries: int = 3) -> str:
        """
        Generate wrong answers using Gemini API with token rotation on failure
        
        Args:
            question: Input question
            context: Context from web sources
            correct_answer: The correct answer
            max_retries: Maximum number of retries with different tokens
            
        Returns:
            Generated wrong answers
        """
        if not self.gemini_api_keys:
            raise ValueError("No Gemini API keys provided")
            
        prompt = f"""
Based on the following context information, question, and true answer, please create a multiple choice question by providing 3 wrong answers that can be found (priority) in the context or from your knowledge.

Context:
{context}

Question: {question}
True Answer: {correct_answer}

IMPORTANT:
- Your wrong answers must not be too obvious but should be more challenging.
- You should prioritize the information in the provided context to generate wrong answers.
- Consider the query time context when generating answers if it's relevant.
- Your answer must follow exactly this format so the program can read your wrong answers: your_thinking<answer>option_1<answer>option_2<answer>option_3</eos>
- Your generated options must have similar format with the `True Answer`

Format example for `True Answer` 'It is France':
I need to create plausible but incorrect alternatives...<answer>It is Germany<answer>It is Vietnam<answer>It is Japan</eos>
"""
        
        attempts = 0
        while attempts < max_retries and attempts < len(self.gemini_api_keys):
            try:
                if not self.gemini_model:
                    self._setup_gemini()
                
                print(f"Attempting Gemini API call (attempt {attempts + 1})")
                response = self.gemini_model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                print(f"Gemini API error with key {self.current_key_index}: {e}")
                attempts += 1
                if attempts < len(self.gemini_api_keys):
                    self._rotate_api_key()
                time.sleep(2)  # Wait before retry
        
        print("All Gemini API keys failed, falling back to local model")
        return ""
    
    def process_question(self, question: str, search_results: List[str], correct_answer: str) -> str:                
        # Combine and chunk content
        combined_content = '\n\n'.join([i['page_result'] for i in search_results])
        chunks = self.chunk_text(combined_content)
        
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(question, chunks)
        context = '\n\n'.join(relevant_chunks)
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Try Gemini first if available
        generated_answer = self.answer_with_gemini(question, context, correct_answer)

        return generated_answer

def save_results_json(results: List[Dict], filename: str, split_ratio: float = 0.5):
    """
    Save results to JSON file in CRAG format similar to multi_new_mcqa.py
    
    Args:
        results: List of result dictionaries
        filename: Output JSON filename
        split_ratio: Ratio to split between calibration and test sets
    """
    # Filter out failed samples to ensure only successful results are saved
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("Warning: No successful results to save!")
        return
    
    # Split samples into calibration and test
    split_idx = int(len(successful_results) * split_ratio)
    calibration_samples = successful_results[:split_idx]
    test_samples = successful_results[split_idx:]
    
    # Create CRAG format dataset
    dataset = {
        "name": "CRAG_MCQA",
        "description": "CRAG dataset with generated multiple choice questions using RAG system",
        "version": "1.0",
        "total_samples": len(successful_results),
        "calibration_samples": len(calibration_samples),
        "test_samples": len(test_samples),
        "calibration": calibration_samples,
        "test": test_samples
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(successful_results)} successful results to {filename}")
    print(f"  - Calibration: {len(calibration_samples)} samples")
    print(f"  - Test: {len(test_samples)} samples")
    print(f"  - Filtered out {len(results) - len(successful_results)} failed results")


def main():
    """
    Main function to process CRAG dataset
    """
    # Configuration - Add multiple API keys here
    GEMINI_API_KEYS = [
        'tokens',
    ]
    
    input_file = './dev_data.jsonl'
    json_output_file = './wrong_answers_results.json'  # New JSON output file
    
    # Initialize RAG system
    rag_system = MultiTokenWebRAGSystem(gemini_api_keys=GEMINI_API_KEYS, use_local_model=True)
    all_results = []  # Store all results for JSON output
    
    try:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line_num, line in enumerate(in_f):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        question = sample['query']
                        search_results = sample['search_results'] # list of dicts: {page_url: str, page_snippet: str, page_title: str, page_result: str}
                        correct_answer = sample['answer']
                        query_time = sample.get('query_time', '')
                        domain = sample.get('domain', '')
                        question_type = sample.get('question_type', '')
                        static_or_dynamic = sample.get('static_or_dynamic', '')

                        # Process question
                        try:
                            llm_answer = rag_system.process_question(question, search_results, correct_answer)
                        except Exception as e:
                            continue
                        
                        answers = llm_answer.split('</eos>')[0].split('<answer>')[1:]
                        correct_answer_target_index = line_num % len(answers)
                        answers = answers[:correct_answer_target_index] + [correct_answer] + answers[correct_answer_target_index:]

                        result = {
                            'question': question + '\n' + '\n'.join([f"{chr(65 + i)}. {answer}" for i, answer in enumerate(answers)]),
                            'query_time': query_time,
                            'domain': domain,
                            'question_type': question_type,
                            'static_or_dynamic': static_or_dynamic,
                            'options': [chr(65 + i) for i in range(len(answers))],
                            'correct_answer': chr(65 + correct_answer_target_index),
                            'search_results': search_results
                        }

                        all_results.append(result)
                    except Exception as e:
                        print(f"Error parsing line {line_num + 1}: {e}")
                        continue
    
    except FileNotFoundError:
        print(f"Input file {input_file} not found!")
        print("Please run the data download commands first:")
        print("!curl -L -o dev_data.jsonl.bz2 https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2")
        print("!bunzip2 -c dev_data.jsonl.bz2 > ./dev_data.jsonl")
        return
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        # Save results in JSON format
        print(f"\nSaving results in JSON format...")
        save_results_json(all_results, json_output_file, split_ratio=0.5)
        
        print(f"\n{'='*50}")
        print(f"Processing completed!")
        print(f"Results saved to:")
        print(f"  - JSON: {json_output_file}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()

