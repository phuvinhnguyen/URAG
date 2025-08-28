# !curl -L -o dev_data.jsonl.bz2 https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2?download=
# !bunzip2 -c dev_data.jsonl.bz2 > ./dev_data.jsonl
#!/usr/bin/env python3
"""
RAG System for Question Answering using Gemini API and HuggingFace Models
Compatible with Kaggle environment - Modified for wrong answer generation with token rotation
"""

import requests
from typing import List, Dict, Optional
import warnings
import json
import time
warnings.filterwarnings("ignore")

# Install required packages (run this in Kaggle cell first)
"""
!pip install transformers torch sentence-transformers google-generativeai beautifulsoup4 numpy scikit-learn
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import random
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

class MultiTokenWebRAGSystem:
    def __init__(self, gemini_api_keys: List[str] = None, use_local_model: bool = True):
        """
        Initialize RAG system with multiple Gemini API tokens and local HuggingFace model support
        
        Args:
            gemini_api_keys: List of Google Gemini API keys for rotation
            use_local_model: Whether to use local HuggingFace model as fallback
        """
        self.gemini_api_keys = gemini_api_keys or []
        self.current_key_index = 0
        self.use_local_model = use_local_model
        self.gemini_model = None
        
        # Initialize first Gemini API key
        self._setup_gemini()
            
        # Initialize local model and embeddings
        if use_local_model:
            print("Loading local models...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Use a smaller model that works well on Kaggle
            model_name = "microsoft/DialoGPT-medium"  # Alternative: "gpt2", "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Models loaded successfully!")
    
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
    
    def scrape_website(self, url: str, max_length: int = 2000) -> str:
        """
        Scrape content from a website
        
        Args:
            url: Website URL
            max_length: Maximum length of content to return
            
        Returns:
            Scraped text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_length]
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
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
    
    def find_relevant_chunks(self, question: str, chunks: List[str], top_k: int = 10) -> List[str]:
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
    
    def answer_with_local_model(self, question: str, context: str, correct_answer: str) -> str:
        """
        Generate wrong answers using local HuggingFace model
        
        Args:
            question: Input question
            context: Context from web sources
            correct_answer: The correct answer
            
        Returns:
            Generated wrong answers in specified format
        """
        if not self.use_local_model:
            raise ValueError("Local model not initialized")
            
        # Create prompt for wrong answer generation
        prompt = f"""Generate 3 plausible but incorrect answers for this question based on the context.

Context: {context[:800]}

Question: {question}
Correct Answer: {correct_answer}

Generate 3 wrong answers. Format: option1|option2|option3

Wrong answers:"""
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            generated = response[len(prompt):].strip()
            
            # Try to format as required format
            if '|' in generated:
                options = generated.split('|')[:3]
                formatted_response = f"Local model thinking<answer>{'<answer>'.join(options)}</eos>"
            else:
                # Fallback: create simple wrong answers
                words = generated.split()[:20]  # Take first 20 words
                wrong_ans = ' '.join(words) if words else "Alternative answer"
                formatted_response = f"Local model fallback<answer>{wrong_ans} (option 1)<answer>{wrong_ans} (option 2)<answer>{wrong_ans} (option 3)</eos>"
            
            return formatted_response
            
        except Exception as e:
            print(f"Local model error: {e}")
            # Ultimate fallback
            return f"Error in generation<answer>Alternative 1<answer>Alternative 2<answer>Alternative 3</eos>"
    
    def process_question(self, question: str, websites: List[str], correct_answer: str,
                        web_contents: Optional[List[str]] = None) -> Dict:
        """
        Main function to process question and generate wrong answers
        
        Args:
            question: Input question
            websites: List of website URLs
            correct_answer: The correct answer
            web_contents: Pre-scraped web contents (optional)
            
        Returns:
            Dictionary with results
        """
        print(f"Processing question: {question[:100]}...")
        
        result = {
            'question': question,
            'correct_answer': correct_answer,
            'websites': websites,
            'generated_content': '',
            'model_used': '',
            'success': False,
            'error': None
        }
        
        # Get web content
        all_content = []
        
        if web_contents:
            all_content = web_contents
        else:
            print("Scraping websites...")
            for url in websites[:3]:  # Limit to first 3 URLs to avoid timeout
                print(f"Scraping: {url}")
                content = self.scrape_website(url)
                if content:
                    all_content.append(content)
        
        if not all_content:
            result['error'] = "No content could be retrieved from the provided sources"
            return result
        
        # Combine and chunk content
        combined_content = '\n\n'.join(all_content)
        chunks = self.chunk_text(combined_content)
        
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(question, chunks)
        context = '\n\n'.join(relevant_chunks)
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Generate wrong answers
        generated_answer = ""
        
        # Try Gemini first if available
        if self.gemini_api_keys:
            print("Generating wrong answers with Gemini...")
            generated_answer = self.answer_with_gemini(question, context, correct_answer)
            if generated_answer:
                result['model_used'] = f'gemini_key_{self.current_key_index}'
        
        # Fallback to local model if Gemini fails or not available
        if not generated_answer and self.use_local_model:
            print("Generating wrong answers with local model...")
            generated_answer = self.answer_with_local_model(question, context, correct_answer)
            if generated_answer:
                result['model_used'] = 'local_model'
        
        if generated_answer:
            result['generated_content'] = generated_answer
            result['success'] = True
        else:
            result['error'] = "Failed to generate wrong answers with any available model"
        
        return result


def save_result(result: Dict, output_file: str):
    """Save result to JSONL file"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    """
    Main function to process CRAG dataset
    """
    # Configuration - Add multiple API keys here
    GEMINI_API_KEYS = [
        'tokens',
    ]
    
    input_file = './dev_data.jsonl'
    output_file = './wrong_answers_results.jsonl'
    
    # Initialize RAG system
    rag_system = MultiTokenWebRAGSystem(gemini_api_keys=GEMINI_API_KEYS, use_local_model=True)
    
    # Clear output file
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    
    processed_count = 0
    success_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line_num, line in enumerate(in_f):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        question = sample['query']
                        websites = [i['page_url'] for i in sample['search_results']]
                        correct_answer = sample['answer']
                        
                        print(f"\n{'='*20} Processing sample {line_num + 1} {'='*20}")
                        
                        # Process question
                        result = rag_system.process_question(question, websites, correct_answer)
                        
                        # Save result
                        save_result(result, output_file)
                        
                        processed_count += 1
                        if result['success']:
                            success_count += 1
                            
                        print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
                        if result.get('generated_content'):
                            print(f"Generated: {result['generated_content'][:200]}...")
                        
                        # Progress update
                        if processed_count % 10 == 0:
                            print(f"\nProgress: {processed_count} processed, {success_count} successful")
                            
                        # # Optional: Limit processing for testing
                        # if processed_count >= 3:  # Process only first 3 for testing
                        #     break
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num + 1}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num + 1}: {e}")
                        # Save error result
                        error_result = {
                            'question': 'Parse error',
                            'correct_answer': '',
                            'websites': [],
                            'generated_content': '',
                            'model_used': '',
                            'success': False,
                            'error': str(e)
                        }
                        save_result(error_result, output_file)
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
        print(f"\n{'='*50}")
        print(f"Processing completed!")
        print(f"Total processed: {processed_count}")
        print(f"Successful: {success_count}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()

