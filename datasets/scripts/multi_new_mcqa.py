import random
import json
from typing import List, Dict, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

class SimpleMCQAConverter:
    def __init__(self, config: Dict = None):
        """Initialize with configuration"""
        self.config = {
            # RAG settings
            'chunk_size': 300,
            'chunk_overlap': 100,
            'top_k_chunks': 5,
            'embedding_model': 'all-MiniLM-L6-v2',
            
            # Generation settings  
            'gemini_api_keys': [],
            'gemini_models': ['gemini-2.0-flash-exp', 'gemini-exp-1206'],
            'num_distractors': 3,
            'max_summary_length': 900,
            
            # Distractor types
            'distractor_types': [
                'hallucination',      # Add fake info
                'contradiction',      # Contradict facts
                'partial_truth',      # Missing key details
                'wrong_conclusion',   # Wrong inference
                'exaggeration',       # Over/under state
                'conflation',         # Mix up entities
                'temporal_error',     # Wrong timing
                'scope_error'         # Wrong scope
            ]
        }
        
        if config:
            self.config.update(config)
            
        # Initialize models
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        self.current_key_idx = 0
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Setup Gemini API"""
        if self.config['gemini_api_keys'] and self.current_key_idx < len(self.config['gemini_api_keys']):
            genai.configure(api_key=self.config['gemini_api_keys'][self.current_key_idx])
            model_name = random.choice(self.config['gemini_models'])
            self.gemini_model = genai.GenerativeModel(model_name)
        else:
            self.gemini_model = None
    
    def _rotate_key(self):
        """Rotate to next API key"""
        self.current_key_idx = (self.current_key_idx + 1) % len(self.config['gemini_api_keys'])
        self._setup_gemini()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        chunk_size = self.config['chunk_size']
        overlap = self.config['chunk_overlap']
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 20:  # Skip very short chunks
                chunks.append(chunk.strip())
        return chunks
    
    def retrieve_relevant_chunks(self, query: str, document: str) -> List[str]:
        """Use embedding similarity to find relevant chunks"""
        chunks = self.chunk_text(document)
        if not chunks:
            return []
            
        # Get embeddings
        query_embedding = self.embedding_model.encode([query])
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        # Calculate similarities and get top-k
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-self.config['top_k_chunks']:][::-1]
        
        return [chunks[i] for i in top_indices]
    
    def create_distractor_prompt(self, correct_summary: str, relevant_chunks: List[str], 
                               distractor_type: str) -> str:
        """Create prompt for specific distractor type"""
        
        evidence_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks)])
        
        prompts = {
            'hallucination': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: Create a summary that adds 1-2 plausible but FABRICATED details not mentioned in the evidence. Make it seem realistic but completely made up.

Wrong summary:""",

            'contradiction': f"""
Evidence from document:  
{evidence_text}

Correct summary: {correct_summary}

Task: Create a summary that CONTRADICTS key facts from the evidence. Flip important details while keeping similar structure.

Wrong summary:""",

            'partial_truth': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: Create a summary that omits CRUCIAL information from the evidence, making it misleading despite being partially correct.

Wrong summary:""",

            'wrong_conclusion': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: Use facts from evidence but draw WRONG conclusions or make inappropriate inferences.

Wrong summary:""",

            'exaggeration': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: Significantly EXAGGERATE or minimize the importance/scale of events mentioned in the evidence.

Wrong summary:""",

            'conflation': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: MIX UP different entities, events, or concepts from the evidence chunks.

Wrong summary:""",

            'temporal_error': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: Change the TIMING or sequence of events mentioned in the evidence.

Wrong summary:""",

            'scope_error': f"""
Evidence from document:
{evidence_text}

Correct summary: {correct_summary}

Task: Change the SCOPE (local→global, specific→general, individual→group) inappropriately.

Wrong summary:"""
        }
        
        return prompts.get(distractor_type, prompts['hallucination'])
    
    def generate_distractor(self, correct_summary: str, document: str) -> Dict:
        """Generate one distractor using RAG"""
        distractor_type = random.choice(self.config['distractor_types'])
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(correct_summary, document)
        
        # Create prompt
        prompt = self.create_distractor_prompt(correct_summary, relevant_chunks, distractor_type)
        
        result = {
            'text': '',
            'type': distractor_type,
            'chunks_used': len(relevant_chunks),
            'success': False
        }
        
        # Try generation with retries
        for attempt in range(len(self.config['gemini_api_keys']) if self.config['gemini_api_keys'] else 1):
            try:
                if self.gemini_model:
                    response = self.gemini_model.generate_content(prompt)
                    result['text'] = response.text.strip()
                    result['success'] = True
                    break
            except Exception as e:
                print(f"API error (attempt {attempt + 1}): {e}")
                if attempt < len(self.config['gemini_api_keys']) - 1:
                    self._rotate_key()
        
        # Fallback if generation fails
        if not result['success']:
            result['text'] = self._create_fallback_distractor(correct_summary, distractor_type)
            result['success'] = False
        
        return result
    
    def _create_fallback_distractor(self, correct_summary: str, distractor_type: str) -> str:
        """Simple rule-based fallback"""
        if distractor_type == 'contradiction':
            return correct_summary.replace('increased', 'decreased').replace('positive', 'negative')
        elif distractor_type == 'exaggeration':
            return correct_summary + " This represents a major breakthrough that will revolutionize the entire industry."
        else:
            # Generic fallback
            words = correct_summary.split()
            return ' '.join(words[:len(words)//2]) + " However, the situation remains unclear pending further investigation."
    
    def convert_to_mcqa(self, document: str, correct_summary: str, sample_id: int) -> Dict:
        """Convert one sample to MCQA format matching CRAG dataset structure"""
        
        # Generate distractors
        distractors = []
        metadata = {'types': [], 'chunks_used': [], 'success': []}
        
        for i in range(self.config['num_distractors']):
            result = self.generate_distractor(correct_summary, document)
            distractors.append(result['text'])
            metadata['types'].append(result['type'])
            metadata['chunks_used'].append(result['chunks_used'])
            metadata['success'].append(result['success'])
        
        # Create options and shuffle
        all_options = [correct_summary] + distractors
        random.shuffle(all_options)
        correct_idx = all_options.index(correct_summary)
        
        # Convert to letter format (A, B, C, D)
        correct_answer = chr(65 + correct_idx)  # Convert 0,1,2,3 to A,B,C,D
        
        # Create options list with letters
        options_letters = [chr(65 + i) for i in range(len(all_options))]
        
        # Create search_results structure matching CRAG format
        search_results = [
            {
                "page_result": document,  # Full document as page_result
                "page_snippet": "",       # Empty as required
                "page_title": "",         # Empty field
                "page_url": "",           # Empty field
                "page_rank": 1,           # Default rank
                "page_score": 1.0,        # Default score
                "page_source": "generated", # Source identifier
                "page_domain": "",        # Empty field
                "page_language": "en",    # Default language
                "page_metadata": {}       # Empty metadata
            }
        ]
        
        return {
            'id': sample_id,
            'question': f"Which of the following best summarizes the given document?\nA. {all_options[0]}\nB. {all_options[1]}\nC. {all_options[2]}\nD. {all_options[3]}",
            'correct_answer': correct_answer,
            'options': options_letters,
            'search_results': search_results,
            'query_time': "March 1, 2024",  # Default query time
            'technique': "rag",             # Default technique
            'metadata': metadata            # Keep original metadata for debugging
        }
    
    def process_dataset(self, dataset, num_samples: int = 100, start_idx: int = 0) -> List[Dict]:
        """Process dataset samples"""
        mcqa_samples = []
        processed = 0
        current_idx = start_idx
        
        print(f"Processing {num_samples} samples...")
        
        while processed < num_samples and current_idx < len(dataset):
            sample = dataset[current_idx]
            current_idx += 1
            
            # Filter by length
            if len(sample['summary']) >= self.config['max_summary_length']:
                continue
            
            try:
                mcqa_sample = self.convert_to_mcqa(sample['document'], sample['summary'], processed + 1)
                mcqa_samples.append(mcqa_sample)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Processed {processed}/{num_samples}")
                    
            except Exception as e:
                print(f"Error processing sample {current_idx-1}: {e}")
        
        return mcqa_samples
    
    def save_dataset(self, samples: List[Dict], filename: str, split_ratio: float = 0.5):
        """Save samples to JSON in CRAG format with calibration/test split"""
        
        # Split samples into calibration and test
        split_idx = int(len(samples) * split_ratio)
        calibration_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        # Create CRAG format dataset
        dataset = {
            "name": "CRAG_MCQA",
            "description": "CRAG Multiple Choice Question Answering dataset generated from Multi-News",
            "version": "1.0",
            "total_samples": len(samples),
            "calibration_samples": len(calibration_samples),
            "test_samples": len(test_samples),
            "calibration": calibration_samples,
            "test": test_samples
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(samples)} samples to {filename}")
        print(f"  - Calibration: {len(calibration_samples)} samples")
        print(f"  - Test: {len(test_samples)} samples")


# Simple usage example
if __name__ == "__main__":
    # Configuration
    config = {
        'gemini_api_keys': ["tokens"],
        'chunk_size': 40,           # Configurable chunk size
        'chunk_overlap': 10,         # Configurable overlap
        'top_k_chunks': 8,           # Number of chunks to retrieve
        'num_distractors': 3,        # Number of wrong answers
        'max_summary_length': 900    # Filter criterion
    }
    
    # Load data and initialize
    data = load_dataset('alexfabbri/multi_news', split='test', trust_remote_code=True)
    converter = SimpleMCQAConverter(config)
    
    # Process samples  
    mcqa_samples = converter.process_dataset(
        dataset=data,
        num_samples=10,    # Start small for testing
        start_idx=0
    )
    
    # Save results in CRAG format
    converter.save_dataset(mcqa_samples, "crag_mcqa_generated.json", split_ratio=0.5)
    
    # Print example
    if mcqa_samples:
        sample = mcqa_samples[0]
        print(f"\n=== Example MCQA ===")
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Correct Answer: {sample['correct_answer']}")
        print(f"Options: {sample['options']}")
        print(f"Search Results: {len(sample['search_results'])} items")
        print(f"Query Time: {sample['query_time']}")
        print(f"Technique: {sample['technique']}")
        print(f"\nDistractor types: {sample['metadata']['types']}")
        print(f"Success rate: {sum(sample['metadata']['success'])}/{len(sample['metadata']['success'])}")