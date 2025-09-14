from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import re

SYSTEM_PROMPT = """You are provided with a multiple choice question and various references. Your task is to answer the question succinctly, using format Answer|X (mandatory) where X is your answer for the multiple choice question, which can be A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts. You must provide the final answer and finish the chat as soon as possible.

For example:
Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid
Answer|C

Question: What is the result of 2 + 2?
A. 3
B. 4
C. 5
D. 6
Answer|B
"""

RAT_REVISE_PROMPT = """You are tasked with revising a specific thought in a chain of reasoning using new evidence. Keep the reasoning structure and only revise the given thought. Always end with Answer|X where X is your answer.

Instructions:
1. Review the current reasoning so far
2. Focus on revising ONLY the specific thought provided
3. Use the evidence to improve or correct that thought
4. Keep all other reasoning intact
5. Provide the complete revised reasoning ending with Answer|X

Example:
Current reasoning: "1. France is in Europe. 2. Paris is a major city. 3. Therefore the answer is C."
Thought to revise: "Paris is a major city"
Evidence: "Paris is the capital and largest city of France, serving as the country's political and cultural center."
Revised reasoning: "1. France is in Europe. 2. Paris is the capital and largest city of France, serving as the political and cultural center. 3. Therefore the answer is C."
"""

class RATLLMSystem(AbstractRAGSystem):
    """
    RAT (Retrieval Augmented Thoughts) LLM system that supports chain-of-thought reasoning.
    
    This system can be used standalone or as part of RATRAGSystem for retrieval-augmented reasoning.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", 
                 technique: str = "direct", max_new_tokens: int = 512, temperature: float = 0.1, 
                 method: str = 'normal'):
        self.device = device
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int:
        return 1
    
    def _create_unified_prompt(self, system_message: str, user_message: str) -> str:
        """Create unified prompt following the same pattern as other LLM systems."""
        try:
            return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ], tokenize=False, add_generation_prompt=True)
        except:
            model_lower = self.model_name.lower()
            
            if "llama" in model_lower:
                return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]\n\n"
            elif "mistral" in model_lower:
                return f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
            elif "falcon" in model_lower:
                return f"User: {system_message}\n\n{user_message}\n\nAssistant:"
            elif "mpt" in model_lower:
                return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            elif any(x in model_lower for x in ["dialogpt", "gpt"]):
                return f"{system_message}\n\n{user_message}\n\nResponse:"
            else:
                return f"### Instruction:\n{system_message}\n\n### Input:\n{user_message}\n\n### Response:"
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        
        if self.technique == 'cot':
            system_message = SYSTEM_PROMPT + " Think step by step and provide detailed reasoning."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif self.technique == 'rag' or self.technique == 'rat':
            context = sample.get('context', '')
            if context:
                system_message = SYSTEM_PROMPT + " Use the provided context to inform your answer."
                user_message = f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
            else:
                system_message = SYSTEM_PROMPT
                user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        else:
            system_message = SYSTEM_PROMPT
            user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        
        return self._create_unified_prompt(system_message, user_message)
    
    def extract_thoughts(self, reasoning_text: str) -> List[str]:
        """Extract individual thoughts from reasoning text."""
        # Remove Answer|X part for processing
        text = re.sub(r'Answer\|[A-Z].*$', '', reasoning_text, flags=re.MULTILINE).strip()
        
        thoughts = []
        
        # Method 1: Split by numbered steps (1., 2., 3., etc.)
        numbered_pattern = r'(\d+\.\s*[^0-9]+?)(?=\d+\.|$)'
        numbered_matches = re.findall(numbered_pattern, text, re.DOTALL)
        if numbered_matches:
            thoughts = [match.strip() for match in numbered_matches]
        
        # Method 2: Split by bullet points or dashes
        elif '- ' in text or '* ' in text:
            lines = text.split('\n')
            current_thought = []
            for line in lines:
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    if current_thought:
                        thoughts.append(' '.join(current_thought))
                    current_thought = [line]
                elif line and current_thought:
                    current_thought.append(line)
            if current_thought:
                thoughts.append(' '.join(current_thought))
        
        # Method 3: Split by double newlines or sentence boundaries
        else:
            # Split by double newlines first
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                # If paragraph is long, split by sentences
                if len(para) > 100:
                    sentences = re.split(r'[.!?]+', para)
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 20:  # Minimum thought length
                            thoughts.append(sent)
                else:
                    para = para.strip()
                    if len(para) > 20:
                        thoughts.append(para)
        
        # Clean and filter thoughts
        cleaned_thoughts = []
        for thought in thoughts:
            thought = thought.strip()
            if len(thought) > 10 and not thought.startswith('Answer|'):
                cleaned_thoughts.append(thought)
        
        return cleaned_thoughts[:10]  # Limit to max 10 thoughts
    
    def generate_query_for_thought(self, question: str, thought: str, reasoning_so_far: str) -> str:
        """Generate search query for a specific thought."""
        # Simple approach: combine question context with thought keywords
        thought_clean = re.sub(r'^\d+\.\s*', '', thought)  # Remove numbering
        thought_clean = re.sub(r'^[-*]\s*', '', thought_clean)  # Remove bullets
        
        # Extract key phrases from thought (simple keyword extraction)
        words = thought_clean.split()
        key_words = [word for word in words if len(word) > 3 and word.lower() not in 
                    ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will']]
        
        if len(key_words) > 3:
            query = f"{question[:100]} {' '.join(key_words[:5])}"
        else:
            query = f"{question[:100]} {thought_clean[:100]}"
        
        return query.strip()
    
    def revise_thought_with_evidence(self, question: str, current_reasoning: str, 
                                   thought_to_revise: str, evidence: str, options: List[str]) -> str:
        """Revise a specific thought using evidence while preserving overall reasoning structure."""
        system_message = RAT_REVISE_PROMPT
        user_message = f"""Question: {question}

Current reasoning so far:
{current_reasoning}

Thought to revise: {thought_to_revise}

Evidence: {evidence}

Please revise ONLY the specified thought using the evidence. Keep all other reasoning intact and provide the complete revised reasoning ending with Answer|X where X is your answer for the multiple choice question."""
        
        prompt = self._create_unified_prompt(system_message, user_message)
        
        # Use lower temperature for revision to be more conservative
        original_temp = self.temperature
        self.temperature = min(0.3, self.temperature)
        
        response, _ = self._generate_response_with_probabilities(prompt, options)
        
        # Restore original temperature
        self.temperature = original_temp
        
        return response
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._generate_prompt(sample)
        options = sample.get('options', [])
        response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
        
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': max(conformal_probabilities.items(), key=lambda x: x[1])[0],
            'conformal_probabilities': conformal_probabilities,
            'technique': self.technique
        }
