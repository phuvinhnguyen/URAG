from systems.abstract import AbstractRAGSystem
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# RAT-specific prompts
INITIAL_COT_PROMPT = """You are provided with a multiple choice question. Your task is to think step by step and provide your reasoning using a chain of thoughts approach.

Please structure your response as follows:
1. First, provide your initial reasoning step by step
2. Then provide your final answer in the format Answer|X where X is your answer (A, B, C, D, ...)

For example:
Question: What is the capital of France?
A. London
B. Berlin  
C. Paris
D. Madrid

Let me think step by step:
1. France is a country in Western Europe
2. The capital of France is the city where the French government is located
3. Paris is the largest city in France and serves as its political and cultural center
4. London is the capital of the UK, Berlin is the capital of Germany, Madrid is the capital of Spain

Answer|C
"""

REVISION_PROMPT = """You are provided with a multiple choice question, your previous reasoning, and some additional relevant information. 

Your task is to revise your previous reasoning step by step using the new information, then provide your final answer.

Previous reasoning: {previous_reasoning}

Additional information: {retrieved_info}

Please revise your reasoning considering this new information and provide your final answer in the format Answer|X where X is your answer (A, B, C, D, ...)
"""

class RATLLMSystem(AbstractRAGSystem):
    """
    RAT (Retrieval Augmented Thoughts) LLM system that implements iterative refinement of chain-of-thoughts.
    
    This system implements the RAT technique:
    1. Generate initial zero-shot chain-of-thoughts
    2. Iteratively revise each thought step with retrieved information
    3. Refine reasoning based on relevant context
    4. Generate final answer with improved reasoning
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", 
                 max_iterations: int = 2, max_new_tokens: int = 512, temperature: float = 0.1, 
                 method: str = 'normal'):
        self.device = device
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        
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
    
    def _generate_initial_cot(self, question: str) -> str:
        """Generate initial chain-of-thoughts for the question."""
        prompt = f"{question}\n\n{INITIAL_COT_PROMPT}"
        
        try:
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
        except:
            pass
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        generated_text = generated_text[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        return generated_text.strip()
    
    def _revise_reasoning(self, question: str, previous_reasoning: str, retrieved_info: str, options: List[str]) -> tuple:
        """Revise reasoning with retrieved information."""
        revision_prompt = REVISION_PROMPT.format(
            previous_reasoning=previous_reasoning,
            retrieved_info=retrieved_info
        )
        
        full_prompt = f"{question}\n\n{revision_prompt}"
        
        try:
            full_prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a helpful assistant that revises reasoning based on new information."},
                {"role": "user", "content": full_prompt}
            ], tokenize=False, add_generation_prompt=True)
        except:
            pass
        
        return self._generate_response_with_probabilities(full_prompt, options)
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract individual reasoning steps from generated text."""
        # Look for numbered steps or bullet points
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('- ') or line.startswith('* '):
                steps.append(line)
            elif line and not line.startswith('Answer|'):
                # Consider non-empty lines as reasoning steps
                steps.append(line)
        
        return steps
    
    def _simulate_retrieval_for_step(self, question: str, reasoning_step: str) -> str:
        """
        Simulate retrieval for a reasoning step.
        In a full RAT implementation, this would query external knowledge bases.
        For now, we return a placeholder that encourages deeper reasoning.
        """
        # This is a simplified version - in real RAT, this would involve:
        # 1. Formulating queries based on the reasoning step
        # 2. Retrieving relevant documents
        # 3. Extracting relevant information
        
        return f"Additional context for reasoning step: '{reasoning_step}' - Consider domain-specific knowledge and factual accuracy."
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample using RAT methodology."""
        question = sample.get('question', '')
        options = sample.get('options', ['A', 'B', 'C', 'D'])
        
        # Step 1: Generate initial chain-of-thoughts
        initial_reasoning = self._generate_initial_cot(question)
        current_reasoning = initial_reasoning
        
        reasoning_history = [{"iteration": 0, "reasoning": initial_reasoning, "type": "initial"}]
        
        # Step 2: Iteratively refine reasoning
        for iteration in range(1, self.max_iterations + 1):
            # Extract reasoning steps
            reasoning_steps = self._extract_reasoning_steps(current_reasoning)
            
            # Simulate retrieval for each step (in real implementation, this would be actual retrieval)
            retrieved_info_list = []
            for step in reasoning_steps:
                retrieved_info = self._simulate_retrieval_for_step(question, step)
                retrieved_info_list.append(retrieved_info)
            
            # Combine retrieved information
            combined_retrieved_info = "\n".join(retrieved_info_list)
            
            # Revise reasoning with retrieved information
            revised_response, conformal_probabilities = self._revise_reasoning(
                question, current_reasoning, combined_retrieved_info, options
            )
            
            current_reasoning = revised_response
            reasoning_history.append({
                "iteration": iteration,
                "reasoning": revised_response,
                "retrieved_info": combined_retrieved_info,
                "type": "revision"
            })
        
        # Extract final answer
        final_response, final_probabilities = self._generate_response_with_probabilities(
            f"{question}\n\nFinal reasoning: {current_reasoning}\n\nPlease provide your final answer in the format Answer|X where X is your answer (A, B, C, D, ...)",
            options
        )
        
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': final_response,
            'predicted_answer': max(final_probabilities.items(), key=lambda x: x[1])[0],
            'conformal_probabilities': final_probabilities,
            'reasoning_history': reasoning_history,
            'technique': 'rat_llm',
            'iterations_used': self.max_iterations
        }
