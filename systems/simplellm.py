from systems.abstract import AbstractRAGSystem
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """You are provided with a multiple choice question and various references. Your task is to answer the question succinctly, using format Answer|X (mandatory) where X is your answer for the multiple choice question, which can be A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts. You must provide the final answer and finish the chat as soon as possible.
For example:
Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid
Answer|A

Question: What is the result of 2 + 2?
A. 3
B. 4
C. 5
D. 6
Answer|B
"""

class EndSequenceCriteria(StoppingCriteria):
    def __init__(self, end_ids):
        super().__init__()
        self.end_ids = end_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) < len(self.end_ids) + 1:
            return False
        result = torch.equal(input_ids[0][-len(self.end_ids)-1:-1], torch.tensor(self.end_ids, device=input_ids.device))
        return result

class SimpleLLMSystem(AbstractRAGSystem):
    """
    Simple LLM system without RAG - just direct prompting, this serves as a baseline and example implementation.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", technique: str = "direct", max_new_tokens: int = 100, temperature: float = 0.1):
        self.device = device
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int: return 1
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique."""
        question = sample.get('question', '')
        
        if self.technique == 'cot': prompt = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif self.technique == 'rag': prompt = f"Context information: {sample.get('context', '')}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        else: prompt = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        
        try:
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
                ], tokenize=False, add_generation_prompt=True)
        except: pass

        return prompt
    
    def _generate_response_with_probabilities(self, prompt: str, options: List[str]):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start_ids = self.tokenizer.encode("Answer|", add_special_tokens=False)
        stopping = StoppingCriteriaList([EndSequenceCriteria(start_ids)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        last_token_probs = F.softmax(outputs.scores[-1][0], dim=-1)
        option_tokens = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
        last_token_probs = last_token_probs[option_tokens] + 1e-10
        last_token_probs = last_token_probs / last_token_probs.sum()
        conformal_probabilities = {option: last_token_probs[i].item() for i, option in enumerate(options)}
        return generated_text, conformal_probabilities

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