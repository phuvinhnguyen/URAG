from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from loguru import logger
import torch.nn.functional as F
from typing import Dict, Any, List
import random

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
        if len(input_ids[0]) < len(self.end_ids) + 1: return False
        return torch.equal(input_ids[0][-len(self.end_ids)-1:-1], torch.tensor(self.end_ids, device=input_ids.device))

class FusionLLMSystem(AbstractRAGSystem):
    """
    Fusion LLM system that generates multiple diverse queries for better retrieval.
    
    This system implements the query fusion technique:
    1. Generate multiple diverse queries from the original question
    2. Use these queries for enhanced retrieval
    3. Apply Reciprocal Rank Fusion to combine results
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda", num_samples: int = 20, num_queries: int = 3, technique: str = "direct", max_new_tokens: int = 512, temperature: float = 0.1):
        """Initialize the Fusion LLM system."""
        self.device = device

        self.technique = technique
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_samples = num_samples
        self.num_queries = num_queries
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None)
        self.model.eval()
        
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_batch_size(self) -> int: return 1
    
    def _create_unified_prompt(self, system_message: str, user_message: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                tokenize=False, add_generation_prompt=True
            )
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
    
    def generate_diverse_queries(self, questions: List[str]) -> List[str]:
        systems = [
            "You are a helpful assistant that rephrases questions while keeping the same meaning.",
            "You are a helpful assistant that creates alternative question formulations.",
            "You are a helpful assistant that generates related questions.",
            "You are a helpful assistant that makes questions more specific.",
            "You are a helpful assistant that simplifies complex questions."
        ]
        
        user_messages = [
            "Create an alternative question: {}",
            "Generate a related question: {}",
            "Make this question more specific: {}",
            "Simplify this question: {}",
            "What is another way to ask about the same topic as this question: {}",
            "Generate a related question that might help answer this question: {}",
        ]
        
        prompts = []
        for question in questions: prompts += [self._create_unified_prompt(random.choice(systems), random.choice(user_messages).format(question)) for _ in range(self.num_queries)]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=224,
            temperature=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        queries = [[question] for _, question in enumerate(questions)]
        for i, output in enumerate(outputs):
            index = i // self.num_queries
            prompt_len = len(inputs.input_ids[i])
            query = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            if query and query not in queries[index]: queries[index].append(query)
        
        return queries
        
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        technique = self.technique
        
        if technique == 'cot':
            system_message = SYSTEM_PROMPT + "Reasoning step by step."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is one of A, B, C, or D."
        elif technique == 'rag' or technique == 'fusion':
            context = sample.get('context', '')
            if context:
                system_message = SYSTEM_PROMPT + "Use the context to inform your answer."
                user_message = f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, or D."
            else:
                system_message = SYSTEM_PROMPT + "Provide your final answer for the question of the user."
                user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, or D."
        else:
            system_message = SYSTEM_PROMPT + "Provide your final answer for the question of the user."
            user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, or D."
        
        return self._create_unified_prompt(system_message, user_message)

    def _generate_response_with_probabilities(self, prompt: str, options: List[str] = None):
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
                eos_token_id=self.tokenizer.eos_token_id,
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
            'technique': self.technique,
        }
