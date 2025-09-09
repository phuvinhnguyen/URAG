from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from loguru import logger
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter

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

class SelfLLMSystem(AbstractRAGSystem):
    """
    Self-RAG LLM system that implements self-reflection for retrieval and generation.
    
    This system uses special reflection tokens to evaluate:
    - Whether retrieval is needed
    - Whether retrieved documents are relevant
    - Whether the generated answer is supported by the context
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda", technique: str = "direct", max_new_tokens: int = 512, temperature: float = 0.1):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.technique = technique
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None)
        self.model_name = model_name
        self.model.eval()
        
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int: return 1

    def generate(self, queries: List[str]) -> List[str]:
        systems = '''You are a helpful assistant that answers questions in the xml format, some examples are:
    <retrieve>Yes</retrieve>
    <relevant>Yes</relevant>
    <support>Fully supported</support>
    <utility>Helpful</utility>
    '''
        prompts = [self._create_unified_prompt(systems, query) for query in queries]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return [self.tokenizer.decode(output[len(inputs.input_ids[i]):], skip_special_tokens=True).strip() for i, output in enumerate(outputs)]        

    def evaluate_retrieval_need(self, questions: List[str]) -> List[bool]:
        prompt = """Question: {question}

Do you need to retrieve additional information to answer this question accurately?
Please respond with <retrieve>Yes</retrieve> or <retrieve>No</retrieve>.

Response: """
        
        response = self.generate([prompt.format(question=question) for question in questions])
        return [('<retrieve>yes</retrieve>' in resp.lower()) for resp in response]
    
    def evaluate_relevance(self, questions: List[str], batch_contexts: List[List[str]]) -> bool:
        prompt = """Question: {question}

Retrieved Context: {context}

Is this context relevant and helpful for answering the question?
Please respond with <relevant>Yes</relevant> or <relevant>No</relevant>.

Response: """

        prompts = []
        for question, contexts in zip(questions, batch_contexts):
            for context in contexts:
                prompts.append(prompt.format(question=question, context=context))
        
        is_relevant = [('<relevant>yes</relevant>' in response.lower()) for response in self.generate(prompts)]
        return [is_relevant[val-len(batch_contexts[i]):val] for i, val in enumerate(np.cumsum([len(contexts) for contexts in batch_contexts]))]
    
    def evaluate_support(self, questions: List[str], contexts: List[str], answers: List[str]) -> str:
        prompt = """Question: {question}

Context: {context}

Answer: {answer}

Is this answer fully supported by the provided context?
Please respond with one of:
- <support>Fully supported</support>
- <support>Partially supported</support>
- <support>Not supported</support>

Response: """
        
        responses = self.generate([prompt.format(question=question, context=context, answer=answer) for question, context, answer in zip(questions, contexts, answers)])
        
        results = []
        for response in responses:
            if '<support>Fully supported</support>' in response: results.append(2) # Fully supported
            elif '<support>Partially supported</support>' in response: results.append(1) # Partially supported
            else: results.append(0) # Not supported
        return results
    
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

    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        
        if self.technique == 'cot':
            system_message = SYSTEM_PROMPT + "Reasoning step by step."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is one of A, B, C, or D."
        elif self.technique == 'rag' or self.technique == 'self':
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
            'technique': self.technique,
        }
