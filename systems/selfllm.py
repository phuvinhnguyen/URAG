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

class SelfLLMSystem(AbstractRAGSystem):
    """
    Self-RAG LLM system that implements self-reflection for retrieval and generation.
    
    This system uses special reflection tokens to evaluate:
    - Whether retrieval is needed
    - Whether retrieved documents are relevant
    - Whether the generated answer is supported by the context
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda", technique: str = "direct", max_new_tokens: int = 512, temperature: float = 0.1, method: str = 'normal'):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.technique = technique
        self.method = method
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
    <utility>5</utility>
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
        retrieval_decisions = [('<retrieve>yes</retrieve>' in resp.lower()) for resp in response]
        
        # Log retrieval decisions
        for i, (question, resp, decision) in enumerate(zip(questions, response, retrieval_decisions)):
            logger.info(f"[SELFLLM] Question {i+1}: {question[:100]}...")
            logger.info(f"[SELFLLM] Retrieval decision response: {resp.strip()}")
            logger.info(f"[SELFLLM] Needs retrieval: {decision}")
            logger.info("---")
        
        return retrieval_decisions
    
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
        
        responses = self.generate(prompts)
        is_relevant = [('<relevant>yes</relevant>' in response.lower()) for response in responses]
        
        # Log relevance decisions
        prompt_idx = 0
        for q_idx, (question, contexts) in enumerate(zip(questions, batch_contexts)):
            logger.info(f"[SELFLLM] Relevance evaluation for question {q_idx+1}: {question[:100]}...")
            for c_idx, context in enumerate(contexts):
                resp = responses[prompt_idx]
                relevant = is_relevant[prompt_idx]
                logger.info(f"[SELFLLM] Context {c_idx+1}: {context[:150]}...")
                logger.info(f"[SELFLLM] Relevance response: {resp.strip()}")
                logger.info(f"[SELFLLM] Is relevant: {relevant}")
                prompt_idx += 1
            logger.info("---")
        
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
        for i, (question, context, answer, response) in enumerate(zip(questions, contexts, answers, responses)):
            if '<support>Fully supported</support>' in response: 
                support_score = 2
                support_text = "Fully supported"
            elif '<support>Partially supported</support>' in response: 
                support_score = 1
                support_text = "Partially supported"
            else: 
                support_score = 0
                support_text = "Not supported"
            
            results.append(support_score)
            
            # Log support evaluation
            logger.info(f"[SELFLLM] Support evaluation {i+1}:")
            logger.info(f"[SELFLLM] Question: {question[:100]}...")
            logger.info(f"[SELFLLM] Context: {context[:150]}...")
            logger.info(f"[SELFLLM] Answer: {answer[:100]}...")
            logger.info(f"[SELFLLM] Support response: {response.strip()}")
            logger.info(f"[SELFLLM] Support score: {support_score} ({support_text})")
            logger.info("---")
            
        return results
    
    def evaluate_utility(self, questions: List[str], contexts: List[str], answers: List[str]) -> List[int]:
        """Evaluate the utility of generated answers using utility reflection tokens."""
        prompt = """Question: {question}

Context: {context}

Answer: {answer}

How useful is this answer for addressing the question? Consider completeness, accuracy, and helpfulness.
Please respond with one of:
- <utility>5</utility> - Extremely useful, comprehensive and accurate
- <utility>4</utility> - Very useful, mostly complete and accurate  
- <utility>3</utility> - Moderately useful, adequate but could be better
- <utility>2</utility> - Somewhat useful, limited help
- <utility>1</utility> - Not useful, unhelpful or inaccurate

Response: """
        
        responses = self.generate([prompt.format(question=question, context=context, answer=answer) for question, context, answer in zip(questions, contexts, answers)])
        
        results = []
        for i, (question, context, answer, response) in enumerate(zip(questions, contexts, answers, responses)):
            if '<utility>5</utility>' in response: 
                utility_score = 5
                utility_text = "Extremely useful"
            elif '<utility>4</utility>' in response: 
                utility_score = 4
                utility_text = "Very useful"
            elif '<utility>3</utility>' in response: 
                utility_score = 3
                utility_text = "Moderately useful"
            elif '<utility>2</utility>' in response: 
                utility_score = 2
                utility_text = "Somewhat useful"
            elif '<utility>1</utility>' in response: 
                utility_score = 1
                utility_text = "Not useful"
            else: 
                utility_score = 3
                utility_text = "Moderately useful (default)"
            
            results.append(utility_score)
            
            # Log utility evaluation
            logger.info(f"[SELFLLM] Utility evaluation {i+1}:")
            logger.info(f"[SELFLLM] Question: {question[:100]}...")
            logger.info(f"[SELFLLM] Context: {context[:150]}...")
            logger.info(f"[SELFLLM] Answer: {answer[:100]}...")
            logger.info(f"[SELFLLM] Utility response: {response.strip()}")
            logger.info(f"[SELFLLM] Utility score: {utility_score} ({utility_text})")
            logger.info("---")
            
        return results
    
    def generate_with_segments(self, questions: List[str], contexts: List[str] = None, max_segments: int = 3) -> List[Dict[str, Any]]:
        """Generate answers segment by segment with inline reflection tokens."""
        if contexts is None:
            contexts = [""] * len(questions)
            
        batch_results = []
        
        # Process each question individually for segment generation
        # (Segment generation requires iterative prompting which is harder to batch efficiently)
        for question, context in zip(questions, contexts):
            segments = []
            current_context = context
            final_answer = ""
            
            for segment_idx in range(max_segments):
                # Generate segment
                segment_prompt = f"""Question: {question}

Context: {current_context}

Previous segments: {' '.join(segments)}

Generate the next segment of your answer. Keep it focused and concise.
After generating, evaluate if you need to retrieve more information using <retrieve>Yes/No</retrieve>.

Response: """
                
                segment_response = self.generate([segment_prompt])[0]
                segments.append(segment_response)
                
                # Check if retrieval is needed for next segment
                if '<retrieve>yes</retrieve>' in segment_response.lower() and segment_idx < max_segments - 1:
                    # This would trigger additional retrieval in the main RAG system
                    continue
                else:
                    # Finalize answer
                    final_answer = ' '.join(segments)
                    break
            
            batch_results.append({
                'question': question,
                'context': current_context,
                'segments': segments,
                'final_answer': final_answer,
                'needs_retrieval': '<retrieve>yes</retrieve>' in segments[-1].lower() if segments else False
            })
        
        # Batch evaluate support and utility for all final answers
        final_answers = [result['final_answer'] for result in batch_results]
        final_contexts = [result['context'] for result in batch_results]
        
        if any(ctx for ctx in final_contexts):  # Only evaluate if there are contexts
            support_scores = self.evaluate_support(questions, final_contexts, final_answers)
            utility_scores = self.evaluate_utility(questions, final_contexts, final_answers)
        else:
            support_scores = [2] * len(questions)  # Default to fully supported if no context
            utility_scores = self.evaluate_utility(questions, [""] * len(questions), final_answers)
        
        # Add scores to results
        for i, result in enumerate(batch_results):
            result['support_score'] = support_scores[i]
            result['utility_score'] = utility_scores[i]
        
        return batch_results
    
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
