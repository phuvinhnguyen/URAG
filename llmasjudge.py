import os
import re
import time
from typing import List, Tuple, Set, Union
from loguru import logger

try:
    from transformers import pipeline
    
    transformer_llm = pipeline(
        "text-generation", 
        model=os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        device_map="auto"
    )
    use_transformer = True
except:
    use_transformer = False

try:
    from vllm import LLM, SamplingParams
    
    vllm_llm = LLM(model=os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"))
    use_vllm = True
except:
    use_vllm = False

try:
    import litellm
    api_model_name = os.getenv("API_MODEL_NAME", "gpt-4o-mini")
    use_api = True
except:
    use_api = False

def _get_few_shot_prompt(prediction: str, labels: str) -> str:
    """Generate evaluation prompt with few-shot examples."""    
    return f"""Here are examples of evaluating predictions:

Prediction: The capital of France is Paris
Label(s): Paris
<answer>Correct</answer>

Prediction: London is a city
Label(s): London is the capital of England
<answer>Wrong</answer>

Prediction: Einstein developed the theory of relativity
Label(s): theory of relativity; relativity theory
<answer>Correct</answer>

Now evaluate:
Prediction: {prediction}
Label(s): {labels}
<answer>"""

def _parse_responses(responses: List[str], predictions: List[str]) -> Tuple[List[int], Set[str]]:
    """Parse responses and return correctness list and correct predictions set."""
    correctness = []
    correct_predictions = set()
    
    for i, response in enumerate(responses):
        match = re.search(r'<answer>\s*(Correct|Wrong)\s*</answer>', response, re.IGNORECASE)
        is_correct = match and match.group(1).lower() == 'correct' if match else 'correct' in response.lower()
        
        correctness.append(1 if is_correct else 0)
        if is_correct:
            correct_predictions.add(predictions[i])
    
    return correctness, correct_predictions

def call_api(prompts: List[str]) -> Tuple[List[int], Set[str]]:
    """Call API with rate limiting and retry logic."""
    global use_api
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        use_api = False
        raise RuntimeError("No API key found")
    
    try:
        responses = []
        max_retries = 3
        
        requests = [
            {
                'model': api_model_name,
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': 1024,
                'temperature': 0.7,
                'api_key': api_key,
                'stop': ["</answer>"]
            }
            for prompt in prompts
        ]

        responses = litellm.batch_completion(requests, max_retries=max_retries, timeout=60)

        for response in responses:
            content = response.choices[0].message.content.strip()
            if not content.endswith('</answer>'):
                content += '</answer>'
            responses.append(content)

        return _parse_responses(responses, [])
        
    except Exception as e:
        logger.error(f"API call failed: {e}")
        use_api = False
        raise e

def call_vllm(prompts: List[str]) -> Tuple[List[int], Set[str]]:
    """Call VLLM for inference."""
    global use_vllm
    
    try:
        sampling_params = SamplingParams(
            temperature=0.1, 
            max_tokens=10, 
            stop=["</answer>"]
        )
        
        outputs = vllm_llm.generate(prompts, sampling_params)
        responses = []
        
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            if not generated_text.endswith('</answer>'):
                generated_text += '</answer>'
            responses.append(generated_text)
        
        return _parse_responses(responses, [])  # predictions will be passed separately
        
    except Exception as e:
        logger.error(f"VLLM call failed: {e}")
        use_vllm = False
        raise e

def call_transformer(prompts: List[str]) -> Tuple[List[int], Set[str]]:
    """Call Transformers pipeline for inference."""
    global use_transformer
    
    try:        
        outputs = transformer_llm(
            prompts, 
            max_new_tokens=10, 
            temperature=0.1, 
            do_sample=True,
            return_full_text=False
        )
        
        responses = []
        for output in outputs:
            if isinstance(output, list):
                text = output[0]['generated_text']
            else:
                text = output['generated_text']
            
            text = text.strip()
            if not text.endswith('</answer>'):
                text += '</answer>'
            responses.append(text)
        
        return _parse_responses(responses, [])  # predictions will be passed separately
        
    except Exception as e:
        logger.error(f"Transformer call failed: {e}")
        use_transformer = False
        raise e

def correct_prediction(predictions: List[str], 
                      labels: List[Union[str, List[str]]],
                      ) -> Tuple[List[int], Set[str]]:
    """
    Evaluate predictions against labels using available backends.
    
    Returns:
        Tuple of (correctness_list, correct_predictions_set)
    """
    # Using exact matching before using llm judge
    correctness = []
    correct_predictions = set()

    if isinstance(labels, str): labels = [labels]
    for pred in predictions:
        for label in labels:
            if pred.lower().strip() == label.lower().strip():
                correctness.append(1)
                correct_predictions.add(pred)
                break
        else:
            correctness.append(0)

    if len(''.join(predictions)) == len(predictions) and ((isinstance(labels, List) and len(labels) == len(''.join(labels))) or (isinstance(labels, str) and len(labels) == 1)):
        return correctness, correct_predictions
    
    if isinstance(labels, List): labels = '; '.join(labels)
    
    # Generate prompts
    prompts = [_get_few_shot_prompt(pred, labels) for pred, corr in zip(predictions, correctness) if corr == 0]
    
    # Try backends in order of preference
    if use_api:
        try:
            correct, _ = call_api(prompts)
            correct_prediction = {predictions[i] for i, c in enumerate(correct) if c == 1}
            logger.info(f"Used API backend: {sum(correct)}/{len(predictions)} correct")
            cur_index = 0
            for i in range(len(correctness)):
                if correctness[i] == 0:
                    correctness[i] = correct[cur_index]
                    cur_index += 1
            return correctness, correct_prediction | correct_predictions
        except:
            pass
    if use_vllm:
        try:
            correct, _ = call_vllm(prompts)
            correct_prediction = {predictions[i] for i, c in enumerate(correct) if c == 1}
            logger.info(f"Used VLLM backend: {sum(correct)}/{len(predictions)} correct")
            cur_index = 0
            for i in range(len(correctness)):
                if correctness[i] == 0:
                    correctness[i] = correct[cur_index]
                    cur_index += 1
            return correctness, correct_prediction | correct_predictions
        except:
            pass
    if use_transformer:
        try:
            correct, _ = call_transformer(prompts)
            correct_prediction = {predictions[i] for i, c in enumerate(correct) if c == 1}
            logger.info(f"Used Transformer backend: {sum(correct)}/{len(predictions)} correct")
            cur_index = 0
            for i in range(len(correctness)):
                if correctness[i] == 0:
                    correctness[i] = correct[cur_index]
                    cur_index += 1
            return correctness, correct_prediction | correct_predictions
        except:
            pass
    
    raise RuntimeError("All backends failed or are disabled")

# Example usage
if __name__ == "__main__":
    predictions = ["Paris is the capital", "London", "Tokyo is in Japan"]
    labels = ["Paris", ["London", "UK capital"], "Tokyo"]
    
    correctness, correct_set = correct_prediction(predictions, labels)
    print(f"Correctness: {correctness}")
    print(f"Correct predictions: {correct_set}")