from datasets import load_dataset
import json
import random
import time
import google.generativeai as genai
from typing import List
import re

class GeminiTokenRotator:
    def __init__(self, api_tokens: List[str]):
        """
        Initialize with a list of API tokens for rotation
        """
        self.api_tokens = api_tokens
        self.current_token_index = 0
        self.configure_current_token()
    
    def configure_current_token(self):
        """Configure Gemini with current token"""
        current_token = self.api_tokens[self.current_token_index]
        genai.configure(api_key=current_token)
        self.model = genai.GenerativeModel(random.choice(['gemini-2.5-pro',
                                                          'gemini-2.5-flash'
                                                          'gemini-2.0-flash',
                                                          'gemini-2.0-flash-001']))
        print(f"Using token {self.current_token_index + 1}/{len(self.api_tokens)}")
    
    def rotate_token(self):
        """Rotate to next token"""
        self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
        self.configure_current_token()
        print(f"Rotated to token {self.current_token_index + 1}")
    
    def generate_content(self, prompt: str, max_retries: int = 3):
        """
        Generate content with automatic token rotation on rate limit
        """
        for attempt in range(max_retries):
            time.sleep(15)  # Brief pause
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    self.rotate_token()
                else:
                    print(f"Max retries reached. Error: {e}")
                    raise e
        
        raise Exception("Failed to generate content after all retries")

def parse_fake_answers(fake_answer_text: str):
    """
    Parse the generated fake answers from Gemini response
    """
    try:
        # Split by <eos> first
        content = fake_answer_text.split('<eos>')[0].split('<answer>')[1:]
        return content
    except Exception as e:
        print(f"Error parsing fake answers: {e}")
        print(f"Raw response: {fake_answer_text}")
        # Return some default wrong answers as fallback
        return ["0", "1", "-1"]

def create_prompt(question: str, solution: str, correct_answer: str):
    """Create prompt for generating fake answers"""
    return f"""
Generate 3 wrong but plausible answers based on the given math question, solution, and correct answer.

Question: {question}
Solution: {solution}
Correct Answer: {correct_answer}

You should assume people might make errors at various steps (mis-computation, using wrong theorem, using wrong formula, overcomplicating the problem, etc.) and generate wrong answers from there.

Your wrong answers must have similar format to the correct answer (LaTeX format).

Format your response exactly like this:
reasoning<answer>first wrong answer<answer>second wrong answer<answer>third wrong answer<eos>

For example:
1+1=2, wrong answers should be 3,4,1, provided correct answer is $2$, so wrong answers must be formated similarly<answer>$3$<answer>$4$<answer>$1$<eos>

Make sure each wrong answer is mathematically plausible but incorrect.
"""

def main():
    # Initialize API tokens - replace with your actual tokens
    API_TOKENS = ['YOUR_GEMINI_API_TOKEN_1']
    
    # Validate that tokens are provided
    if API_TOKENS[0] == "YOUR_GEMINI_API_TOKEN_1":
        print("Please replace API_TOKENS with your actual Gemini API tokens!")
        return
    
    # Initialize Gemini with token rotation
    gemini_rotator = GeminiTokenRotator(API_TOKENS)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('Hothan/OlympiadBench', 'OE_TO_maths_en_COMP', split='train')
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize data structure
    data = {
        "name": "OlympiadBench",
        "description": "OlympiadBench is a dataset for evaluating the performance of RAG systems in solving math problems.",
        "version": "1.0",
        "total_samples": 0,
        "calibration_samples": 0,
        "test_samples": 0,
        "calibration": [],
        "test": []
    }
    
    # Process each item
    for index, item in enumerate(dataset):
        print(f"\nProcessing item {index + 1}/{len(dataset)}")
        
        try:
            question = item['question']
            answer = item['final_answer'][0]  # Get first final answer
            solution = item['solution']
            subfield = item['subfield']
            answer_type = item['answer_type']
            difficulty = item['difficulty']
            
            # Create prompt for Gemini
            prompt = create_prompt(question, solution, answer)
            
            # Generate fake answers using Gemini with token rotation
            print("Generating fake answers...")
            fake_answer_response = gemini_rotator.generate_content(prompt)
            
            # Parse fake answers
            fake_answers = parse_fake_answers(fake_answer_response)
            
            # Combine with correct answer
            all_answers = fake_answers + [answer]
            random.shuffle(all_answers)
            
            # Find correct answer position
            correct_answer = chr(65 + all_answers.index(answer))  # A, B, C, D
            
            # Create options text
            options_text = '\\n'.join([chr(65 + i) + '. ' + ans for i, ans in enumerate(all_answers)])
            
            # Create question with options
            full_question = question + '\\n\\nOptions:\\n' + options_text
            
            # Create options list
            options = [chr(65 + i) for i in range(len(all_answers))]
            
            # Create result object
            result = {
                'id': f'olympiad_{index}',
                'question': full_question,
                'correct_answer': correct_answer,
                'options': options,
                'answer_type': answer_type,
                'subfield': subfield,  # Fixed typo
                'difficulty': difficulty,
                'search_results': [{
                    'page_url': '',
                    'page_name': '',
                    'page_snippet': '',
                    'page_result': '',
                    'persistent_storage': ['hoskinson-center/proof-pile']
                }]
            }
            
            # Distribute between calibration and test
            if index % 2 == 0:
                data['calibration'].append(result)
                data['calibration_samples'] += 1
            else:
                data['test'].append(result)
                data['test_samples'] += 1
            
            data['total_samples'] += 1
            
            # Add small delay to be respectful to API
            
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            continue
    
    # Save results
    print(f"\\nSaving {data['total_samples']} processed samples...")
    with open('OlympiadBench.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Saved {data['calibration_samples']} calibration and {data['test_samples']} test samples.")

if __name__ == "__main__":
    main()