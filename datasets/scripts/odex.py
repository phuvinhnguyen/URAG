import json
import random
import difflib
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('code-rag-bench/odex')['train']

data = {
    "name": "ODEX",
    "description": "ODEX is a dataset for evaluating the performance of RAG systems in simple coding tasks.",
    "version": "1.0",
    "total_samples": len(dataset),
    "calibration_samples": 0,
    "test_samples": 0,
    "calibration": [],
    "test": []
}

# Precompute all canonical solutions for similarity matching
all_solutions = [item['canonical_solution'] for item in dataset]

for i, item in enumerate(dataset):
    # Find similar solutions using difflib
    current_solution = item['canonical_solution']
    similar_indices = []
    
    # Get similarity scores for all solutions
    for j, solution in enumerate(all_solutions):
        if j != i:  # Exclude current solution
            similarity = difflib.SequenceMatcher(None, current_solution, solution).ratio()
            similar_indices.append((j, similarity))
    
    # Sort by similarity and get top 3
    similar_indices.sort(key=lambda x: x[1], reverse=True)
    top3_indices = [idx for idx, _ in similar_indices[:3]]
    
    # Create options (correct answer + 3 similar ones)
    options = [current_solution] + [all_solutions[idx] for idx in top3_indices]
    random.shuffle(options)
    
    # Identify correct answer index
    correct_index = options.index(current_solution)
    correct_answer = chr(65 + correct_index)  # Convert to A-D
    
    # Format options with letters
    option_text = '\n'.join([f"{chr(65+j)}. {option}" for j, option in enumerate(options)])
    
    # Create question text
    question = f"{item['intent']}\n{item['prompt']}\n\nOptions:\n{option_text}"
    
    # Create search results structure
    search_results = [{
        'page_name': 'Library Documentation',
        'page_url': '',
        'page_snippet': 'Relevant code examples and documentation',
        'page_result': '',
        'persistent_storage': ['code-rag-bench/library-documentation']
    }]
    
    # Create item structure
    citem = {
        'id': i,
        'question': question,
        'correct_answer': correct_answer,
        'options': [chr(65+j) for j in range(4)],
        'search_results': search_results
    }
    
    # Split into calibration and test sets
    if i % 2 == 0:
        data['calibration_samples'] += 1
        data['calibration'].append(citem)
    else:
        data['test_samples'] += 1
        data['test'].append(citem)

# Save to JSON file
with open('./odex.json', 'w') as f:
    json.dump(data, f, indent=4)

print(f"Dataset created with {data['calibration_samples']} calibration and {data['test_samples']} test samples")