import json
from datasets import load_dataset
import random
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load datasets
print("Loading datasets...")
dataset = load_dataset("JetBrains-Research/lca-commit-message-generation", "default", split="test")
code_dataset = load_dataset("JetBrains-Research/lca-commit-message-generation", "retrieval_bm25", split="64k")

print("Creating hash mappings...")
# Create dictionary mapping hash to message from labels dataset
message_hash = {item['hash']: item['message'] for item in dataset}
diff_hash = {item['hash']: '\n\n'.join([f"@old_path\n{mod['old_path']}\n@new_path\n{mod['new_path']}@diff\n{mod['diff']}" for mod in item['mods']]) for item in dataset}

# Handle different possible structures for context_hash
context_hash = {}
for item in code_dataset:
    try:
        if 'context' in item and isinstance(item['context'], dict):
            # Original structure
            context_hash[item['hash']] = '\n\n'.join('@path\n' + item['context']['source'] + '\n@code\n' + item['context']['content'])
        elif 'context' in item and isinstance(item['context'], list):
            # If context is a list
            contexts = []
            for ctx in item['context']:
                if isinstance(ctx, dict) and 'source' in ctx and 'content' in ctx:
                    contexts.append('@path\n' + ctx['source'] + '\n@code\n' + ctx['content'])
            context_hash[item['hash']] = '\n\n'.join(contexts)
        elif 'source' in item and 'content' in item:
            # If source and content are directly in item
            context_hash[item['hash']] = '@path\n' + item['source'] + '\n@code\n' + item['content']
        else:
            # Fallback: use available text fields
            available_text = []
            for key in ['content', 'source', 'text']:
                if key in item and item[key]:
                    available_text.append(f"@{key}\n{item[key]}")
            context_hash[item['hash']] = '\n\n'.join(available_text)
    except Exception as e:
        print(f"Error processing item {item.get('hash', 'unknown')}: {e}")
        context_hash[item.get('hash', f'item_{len(context_hash)}')] = ""

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models like 'all-mpnet-base-v2' for better quality

# Get all commit messages for similarity search
all_messages = list(message_hash.values())
all_hashes = list(message_hash.keys())

print("Computing embeddings for all commit messages...")
# Compute embeddings for all messages
all_message_embeddings = model.encode(all_messages, show_progress_bar=True)

def find_similar_messages(target_message, target_hash, top_k=4):
    """Find top_k most similar messages to the target message, excluding the target itself"""
    # Get embedding for target message
    target_embedding = model.encode([target_message])
    
    # Compute similarities
    similarities = cosine_similarity(target_embedding, all_message_embeddings)[0]
    
    # Get indices of most similar messages (excluding the target message itself)
    similar_indices = []
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    
    for idx in sorted_indices:
        if all_hashes[idx] != target_hash:  # Exclude the target message itself
            similar_indices.append(idx)
        if len(similar_indices) == top_k:
            break
    
    # Return the most similar messages
    return [all_messages[idx] for idx in similar_indices]

print("Finding similar messages and generating QA samples...")
# Create similarity hash with proper similar message finding
sim_hash = {}
for hash_text, message in message_hash.items():
    similar_messages = find_similar_messages(message, hash_text, top_k=4)
    sim_hash[hash_text] = similar_messages

output_samples = []
for idx, (hash_text, message) in enumerate(message_hash.items()):
    if idx % 100 == 0:
        print(f"Processing sample {idx}/{len(message_hash)}")
    
    # Get diff and context, with fallback to empty string
    diff = diff_hash.get(hash_text, "")
    context = context_hash.get(hash_text, "")
    sim = sim_hash[hash_text]
    
    # Debug: Check if we have content
    if idx < 3:  # Print first 3 samples for debugging
        print(f"Sample {idx}:")
        print(f"  Hash: {hash_text}")
        print(f"  Diff length: {len(diff)}")
        print(f"  Context length: {len(context)}")
        print(f"  Similar messages: {len(sim)}")
    
    # Combine diff and context - ensure we have some content
    full_context_parts = []
    if diff.strip():
        full_context_parts.append(f"@diff\n{diff}")
    if context.strip():
        full_context_parts.append(f"@context\n{context}")
    
    # If no context, add a placeholder
    if not full_context_parts:
        full_context_parts.append(f"@info\nCommit hash: {hash_text}")
    
    full_context = '\n\n'.join(full_context_parts)
    
    # Create options (4 similar + 1 correct)
    options = sim + [message]
    random.shuffle(options)
    correct_answer = chr(65 + options.index(message))  # A, B, C, D, E
    
    # Format options text
    options_text = '\n'.join([chr(65 + i) + '. ' + msg for i, msg in enumerate(options)])
    option_labels = [chr(65 + i) for i in range(len(options))]
    
    # Create output sample
    output_sample = {
        "id": hash_text,
        "question": "Given the context, which commit message best describes the following code changes?\n" + options_text,
        "options": option_labels,
        "correct_answer": correct_answer,
        "search_results": [{
            'page_url': f'https://github.com/commit/{hash_text}',
            'page_name': f'Commit {hash_text[:8]}',
            "page_snippet": f"Code changes and context for commit {hash_text[:8]}", 
            "page_result": full_context,
        }]
    }
    
    output_samples.append(output_sample)

# Save results
print("Saving results...")
with open("commit_message_qa.json", "w", encoding='utf-8') as f:
    json.dump(output_samples, f, indent=2, ensure_ascii=False)

print(f"Generated {len(output_samples)} QA samples and saved to commit_message_qa.json")