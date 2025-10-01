import json as js
import os

if __name__ == "__main__":
    dataset_path = ['/media/volume/LLMRag/URAG/datasets/commit_message_qa.json', '/media/volume/LLMRag/URAG/datasets/crag_task_1_and_2_mcqa.json',
    '/media/volume/LLMRag/URAG/datasets/dialfact.json', '/media/volume/LLMRag/URAG/datasets/healthver_mcqa.json',
    '/media/volume/LLMRag/URAG/datasets/multinewsum_mcqa.json', '/media/volume/LLMRag/URAG/datasets/odex.json',
    '/media/volume/LLMRag/URAG/datasets/OlympiadBench.json', '/media/volume/LLMRag/URAG/datasets/scifact_mcqa.json']
    for file in dataset_path:
        dataset = js.load(open(os.path.join(os.path.dirname(__file__), file), "r"))
        print('==========================================================================')
        print(f"Dataset description: {dataset['description']}")
        print(f"Dataset {dataset['name']} has {dataset['total_samples']} samples")
        print(f"Dataset {file} has {dataset['calibration_samples']} calibration samples")
        print(f"Dataset {file} has {dataset['test_samples']} test samples")


