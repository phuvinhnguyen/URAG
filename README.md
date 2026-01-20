# URAG: Uncertainty-aware RAG Evaluation with Conformal Prediction

A framework for evaluating RAG (Retrieval-Augmented Generation) systems using conformal prediction techniques. This provides statistical guarantees on prediction uncertainty and enables rigorous comparison between different system architectures.

## Quick Start

### Installation

```bash
git clone <repository-url>
cd URAG
pip install -r requirements.txt
```

### Basic Usage

Evaluate a simple LLM system on the example dataset:

```bash
python cli.py --system simplellm --dataset datasets/example.json
```

## Command Line Interface

### Basic Command

```bash
python cli.py --system <system_name> --dataset <dataset_path>
```

### YAML Configuration

```bash
python cli.py --config <config_file.yaml>
```

### Full Options

```bash
python cli.py \
  --system simplellm \
  --dataset datasets/example.json \
  --output results/ \
  --alpha 0.1 \
  --model microsoft/DialoGPT-small \
  --device auto \
  --verbose
```

### Parameters

- `--config`: Path to YAML configuration file (alternative to individual options)
- `--system`: System to evaluate (required if not using config)
  - `simplellm`: Simple LLM baseline without RAG
  - `simplerag`: Simple RAG system with keyword-based retrieval
  - `HyDErag`: HyDE RAG system with hypothetical document embeddings
- `--dataset`: Path to dataset JSON file (required if not using config)
- `--output`: Output directory for results (default: `results/`)
- `--alpha`: Conformal prediction error rate (default: `0.1` for 90% coverage)
- `--model`: HuggingFace model name (default: `microsoft/DialoGPT-small`)
- `--device`: Device to use - `auto`, `cpu`, or `cuda` (default: `auto`, auto-detects best available)
- `--verbose`: Enable detailed logging

## Example Commands

### Basic Evaluation
```bash
python cli.py --system simplellm --dataset datasets/example.json
```

### High Coverage (95%)
```bash
python cli.py --system simplellm --dataset datasets/example.json --alpha 0.05
```

### Using Different Model
```bash
python cli.py --system simplellm --dataset datasets/example.json --model microsoft/DialoGPT-medium
```

### Verbose Output
```bash
python cli.py --system simplellm --dataset datasets/example.json --verbose
```

### YAML Configuration Files
```bash
# Use predefined config
python cli.py --config configs/simplellm_example.yaml

# Use comprehensive dataset with RAG
python cli.py --config configs/simplerag_comprehensive.yaml

# Use HyDE RAG system
python cli.py --config configs/hyde_example.yaml
```

### Compare Systems
```bash
python compare_performance.py <path_to_result_1.json> <path_to_result_2.json> <output_path>
```
Example:
```bash
python compare_performance.py ../results/a.json ../results/b.json ../test.json
```

## Dataset Format

Datasets should be JSON files with the following structure:

```json
{
  "name": "Dataset Name",
  "description": "Dataset description",
  "calibration": [
    {
      "id": "cal_001",
      "question": "What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "C",
      "technique": "direct",
      "domain": "geography"
    }
  ],
  "test": [
    {
      "id": "test_001", 
      "question": "What is the smallest unit of matter?\nA. Molecule\nB. Atom\nC. Electron\nD. Proton",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B",
      "technique": "direct",
      "domain": "science"
    }
  ]
}
```

### Required Fields

For each sample in `calibration` and `test`:

- `id`: Unique identifier
- `question`: Multiple choice question with options
- `options`: List of option labels (e.g., ["A", "B", "C", "D"])
- `correct_answer`: Correct option label

### Optional Fields

- `technique`: Prompting technique (`direct`, `cot`, `rag`)
- `domain`: Question domain/category
- `context`: Additional context for RAG systems
- `search_results`: Pre-retrieved context

## YAML Configuration Format

Configuration files use YAML format for easy editing and version control:

```yaml
# System configuration
system:
  name: simplellm  # System name (must match filename in systems/)
  args:
    model_name: microsoft/DialoGPT-small  # Model parameters
    alpha: 0.1  # Conformal prediction confidence level
    # device: cuda  # Optional: override auto-detection

# Data and output paths
dataset: datasets/example.json
output: results/my_experiment

# Additional system-specific parameters can be added under args:
```

### Example Configurations

See the `configs/` directory for examples:
- `configs/simplellm_example.yaml`: Basic LLM evaluation
- `configs/simplerag_comprehensive.yaml`: RAG system with larger dataset
- `configs/hyde_example.yaml`: HyDE RAG system with hypothetical document embeddings
- `configs/comparison.yaml`: Template for comparing systems

## Metrics Explained

### Core Metrics

- **Accuracy**: Standard classification accuracy (higher is better)
- **LAC Coverage**: Proportion of correct answers in LAC prediction sets
- **APS Coverage**: Proportion of correct answers in APS prediction sets  
- **Average Coverage**: Average of LAC Coverage and APS Coverage
- **LAC Average Set Size**: Average size of LAC prediction sets (smaller is better)
- **APS Average Set Size**: Average size of APS prediction sets (smaller is better)
- **Average Set Size**: Average of APS Average Set Size and LAC Average Set Size
- **LAC Set Size**: In comparison, this metric compares LAC sample by sample before taking average
- **APS Set Size**: In comparison, this metric compares APS sample by sample before taking average
- **Set Size**: In comparison, this metric computes Set Size sample by sample, then compare this score sample by sample before taking average

### Conformal Prediction Methods

- **LAC (Least Ambiguous Classifier)**: Simple threshold on class probabilities
- **APS (Adaptive Prediction Sets)**: Adaptive threshold based on cumulative probability
- **Set size**: $\frac{LAC+APS}{2}$

Both methods provide **statistical coverage guarantees**: for confidence level (1-α), the prediction sets satisfy P(y_true ∈ C(x)) ≥ 1-α.

## Output Files

The evaluation produces three main output files in the specified output directory:

1. **Calibration Results** (`calibration_results_TIMESTAMP.json`)
   - Detailed results for calibration set
   - Individual sample predictions and probabilities

2. **Test Results** (`test_results_TIMESTAMP.json`)
   - Detailed results for test set  
   - Predictions, probabilities, and system responses

3. **Evaluation Metrics** (`evaluation_metrics_TIMESTAMP.json`)
   - Summary metrics and statistics
   - Threshold values and coverage statistics

## Available Systems

### SimpleLLM (`simplellm`)

Simple LLM system without RAG capabilities. Serves as a baseline for comparison.

**Features:**
- Direct prompting with multiple choice questions
- Probability estimation via logit analysis
- Support for different prompting techniques

**Usage:**
```bash
python cli.py --system simplellm --dataset datasets/example.json
```

### SimpleRAG (`simplerag`)

Simple RAG system with keyword-based retrieval from a built-in knowledge base.

**Features:**
- Keyword-based document retrieval
- Context augmentation for LLM prompts
- Built-in knowledge base covering multiple domains
- Retrieval scoring and ranking

**Usage:**
```bash
python cli.py --system simplerag --dataset datasets/example.json
```

### HyDERAG (`HyDErag`)

Advanced RAG system using HyDE (Hypothetical Document Embeddings) technique for improved retrieval.

**Features:**
- Hypothetical document generation for better semantic retrieval
- Combines semantic similarity with keyword matching
- Enhanced context augmentation
- Fallback to traditional retrieval when needed
- Built-in comprehensive knowledge base

**Usage:**
```bash
python cli.py --system HyDErag --dataset datasets/example.json
```

**How HyDE Works:**
1. Generate a hypothetical document that would answer the query
2. Use the hypothetical document for semantic retrieval instead of the raw query  
3. Retrieve relevant documents based on semantic similarity
4. Generate final answer using retrieved context

**Compare HyDE vs SimpleRAG:**
```bash
# Test HyDE
python cli.py --system HyDErag --dataset datasets/example.json --output results/hyde_test

# Test SimpleRAG
python cli.py --system simplerag --dataset datasets/example.json --output results/simplerag_test

# Compare results
python compare_performance.py results/hyde_test/evaluation_metrics_*.json results/simplerag_test/evaluation_metrics_*.json results/comparison.json
```

## Adding New Systems

Adding new systems is now automatic! Just create a new file in the `systems/` directory.

### Step 1: Create System File

Create a new Python file in `systems/` directory (e.g., `systems/mysystem.py`):

```python
# systems/mysystem.py
from systems.abstract import AbstractRAGSystem
from typing import Dict, Any

class MyRAGSystem(AbstractRAGSystem):
    def __init__(self, **kwargs):
        # Initialize your system with any parameters
        # All kwargs from config file 'args' section will be passed here
        self.my_parameter = kwargs.get('my_parameter', 'default_value')
    
    def get_batch_size(self) -> int:
        return 1
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Process sample and return required format
        return {
            'id': sample.get('id'),
            'generated_response': response,
            'predicted_answer': answer,
            'option_probabilities': probabilities
        }
```

### Step 2: Use Immediately

The system is automatically discovered! Use it right away:

**Command Line:**
```bash
python cli.py --system mysystem --dataset datasets/example.json
```

**YAML Config:**
```yaml
system:
  name: mysystem  # Filename without .py extension
  args:
    my_parameter: custom_value
    model_name: microsoft/DialoGPT-medium

dataset: datasets/example.json
output: results/mysystem_test
```
## Example Results

```
============================================================
EVALUATION RESULTS  
============================================================
Total Samples: 5
Accuracy: 0.8000
LAC Coverage: 0.9000
APS Coverage: 0.9000
LAC Avg Set Size: 2.4000
APS Avg Set Size: 2.2000

Thresholds:
  LAC Threshold: 0.3457
  APS Threshold: 0.4123

Output Files:
  Calibration: results/calibration_results_20241201_143022.json
  Test: results/test_results_20241201_143022.json
  Metrics: results/evaluation_metrics_20241201_143022.json
============================================================
```

## Reproduce

Please refer (and switch) to the `dev` branch for more information.
