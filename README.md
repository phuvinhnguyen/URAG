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
```

### Compare Systems
```bash
# Evaluate SimpleLLM
python cli.py --system simplellm --dataset datasets/comprehensive.json --output results_simplellm/

# Evaluate SimpleRAG  
python cli.py --system simplerag --dataset datasets/comprehensive.json --output results_simplerag/
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

### YAML Configuration Features

- **Auto Device Detection**: Device is automatically detected (CUDA > MPS > CPU) unless overridden
- **Flexible Parameters**: All system constructor arguments can be specified in `args`
- **System Auto-Discovery**: Systems are automatically discovered from `systems/` directory
- **Validation**: Configuration is validated before execution

### Example Configurations

See the `configs/` directory for examples:
- `configs/simplellm_example.yaml`: Basic LLM evaluation
- `configs/simplerag_comprehensive.yaml`: RAG system with larger dataset
- `configs/comparison.yaml`: Template for comparing systems

## Metrics Explained

### Core Metrics

- **Accuracy**: Standard classification accuracy (higher is better)
- **LAC Coverage**: Proportion of correct answers in LAC prediction sets
- **APS Coverage**: Proportion of correct answers in APS prediction sets  
- **LAC Avg Set Size**: Average size of LAC prediction sets (smaller is better)
- **APS Avg Set Size**: Average size of APS prediction sets (smaller is better)

### Conformal Prediction Methods

- **LAC (Least Ambiguous Classifier)**: Simple threshold on class probabilities
- **APS (Adaptive Prediction Sets)**: Adaptive threshold based on cumulative probability

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

### Auto-Discovery Features

- ✅ **No Registration Required**: Systems are found automatically using `importlib`
- ✅ **Filename = System Name**: Use the filename (without `.py`) as the system name
- ✅ **Flexible Parameters**: Pass any parameters via config file `args` section
- ✅ **Immediate Availability**: New systems work immediately without code changes
- ✅ **Error Handling**: Import errors are logged but don't break other systems

### System Naming Convention

- File: `systems/myawesomesystem.py` → System name: `myawesomesystem`
- File: `systems/gpt4_rag.py` → System name: `gpt4_rag`
- File: `systems/experimental.py` → System name: `experimental`

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

## Advanced Usage

### Custom Model Configuration

```bash
# Use a larger model
python cli.py --system simplellm --dataset datasets/example.json --model microsoft/DialoGPT-medium

# Force CPU usage
python cli.py --system simplellm --dataset datasets/example.json --device cpu
```

### Different Confidence Levels

```bash
# 95% coverage (more conservative)
python cli.py --system simplellm --dataset datasets/example.json --alpha 0.05

# 80% coverage (less conservative) 
python cli.py --system simplellm --dataset datasets/example.json --alpha 0.2
```

### Batch Evaluation

For evaluating multiple systems:

```bash
#!/bin/bash
for system in simplellm; do
  for alpha in 0.05 0.1 0.2; do
    echo "Evaluating $system with alpha=$alpha"
    python cli.py --system $system --dataset datasets/example.json --alpha $alpha --output results/${system}_alpha${alpha}/
  done
done
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use smaller model or CPU
   ```bash
   python cli.py --system simplellm --dataset datasets/example.json --device cpu
   ```

2. **Model download fails**: Check internet connection and HuggingFace model name

3. **Dataset format errors**: Validate JSON structure and required fields

### Debug Mode

Enable verbose logging for detailed information:

```bash
python cli.py --system simplellm --dataset datasets/example.json --verbose
```

## Performance Tips

1. **Use appropriate device**: GPU for larger models, CPU for smaller ones
2. **Batch size optimization**: Larger systems can process multiple samples at once
3. **Model selection**: Balance between accuracy and speed based on your needs

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{urag2024,
  title={URAG: Uncertainty-aware RAG Evaluation with Conformal Prediction},
  year={2024},
  url={<repository-url>}
}
```

## References

- Romano et al. "Conformalized Quantile Regression" (2019)
- Angelopoulos et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction" (2020)

## License

[License information here]
