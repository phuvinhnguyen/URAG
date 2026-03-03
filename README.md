# URAG: Uncertainty-aware RAG Evaluation

URAG is a framework for evaluating RAG (Retrieval-Augmented Generation) systems with conformal prediction on multiple-choice QA benchmarks.

## Running experiments (YAML only)

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Run everything via a config file**:

```bash
python cli.py --config <path/to/config.yaml>
```

- **Examples** (LCA commit-message dataset):

```bash
python cli.py --config URAG/configs/simplellm_normal_lca.yaml
python cli.py --config URAG/configs/simplerag_normal_lca.yaml
python cli.py --config URAG/configs/hyderag_normal_lca.yaml
```

Each config specifies:
- `system`: which system to run (e.g., `simplellm`, `simplerag`, `hyderag`, `fusionrag`, `ratrag`, `raptorrag`, `replugrag`, `selfrag`).
- `dataset`: path to the MCQA dataset JSON (e.g., `datasets/commit_message_qa.json`).
- `output`: directory where calibration/test results and metrics JSON files are written.

For available datasets, download commands, and statistics, see `datasets/DATASETS.md`.

## Implementing a new system

Systems are defined by subclassing `AbstractRAGSystem` in `systems/abstract.py`.

- **Minimal implementation**:
  - Create `systems/mysystem.py`.
  - Subclass `AbstractRAGSystem`.
  - Implement:
    - `get_batch_size(self) -> int`
    - Either:
      - `process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]` (single-sample),
      - or `batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]` for true batching.

- **Expected output format per sample** (from `process_sample` or each element from `batch_process_samples`):

```python
{
    "id": sample.get("id"),
    "generated_response": "... raw model output ...",
    "predicted_answer": "A",  # single option label
    "conformal_probabilities": {"A": 0.7, "B": 0.2, "C": 0.1},  # or similar dict of option → prob
    # optional: "reasoning", "retrieved_docs", "confidence", "processing_time", ...
}
```

The base class also provides helper methods (e.g., `_generate_response_with_probabilities`, `_generate_response_with_probabilities_normal`, `_extract_answer_probabilities`) that you can reuse to get answer probabilities from a HF model.

## YAML config format

A config file fully specifies **which system**, **which dataset**, and **where to write results**:

```yaml
system:
  name: simplellm          # filename in systems/ without .py
  alpha: 0.1               # conformal prediction error rate (1 - alpha coverage)
  args:
    model_name: meta-llama/Llama-3.1-8B-Instruct
    method: normal         # prompting methods (normal, aware, ...)

dataset: datasets/commit_message_qa.json # path to dataset as a json file
output: results/simplellm/normal/llama_3.1_8b_instruct/lca_commit_message_0.1 # location to save the result
```

- `system.name` must match a file `systems/<name>.py` that defines a subclass of `AbstractRAGSystem`.
- All keys under `system.args` are forwarded as keyword arguments to that system’s constructor.
- `alpha`, `dataset`, and `output` control conformal calibration level, input data, and output location respectively.

To run a new system, create a system file under `systems/`, add a YAML config in `configs/`, then execute:

```bash
python cli.py --config <your_new_config.yaml>
```

## Result

After running, you will see something like this:
```txt
============================================================
EVALUATION RESULTS
============================================================
Total Samples: 82
Accuracy: $a
LAC Coverage: $lacc
APS Coverage: $apsc
LAC Avg Set Size: $lac_ss
APS Avg Set Size: $aps_ss

Thresholds:
  LAC Threshold: 0.9631
  APS Threshold: 0.9998

Output Files:
  Calibration: $x
  Test: $y
  Metrics: $z
```

where:
- Accuracy: $a
- Coverage: ($lacc + $apsc)/2
- Set Size: ($lac_ss + $aps_ss)/2
- LLM's answer (what the system returns in `process_sample` and `batch_process_samples` methods) for all samples are saved in $x (calibration set) and $y (test set)
- Performance, set size, coverage rate, conformal probabilities are saved in $z
