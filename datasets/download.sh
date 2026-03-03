#!/usr/bin/env bash
set -e

TMP_DIR="_tmp_downloads"

mkdir -p "$TMP_DIR"

echo "=== Downloading MCQA CRAG ==="
if [ ! -f "crag_mcqa_task_1_and_2.json" ]; then
  curl -L -o "$TMP_DIR/mcqa-crag.zip" \
    https://www.kaggle.com/api/v1/datasets/download/curiouskat/mcqa-crag
  unzip -o "$TMP_DIR/mcqa-crag.zip" -d "$TMP_DIR/mcqa-crag"
  cp "$TMP_DIR/mcqa-crag/crag_task_1_and_2_mcqa.json" "crag_mcqa_task_1_and_2.json"
fi

echo "=== SciFact ==="
if [ ! -f "scifact.json" ]; then
  curl -L -o "$TMP_DIR/scifact.zip" \
    https://www.kaggle.com/api/v1/datasets/download/suzhentxt/scifact-mcqa
  unzip -o "$TMP_DIR/scifact.zip" -d "$TMP_DIR/scifact"
  cp "$TMP_DIR/scifact/scifact_mcqa.json" "scifact.json"
fi

echo "=== HealthVer ==="
if [ ! -f "healthver.json" ]; then
  curl -L -o "$TMP_DIR/healthver.zip" \
    https://www.kaggle.com/api/v1/datasets/download/suzhentxt/healthver-en-mcqa
  unzip -o "$TMP_DIR/healthver.zip" -d "$TMP_DIR/healthver"
  cp "$TMP_DIR/healthver/healthver_mcqa.json" "healthver.json"
fi

echo "=== DialFact ==="
if [ ! -f "dialfact.json" ]; then
  curl -L -o "$TMP_DIR/dialfact.zip" \
    https://www.kaggle.com/api/v1/datasets/download/suzhentxt/dialfact-mcqa-ver2
  unzip -o "$TMP_DIR/dialfact.zip" -d "$TMP_DIR/dialfact"
  cp "$TMP_DIR/dialfact/dialfact_mcqa_subset.json" "dialfact.json"
fi

echo "=== Done ==="


# COMMENTED OUT DATASETS

# # MCQA CRAG dataset
# Downloaded link: https://www.kaggle.com/datasets/curiouskat/mcqa-crag
# local path: datasets/crag_mcqa_task_1_and_2.json

# ```bash
# curl -L -o ./mcqa-crag.zip https://www.kaggle.com/api/v1/datasets/download/curiouskat/mcqa-crag
# unzip mcqa-crag.zip
# ```

# ## Multinew Sum
# local path: datasets/multinewsum_mcqa.json
# Link: https://huggingface.co/datasets/alexfabbri/multi_news

# ## Scifact
# link: https://huggingface.co/datasets/allenai/scifact
# local path: datasets/scifact.json
# ```bash
# #!/bin/bash
# curl -L -o ./scifact-mcqa.zip https://www.kaggle.com/api/v1/datasets/download/suzhentxt/scifact-mcqa
# unzip ./scifact-mcqa.zip
# ```


# ## Healthver
# link: https://huggingface.co/datasets/dwadden/healthver_entailment
# local path: datasets/healthver.json
# ```bash
# #!/bin/bash
# curl -L -o ./healthver-en-mcqa.zip https://www.kaggle.com/api/v1/datasets/download/suzhentxt/healthver-en-mcqa
# unzip healthver-en-mcqa.zip
# rm healthver-en-mcqa.zip
# ```

# ## ODEX
# link: https://huggingface.co/datasets/code-rag-bench/odex/
# local path: datasets/odex.json
# find similar answers

# ## DialFact
# link: https://github.com/salesforce/DialFact
# Download link: https://www.kaggle.com/api/v1/datasets/download/suzhentxt/dialfact-mcqa-ver2
# local path: datasets/dialfact.json
# ```bash
# curl -L -o ./dialfact-mcqa.zip https://www.kaggle.com/api/v1/datasets/download/suzhentxt/dialfact-mcqa-ver2
# unzip dialfact-mcqa.zip 
# ```

# ## lca-commit-message-generation
# link: https://huggingface.co/datasets/JetBrains-Research/lca-commit-message-generation
# local path: datasets/commit_message_qa.json

# ## Olympiad bench + MathPile docs
# link: https://huggingface.co/datasets/GAIR/MathPile
# link: huggingface.co/datasets/Hothan/OlympiadBench
# local path: datasets/OlympiadBench.json

# ## W/Odex
# Odex with database is "Amod/mental_health_counseling_conversations"

# ## W/DialFact
# DialFact with database is "Amod/mental_health_counseling_conversations"