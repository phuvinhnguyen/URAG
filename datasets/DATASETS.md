## MCQA CRAG dataset
Downloaded link: https://www.kaggle.com/datasets/curiouskat/mcqa-crag
local path: datasets/crag_mcqa_task_1_and_2.json

```bash
curl -L -o ./mcqa-crag.zip https://www.kaggle.com/api/v1/datasets/download/curiouskat/mcqa-crag
unzip mcqa-crag.zip
```

## MCQA CRAG tiny v
Downloaded link: https://www.kaggle.com/datasets/curiouskat/mcqa-crag
local path: datasets/crag_mcqa_task_1_and_2_tiny.json

```bash
curl -L -o ./mcqa-crag.zip https://www.kaggle.com/api/v1/datasets/download/curiouskat/mcqa-crag
unzip mcqa-crag.zip
rm mcqa-crag.zip
```

## Multinew Sum v
local path: datasets/multinewsum_mcqa.json
Link: https://huggingface.co/datasets/alexfabbri/multi_news

## Scifact
link: https://huggingface.co/datasets/allenai/scifact
local path: datasets/scifact.json
```bash
#!/bin/bash
curl -L -o ./scifact-mcqa.zip https://www.kaggle.com/api/v1/datasets/download/suzhentxt/scifact-mcqa
unzip ./scifact-mcqa.zip
```


## Healthver v
link: https://huggingface.co/datasets/dwadden/healthver_entailment
local path: datasets/healthver.json
```bash
#!/bin/bash
curl -L -o ./healthver-en-mcqa.zip https://www.kaggle.com/api/v1/datasets/download/suzhentxt/healthver-en-mcqa
unzip healthver-en-mcqa.zip
rm healthver-en-mcqa.zip
```

## ODEX v
link: https://huggingface.co/datasets/code-rag-bench/odex/
local path: datasets/odex.json
find similar answers

## DialFact v
link: https://github.com/salesforce/DialFact
Download link: https://www.kaggle.com/api/v1/datasets/download/suzhentxt/dialfact-mcqa-ver2
local path: datasets/dialfact.json
```bash
curl -L -o ./dialfact-mcqa.zip https://www.kaggle.com/api/v1/datasets/download/suzhentxt/dialfact-mcqa-ver2
unzip dialfact-mcqa.zip 
```

## lca-commit-message-generation v
link: https://huggingface.co/datasets/JetBrains-Research/lca-commit-message-generation
local path: datasets/commit_message_qa.json

## Olympiad bench + MathPile docs v
link: https://huggingface.co/datasets/GAIR/MathPile
link: huggingface.co/datasets/Hothan/OlympiadBench
local path: datasets/OlympiadBench.json

## W/Odex
Odex with database is "Amod/mental_health_counseling_conversations"

## W/DialFact
DialFact with database is "Amod/mental_health_counseling_conversations"



# Database statistic

| Dataset file | Database used | Samples | Total docs | Avg docs/sample | Total text chars | Avg chars/doc | Total words | Avg words/doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OlympiadBench.json | `persistent:hoskinson-center/proof-pile` | 661 | 20,672 (shared DB) | N/A (shared) | 51,167,805 | 2,475.22 | 8,910,741 | 431.05 |
| commit_message_qa.json | inline search_results | 163 | 163 | 1.00 | 42,177,274 | 258,756.28 | 3,879,366 | 23,799.79 |
| crag_task_1_and_2_mcqa.json | inline search_results | 2,330 | 11,650 | 5.00 | 4,208,072,919 | 361,207.98 | 179,542,626 | 15,411.38 |
| dialfact.json | inline search_results | 2,000 | 2,435 | 1.22 | 11,835,703 | 4,860.66 | 1,877,746 | 771.15 |
| healthver_mcqa.json | inline search_results | 1,332 | 4,191 | 3.15 | 88,159,278 | 21,035.38 | 13,167,923 | 3,141.95 |
| multinewsum_mcqa.json | inline search_results | 950 | 950 | 1.00 | 6,109,017 | 6,430.54 | 993,322 | 1,045.60 |
| odex.json | `persistent:code-rag-bench/library-documentation` | 439 | 34,003 (shared DB) | N/A (shared) | 60,254,653 | 1,772.04 | 8,113,623 | 238.61 |
| wodex.json | `persistent:Amod/mental_health_counseling_conversations` | 439 | 3,512 (shared DB) | N/A (shared) | 4,666,469 | 1,328.72 | 818,938 | 233.18 |
| scifact_mcqa.json | inline search_results | 374 | 395 | 1.06 | 26,929,250 | 68,175.32 | 3,872,744 | 9,804.42 |
