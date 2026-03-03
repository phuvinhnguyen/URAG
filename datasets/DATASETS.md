# DATASET

## Format

**One question, one database**
```json
{
    "name": "dataset name",
    "description": "dataset description",
    "version": "1.0",
    "total_samples": 100,
    "calibration_samples": 50,
    "test_samples": 50,
    "calibration": [
    {
        "id": 1,
        "question": "Some question ...",
        "correct_answer": "D",
        "options": [
            "A",
            "B",
            "C",
            "D"
        ],
        "search_results": [
        {
          "page_result": "Some contents to retrieve from",
          "page_snippet": "",
          "page_title": "",
          "page_url": "",
          "page_rank": 1,
          "page_score": 1.0,
          "page_source": "generated",
          "page_domain": "",
          "page_language": "en",
          "page_metadata": {}
        }
        ],
        "query_time": "March 1, 2024",
        "technique": "rag",
        "metadata": {}
    }
    ],
    "test": [
        {
        "id": 1,
        "question": "Some question ...",
        "correct_answer": "D",
        "options": [
            "A",
            "B",
            "C",
            "D"
        ],
        "search_results": [
            {
            "page_result": "Some contents to retrieve from",
            "page_snippet": "",
            "page_title": "",
            "page_url": "",
            "page_rank": 1,
            "page_score": 1.0,
            "page_source": "generated",
            "page_domain": "",
            "page_language": "en",
            "page_metadata": {}
            }
        ],
        "query_time": "March 1, 2024",
        "technique": "rag",
        "metadata": {}
        }
    ]
```

**All questions, shared database**

```json
{
    "name": "dataset name",
    "description": "dataset description",
    "version": "1.0",
    "total_samples": 100,
    "calibration_samples": 50,
    "test_samples": 50,
    "calibration": [
    {
        "id": 1,
        "question": "Some question ...",
        "correct_answer": "D",
        "options": [
            "A",
            "B",
            "C",
            "D"
        ],
        "search_results": [
            {
                "page_name": "",
                "page_url": "",
                "page_snippet": "", 
                "page_result": "",
                "persistent_storage": [
                    "name of the database to be constructed in URAG/utils/storage.py"
                ]
            }
        ],
        "query_time": "March 1, 2024",
        "technique": "rag",
        "metadata": {}
    }
    ],
    "test": [
        {
        "id": 1,
        "question": "Some question ...",
        "correct_answer": "D",
        "options": [
            "A",
            "B",
            "C",
            "D"
        ],
        "search_results": [
            {
                "page_name": "",
                "page_url": "",
                "page_snippet": "", 
                "page_result": "",
                "persistent_storage": [
                    "name of the database to be constructed in URAG/utils/storage.py"
                ]
            }
        ]
        "query_time": "March 1, 2024",
        "technique": "rag",
        "metadata": {}
        }
    ]
```
Where URAG/utils/storage.py can be found [here](https://github.com/phuvinhnguyen/URAG/blob/master/utils/storage.py)

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


# Download

To download all missing datasets we used in the evaluation, run this command:
```bash
bash download.sh
```