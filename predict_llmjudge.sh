#!/bin/bash

datadir="../data/llmjudge"  # configure to the directory where the walkthrough-llmjudge.sh will write the data files

python -m rubric_llm_judge.llmjudge-relevance-label ${datadir}/questions-rate--llmjudge-passages_dev.json.gz  --input-qrel-path LLMJudge/data/llm4eval_dev_qrel_2024.txt --output-qrel-path llm4eval_dev_qrel_2024-predicted.txt
