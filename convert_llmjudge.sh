#!/bin/bash

# Create input files for Rubric

python -m rubric_llm_judge.llmjudge-inputs --query-path LLMJudge/data/llm4eval_query_2024.txt --input-qrel-path LLMJudge/data/llm4eval_dev_qrel_2024.txt -p data/llmjudge/llmjudge-passages_dev.json.gz LLMJudge/data/llm4eval_document_2024.jsonl  --query-out data/llmjudge/llmjudge_queries_dev.jsonl

python -m rubric_llm_judge.llmjudge-inputs --query-path LLMJudge/data/llm4eval_query_2024.txt --input-qrel-path LLMJudge/data/llm4eval_test_qrel_2024.txt -p data/llmjudge/llmjudge-passages_test.json.gz LLMJudge/data/llm4eval_document_2024.jsonl  --query-out data/llmjudge/llmjudge_queries_test.jsonl


datadir="../data/llmjudge"

cp llmjudge_queries_dev.jsonl ${datadir}
cp llmjudge_queries_test.jsonl ${datadir}
cp llmjudge-passages_dev.json.gz ${datadir}
cp llmjudge-passages_test.json.gz ${datadir}
