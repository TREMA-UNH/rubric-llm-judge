#!/bin/bash

python -m rubric_llm_judge.llmjudge-inputs --query-path LLMJudge/data/llm4eval_query_2024.txt --input-qrel-path LLMJudge/data/llm4eval_dev_qrel_2024.txt -p llmjudge-passages_dev.json.gz LLMJudge/data/llm4eval_document_2024.jsonl  --query-out llmeval_dev.jsonl.gz

python -m rubric_llm_judge.llmjudge-inputs --query-path LLMJudge/data/llm4eval_query_2024.txt --input-qrel-path LLMJudge/data/llm4eval_test_qrel_2024.txt -p llmjudge-passages_test.json.gz LLMJudge/data/llm4eval_document_2024.jsonl  --query-out llmeval_test.jsonl.gz
