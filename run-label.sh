#!/usr/bin/env bash

set -e -x

name="hi"
dataset="/home/ben/rubric-llm-judge/LLMJudge/data"
judgement_dir="/home/dietz/jelly-home/peanut-jupyter/exampp/data/llmjudge"
judgements="$judgement_dir/questions-explain--questions-rate--llmjudge-passages_dev.json.gz"
#judgements="$judgement_dir/nuggets-rate--all-llmjudge-passages_dev.json.gz"
#judgements="$judgement_dir/Thomas-Sun_few-HELM-FagB_few-Sun-FagB-llmjudge-passages_dev.json.gz"
test_judgements="$judgement_dir/nuggets-explain--nuggets-rate--all-llmjudge-passages_test.json.gz"

train() {
    python -m rubric_llm_judge.label \
        train \
        --qrel $dataset/llm4eval_dev_qrel_2024.txt \
        -j $judgements \
        -c LogReg \
        -o $name.model
}

predict() {
    python -m rubric_llm_judge.label \
        predict \
        -m $name.model \
        --qrel $dataset/llm4eval_dev_qrel_2024.txt \
        -j $judgements \
        -o $name-dev.jsonl.gz \
        --output-qrel $name-dev.qrel
}

test() {
    python -m rubric_llm_judge.label \
        predict \
        -m $name.model \
        --qrel $dataset/llm4eval_test_qrel_2024.txt \
        -j $test_judgements \
        -o $name-test.jsonl.gz \
        --output-qrel $name-test.qrel
}

exampp_balance() {
    zcat $1 | jq .[1].[].grades.[].self_ratings | sort | uniq -c
}

qrel_balance() {
    cut -f4 -d' ' $1 | sort | uniq -c
}

$@
