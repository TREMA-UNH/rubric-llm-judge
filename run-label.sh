#!/usr/bin/env bash

set -e -x -o pipefail

if [ -z "$CLASSIFIER" ]; then CLASSIFIER=LinearSVM; fi
if [ -z "$RESTARTS" ]; then RESTARTS=1; fi
if [ -z "$OUT" ]; then OUT=out; fi

dataset="/home/ben/rubric-llm-judge/LLMJudge/data"
judgement_dir="/home/dietz/jelly-home/peanut-jupyter/exampp/data/llmjudge"

#judgements="$judgement_dir/questions-explain--questions-rate--llmjudge-passages_dev.json.gz"
#judgements="$judgement_dir/nuggets-rate--all-llmjudge-passages_dev.json.gz"
#judgements="$judgement_dir/Thomas-Sun_few-HELM-FagB_few-Sun-FagB-llmjudge-passages_dev.json.gz"
judgements="$judgement_dir/all-llmjudge-passages_dev.json.gz"

test_judgements="$judgement_dir/nuggets-explain--nuggets-rate--all-llmjudge-passages_test.json.gz"

CLASSIFIERS=(
    DecisionTree
    RandomForest
    MLP
    #LogRegCV
    #LogReg
    LinearSVM
    SVM
    HistGradientBoostedClassifier
    ExtraTrees
)

model_dir() {
    local dir="$OUT/$CLASSIFIER"
    mkdir -p "$dir"
    printf "$dir"
}

train() {
    python -m rubric_llm_judge.label \
        train \
        --qrel $dataset/llm4eval_dev_qrel_2024.txt \
        --judgements $judgements \
        --classifier $CLASSIFIER \
        --restarts $RESTARTS \
        --output $(model_dir)/model
}

predict() {
    python -m rubric_llm_judge.label \
        predict \
        --model $(model_dir)/model \
        --qrel $dataset/llm4eval_dev_qrel_2024.txt \
        --judgements $judgements \
        --output $(model_dir)/dev.jsonl.gz \
        --output-qrel $(model_dir)/dev.qrel
}

test() {
    python -m rubric_llm_judge.label \
        predict \
        -m $(model_dir)/model \
        --qrel $dataset/llm4eval_test_qrel_2024.txt \
        -j $test_judgements \
        -o $(model_dir)/test.jsonl.gz \
        --output-qrel $(model_dir)/test.qrel
}

exampp_balance() {
    zcat $1 | jq .[1].[].grades.[].self_ratings | sort | uniq -c
}

qrel_balance() {
    cut -f4 -d' ' $1 | sort | uniq -c
}

run_all() {
    for m in ${CLASSIFIERS[@]}; do
        CLASSIFIER=$m train || echo "$m failed"
    done
}

final_run() {
    RESTARTS=5
    export LABEL_PROMPTS

    printf "Running $OUT dev..."
    OUT="out-final/$NAME-$CLASSIFIER"
    judgements="$judgement_dir/all-llmjudge-passages_dev.json.gz"
    test_judgements="$judgement_dir/all-llmjudge-passages_test.json.gz"
    (git rev-parse HEAD; train; predict) |& tee $(model_dir)/log-dev

    printf "Running $OUT test..."
    (git rev-parse HEAD; test) |& tee $(model_dir)/log-test

    cp $(model_dir)/test.qrel submission/llm4eval_test_qrel_2024-all-$CLASSIFIER-$NAME.txt
}

final_runs() {
    LABEL_PROMPTS="nuggets questions direct" NAME="all" CLASSIFIER=ExtraTrees final_run
    LABEL_PROMPTS="nuggets questions direct" NAME="all" CLASSIFIER=RandomForest final_run
    LABEL_PROMPTS="nuggets" NAME="nuggets" CLASSIFIER=ExtraTrees final_run
    LABEL_PROMPTS="nuggets" NAME="nuggets" CLASSIFIER=RandomForest final_run
    LABEL_PROMPTS="questions" NAME="questions" CLASSIFIER=ExtraTrees final_run
    LABEL_PROMPTS="questions" NAME="questions" CLASSIFIER=RandomForest final_run
    LABEL_PROMPTS="direct" NAME="direct" CLASSIFIER=ExtraTrees final_run
    LABEL_PROMPTS="direct" NAME="direct" CLASSIFIER=RandomForest final_run
}

$@
# call  `./run-label.sh $function`
