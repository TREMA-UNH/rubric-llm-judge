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
        --output $(model_dir)/model \
        > $(model_dir)/train.stdout.log \
        2> $(model_dir)/train.stderr.log
}

predict() {
    python -m rubric_llm_judge.label \
        predict \
        --model $(model_dir)/model \
        --qrel $dataset/llm4eval_dev_qrel_2024.txt \
        --judgements $judgements \
        --output $(model_dir)/dev.jsonl.gz \
        --output-qrel $(model_dir)/dev.qrel \
        > $(model_dir)/predict.stdout.log \
        2> $(model_dir)/predict.stderr.log
}

run_test() {
    python -m rubric_llm_judge.label \
        predict \
        -m $(model_dir)/model \
        --qrel $dataset/llm4eval_test_qrel_2024.txt \
        -j $test_judgements \
        -o $(model_dir)/test.jsonl.gz \
        --output-qrel $(model_dir)/test.qrel \
        > $(model_dir)/test.stdout.log \
        2> $(model_dir)/test.stderr.log
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
    git rev-parse HEAD > $(model_dir)/commit
    train
    predict

    printf "Running $OUT test..."
    run_test

    cp $(model_dir)/dev.qrel submission/llm4eval_dev_qrel_2024-all-$CLASSIFIER-$NAME.txt
    cp $(model_dir)/test.qrel submission/llm4eval_test_qrel_2024-all-$CLASSIFIER-$NAME.txt

    sed -e "s/^/$NAME /" $(model_dir)/train.stdout.log >> out-final/summary
}

final_runs() {
    rm -f out-final/summary
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
