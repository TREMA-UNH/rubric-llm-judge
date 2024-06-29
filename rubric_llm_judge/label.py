#!/usr/bin/env python3

"""

We have a set of documents (namely, paragraphs) $p_i$ and queries $q_j$. The task is to
assign a label $l_{ij} \in \{0,1,2,3\}$ for a subset of passage/query pairs.

To address this task, we derive a set of questions $Q_{jk}$ for each query
$q_j$. We then prompt a language model to determine:

 * a self-rating $r_{ik}$ characterising whether the question $Q_{jk}$ can be
   answered with the content of document $p_i$
 * a textual string $a_{ik}$ extracted from the document $p_i$ answering $Q_{jk}$.

We want to predict a label $l_{ij}$ for each document/query pair derived from
$r_{ik}$ and $a_{ik}$. For this we learn a classifier based upon the following
features:

 * [self-ratings]: $\\vec r_{ik}$
 * [answerability]: 

"""

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from exam_pp.data_model import *

import numpy as np
import abc
from typing import List, Tuple, Iterator, Dict, NewType
from io import TextIOBase
from pathlib import Path


QuestionId = NewType("QuestionId", str)
QueryId = NewType("QueryId", str)
DocId = NewType("DocId", str)


class Classifier(abc.ABC):
    @abc.abstractmethod
    def predict(x, y) -> np.ndarray:
        pass


def read_qrel(f: Path) -> Iterator[Tuple[QueryId, DocId, int]]:
    for l in f.open('r').readlines():
        parts = l.split()
        qid = QueryId(parts[0])
        did = DocId(parts[2])
        rel = int(parts[3])
        yield (qid, did, rel)


def read_test_pairs(f: Path) -> Iterator[Tuple[QueryId, DocId]]:
    for l in f.open('r').readlines():
        parts = l.split()
        qid = QueryId(parts[0])
        did = DocId(parts[2])
        yield (qid, did)


def rating_histogram(queries: List[QueryWithFullParagraphList]
                     ) -> Dict[QuestionId, Dict[int, int]]:
    result: Dict[QuestionId, Dict[int, int]]
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for q in queries:
        para: FullParagraphData
        for para in q.paragraphs:
            for grades in para.exam_grades or []:
                for s in grades.self_ratings or []:
                    assert s.question_id
                    result[QuestionId(s.question_id)][int(s.self_rating)] += 1

    return dict(result)


def build_features(queries: List[QueryWithFullParagraphList],
                   rels: Optional[Dict[Tuple[QueryId, DocId], int]]
                   ) -> Tuple[Dict[Tuple[QueryId, DocId], int], np.ndarray, Optional[np.ndarray]]:
    hist: Dict[QuestionId, Dict[int, int]]
    hist = rating_histogram(queries)

    queryDocMap: Dict[Tuple[QueryId, DocId], int]
    queryDocMap = {}

    X: List[List[float]]
    X = []

    y: Optional[List[float]]
    y = []

    for q in queries:
        para: FullParagraphData
        for para in q.paragraphs:
            did = DocId(para.paragraph_id)
            qid = QueryId(q.queryId)
            queryDocMap[(qid, did)] = len(X)
            ratings: List[Tuple[QuestionId, int]]
            ratings = [
                (QuestionId(s.question_id), s.self_rating)
                for grades in para.exam_grades or []
                for s in grades.self_ratings or []
                if s.question_id
                ]
            feats = [
                float(rating)
                for _qstid, rating in sorted(ratings, key=lambda q: hist[q[0]][5], reverse=True)
                ]
            X.append(feats)

            if rels:
                rel = rels[(qid, did)]
                y.append(rel)

    return (queryDocMap, np.array(X), np.array(y) if rels else None)


def train(qrel: Path, judgements: Path) -> Classifier:
    rels = {
        (qid, did): rel
        for (qid, did, rel) in read_qrel(qrel)
    }

    queries: List[QueryWithFullParagraphList]
    queries = parseQueryWithFullParagraphs(judgements)

    _, X, y = build_features(queries, rels)

    #clf = MLPClassifier(hidden_layer_sizes=(5, 2))
    clf = LogisticRegression()
    clf.fit(X, y)
    print(cross_val_score(clf, X, y, cv=5))
    print(clf.score(X, y))
    return clf


def test(clf: Classifier, test_pairs: Path, judgements: Path, out: TextIOBase) -> None:
    queries: List[QueryWithFullParagraphList]
    queries = parseQueryWithFullParagraphs(judgements)
    queryDocMap, X, _y = build_features(queries, None)
    y = clf.predict(X)

    for qid, did in read_test_pairs(test_pairs):
        rel = y[queryDocMap[(qid,did)]]
        print(f'{qid} 0 {did} {rel}', file=out)


def main() -> None:
    dataset_root = Path('/home/ben/rubric-llm-judge/LLMJudge/data')
    judgements_root = Path('/home/dietz/jelly-home/peanut-jupyter/exampp/data/llmjudge/old')
    train_qrel = dataset_root / 'llm4eval_dev_qrel_2024.txt'
    train_judgements = judgements_root / 'questions-explain--questions-rate--llmjudge-passages_dev.json.gz'
    clf = train(qrel=train_qrel, judgements=train_judgements)

    test_qrel = dataset_root / 'llm4eval_test_qrel_2024.txt'
    test_judgements = judgements_root / 'questions-explain--questions-rate--llmjudge-passages_test.json.gz'
    test(clf, test_qrel, test_judgements, open('out.qrel', 'w'))


if __name__ == '__main__':
    main()
