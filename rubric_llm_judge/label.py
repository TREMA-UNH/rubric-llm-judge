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
from sklearn.metrics import cohen_kappa_score

from exam_pp.data_model import *

import pickle
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


def read_qrel(f: Path) -> Iterator[Tuple[QueryId, DocId, Optional[int]]]:
    for l in f.open('r').readlines():
        parts = l.split()
        qid = QueryId(parts[0])
        did = DocId(parts[2])
        rel = int(parts[3]) if len(parts) == 4 else None
        yield (qid, did, rel)


def rating_histogram(queries: List[QueryWithFullParagraphList]
                     ) -> Dict[QuestionId, Dict[int, int]]:
    """
    Histogram the number of times each question had the given self-rating.
    result[question][rating] = count.
    """
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
                if s.question_id is not None
                ]
            feats = [
                float(rating)
                for _qstid, rating in sorted(ratings, key=lambda q: hist[q[0]][5], reverse=True)
                ]
            assert len(feats) == 10
            X.append(feats)

            if rels is not None:
                rel = rels[(qid, did)]
                y.append(rel)

    return (queryDocMap, np.array(X), np.array(y) if rels is not None else None)


def train(qrel: Path, judgements: Path) -> Classifier:
    rels = {
        (qid, did): rel
        for (qid, did, rel) in read_qrel(qrel)
        if rel is not None
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


def predict(clf: Classifier,
            test_pairs: List[Tuple[QueryId, DocId]],
            judgements: Path,
            truth: Optional[Dict[Tuple[QueryId, DocId], int]],
            out_qrel: Optional[TextIOBase],
            out_exampp: Optional[Path]
            ) -> None:
    """
    clf: classifier
    test_qrel: path to (partial) qrel containing test query/document pairs
    truth: 
    """

    queries: List[QueryWithFullParagraphList]
    queries = parseQueryWithFullParagraphs(judgements)
    queryDocMap, X, _y = build_features(queries, None)
    y = clf.predict(X)

    if truth is not None:
        y_truth = list(truth.values())
        y_test = [ y[queryDocMap[(qid, did)]] for qid,did in truth.keys() ]
        print(cohen_kappa_score(y_truth, y_test))

    if out_qrel is not None:
        for qid, did in test_pairs:
            rel = y[queryDocMap[(qid,did)]]
            print(f'{qid} 0 {did} {rel}', file=out_qrel)

    if out_exampp is not None:
        for q in queries:
            para: FullParagraphData
            for para in q.paragraphs:
                did = DocId(para.paragraph_id)
                qid = QueryId(q.queryId)
                rel = y[queryDocMap[(qid,did)]]
                para.grades = [
                    Grades(
                        correctAnswered=rel > 3,
                        answer='',
                        llm='flan-t5-large',
                        llm_options={},
                        prompt_info={},
                        self_ratings=rel,
                        prompt_type='exampp-logistic-regression-labelling',
                    )
                ]

        writeQueryWithFullParagraphs(out_exampp, queries)


def main() -> None:
    import argparse
    from argparse import FileType
    parser = argparse.ArgumentParser()
    parser.set_defaults(mode=None)
    subp = parser.add_subparsers()

    p = subp.add_parser('train')
    p.set_defaults(mode='train')
    p.add_argument('--qrel', '-q', type=Path, required=True, help='Query relevance file')
    p.add_argument('--judgements', '-j', type=Path, required=True, help='exampp judgements file')
    p.add_argument('--output', '-o', type=FileType('wb'), required=True, help='Output model file')

    p = subp.add_parser('predict')
    p.set_defaults(mode='predict')
    p.add_argument('--model', '-m', type=FileType('rb'), required=True, help='Model file')
    p.add_argument('--qrel', '-q', type=Path, required=True, help='Query/documents to predict in form of a partial .qrel file')
    p.add_argument('--judgements', '-j', type=Path, required=True, help='exampp judgements file')
    p.add_argument('--output', '-o', type=Path, required=True, help='Output exampp judgements file')
    p.add_argument('--output-qrel', type=FileType('wt'), required=True, help='Output qrel file')

    args = parser.parse_args()

    if args.mode == 'train':
        clf = train(qrel=args.qrel, judgements=args.judgements)
        pickle.dump(clf, args.output)
    elif args.mode == 'predict':
        clf = pickle.load(args.model)
        qrel = list(read_qrel(args.qrel))
        test_pairs = [ (qid,did) for qid, did, _ in qrel ]
        truth = {(qid,did): rel
                 for qid, did, rel in qrel
                 if rel is not None }
        predict(clf=clf,
                test_pairs=test_pairs,
                truth=truth if truth != {} else None,
                judgements=args.judgements,
                out_qrel=args.output_qrel,
                out_exampp=args.output)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
