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

import sklearn.tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as pl

from exam_pp.data_model import *

import enum
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
    gfilter = GradeFilter.noFilter()
    gfilter.is_self_rated = True
    for q in queries:
        para: FullParagraphData
        for para in q.paragraphs:
            for grades in para.retrieve_exam_grade_all(gfilter):
                for s in grades.self_ratings or []:
                    result[QuestionId(s.get_id())][int(s.self_rating)] += 1

    return dict(result)


def build_features(queries: List[QueryWithFullParagraphList],
                   rels: Optional[Dict[Tuple[QueryId, DocId], int]]
                   ) -> Tuple[Dict[Tuple[QueryId, DocId], int], np.ndarray, Optional[np.ndarray]]:
    hist: Dict[QuestionId, Dict[int, int]]
    hist = rating_histogram(queries)

    queryDocMap: Dict[Tuple[QueryId, DocId], int]
    queryDocMap = {}

    X: List[np.ndarray]
    X = []

    y: List[float]
    y = []

    def encode_rating(i):
        x = np.zeros((6,))
        x[i] = 1
        return x

    gfilter = GradeFilter.noFilter()
    gfilter.is_self_rated = True

    for q in queries:
        para: FullParagraphData
        for para in q.paragraphs:
            did = DocId(para.paragraph_id)
            qid = QueryId(q.queryId)
            queryDocMap[(qid, did)] = len(X)

            ratings: List[Tuple[QuestionId, int]]
            ratings = [
                (QuestionId(s.get_id()), s.self_rating)
                for grades in para.retrieve_exam_grade_all(gfilter)
                for s in grades.self_ratings or []
                #if s.get_id() is not None
                ]

            expected_ratings = 10
            if len(ratings) != expected_ratings:
                print(f'Query {qid} document {did} has {len(ratings)} ratings')

            def pad_ratings(xs):
                if len(xs) < expected_ratings:
                    return [rating for qstid,rating in xs] + [0] * (expected_ratings - len(ratings))
                else:
                    return [rating for qstid,rating in xs[:expected_ratings]]

            feats: List[np.ndarray]
            feats = []

            def rating_feature(sort_key, encoding):
                nonlocal feats
                feats += [
                        encoding(rating)
                        for rating in pad_ratings(sorted(ratings, key=sort_key, reverse=True))
                        ]

            # Integer ratings sorted by question informativeness
            rating_feature(lambda q: hist[q[0]][5], lambda x: x)

            # One-hot ratings sorted by question informativeness
            #rating_feature(lambda q: hist[q[0]][5], encode_rating)

            # Integer ratings sorted by rating
            #rating_feature(lambda q: q[1], lambda x: x)

            # One-hot ratings sorted by rating
            #rating_feature(lambda q: q[1], encode_rating)

            X.append(np.hstack(feats))

            if rels is not None:
                rel = rels[(qid, did)]
                y.append(rel)

    XX = np.array(X)
    np.savetxt('feats.csv', XX)
    return (queryDocMap, XX, np.array(y) if rels is not None else None)


class Method(enum.Enum):
    DecisionTree = enum.auto()
    MLP = enum.auto()
    LogReg = enum.auto()


def train(qrel: Path, judgements: Path, method: Method) -> Classifier:
    rels = {
        (qid, did): rel
        for (qid, did, rel) in read_qrel(qrel)
        if rel is not None
    }

    queries: List[QueryWithFullParagraphList]
    queries = parseQueryWithFullParagraphs(judgements)

    _, X, y = build_features(queries, rels)

    if method == Method.MLP:
        clf = MLPClassifier(hidden_layer_sizes=(5, 5))
    elif method == Method.DecisionTree:
        clf = sklearn.tree.DecisionTreeClassifier()
    elif method == Method.LogReg:
        clf = LogisticRegressionCV(
                cv=StratifiedKFold(5),
                #class_weight='balanced',
                penalty='l2',
                dual=False,
                #scoring='accuracy',
                scoring=make_scorer(cohen_kappa_score),
                solver='saga', multi_class='multinomial'
                #solver='liblinear', multi_class='ovr'
                )
    else:
        assert False

    clf.fit(X, y)

    print('cross-validation: ', cross_val_score(clf, X, y, cv=5))
    print('score', clf.score(X, y))
    if method == Method.DecisionTree:
        sklearn.tree.plot_tree(clf)
        pl.savefig('tree.svg')

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
        print('Kappa', cohen_kappa_score(y_truth, y_test))
        print(confusion_matrix(y_truth, y_test))

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
    p.add_argument('--classifier', '-c', type=Method, default=Method.DecisionTree, help=f'Classification method (one of {", ".join(m.name for m in Method)})')

    p = subp.add_parser('predict')
    p.set_defaults(mode='predict')
    p.add_argument('--model', '-m', type=FileType('rb'), required=True, help='Model file')
    p.add_argument('--qrel', '-q', type=Path, required=True, help='Query/documents to predict in form of a partial .qrel file')
    p.add_argument('--judgements', '-j', type=Path, required=True, help='exampp judgements file')
    p.add_argument('--output', '-o', type=Path, required=True, help='Output exampp judgements file')
    p.add_argument('--output-qrel', type=FileType('wt'), required=True, help='Output qrel file')

    args = parser.parse_args()

    if args.mode == 'train':
        clf = train(qrel=args.qrel, judgements=args.judgements, method=args.classifier)
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
