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

from sklearn.calibration import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import sklearn.tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, BaseCrossValidator
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as pl

from exam_pp.data_model import *

import enum
import pickle
import numpy as np
import abc
from typing import List, Tuple, Iterator, Dict, NewType, Callable, TypeVar
from io import TextIOBase
from pathlib import Path


QuestionId = NewType("QuestionId", str)
QueryId = NewType("QueryId", str)
DocId = NewType("DocId", str)

SELF_GRADED = GradeFilter.noFilter()
SELF_GRADED.is_self_rated = True


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
            for grades in para.retrieve_exam_grade_all(SELF_GRADED):
                for s in grades.self_ratings or []:
                    result[QuestionId(s.get_id())][int(s.self_rating)] += 1

    return dict(result)


def build_features(queries: List[QueryWithFullParagraphList],
                   rels: Optional[Dict[Tuple[QueryId, DocId], int]]
                   ) -> Tuple[Dict[Tuple[QueryId, DocId], int], np.ndarray, Optional[np.ndarray]]:
    """
    Result: (queryDocMap, feature tensor, training vector)
    Build a feature tensor for the given set of queries. Also produces a
    dictionary mapping query/document pairs to their respective row indexes in
    the feature tensor.

    Finally, if ground-truth relevances are given with `rels`, the result will
    also include a training vector.
    """
    hist: Dict[QuestionId, Dict[int, int]]
    hist = rating_histogram(queries)

    # associate (query,doc) to row in the feature matrix X
    queryDocMap: Dict[Tuple[QueryId, DocId], int]
    queryDocMap = {}

    X: List[np.ndarray]
    X = []

    y: List[float]
    y = []

    def one_hot_rating(i:int) -> np.ndarray:
        x = np.zeros((6,))
        x[i] = 1
        return x

    for q in queries:
        para: FullParagraphData
        for para in q.paragraphs:
            did = DocId(para.paragraph_id)
            qid = QueryId(q.queryId)
            queryDocMap[(qid, did)] = len(X)

            ratings: List[Tuple[QuestionId, int]]
            ratings = [
                (QuestionId(s.get_id()), s.self_rating)
                for grades in para.retrieve_exam_grade_all(SELF_GRADED)
                for s in grades.self_ratings or []
                #if s.get_id() is not None
                ]

            expected_ratings = 2
            if len(ratings) != expected_ratings:
                #print(f'Query {qid} document {did} has {len(ratings)} ratings')
                pass

            feats: List[np.ndarray]
            feats = []

            def rating_feature(sort_key: Callable[[Tuple[QuestionId, int]], Any],
                               encoding: Callable[[int], np.ndarray]):
                """
                Introduce a set of features based on the document's ratings
                represented as a vector using the given function, sorting the
                ratings using the given sort key.
                """
                nonlocal feats, ratings
                sorted_ratings = sorted(ratings, key=sort_key, reverse=True)
                if len(sorted_ratings) < expected_ratings:
                    padded_ratings = [rating for qstid,rating in sorted_ratings] + [0] * (expected_ratings - len(ratings))
                else:
                    padded_ratings = [rating for qstid,rating in sorted_ratings[:expected_ratings]]

                feats += [ encoding(rating) for rating in padded_ratings ]

            # Integer ratings sorted by question informativeness
            # rating_feature(sort_key=lambda q: hist[q[0]][5], encoding=lambda x: x)

            # One-hot ratings sorted by question informativeness
            # rating_feature(sort_key=lambda q: hist[q[0]][5], encoding=one_hot_rating)

            # Integer ratings sorted by rating
            # rating_feature(sort_key=lambda q: q[1], encoding=lambda x: np.array([x]))

            # One-hot ratings sorted by rating
            rating_feature(sort_key=lambda q: q[1], encoding=one_hot_rating)

            # One-hot maximum rating
            #feats += [ one_hot_rating(max(rating for qstid, rating in ratings)) ]

            # Integer maximum rating
            #feats += [ [max(rating for qstid, rating in ratings)] ]


            X.append(np.hstack(feats))

            if rels is not None:
                rel = rels[(qid, did)]
                y.append(rel)

    XX = np.array(X)
    # todo: z-score normalization

    np.savetxt('feats.csv', XX)
    return (queryDocMap, XX, np.array(y) if rels is not None else None)


class Method(enum.Enum):
    DecisionTree = enum.auto()
    MLP = enum.auto()
    LogRegCV = enum.auto()
    LogReg = enum.auto()
    SVM = enum.auto()
    LinearSVM = enum.auto()
    RandomForest = enum.auto()
    HistGradientBoostedClassifier = enum.auto()


def train(qrel: Path, queries: List[QueryWithFullParagraphList], method: Method) -> Classifier:
    rels = {
        (qid, did): rel
        for (qid, did, rel) in read_qrel(qrel)
        if rel is not None
    }

    _, X, y = build_features(queries, rels)

    if method == Method.MLP:
        clf = MLPClassifier(hidden_layer_sizes=(5,1),
                             activation='tanh', 
                             learning_rate='constant',
                               solver='adam'
                               )
    elif method == Method.DecisionTree:
        clf = sklearn.tree.DecisionTreeClassifier()
    elif method == Method.LogRegCV:
        clf = LogisticRegressionCV(
                cv=StratifiedKFold(5, shuffle=False),
                # cv=KFold(n_splits=2, shuffle=False),
                class_weight='balanced',
                max_iter=10000,
                penalty='l2',
                dual=False,
                fit_intercept=True,
                #scoring='accuracy',
                scoring=make_scorer(cohen_kappa_score),
                #solver='saga', multi_class='multinomial'
                # solver='liblinear', multi_class='ovr'
                )
    elif method == Method.LogReg:
        clf = LogisticRegression(
                #class_weight='balanced',
                class_weight="balanced",
                max_iter=10000,
                penalty='l2',
                dual=False,
                #solver='saga', multi_class='multinomial'
                # solver='liblinear',  multi_class='ovr'
                fit_intercept=True
                )
    elif method == Method.SVM:
        # implement grid search: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
        clf = SVC(
                decision_function_shape='ovo',
                class_weight="balanced"
                )
    elif method == Method.LinearSVM:
        clf = LinearSVC(
                class_weight="balanced",
                dual=False
              )
    elif method == Method.RandomForest:
        clf = RandomForestClassifier(
                class_weight="balanced",
                max_depth=2,
                n_estimators=5,
                min_samples_split=10
              )
    elif method == Method.HistGradientBoostedClassifier:
        clf = HistGradientBoostingClassifier(
                class_weight="balanced",
                max_depth=2,
                scoring=make_scorer(cohen_kappa_score)
              )
    else:
        assert False

    clf.fit(X, y)

    print('cross-validation: ', cross_val_score(clf, X, y, cv=10))
    print('training score', clf.score(X, y)) # training loss achieved
    if method == Method.DecisionTree:
        print('tree depth: ', clf.get_depth())
        print('parameters: ', len(clf.get_params()))
        sklearn.tree.plot_tree(clf)
        pl.savefig('tree.svg')
    elif method == Method.LogReg:
        print('parameters: ', clf.intercept_, clf.coef_)
        pass
    elif method == Method.LogRegCV:
        print('parameters: ', clf.Cs_, clf.C_)

    # todo custom loss function optimizer
        #     def my_custom_loss_func(y_true,y_pred):
        #    diff3=max((abs(y_true-y_pred))*y_true)
        #    return diff3

        # score=make_scorer(my_custom_loss_func,greater_ is_better=False)
        # clf=RandomForestClassifier()
        # mnn= GridSearchCV(clf,score)
        # knn = mnn.fit(feam,labm)  



    return clf


def predict(clf: Classifier,
            test_pairs: List[Tuple[QueryId, DocId]],
            queries: List[QueryWithFullParagraphList],
            truth: Optional[Dict[Tuple[QueryId, DocId], int]],
            out_qrel: Optional[TextIOBase],
            out_exampp: Optional[Path]
            ) -> None:
    """
    clf: classifier
    test_qrel: path to (partial) qrel containing test query/document pairs
    truth: optional mapping from query/document pairs to ground-truth label.
    """

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
                        correctAnswered=rel >= 1,
                        answer='',
                        llm='flan-t5-large',
                        llm_options={},
                        self_ratings=rel,
                        prompt_type='DirectGrading',
                        prompt_info={
                            'prompt_class': 'exampp-llmjudge-labelling',
                            'is_self_rated': True,
                        },
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
    p.add_argument('--classifier', '-c', type=Method.__getitem__, default=Method.DecisionTree, help=f'Classification method (one of {", ".join(m.name for m in Method)})')

    p = subp.add_parser('predict')
    p.set_defaults(mode='predict')
    p.add_argument('--model', '-m', type=FileType('rb'), required=True, help='Model file')
    p.add_argument('--qrel', '-q', type=Path, required=True, help='Query/documents to predict in form of a partial .qrel file')
    p.add_argument('--judgements', '-j', type=Path, required=True, help='exampp judgements file')
    p.add_argument('--output', '-o', type=Path, required=True, help='Output exampp judgements file')
    p.add_argument('--output-qrel', type=FileType('wt'), required=True, help='Output qrel file')

    args = parser.parse_args()

    if args.mode == 'train':
        queries: List[QueryWithFullParagraphList]
        queries = parseQueryWithFullParagraphs(args.judgements)

        train_queries, test_queries = train_test_split(queries, test_size=0.5)

        clf = train(qrel=args.qrel,
                    queries=train_queries,
                    method=args.classifier)
        pickle.dump(clf, args.output)

        # Compute validation error
        test_pairs = [(QueryId(q.queryId), DocId(para.paragraph_id))
                      for q in test_queries
                      for para in q.paragraphs
                      ]
        truth = {(qid,did): rel
                 for qid, did, rel in read_qrel(args.qrel)
                 if rel is not None }

        print('Validation set prediction')
        predict(clf=clf,
                test_pairs=test_pairs,
                truth=truth,
                queries=queries,
                out_qrel=None,
                out_exampp=None)

    elif args.mode == 'predict':
        clf = pickle.load(args.model)
        qrel = list(read_qrel(args.qrel))
        queries = parseQueryWithFullParagraphs(args.judgements)
        test_pairs = [ (qid,did) for qid, did, _ in qrel ]
        truth = {(qid,did): rel
                 for qid, did, rel in qrel
                 if rel is not None }
        predict(clf=clf,
                test_pairs=test_pairs,
                truth=truth if truth != {} else None,
                queries=queries,
                out_qrel=args.output_qrel,
                out_exampp=args.output)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
