from collections import defaultdict
import heapq
from pathlib import Path
from typing import Dict, List, Tuple

from exam_pp import data_model
from exam_pp.exam_to_qrels import QrelEntry, write_qrel_file 
from exam_pp.data_model import FullParagraphData, GradeFilter, parseQueryWithFullParagraphs, ExamGrades, Grades, SelfRating
import sklearn

def read_llmjudge_qrel_file(qrel_in_file:Path) ->List[QrelEntry]:
    '''Use to read qrel file'''
    with open(qrel_in_file, 'rt') as file:
        qrel_entries:List[QrelEntry] = list()
        for line in file.readlines():
            splits = line.split(" ")
            if len(splits)>=4:
                qrel_entries.append(QrelEntry(query_id=splits[0].strip(), paragraph_id=splits[2].strip(), grade=int(splits[3].strip())))
            elif len(splits)>=3: # we have a qrels file to complete.
                qrel_entries.append(QrelEntry(query_id=splits[0].strip(), paragraph_id=splits[2].strip(), grade=-99))
            else:
                raise RuntimeError(f"All lines in qrels file needs to contain four columns, or three for qrels to be completed. Offending line: \"{line}\"")
    return qrel_entries





def k_best_rating(self_ratings:List[SelfRating], min_answers:int)->int:
    # if exam_grade.self_ratings is None:
    #     raise RuntimeError(f"exam_grades.self_ratings is None. Can't derive relevance label.")
    ratings = (rate.self_rating for rate in self_ratings)
    best_rating:int
    if min_answers > 1:
        best_rating = min( heapq.nlargest(min_answers, ratings ))
    else:
        best_rating = max(ratings)
    return best_rating


def predict_labels_from_exam_ratings(para:FullParagraphData, grade_filter:GradeFilter, min_answers:int=1)->int:
    for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
        if exam_grade.self_ratings is None:
            raise RuntimeError(f"paragraphId: {para.paragraph_id}:  Exam grades have no self ratings!  {exam_grade}")

        return k_best_rating(self_ratings=exam_grade.self_ratings, min_answers=min_answers)
            
    return 0

def predict_labels_from_grade_rating(para:FullParagraphData, grade_filter:GradeFilter)->int:
    if para.grades is None:
        raise RuntimeError(f"paragraph \"{para.paragraph_id}\"does not have annotated `grades`. Data: {para}")

    grade: Grades
    for grade in para.retrieve_grade_any(grade_filter=grade_filter): # there will be 1 or 0
        if grade.self_ratings is not None:
            return grade.self_ratings
    raise RuntimeError(f"paragraph \"{para.paragraph_id}\"does not have self_ratings in \"grades\". Data: {para}")


def extract_heuristic_question_relevance_label(rubric_paragraph:FullParagraphData, grade_filter:GradeFilter, min_answers:int)-> int:
    exam_grades = rubric_paragraph.retrieve_exam_grade_any(grade_filter)

    if len(exam_grades)<1:
        raise RuntimeError(f"Cannot obtain exam_grades for grade filter {grade_filter} in rubric paragraph {rubric_paragraph}")
    
    exam_grade=exam_grades[0]
    # best_grade = max([r.self_rating for r in exam_grades[0].self_ratings_as_iterable()])
    best_grade = k_best_rating(self_ratings=exam_grade.self_ratings, min_answers=min_answers)
    if best_grade >= 5:
        return 3
    if best_grade >= 4:
        return 1
    if best_grade >= 1:
        return 0
    else:
        return 0

def extract_heuristic_nugget_relevance_label(rubric_paragraph:FullParagraphData, grade_filter:GradeFilter, min_answers:int)-> int:
    exam_grades = rubric_paragraph.retrieve_exam_grade_any(grade_filter)

    if len(exam_grades)<1:
        raise RuntimeError(f"Cannot obtain exam_grades for grade filter {grade_filter} in rubric paragraph {rubric_paragraph}")
    
    exam_grade=exam_grades[0]
    # best_grade = max([r.self_rating for r in exam_grades[0].self_ratings_as_iterable()])
    best_grade = k_best_rating(self_ratings=exam_grade.self_ratings, min_answers=min_answers)
    if best_grade >= 5:
        return 3
    if best_grade >= 4:
        return 2
    if best_grade >= 3:
        return 1
    else:
        return 0


def evaluate(train_labels:List[int], predict_labels:List[int]):
    kappa = sklearn.metrics.cohen_kappa_score(train_labels, predict_labels)
    print(f"multiclass kappa={kappa}")

def main(cmdargs=None):
    """Convert EXAM/RUBRIC grades into LLMJudge relevance label predictions."""

    import argparse

    desc = f'''Convert EXAM/RUBRIC data to predictions of relevance labels for LLMJudge. \n
              The RUBRIC grades will to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {data_model.FullParagraphData.schema_json(indent=2)}
             '''
    
    parser = argparse.ArgumentParser(description="Convert RUBRIC grades to LLMJudge relevance labels."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )

    parser.add_argument('grade_file', type=str, metavar='xxx.jsonl.gz'
                        , help='RUBRIC grades json file.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('--input-qrel-path', type=str, metavar='PATH', help='Path to read LLMJudge qrels (to be completed)')
    parser.add_argument('--output-qrel-path', type=str, metavar='PATH', help='Path to write completed LLMJudge qrels to')
    parser.add_argument('--min-answers', type=int, metavar='K', help='Considers the K\'th best self-rating. (For K=1, uses the best grade)')

    

    # parser.add_argument('--max-queries', type=int, metavar='INT', default=None, help='limit the number of queries that will be processed (for debugging)')
    # parser.add_argument('--max-paragraphs', type=int, metavar='INT', default=None, help='limit the number of paragraphs that will be processed (for debugging)')


    # Parse the arguments
    args = parser.parse_args(args = cmdargs)  

    # Fetch the qrels file to complete
    input_qrels = read_llmjudge_qrel_file(qrel_in_file=args.input_qrel_path)
    qrel_query_ids = {q.query_id  for q in input_qrels}
    input_qrels_by_qid:Dict[str,List[QrelEntry]] = defaultdict(list)
    for qrel_entry in input_qrels:
        input_qrels_by_qid[qrel_entry.query_id].append(qrel_entry)
    

    # filter query set to the queries in the qrels file only
    # query_set = {qid:qstr  for qid,qstr in query_set.items() if qid in qrel_query_ids}

    # print(f"query_set = {query_set}")

    # Open RUBRIC grades
    rubric_data = parseQueryWithFullParagraphs(file_path=args.grade_file)
    rubric_lookup:Dict[Tuple[str,str],FullParagraphData] = dict()
    for rubric_entry in rubric_data:
        query_id = rubric_entry.queryId
        for para in rubric_entry.paragraphs:
            rubric_lookup[(query_id, para.paragraph_id)] = para



    # grade filter
    grade_filter = GradeFilter.noFilter()
    grade_filter.is_self_rated=True

    # now emit the input files for RUBRIC/EXAM
    completed_qrels:List[QrelEntry] = list()


    train_labels = list()
    predict_labels = list()

    for qrel_entry in input_qrels:
        qrel_entry.query_id
        qrel_entry.paragraph_id
        rubric_paragraph = rubric_lookup.get((qrel_entry.query_id, qrel_entry.paragraph_id))
        if rubric_paragraph is None:
            raise RuntimeError(f"Cannot find paragraph for qrel entry {qrel_entry} in rubric_lookup (loaded from {args.grade_file})")

 
        relevance_label = extract_heuristic_question_relevance_label(rubric_paragraph, grade_filter=grade_filter, min_answers=args.min_answers)

        # keep stats for evaluation
        train_labels.append(qrel_entry.grade)
        predict_labels.append(relevance_label)

        completed_qrels.append(QrelEntry(query_id = qrel_entry.query_id, paragraph_id = qrel_entry.paragraph_id, grade = relevance_label))



    write_qrel_file(qrel_out_file=args.output_qrel_path, qrel_entries=completed_qrels)

    evaluate(train_labels, predict_labels)


if __name__ == "__main__":
    main()
