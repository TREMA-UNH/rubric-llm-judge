from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from exam_pp import data_model
from exam_pp.exam_to_qrels import QrelEntry, write_qrel_file 
from exam_pp.data_model import FullParagraphData, GradeFilter, parseQueryWithFullParagraphs

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

def extract_relevance_label(rubric_paragraph:FullParagraphData, grade_filter:GradeFilter)-> int:
    exam_grades = rubric_paragraph.retrieve_exam_grade_any(grade_filter)

    if len(exam_grades)<1:
        raise RuntimeError(f"Cannot obtain exam_grades for grade filter {grade_filter} in rubric paragraph {rubric_paragraph}")
    
    best_grade = max([r.self_rating for r in exam_grades[0].self_ratings_as_iterable()])
    if best_grade >= 5:
        return 3
    if best_grade >= 4:
        return 2
    if best_grade >= 1:
        return 1
    else:
        return 0


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

    for qrel_entry in input_qrels:
        qrel_entry.query_id
        qrel_entry.paragraph_id
        rubric_paragraph = rubric_lookup.get((qrel_entry.query_id, qrel_entry.paragraph_id))
        if rubric_paragraph is None:
            raise RuntimeError(f"Cannot find paragraph for qrel entry {qrel_entry} in rubric_lookup (loaded from {args.grade_file})")

        relevance_label = extract_relevance_label(rubric_paragraph, grade_filter=grade_filter)

        completed_qrels.append(QrelEntry(query_id = qrel_entry.query_id, paragraph_id = qrel_entry.paragraph_id, grade = relevance_label))



    write_qrel_file(qrel_out_file=args.output_qrel_path, qrel_entries=completed_qrels)



if __name__ == "__main__":
    main()
