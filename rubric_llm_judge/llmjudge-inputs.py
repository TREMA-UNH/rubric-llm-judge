from collections import defaultdict
import gzip
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

from exam_pp import query_loader, data_model
from exam_pp.exam_to_qrels import QrelEntry 
from exam_pp.data_model import *


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

def read_llmjudge_query_file(query_file:Path, max_queries:Optional[int]=None) -> Dict[str,str]:
    with open(query_file, 'rt') as file:
        query_dict = dict()
        for line in itertools.islice(file.readlines(), max_queries):
            splits = line.split("\t")
            if len(splits)>=2:
                query_dict[splits[0].strip()]=splits[1].strip()
            else:
                raise RuntimeError(f"each line in query file {query_file} must contain two tab-separated columns. Offending line: \"{line}\"")

    return query_dict



class LLMJudgeDocument(BaseModel):
    docid:str
    doc:str

def parseLLMJudgeDocument(line:str) -> LLMJudgeDocument:
    # Parse the JSON content of the line
    # print(line)
    return LLMJudgeDocument.parse_raw(line)

def loadLLMJudgeCorpus(file_path:Path, max_paragraphs:Optional[int]) -> List[LLMJudgeDocument]:
    '''Load LLMJudge document corpus'''

    result:List[LLMJudgeDocument] = list()
    try: 
        with open(file_path, 'rt', encoding='utf-8') as file:
            # return [parseQueryWithFullParagraphList(line) for line in file]
            for line in itertools.islice(file.readlines(), max_paragraphs):
                result.append(parseLLMJudgeDocument(line))
    except  EOFError as e:
        print(f"Warning: File EOFError on {file_path}. Use truncated data....\nFull Error:\n{e} \n offending line: \n {line}")
    return result

def write_query_file(file_path:Path, queries:Dict[str,str])->None:
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        json.dump(obj=queries,fp=file)


def main(cmdargs=None):
    """Convert LLMJudge data to inputs for EXAM/RUBRIC."""

    import argparse

    desc = f'''Convert LLMJudge data to inputs for EXAM/RUBRIC. \n
              The RUBRIC input will to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {data_model.FullParagraphData.schema_json(indent=2)}
             '''
    
    parser = argparse.ArgumentParser(description="Convert LLMJudge data to RUBRIC inputs."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('llmjudge_corpus', type=str, metavar='xxx.jsonl.gz'
                        , help='input json file with corpus from the LLMJudge collection'
                        )


    parser.add_argument('-p', '--paragraph-file', type=str, metavar='xxx.jsonl.gz'
                        , help='output json file with paragraph to grade with exam questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('--query-path', type=str, metavar='PATH', help='Path to read LLMJudge queries')
    parser.add_argument('--input-qrel-path', type=str, metavar='PATH', help='Path to read LLMJudge qrels (to be completed)')
    parser.add_argument('--query-out', type=str, metavar='PATH', help='Path to write queries for RUBRIC/EXAM to')

    

    parser.add_argument('--max-queries', type=int, metavar='INT', default=None, help='limit the number of queries that will be processed (for debugging)')
    parser.add_argument('--max-paragraphs', type=int, metavar='INT', default=None, help='limit the number of paragraphs that will be processed (for debugging)')


    # Parse the arguments
    args = parser.parse_args(args = cmdargs)  

    # First we load all queries
    query_set:Dict[str,str] 
    query_set = read_llmjudge_query_file(query_file=args.query_path, max_queries = args.max_queries)

    # Fetch the qrels file  ... and munge
    input_qrels = read_llmjudge_qrel_file(qrel_in_file=args.input_qrel_path)
    qrel_query_ids = {q.query_id  for q in input_qrels}
    input_qrels_by_qid:Dict[str,List[QrelEntry]] = defaultdict(list)
    for qrel_entry in input_qrels:
        input_qrels_by_qid[qrel_entry.query_id].append(qrel_entry)
    

    # filter query set to the queries in the qrels file only
    query_set = {qid:qstr  for qid,qstr in query_set.items() if qid in qrel_query_ids}
    write_query_file(file_path=args.query_out, queries=query_set)

    # print(f"query_set = {query_set}")

    # load the paragraph data
    corpus = loadLLMJudgeCorpus(file_path = args.llmjudge_corpus, max_paragraphs = args.max_paragraphs)
    corpus_by_para_id = {para.docid: para  for para in corpus}

    # print(f"corpus = {corpus}")
    




    # now emit the input files for RUBRIC/EXAM
    rubric_data:List[QueryWithFullParagraphList] = list()

    for query_id, query_str in query_set.items():
        paragraphs:List[FullParagraphData] = list()
        for qrels_entry in input_qrels_by_qid[query_id]:
            judgments = []
            if qrel_entry.grade != -99:
                judgment = Judgment(paragraphId= qrels_entry.paragraph_id, query=query_id, relevance=qrel_entry.grade, titleQuery=query_str)
                judgments = [judgment]
            para = corpus_by_para_id[qrels_entry.paragraph_id]
            if para is None:
                raise RuntimeError(f"docid {qrels_entry.paragraph_id} not found in LLMJudge corpus")
            rubric_paragraph= FullParagraphData( paragraph_id= qrels_entry.paragraph_id
                                               , text= para.doc
                                               , paragraph=""
                                               , paragraph_data=ParagraphData(judgments=judgments, rankings=list())
                                               , exam_grades=None
                                               , grades=None
                                               )
            paragraphs.append(rubric_paragraph)
            # print(f"{rubric_paragraph}")


        rubric_data.append(QueryWithFullParagraphList(queryId=query_id, paragraphs= paragraphs))


    writeQueryWithFullParagraphs(args.paragraph_file, queryWithFullParagraphList=rubric_data)



if __name__ == "__main__":
    main()
