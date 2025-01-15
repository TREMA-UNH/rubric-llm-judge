
from collections import defaultdict
import gzip
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from pydantic.v1 import BaseModel

from exam_pp import query_loader, data_model
from exam_pp.exam_to_qrels import QrelEntry 
from exam_pp.data_model import *
from exam_pp.question_bank_loader import *

QREL_IMPORT_PROMPT_CLASS = "QrelImport"

def read_query_file_as_tsv(query_file:Path, max_queries:Optional[int]=None) -> Dict[str,str]:
    with open(query_file, 'rt') as file:
        query_dict = dict()
        for line in itertools.islice(file.readlines(), max_queries):
            splits = line.split("\t")
            if len(splits)>=2:
                query_dict[splits[0].strip()]=splits[1].strip()
            else:
                raise RuntimeError(f"each line in query file {query_file} must contain two tab-separated columns. Offending line: \"{line}\"")

    return query_dict


def read_qrel_file(qrel_in_file:Path) ->List[QrelEntry]:
    '''Use to read qrel file'''
    with open(qrel_in_file, 'rt') as file:
        qrel_entries:List[QrelEntry] = list()
        for line in file.readlines():
            splits = line.split(" ")
            qrel_entry=None
            if len(splits)>=4:
                qrel_entry = QrelEntry(query_id=splits[0].strip(), paragraph_id=splits[2].strip(), grade=int(splits[3].strip()))
            elif len(splits)>=3: # we have a qrels file to complete.
                qrel_entry = QrelEntry(query_id=splits[0].strip(), paragraph_id=splits[2].strip(), grade=-99)
            else:
                raise RuntimeError(f"All lines in qrels file needs to contain four columns, or three for qrels to be completed. Offending line: \"{line}\"")
            
            qrel_entries.append(qrel_entry)
            # print(f"{line}\n {qrel_entry}")
    return qrel_entries

def read_query_file(query_file:Path, max_queries:Optional[int]=None) -> Dict[str,str]:
    with open(query_file, 'rt') as file:
        query_dict = dict()
        for line in itertools.islice(file.readlines(), max_queries):
            splits = line.split("\t")
            if len(splits)>=2:
                query_dict[splits[0].strip()]=splits[1].strip()
            else:
                raise RuntimeError(f"each line in query file {query_file} must contain two tab-separated columns. Offending line: \"{line}\"")

    return query_dict


def convert_paragraphs( rubric_data:List[QueryWithFullParagraphList]
                       , query_mapping:Dict[str,str]
                       )->List[QueryWithFullParagraphList]:

    for entry in rubric_data:
        old_query_id = entry.queryId
        new_query_id = query_mapping[old_query_id]
        entry.queryId = new_query_id

        for para in entry.paragraphs:

            for judgment in para.paragraph_data.judgments:
                    judgment.query = new_query_id

            for ranking in para.paragraph_data.rankings:
                    ranking.queryId = new_query_id

    return rubric_data

def main(cmdargs=None):
    """Convert LLMJudge data to inputs for EXAM/RUBRIC."""

    import argparse

    desc = f'''Change query ids in an EXAM/RUBRIC file according to a mapping\n
              The RUBRIC input will to be a *JSONL.GZ file.  Info about JSON schema with --help-schema
             '''
    help_schema=f'''The input and output file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
                \n  
                    [query_id, [FullParagraphData]] \n
                \n
                where `FullParagraphData` meets the following structure \n
                {FullParagraphData.schema_json(indent=2)}
                \n
                Create a compatible file with 
                exam_pp.data_model.writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList])
                '''
            
    parser = argparse.ArgumentParser(description="Convert TREC Qrels data to RUBRIC judgments or grades."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument("--rubric", dest='rubric_in', type=str, metavar='xxx.jsonl.gz'
                        , help='input RUBRIC file in jsonl.gz format'
                        )
    parser.add_argument("--testbank", type=str, metavar='xxx.jsonl.gz'
                        , help='RUBRIC Test bank file in jsonl.gz format'
                        )
    parser.add_argument('--use-nugget',  action='store_true',  help='Set if the test bank is comprised of nuggets (rather than questions)')


    parser.add_argument('-o', '--output', type=str, metavar='xxx.jsonl.gz'
                        , help='output path for RUBRIC file in jsonl.gz format.'
                        )

    # parser.add_argument('--query-mapping-json', type=str, metavar='PATH', help='Path to read the query mapping as JSON')
    parser.add_argument('--query-mapping-tsv', required=True, type=str, metavar='PATH', help='Path to read the query mapping as TSV')
    parser.add_argument('--doc-mapping-tsv', required=True, type=str, metavar='PATH', help='Path to read the document mapping as TSV')

    parser.add_argument('--help-schema', action='store_true', help="Additional info on required JSON.GZ input format")

    args = parser.parse_args(args = cmdargs)  

    if args.help_schema:
        print(help_schema)
        sys.exit()


    # First we load all queries
    # query_set:Dict[str,str] 
    query_mapping = read_query_file_as_tsv(query_file=args.query_mapping_tsv)
    

    if args.rubric_in is not None:
        rubric_in_file:List[QueryWithFullParagraphList] 
        rubric_in_file = parseQueryWithFullParagraphs(file_path=args.rubric_in)

        # now emit the input files for RUBRIC/EXAM
        rubric_data:List[QueryWithFullParagraphList] 
        rubric_data = convert_paragraphs(query_mapping = query_mapping
                                        , rubric_data=rubric_in_file
                                        )
    

        writeQueryWithFullParagraphs(args.output, queryWithFullParagraphList=rubric_data)


    if args.testbank is not None:
        if not args.use_nugget:
            test_bank:QueryQuestionBank
            test_banks = parseQuestionBank(file_path= args.testbank)
            for test_bank in test_banks:
                # print(test_bank)
                for testpoint in test_bank.items:
                    old_query_id = testpoint.query_id
                    new_query_id = query_mapping[old_query_id]
                    testpoint.query_id = new_query_id
            writeTestBank(args.output, queryTestBanks= test_banks)
        else:
            nugget_bank:QueryNuggetBank
            nugget_banks = parseNuggetBank(file_path= args.testbank)
            for nugget_bank in nugget_banks:
                for testpoint in nugget_bank.items:
                    old_query_id = testpoint.query_id
                    new_query_id = query_mapping[old_query_id]
                    testpoint.query_id = new_query_id
            writeTestBank(args.output, queryTestBanks= nugget_banks)



if __name__ == "__main__":
    main()
