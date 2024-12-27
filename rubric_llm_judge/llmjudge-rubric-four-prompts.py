import asyncio
from typing import *
from exam_pp import data_model
from exam_pp.data_model import *
from exam_pp.t5_qa import *
from exam_pp.exam_grading import noodle
from exam_pp.test_bank_prompts import *
from exam_pp.query_loader import json_query_loader


@dataclass
class FourPrompts(SelfRatingDirectGradingPrompt):
    criterion_name:str
    criterion_desc:str
    my_prompt_type=NuggetPrompt.my_prompt_type

    
    # def __post_init__(self):
    #     super().__post_init__()
        
    def prompt_id(self)->str:
        return self.criterion_name
    
    def prompt_type(self)->str:
        return self.my_prompt_type


    def prompt_template(self, context:str, full_paragraph:FullParagraphData)->str:
        return f'''Please assess how well the provided passage meets specific criteria in
relation to the query. Use the following scoring scale (0-{self.max_valid_rating()}) for evaluation:
0: Not relevant at all / No information provided.
1: Marginally relevant / Partially addresses the criterion.
2: Fairly relevant / Adequately addresses the criterion.
{self.max_valid_rating}: Highly relevant / Fully satisfies the criterion.

Please rate how well the given passage meets the {self.criterion_name} criterion in
relation to the query. The output should be a single score (0-{self.max_valid_rating}) indicating
{self.criterion_desc}
Query: {self.query_text}
Passage: {context}
Score:
'''
    def max_valid_rating(self)->int:
        return 3


    def gpt_json_prompt(self) ->Tuple[str,str]:
        json_instruction= r'''
Give the response in the following JSON format:
```json
{ "score": int }
```'''
        return (json_instruction, "score")





@dataclass
class FourAggregationPrompt(SelfRatingDirectGradingPrompt):
    my_prompt_type=NuggetPrompt.my_prompt_type

    
    # def __post_init__(self):
    #     super().__post_init__()
        
    def prompt_id(self)->str:
        return "aggregate"
    
    def prompt_type(self)->str:
        return self.my_prompt_type


    def prompt_template(self, context:str, full_paragraph:FullParagraphData)->str:
        grade_filter = GradeFilter()
        grade_filter.prompt_class = "FourPrompts"
        grade_filter.model_name=None
        grade_filter.is_self_rated=None
        grade_filter.min_self_rating=None
        grade_filter.question_set=None
        grade_filter.prompt_type=None

        # TODO make sure these exist!
        exactness_score = 0
        topicality_score = 0
        coverage_score = 0
        contextual_fit_score = 0


        grades = grade_filter.fetch_any(full_paragraph.exam_grades, full_paragraph.grades)
        if len(grades)>0 and grades[0].self_ratings is not None:
            grade_set = grades[0]
            # for prompt_name, g in grade_set.answers:
            for rating in grade_set.self_ratings:
                if rating.get_id == "Exactness":
                    exactness_score = rating.self_rating
                if rating.get_id == "Coverage":
                    coverage_score = rating.self_rating
                if rating.get_id == "Topicality":
                    topicality_score = rating.self_rating
                if rating.get_id == "Contextual Fit":
                    contextual_fit_score = rating.self_rating


        return f'''You are a search quality rater evaluating the relevance of passages. Given a
query and passage, you must provide a score on an integer scale of 0 to 3
with the following meanings:
3 = Perfectly relevant: The passage is dedicated to the query and contains
the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the
answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query.
Assume that you are writing an answer to the query. If the passage seems to
be related to the query but does not include any answer to the query, mark
it 1. If you would use any of the information contained in the passage in
such an answer, mark it 2. If the passage is primarily about the query, or
contains vital information about the topic, mark it 3. Otherwise, mark it 0.
Prompt:
Please rate how the given passage is relevant to the query based on the
given scores. The output must be only a score that indicates how relevant
they are.
Query: {self.query_text}
Passage: {context}
Exactness: {exactness_score}
Topicality: {topicality_score}
Coverage: {coverage_score}
Contextual Fit: {contextual_fit_score}
Score:
'''
    def max_valid_rating(self)->int:
        return 3


    def gpt_json_prompt(self) ->Tuple[str,str]:
        json_instruction= r'''
Give the response in the following JSON format:
```json
{ "score": int }
```'''
        return (json_instruction, "score")



def create_grading_prompts(query_id:str, query_text:str)->List[FourPrompts]:
    return [ FourPrompts(query_id=query_id, query_text=query_text, criterion_name="Exactness", criterion_desc="How precisely does the passage answer the query.", facet_id=None, facet_text=None)
            , FourPrompts(query_id=query_id, query_text=query_text, criterion_name="Coverage", criterion_desc="How much of the passage is dedicated to discussing the query and its related topics.", facet_id=None, facet_text=None)
            , FourPrompts(query_id=query_id, query_text=query_text, criterion_name="Topicality", criterion_desc="Is the passage about the same subject as the whole query (not only a single word of it).", facet_id=None, facet_text=None)
            , FourPrompts(query_id=query_id, query_text=query_text, criterion_name="Contextual Fit", criterion_desc="Does the passage provide relevant background or context.", facet_id=None, facet_text=None)
            ]

def create_agggregation_prompts(query_id:str, query_text:str)-> List[FourAggregationPrompt]:
    return [FourAggregationPrompt(query_id=query_id, query_text=query_text, facet_id=None, facet_text=None)]

def get_prompt_classes()-> List[str]:
    return ["FourPrompts","FourAggregationPrompt"]



async def main(cmdargs=None):
    """Score paragraphs by number of questions that are correctly answered."""

    import argparse

    desc = f'''EXAM grading, with special augmentation for Four Prompts
    
            \n
The entries of the given RUBRIC input file will be augmented with exam grades, to be written to a new file
1. Create a RUBRIC inputfile as *JSONL.GZ file. Info about JSON schema with --help-schema
2. Load RUBRIC grading questions via  --question-path $file 
3. Set prompt template via --prompt-class $class
4. Configure the LLM via --name-model $hf_model (as named on huggingface)
5. Different LLM backends and Huggingface pipelines are supported via --model-pipeline these may require additional configuration            
\n
* For vLLM you need to set the url via `export VLLM_URL=http://127.0.0.1:8000/v1`  (also works with ssh port tunnels)
\n
* For OpenAI you need to set the token via `export OPENAI_API_KEY=...`
\n
* For the other pipelines you may need to set the huggingface token via `export HF_TOKEN=...`
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
    
    parser = argparse.ArgumentParser(description="EXAM grading"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('paragraph_file', type=str, metavar='xxx.jsonl.gz'
                        , help='json file with paragraph to grade with exam questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    modelPipelineOpts = {'text2text': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  Text2TextPipeline(model_name, max_token_len=MAX_TOKEN_LEN)
                ,'question-answering': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  QaPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS)
                ,'text-generation': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  TextGenerationPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                , 'llama': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS: LlamaTextGenerationPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS)
                ,'vLLM': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  VllmPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                ,'OpenAI': lambda model_name, MAX_TOKEN_LEN, MAX_OUT_TOKENS:  OpenAIPipeline(model_name, max_token_len=MAX_TOKEN_LEN, max_output_tokens=MAX_OUT_TOKENS) 
                }

    parser.add_argument('-o', '--out-file', type=str, metavar='exam-xxx.jsonl.gz', help='Output file name where paragraphs with exam grade annotations will be written to')
    parser.add_argument('--query-path', type=str, metavar='PATH', help='Path to read queries from')
    parser.add_argument('--use-nuggets', action='store_true', help="if set, assumed --question-path contains nuggets instead of questions")
    parser.add_argument('--question-type', type=str, choices=['question-bank','direct', 'tqa','genq'], default="question-bank", metavar='PATH', help='Grading rubric file format for reading from --question-path')
    

    parser.add_argument('--model-pipeline', type=str, choices=modelPipelineOpts.keys(), required=True, metavar='MODEL', help='the huggingface pipeline used to answer questions. For example, \'sjrhuschlee/flan-t5-large-squad2\' is designed for the question-answering pipeline, where \'google/flan-t5-large\' is designed for the text2text-generation pipeline. Choices: '+", ".join(modelPipelineOpts.keys()))
    parser.add_argument('--model-name', type=str, metavar='MODEL', help='the huggingface model used to answer questions')
    parser.add_argument('--max-tokens', type=int, metavar="N", default=512, help="total number of tokens for input+output (for generative LLMs, just input)")
    parser.add_argument('--max-out-tokens', type=int, metavar="N", default=512, help="total number of tokens for generated output (not used by some HF pipelines)")


    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))

    parser.add_argument('--custom-prompt', type=str,required=False, metavar="PROMPT_TEXT"
                        , help="Custom question prompt text. Variables {question} and {context} will automatically be filled.")

    parser.add_argument('--custom-prompt-name', type=str,required=False, metavar="NAME"
                        , help="Name for the custom prompt. This name will be used instead of --prompt-class during post-processing and leaderboard evaluation")


    parser.add_argument('--max-queries', type=int, metavar="n", default=-1, help="Limit number of queries to be processed")
    parser.add_argument('--max-paragraphs', type=int, metavar="n", default=-1, help="Limit number of paragraphs to be processed")

    parser.add_argument('--restart-paragraphs-file', type=str, metavar='exam-xxx.jsonl.gz', help='Restart logic: Input file name with partial exam grade annotations that we want to copy from. Copies while queries are defined (unless --restart-from-query is set)')
    parser.add_argument('--restart-from-query', type=str, metavar='QUERY_ID', help='Restart logic: Once we encounter Query Id, we stop copying and start re-running the pipeline (Must also set --restart-paragraphs-file)')
 
    parser.add_argument('--help-schema', action='store_true', help="Additional info on required JSON.GZ input format")


    # Parse the arguments
    args = parser.parse_args(args = cmdargs) 
 
    if args.help_schema:
        print(help_schema)
        sys.exit()


    question_set:Dict[str,List[Prompt]]
    queries=json_query_loader(query_json=args.query_path)
    if args.prompt_class == "FourPromptPrompt":
        question_set = {query_id: create_grading_prompts(query_id=query_id, query_text=query_text) for query_id, query_text in queries.items() }
    elif args.prompt_class == "FourAggregationPrompt":
        question_set = {query_id: create_agggregation_prompts(query_id=query_id, query_text=query_text) for query_id, query_text in queries.items() }




    
    llmPipeline = modelPipelineOpts[args.model_pipeline](args.model_name, args.max_tokens, args.max_out_tokens)

    await noodle(
             llmPipeline=llmPipeline
           , question_set=question_set
           , paragraph_file= args.paragraph_file
           , out_file = args.out_file
           , max_queries = args.max_queries
           , max_paragraphs = args.max_paragraphs
           # Restart logic
           , restart_previous_paragraph_file=args.restart_paragraphs_file, restart_from_query=args.restart_from_query
           )

if __name__ == "__main__":
    asyncio.run(main())
    # main()
