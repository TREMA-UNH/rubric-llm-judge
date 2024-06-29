#!/bin/bash


set -eo pipefail


# don't forget to set your OPEN AI API key before question_generation!  `export OPENAI_API_KEY=...`

### External Input
#
# dl-queries.json: Convert queries to a JSON dictionary mapping query ID to query Text
#
# trecDL2020-qrels-runs-with-text.jsonl.gz:  Collect passages from system responses (ranking or generated text) for grading
#    These follow the data interchange model, providing the Query ID, paragraph_id, text. 
#    System's rank information can be stored in paragraph_data.rankings[]
#    If available, manual judgments can be stored in paragraph_data.judgment[]


### Phase 1: Test bank generation
#
# Generating an initial test bank from a set of test nuggets or exam questions.
# 
# The following files are produced:
#
# llmjudge-questions.jsonl.gz: Generated exam questions
#
# llmjudge-nuggets.jsonl.gz Generated test nuggets



echo -e "\n\n\nGenerate llmjudge Nuggets"

#python -O -m exam_pp.question_generation -q data/llmjudge/llmjudge-queries.json -o data/llmjudge/llmjudge-nuggets.jsonl.gz --use-nuggets --test-collection llmjudge --description "A new set of generated nuggets for llmjudge"

echo -e "\n\n\Generate llmjudge Questions"

# python -O -m exam_pp.question_generation -q data/llmjudge/llmjudge_queries_dev.jsonl -o data/llmjudge/llmjudge-questions_dev.jsonl.gz --test-collection llmjudge_dev --description "A new set of generated questions for llmjudge dev set"


# python -O -m exam_pp.question_generation -q data/llmjudge/llmjudge_queries_test.jsonl -o data/llmjudge/llmjudge-questions_test.jsonl.gz --test-collection llmjudge_test --description "A new set of generated questions for llmjudge test set"

echo -e "\n\n\nself-rated llmjudge nuggets"

# subset="dev"
for subset in dev test; do
	ungraded="llmjudge-passages_${subset}.json.gz"
	# also do it for the test set.


	### Phase 2: Grading
	#
	# Passages graded with nuggets and questions using the self-rating prompt
	# (for formal grades) and the answer extraction prompt for manual verification.
	# Grade information is provided in the field exam_grades.
	#
	# Grading proceeds in multiple iterations, one per prompts.
	# starting with the Collected passages. In each phase, the previous output (-o) will be used as input
	#
	# While each iteration produces a file, the final output will include data from all previous iterations.
	#
	# The final produced file is questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2020-qrels-runs-with-text.jsonl.gz



	echo "Grading ${ungraded}. Number of queries:"
	zcat data/llmjudge/$ungraded | wc -l

	# withrate="nuggets-rate--all-${ungraded}"
	# withrateextract="nuggets-explain--${withrate}"

	# grade nuggets

# 	python -O -m exam_pp.exam_grading data/llmjudge/$ungraded -o data/llmjudge/$withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetSelfRatedPrompt --question-path data/llmjudge/llmjudge-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  

	# echo -e "\n\n\ Explained llmjudge Nuggets"

# 	python -O -m exam_pp.exam_grading data/llmjudge/$withrate  -o data/llmjudge/$withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetExtractionPrompt --question-path data/llmjudge/llmjudge-nuggets.jsonl.gz  --question-type question-bank --use-nuggets  

	# grade questions

	echo -e "\n\n\ Rated llmjudge Questions"
	# ungraded="$withrateextract"
	withrate="questions-rate--${ungraded}"
	withrateextract="questions-explain--${withrate}"


# 	python -O -m exam_pp.exam_grading data/llmjudge/$ungraded -o data/llmjudge/$withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path data/llmjudge/llmjudge-questions_${subset}.jsonl.gz  --question-type question-bank 



	echo -e "\n\n\ Explained llmjudge Questions"

# 	python -O -m exam_pp.exam_grading data/llmjudge/$withrate  -o data/llmjudge/$withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConciseUnanswerablePromptWithChoices --question-path data/llmjudge/llmjudge-questions_${subset}.jsonl.gz  --question-type question-bank 


	final=$withrateextract
	# final="Thomas-Sun_few-Sun-HELM-FagB_few-FagB-questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2020-qrels-runs-with-text.jsonl.gz"

	echo "Graded: $final"
done

#### Phase 3: Manual verification and Supervision
# We demonstrate how we support humans conducting a manual supervision of the process
#
# the files produced in this phase are:
# dl-verify-grading.txt : answers to the grading propts selfrated/extraction (grouped by question/nugget)
# llmjudge-bad-question.txt : Questions/nuggets frequently covered by non-relevant passages (should be removed from the test bank)
# llmjudge-uncovered-passages.txt : Relevant passages not covered by any question/nugget (require the addition of new test nuggets/questions.
#

#python -O -m exam_pp.exam_verification --verify-grading data/llmjudge/$final  --question-path data/llmjudge/llmjudge-questions.jsonl.gz  --question-type question-bank  > data/llmjudge/llmjudge-verify-grading.txt

#python -O -m exam_pp.exam_verification --uncovered-passages data/llmjudge/$final --question-path data/llmjudge/llmjudge-questions.jsonl.gz  --question-type question-bank --min-judgment 1 --min-rating 4 > data/llmjudge/llmjudge-uncovered-passages.txt

#python -O -m exam_pp.exam_verification --bad-question data/llmjudge/$final  --question-path data/llmjudge/llmjudge-questions.jsonl.gz  --question-type question-bank --min-judgment 1 --min-rating 4  >  data/llmjudge/llmjudge-bad-question.txt



#### Phase 4: Evaluation
#
# We demonstrate both the Autograder-qrels  and Autograder-cover evaluation approaches
# Both require to select the grades to be used via --model and --prompt_class
# Here we use --model google/flan-t5-large
# and as --prompt_class either QuestionSelfRatedUnanswerablePromptWithChoices or NuggetSelfRatedPrompt.
#
# Alternatively, for test banks with exam questions that have known correct answers (e.g. TQA for CAR-y3), 
# the prompt class QuestionCompleteConcisePromptWithAnswerKey2 can be used to assess answerability.
#
# The files produced in this phase are:
#
# llmjudge-autograde-qrels-\$promptclass-minrating-4.solo.qrels:  Exported Qrel file treating passages with self-ratings >=4 
#
# llmjudge-autograde-qrels-leaderboard-\$promptclass-minrating-4.solo.tsv:  Leaderboard produced with 
#        trec_eval using the exported Qrel file
#
# llmjudge-autograde-cover-leaderboard-\$promptclass-minrating-4.solo.tsv: Leaderboads produced with Autograde Cover treating \
# 	test nuggets/questions as answered when any passage obtains a self-ratings >= 4
#
#
#
#
for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices ; do  #NuggetSelfRatedPrompt
	echo $promptclass

	for minrating in 3 4 5; do
		echo ""
		# python -O -m exam_pp.exam_evaluation data/llmjudge/$final --question-set question-bank --prompt-class $promptclass --min-self-rating $minrating --leaderboard-out data/llmjudge/llmjudge-autograde-cover-leaderboard-$promptclass-minrating-$minrating.solo.$ungraded.tsv 

		# N.B. requires TREC-llmjudge runs to be populated in data/llmjudge/llmjudgeruns
		# python -O -m exam_pp.exam_evaluation data/llmjudge/$final --question-set question-bank --prompt-class $promptclass -q data/llmjudge/llmjudge-autograde-qrels-leaderboard-$promptclass-minrating-$minrating.solo.$ungraded.qrels  --min-self-rating $minrating --qrel-leaderboard-out data/llmjudge/llmjudge-autograde-qrels-$promptclass-minrating-$minrating.solo.$ungraded.tsv --run-dir data/llmjudge/llmjudgeruns 
        
		# Since generative IR systems will not share any passages, we represent them as special run files
		#python -O -m exam_pp.exam_evaluation data/llmjudge/$final --question-set question-bank --prompt-class $promptclass -q data/llmjudge/llmjudge-autograde-qrels-leaderboard-$promptclass-minrating-$minrating.solo.$ungraded.qrels  --min-self-rating $minrating --qrel-leaderboard-out data/llmjudge/llmjudge-autograde-qrels-$promptclass-minrating-$minrating.solo.$ungraded.gen.tsv --run-dir data/llmjudge/llmjudgegen-runs 
	done
done

#### Additional Analyses
# When manual judgments or official leaderboards are available, these can be used for additional analyses and manual oversight
#
# To demonstrate the correlation with official leaderboards, requires the construction of a JSON dictionary
# official_llmjudge_leaderboard.json:  a JSON dictionary mapping method names to official ranks. (these names must match the run files and method names given in `rankings`. In the case of ties, we suggest to assign all tied systems their average rank
#
# For DL, where the judgment 1 is a non-relevant grade, the option `--min-relevant-judgment 2` must be used (default is 1)
#
# Produced outputs `llmjudge*.correlation.tsv` are leaderboards with rank correlation information (Spearman's rank correlation and Kendall's tau correlation)
#
#
# When manual relevance judgments are available Cohen's kappa inter-annotator agreement can be computed. 
# Manual judgments will be taken from the entries `paragraph_data.judgents[].relevance`
# 
# The produced output is
# llmjudge-autograde-inter-annotator-\$promptclass.tex:  LaTeX tables with graded and binarized inter-annotator statistics with Cohen's kappa agreement. ``Min-anwers'' refers to the number of correct answers obtained above a self-rating threshold by a passage. (For \dl{} –-min-relevant-judgment 2 must be set.)
# 

# python -O -m exam_pp.exam_leaderboard_analysis data/llmjudge/$final  --question-set question-bank --prompt-class  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt Thomas FagB FagB_few HELM Sun Sun_few --min-relevant-judgment 2 --trec-eval-metric ndcg_cut.10 ndcg_cut.20 map Rprec recip_rank  --use-ratings --qrel-dir=data/llmjudge --qrel-analysis-out data/llmjudge/llmjudge-autograde-qrels-leaderboard-analysis-graded.correlation.tsv --run-dir data/llmjudge/llmjudgeruns --official-leaderboard data/llmjudge/official_llmjudge_leaderboard.json --question-set-for-facets question-bank --cover-analysis-out data/llmjudge/llmjudge-autograde-cover-leaderboard-analysis-graded.correlation.tsv

final="questions-explain--questions-rate--llmjudge-passages_dev.json.gz"

for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices; do
	echo $promptclass

	for minrating in 3 4 5; do
		# autograde-qrels
		# qrel leaderboard correlation
		# N.B. requires TREC-llmjudge runs to be populated in data/llmjudge/llmjudgeruns
		#python -O -m exam_pp.exam_post_pipeline data/llmjudge/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings --min-trec-eval-level ${minrating} -q data/llmjudge/llmjudge-exam-$promptclass.qrel --qrel-leaderboard-out data/llmjudge/llmjudge-autograde-qrels-leaderboard-$promptclass-minlevel-$minrating.correlation.tsv --run-dir data/llmjudge/llmjudgeruns --official-leaderboard data/llmjudge/official_llmjudge_leaderboard.json 
	
		# autograde-cover 
		 #python -O -m exam_pp.exam_post_pipeline data/llmjudge/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings --min-self-rating ${minrating} --leaderboard-out data/llmjudge/llmjudge-autograde-cover-leaderboard-$promptclass-minlevel-$minrating.correlation.tsv  --official-leaderboard data/llmjudge/official_llmjudge_leaderboard.json
		echo ""
	done



	# inter-annotator agreement
	python -O -m exam_pp.exam_post_pipeline data/llmjudge/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings  --inter-annotator-out data/llmjudge/llmjudge-autograde-inter-annotator-$promptclass.tex
done

