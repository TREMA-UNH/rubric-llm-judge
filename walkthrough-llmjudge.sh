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

subset="dev"


echo -e "\n\n\nGenerate llmjudge Nuggets"
# for subset in dev test; do
# 	python -O -m exam_pp.question_generation -q data/llmjudge/llmjudge_queries_${subset}.jsonl -o data/llmjudge/llmjudge-nuggets_${subset}.jsonl.gz --use-nuggets --test-collection llmjudge_${subset} --description "A new set of generated nuggets for llmjudge ${subset}"
	echo ""
# done
	
echo -e "\n\n\Generate llmjudge Questions"

for subset in dev test; do
# 	python -O -m exam_pp.question_generation -q data/llmjudge/llmjudge_queries_${subset}.jsonl -o data/llmjudge/llmjudge-questions_${subset}.jsonl.gz --test-collection llmjudge_${subset} --description "A new set of generated questions for llmjudge ${subset} set"
	echo ""
done




echo -e "\n\n\nself-rated llmjudge nuggets"

subset="dev"

# for subset in dev test; do
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

	withrate="nuggets-rate--all-${ungraded}"
	withrateextract="nuggets-explain--${withrate}"

	##
	# Nuggets
	#


	#python -O -m exam_pp.exam_grading data/llmjudge/$ungraded -o data/llmjudge/$withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetSelfRatedPrompt --question-path data/llmjudge/llmjudge-nuggets_${subset}.jsonl.gz  --question-type question-bank --use-nuggets 

	echo -e "\n\n\ Explained llmjudge Nuggets"


	#python -O -m exam_pp.exam_grading data/llmjudge/$withrate  -o data/llmjudge/$withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class NuggetExtractionPrompt --question-path data/llmjudge/llmjudge-nuggets_${subset}.jsonl.gz  --question-type question-bank --use-nuggets 

	##
	# Questions
	#


	echo -e "\n\n\ Rated llmjudge Questions"
	# ungraded="$withrateextract"
	withrate="questions-rate--${ungraded}"
	withrateextract="questions-explain--${withrate}"


# 	python -O -m exam_pp.exam_grading data/llmjudge/$ungraded -o data/llmjudge/$withrate --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionSelfRatedUnanswerablePromptWithChoices --question-path data/llmjudge/llmjudge-questions_${subset}.jsonl.gz  --question-type question-bank 



	echo -e "\n\n\ Explained llmjudge Questions"

# 	python -O -m exam_pp.exam_grading data/llmjudge/$withrate  -o data/llmjudge/$withrateextract --model-pipeline text2text --model-name google/flan-t5-large --prompt-class QuestionCompleteConciseUnanswerablePromptWithChoices --question-path data/llmjudge/llmjudge-questions_${subset}.jsonl.gz  --question-type question-bank 


	final=$withrateextract
	
	

	##
	# Direct Grading
	#
	in="$ungraded"
	for direct in FagB Sun FagB_few HELM Sun_few Thomas; do
		echo "direct grading $direct"

		out="$direct-$in"
# 		python -O -m exam_pp.exam_grading data/llmjudge/$in  -o data/llmjudge/$out --model-pipeline text2text --model-name google/flan-t5-large --prompt-class "$direct" --question-path data/llmjudge/llmjudge-questions_${subset}.jsonl.gz  --question-type question-bank 


		in="$out"
		final="$out"
	done
	directfinal="Thomas-Sun_few-Sun-HELM-FagB_few-FagB-questions-explain--questions-rate--nuggets-explain--nuggets-rate--all-trecDL2020-qrels-runs-with-text.jsonl.gz"

	echo "Graded: $final"
	
	
	
# Merge different grades file into one that contains them all

for subset in dev test; do
# 	python -O -m exam_pp.data_model merge data/llmjudge/questions-explain--questions-rate--llmjudge-passages_${subset}.json.gz  data/llmjudge/nuggets-explain--nuggets-rate--all-llmjudge-passages_${subset}.json.gz data/llmjudge/Thomas-Sun_few-HELM-FagB_few-Sun-FagB-llmjudge-passages_${subset}.json.gz -o data/llmjudge/all-llmjudge-passages_${subset}.json.gz
done


# done

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

directfinal="Thomas-Sun_few-HELM-FagB_few-Sun-FagB-llmjudge-passages_dev.json.gz"

for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices; do
	echo $promptclass




	# inter-annotator agreement
# 	python -O -m exam_pp.exam_post_pipeline data/llmjudge/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings  --inter-annotator-out data/llmjudge/llmjudge-autograde-inter-annotator-$promptclass.tex
done


final="nuggets-explain--nuggets-rate--all-llmjudge-passages_dev.json.gz"

for promptclass in  NuggetSelfRatedPrompt; do
	echo $promptclass



	# inter-annotator agreement
	# python -O -m exam_pp.exam_post_pipeline data/llmjudge/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings  --inter-annotator-out data/llmjudge/llmjudge-autograde-inter-annotator-$promptclass.tex
done

for promptclass in  FagB Sun FagB_few HELM Sun_few Thomas; do
	echo $promptclass



	# inter-annotator agreement
	python -O -m exam_pp.exam_post_pipeline data/llmjudge/$directfinal  --question-set question-bank --prompt-class $promptclass --use-relevance-prompt  --min-relevant-judgment 2 --use-ratings  --inter-annotator-out data/llmjudge/llmjudge-autograde-inter-annotator-$promptclass.tex
done
