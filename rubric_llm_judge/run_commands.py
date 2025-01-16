import os
import subprocess
import time

# Input directory



# input_dir = "/home/nf1104/work/data/ablation_results/llama70b/dl23"
# output_dir = "/home/nf1104/work/rubric-llm-judge/postanalysis/ablation_results/llama70b/dl23"
# prompt_class = "FourAggregationPrompt"
# min_self_rating = "--min-self-rating 2"
# min_judgement = "--min-relevant-judgment 2"
# prompt_type = "--prompt-type nugget"



# input_dir = "/home/nf1104/work/data/rubric_format_inputs/flant5large"
# output_dir = "/home/nf1104/work/rubric-llm-judge/postanalysis/flant5large"

# input_dir = "/home/nf1104/work/data/rubric_format_inputs/laura/llama70b"

input_dir = "/home/nf1104/work/data/rubric_format_inputs/laura/llama70b/dl23"
output_dir = "/home/nf1104/work/data/postanalysis/laura/llama70b/dl23"
prompt_class = "FagB"
min_self_rating = ""
min_judgement = ""
prompt_type = "--prompt-type direct_grading"
# Leaderboard and run_dir paths
leaderboards = {
    "test": "/home/nf1104/work/data/leaderboards/official_leaderboard_trecdl2023.json",
    "dl2019": "/home/nf1104/work/data/leaderboards/official_leaderboard_trecdl2019.json",
    "dl2020": "/home/nf1104/work/data/leaderboards/official_leaderboard_trecdl2020.json",
}

run_dirs = {
    "test": "/home/nf1104/work/data/runs/runs_trecdl2023",
    "dl2019": "/home/nf1104/work/data/runs/runs_trecdl2019",
    "dl2020": "/home/nf1104/work/data/runs/runs_trecdl2020",
}

# Function to determine the leaderboard and run_dir based on the file name
def get_leaderboard_and_run_dir(file_name):
    
    if "dl2019" in file_name or "dl19" in file_name:
        return leaderboards["dl2019"], run_dirs["dl2019"]
    elif "dl2020" in file_name or "dl20" in file_name or "DL2020" in file_name:
        return leaderboards["dl2020"], run_dirs["dl2020"]
    else:
        # if file_name.startswith("test"):
        return leaderboards["test"], run_dirs["test"]
    return None, None  # Default case (if needed)

# Iterate over all `.jsonl.gz` files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".jsonl.gz") and file_name=="oneline.jsonl.gz":
        # Extract base name (without extension)
        base_name = file_name.replace(".jsonl.gz", "")

        # Full path to the input file
        input_file = os.path.join(input_dir, file_name)

        # Get the appropriate leaderboard path and run directory
        official_leaderboard, run_dir = get_leaderboard_and_run_dir(file_name)
        if not official_leaderboard or not run_dir:
            print(f"No leaderboard or run_dir match for {file_name}. Skipping...")
            continue

        # Define output files based on base_name
        qrel_file = os.path.join(output_dir, f"{base_name}.qrel")
        leaderboard_file = os.path.join(output_dir, f"{base_name}_ndcg_cut.10.tsv")
        qrel_leaderboard_json = os.path.join(output_dir, f"exampp-{base_name}_ndcg_cut.10.json")
        qrel_analysis_out = os.path.join(output_dir, f"{base_name}_analysis.tsv")
        correlation_out = os.path.join(output_dir, f"exampp_{base_name}_interannotator.tex")

        # Define the three commands
        commands = [
            f"python -O -m exam_pp.exam_evaluation {input_file} --prompt-class {prompt_class} -q {qrel_file}",
            
            # f"python -O -m exam_pp.exam_evaluation {input_file} --prompt-class {prompt_class} -q {qrel_file} --qrel-leaderboard-out {leaderboard_file} --trec-eval-metric ndcg_cut.10 -r {min_self_rating} --run-dir {run_dir}",
            # f"python -O -m exam_pp.exam_post_pipeline {input_file} --prompt-class {prompt_class} --qrel-leaderboard-out {qrel_leaderboard_json} --trec-eval-metric Rndcg ndcg_cut.10 map recip_rank -r --run-dir {run_dir} -q {qrel_file} --dont-check-prompt-class {prompt_type} --official-leaderboard {official_leaderboard} {min_judgement} {min_self_rating} --qrel-analysis-out {qrel_analysis_out}",
            # f"python -O -m exam_pp.exam_post_pipeline {input_file} --prompt-class {prompt_class} -r --correlation-out {correlation_out} --dont-check-prompt-class {prompt_type} {min_judgement}"
        ]

        # Run each command
        for command in commands:
            print(f"Running: {command}")
            subprocess.run(command, shell=True)
            time.sleep(10)
        # break