import json 
import sys
import os 
import pickle as pkl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import GENERAL_CONFIGS, MODEL_CONFIGS
from label_balanced_counterfactuals_generation_utils import run_counterfactual_generation_pipeline

counterfactual_code_generation = GENERAL_CONFIGS['counterfactuals']
num_counterfactual_qs = GENERAL_CONFIGS['num_counterfactual_qs']
num_examples = GENERAL_CONFIGS['num_examples']
EX_IDXS = range(0, num_examples)

assert counterfactual_code_generation=='LABEL_BALANCED'

#Step 1: Run the pipeline to get all data
# step_1_out, step_2_out, step_4_out = run_counterfactual_generation_pipeline()
step_1_out = 'data/hiring_decisions_task_qa_out_gpt-4o-mini_cot_2.pkl'
step_2_out = 'data/hiring_decisions_simulation_question_gen_out_gpt-4o-mini_simqg_gpt-4o-mini_cot_2.pkl'
step_4_out = 'data/hiring_decisions_task_qa_simulation_questions_out_gpt-4o-mini_simqg_gpt-4o-mini_taskqa_gpt-4o-mini_cot_2.pkl'

#Step 1: Parse all the questions and its corresponding answers from pickle file
with open(step_2_out, "rb") as f:
    qs = pkl.load(f)

with open(step_4_out, "rb") as f:
    ans = pkl.load(f)

#Step 2: Extract questions and its corresponding answers into a single list
questions_yes = []
questions_no = []
pred_ans_yes = []
pred_ans_no = []

for idx in ans:
    for i, entry in enumerate(ans[idx]):
        if entry["pred_ans"] == 'yes':
            questions_yes.append(qs[idx]["questions"][i])
            pred_ans_yes.append(entry)
        elif entry["pred_ans"] == 'no':
            questions_no.append(qs[idx]["questions"][i])
            pred_ans_no.append(entry)

# Step 3: Ensure balanced yes/no lists
min_len = min(len(questions_yes), len(questions_no))
questions_yes_bal = questions_yes[:min_len]
questions_no_bal = questions_no[:min_len]
pred_ans_yes_bal = pred_ans_yes[:min_len]
pred_ans_no_bal = pred_ans_no[:min_len]

# Step 4: Create dictionary with one yes and one no question per index
qs_dict = {}
balanced_dict = {}
for i in range(min_len):
    qs_dict[i] = {"questions": [questions_yes_bal[i], questions_no_bal[i]]} #yes question followed by no question
    balanced_dict[i] = [pred_ans_yes_bal[i], pred_ans_no_bal[i]]

# Step 5: Save to pickle
with open(f'data/label_balanced_counterfactuals.pkl', "wb") as f:
    pkl.dump(qs_dict, f)
with open(f'data/label_balanced_counterfactuals_answers.pkl', "wb") as f:
    pkl.dump(balanced_dict, f)

print(f"Saved {min_len} counterfactual pairs")