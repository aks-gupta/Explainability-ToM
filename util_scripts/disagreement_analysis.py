import os
import pickle
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utilities import return_last_max_version
import configs
from configs import GENERAL_CONFIGS, MODEL_CONFIGS

num_examples = configs.GENERAL_CONFIGS['num_examples']
domain = configs.DOMAIN

def match_cf_to_ans(cf_file, task_ans_file, sim_ans_file, domain, num_examples=None):
    with open(cf_file, "rb") as f:
        cf_data = pickle.load(f)
    with open(task_ans_file, "rb") as f:
        task_ans_data = pickle.load(f)
    with open(sim_ans_file, "rb") as f:
        sim_ans_data = pickle.load(f)
    
    # Only slice if num_examples is provided
    if num_examples:
        cf_data = dict(list(cf_data.items())[:num_examples])
    
    all_matched_data = {}
    disagreed_matched_data = {}
    
    for key in cf_data.keys():
        questions = cf_data[key]['questions']
        task_answers = task_ans_data[key]
        sim_answers = sim_ans_data[key]
        
        questions_with_answers = []
        disagreed_questions = []

        for i, question in enumerate(questions):
            question_data = {
                'question': question,
                'task_answer': task_answers[i]['pred_ans'],
                'sim_answer': sim_answers[i]['pred_ans']
            }
            questions_with_answers.append(question_data)
            
            # Check for disagreement
            if task_answers[i]['pred_ans'] != sim_answers[i]['pred_ans']:
                disagreed_questions.append(question_data)
        
        # Initialize the key in all_matched_data with original cf_data structure
        all_matched_data[key] = cf_data[key].copy()
        all_matched_data[key]['matched_questions'] = questions_with_answers
        
        # Only add to disagreed_matched_data if there are disagreements
        if disagreed_questions:
            disagreed_matched_data[key] = cf_data[key].copy()
            disagreed_matched_data[key]['matched_questions'] = disagreed_questions
    
    # Save all matched data
    with open(f"./matched_counterfactuals_with_answers_{domain}.json", "w") as f:
        json.dump(all_matched_data, f, indent=4)
    
    # Save disagreed data
    with open(f"./disagreed_counterfactuals_with_answers_{domain}.json", "w") as f:
        json.dump(disagreed_matched_data, f, indent=4)

    print(f"Total entries: {len(cf_data)}")
    print(f"Task answers: {len(task_ans_data)}, Sim answers: {len(sim_ans_data)}")
    print(f"Entries with disagreements: {len(disagreed_matched_data)}")

if __name__ == "__main__":
    cf_file = f"data/preprocessed/label_balanced_counterfactuals_{domain}.pkl"
    folder_name = f"outputs/{domain}_{configs.MODEL_CONFIGS['taskqa_model'].split('/')[0]}_{num_examples}"
    full_path = return_last_max_version(folder_path=folder_name)
    taskqa_model = MODEL_CONFIGS['taskqa_model']
    taskqa_expl_type = MODEL_CONFIGS['taskqa_expl_type']
    simqg_model = MODEL_CONFIGS['simqg_model']
    simqa_model = MODEL_CONFIGS['simqa_model']
    sim_file = f"{full_path}/{domain}_{GENERAL_CONFIGS['step_3_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_simqa_{simqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
    task_file = f"{full_path}/{domain}_{GENERAL_CONFIGS['step_4_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_taskqa_{taskqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
    match_cf_to_ans(cf_file, task_file, sim_file, domain, num_examples=num_examples)
    