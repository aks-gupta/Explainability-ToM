import sys
import os
import json
import time
import pickle as pkl
from tqdm import trange

from task_qa import task_qa_hiring_decisions, task_qa_hiring_decisions_sim_inputs_list
from simulate_qg import mix_sim_inputs, simulate_qg_hiring_decisions
from simulate_qa import simulate_qa_hiring_decisions
from utilities import create_folder_based_on_version, preprocess_label_balanced_counterfactuals
from configs import GENERAL_CONFIGS, MODEL_CONFIGS, DATASET, DOMAIN, DATA_FILE
from calculate_precision import calculate_precision
from util_scripts.pkl_to_json import pkl_to_json


def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
    if os.path.exists(out_file):
        print(f"Results already exist at {out_file}, skipping...")
        return
    all_preds = {}
    if os.path.exists(out_file):
        all_preds = pkl.load(open(out_file, 'rb'))
    ex_idxs = list(ex_idxs)
    for key in kwargs:
        if type(kwargs[key]) == list or type(kwargs[key]) == dict:
            kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
    preds = task_function(**kwargs)
    for pos, ex_idx in enumerate(ex_idxs):
        all_preds[ex_idx] = preds[pos]
    print(f"DEBUG Pipeline: Saved predictions for examples: {list(all_preds.keys())}")
    assert out_file.endswith('.pkl')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_file_json = out_file.replace('.pkl', '.json')
    pkl.dump(all_preds, open(out_file, 'wb'))
    pkl_to_json(out_file, out_file_json)
    print(f"Results saved to {out_file} and {out_file_json}")

def main():
    print("="*60)
    print("\033[1m\033[94mPIPELINE CONFIGURATION\033[0m")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Domain: {DOMAIN}")
    print(f"Data File: {DATA_FILE}")
    print(f"Number of Examples: {GENERAL_CONFIGS['num_examples']}")
    print(f"Counterfactual Generation: {GENERAL_CONFIGS['counterfactuals']}")
    print(f"Number of Counterfactuals: {GENERAL_CONFIGS['num_counterfactual_qs']}")
    print(f"K-shot: {GENERAL_CONFIGS['k_shot']}")
    print(f"TaskQA Models: {MODEL_CONFIGS['taskqa_model']}")
    print(f"TaskQA Explanation Types: {MODEL_CONFIGS['taskqa_expl_type']}")
    print(f"SimQG Models: {MODEL_CONFIGS['simqg_model']}")
    print(f"SimQA Models: {MODEL_CONFIGS['simqa_model']}")
    print("="*60)
    
    f_log = open('log.txt', 'w')
    timestamp = time.time()
    
    #Create folder based on versions
    folder_name = f"outputs/{DOMAIN}_{MODEL_CONFIGS['taskqa_model'].split('/')[0]}_{GENERAL_CONFIGS['num_examples']}"
    full_path = create_folder_based_on_version(folder_name)
    print(f"Output Directory: {full_path}")

    #Get config values
    num_examples = GENERAL_CONFIGS['num_examples']
    num_disagreement_qs = GENERAL_CONFIGS['num_disagreement_qs']
    counterfactual_code_generation = GENERAL_CONFIGS['counterfactuals']
    num_counterfactual_qs = GENERAL_CONFIGS['num_counterfactual_qs']
    
    taskqa_model = MODEL_CONFIGS['taskqa_model']
    taskqa_expl_type = MODEL_CONFIGS['taskqa_expl_type']
    simqg_model = MODEL_CONFIGS['simqg_model']
    simqa_model = MODEL_CONFIGS['simqa_model']
    simqa_explanation = MODEL_CONFIGS['simqa_expl_type']

    EX_IDXS = range(0, num_examples)
    
    print("Copying config file to output directory for reference.")
    os.system(f'cp configs.py {full_path}/configs.txt')
    print("Config file copied as txt.")

    # STEP 1: TaskQA
    print("\n" + "="*60)
    print("\033[1m\033[94mSTEP 1: TaskQA - Initial Question Answering\033[0m")
    print("="*60)
    # for taskqa_model in MODEL_CONFIGS['taskqa_model']:
    #     for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
            # print(f"Running TaskQA: {taskqa_model} with {taskqa_expl_type}")                
    if counterfactual_code_generation=='LABEL_BALANCED':
        preprocess_label_balanced_counterfactuals(f'./data/disagreement_dataset/disagreement_filtered_{DOMAIN}_{num_disagreement_qs}.json')
        with open(f'./data/preprocessed/label_balanced_original_questions_{DOMAIN}.json', "rb") as f:
            test_inputs = json.load(f)
    else:
        test_inputs = json.load(open(DATA_FILE))['test']
    EX_IDXS = range(0, min(num_examples, len(test_inputs))) 
    step_1_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_1_out']}_{taskqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
    run_task_save_results(task_function=task_qa_hiring_decisions, out_file=step_1_out, ex_idxs=EX_IDXS,
                            model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
    print(f"TaskQA completed: {os.path.basename(step_1_out)}")

    # STEP 2: SimQG
    print("\n" + "="*60)
    print("\033[1m\033[94mSTEP 2: SimQG - Counterfactual Question Generation\033[0m")
    print("="*60)
    if counterfactual_code_generation=='HARDCODED':
        step_2_out = f'data/hardcoded_counterfactuals.pkl'
    elif counterfactual_code_generation=='LABEL_BALANCED':
        step_2_out = f"./data/preprocessed/label_balanced_counterfactuals_{DOMAIN}.pkl"
        print("Using label balanced counterfactuals")
    else:
        print("Generating counterfactuals with SimQG")
        # for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        #     for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
            # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
        for simqg_model in MODEL_CONFIGS['simqg_model']:
            for explanation in ['withexpl']:
                for top_p in [1.0]:
                    print(f"Running SimQG: {simqg_model} with {taskqa_expl_type}")
                    step_2_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_2_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
                    orig_inputs = json.load(open(DATA_FILE))['test']
                    orig_tm_preds = pkl.load(open(step_1_out, 'rb'))
                    run_task_save_results(task_function=simulate_qg_hiring_decisions, ex_idxs=EX_IDXS, out_file=step_2_out,
                                            model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                            top_p=top_p, num_samples=num_counterfactual_qs, with_context=explanation)
                    print(f"SimQG completed: {os.path.basename(step_2_out)}")
    
    # STEP 3: SimQA
    print("\n" + "="*60)
    print("\033[1m\033[94mSTEP 3: SimQA - Answer Counterfactual Questions\033[0m")
    print("="*60)
    # for taskqa_model in MODEL_CONFIGS['taskqa_model']:
    #     for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
    #     # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
    #         for simqg_model in MODEL_CONFIGS['simqg_model']: # expl
    for explanation in [simqa_explanation]:
        for top_p in [1.0]:
            # for simqa_model in MODEL_CONFIGS['simqa_model']:
            print(f"Running SimQA: {simqa_model} with {taskqa_expl_type}")
            step_3_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_3_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_simqa_{simqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
            if counterfactual_code_generation=='LABEL_BALANCED':
                with open(f'./data/preprocessed/label_balanced_original_questions_{DOMAIN}.json', "rb") as f:
                    orig_inputs = json.load(f)
            else:
                orig_inputs = json.load(open(DATA_FILE))['test']
            orig_tm_preds = pkl.load(open(step_1_out, 'rb'))
            sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
            run_task_save_results(task_function=simulate_qa_hiring_decisions, ex_idxs=EX_IDXS, out_file=step_3_out,
                                model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                sim_inputs_list=sim_inputs_list,  include_expl=explanation=='withexpl')
            print(f"SimQA completed: {os.path.basename(step_3_out)}")

    # STEP 4: TaskQA on SimInputs
    print("\n" + "="*60)
    print("\033[1m\033[94mSTEP 4: TaskQA on Simulated Inputs\033[0m")
    print("="*60)
    # for taskqa_model in MODEL_CONFIGS['taskqa_model']:
    #     for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
    #     # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
    #         for simqg_model in MODEL_CONFIGS['simqg_model']:
    for explanation in ['withexpl']:
        for top_p in [1.0]:
            print(f"Running TaskQA on SimInputs: {taskqa_model} with {taskqa_expl_type}")
            step_4_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_4_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_taskqa_{taskqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
            sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
            # all_sim_inputs = [{'question': input} for sim_inputs in sim_inputs_list for input in sim_inputs['questions']]
            run_task_save_results(task_function=task_qa_hiring_decisions_sim_inputs_list, ex_idxs=EX_IDXS, out_file=step_4_out,
                                    model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
            print(f"TaskQA on SimInputs completed: {os.path.basename(step_4_out)}")
    
    # Pipeline Completion
    print("\n" + "="*60)
    print("\033[1m\033[94mPIPELINE COMPLETED SUCCESSFULLY!\033[0m")
    print("="*60)
    print(f"Total Time Taken: {time.time() - timestamp:.2f} seconds")
    print(f"All results saved in: {full_path}")
    print(f"Log file: log.txt")
    print("="*60)
    
    print("\n" + "="*60)
    print("\033[1m\033[94mCALCULATING PRECISION\033[0m")
    print("="*60)
    calculate_precision(taskqa_model, taskqa_expl_type)
    
if __name__ == '__main__':
	main()