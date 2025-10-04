import sys
import os
import json
import time
import pickle as pkl
from tqdm import trange

from task_qa import task_qa_hiring_decisions, task_qa_hiring_decisions_sim_inputs_list
from simulate_qg import mix_sim_inputs, simulate_qg_hiring_decisions
from simulate_qa import simulate_qa_hiring_decisions
from utilities import create_folder_based_on_version
from configs import GENERAL_CONFIGS, MODEL_CONFIGS, DATASET, DOMAIN, DATA_FILE


def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
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
    pkl.dump(all_preds, open(out_file, 'wb'))


def main():
    print("="*60)
    print("PIPELINE CONFIGURATION")
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
    full_path = create_folder_based_on_version()
    print(f"Output Directory: {full_path}")

    #Get config values
    num_examples = GENERAL_CONFIGS['num_examples']
    counterfactual_code_generation = GENERAL_CONFIGS['counterfactuals']
    num_counterfactual_qs = GENERAL_CONFIGS['num_counterfactual_qs']

    EX_IDXS = range(0, num_examples)

    # STEP 1: TaskQA
    print("\n" + "="*60)
    print("STEP 1: TaskQA - Initial Question Answering")
    print("="*60)
    for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
            # print(f"Running TaskQA: {taskqa_model} with {taskqa_expl_type}")                
            if counterfactual_code_generation=='LABEL_BALANCED':
                with open(f'./data/label_balanced_original_questions_{DOMAIN}.json', "rb") as f:
                    test_inputs = json.load(f)
            else:
                test_inputs = json.load(open(DATA_FILE))['test']
            # print(f"DEBUG Pipeline: Loaded {len(test_inputs)} test inputs")
            # print(f"DEBUG Pipeline: Test inputs: {test_inputs[0]}")
            # print(f"DEBUG Pipeline: Test inputs: {test_inputs[1]}")
            
            step_1_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_1_out']}_{taskqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
            run_task_save_results(task_function=task_qa_hiring_decisions, out_file=step_1_out, ex_idxs=EX_IDXS,
									model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
            print(f"TaskQA completed: {os.path.basename(step_1_out)}")

    # STEP 2: SimQG
    print("\n" + "="*60)
    print("STEP 2: SimQG - Counterfactual Question Generation")
    print("="*60)
    if counterfactual_code_generation=='HARDCODED':
        step_2_out = f'data/hardcoded_counterfactuals.pkl'
    elif counterfactual_code_generation=='LABEL_BALANCED':
        step_2_out = f"./data/label_balanced_counterfactuals_{DOMAIN}.pkl"
        print("Using label balanced counterfactuals")
    else:
        print("Generating counterfactuals with SimQG")
        for taskqa_model in MODEL_CONFIGS['taskqa_model']:
            for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
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
    print("STEP 3: SimQA - Answer Counterfactual Questions")
    print("="*60)
    for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in MODEL_CONFIGS['simqg_model']: # expl
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        for simqa_model in MODEL_CONFIGS['simqa_model']:
                            print(f"Running SimQA: {simqa_model} with {taskqa_expl_type}")
                            step_3_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_3_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_simqa_{simqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
                            orig_inputs = json.load(open(DATA_FILE))['test']
                            orig_tm_preds = pkl.load(open(step_1_out, 'rb'))
                            sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
                            run_task_save_results(task_function=simulate_qa_hiring_decisions, ex_idxs=EX_IDXS, out_file=step_3_out,
                                                model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                                sim_inputs_list=sim_inputs_list,  include_expl=explanation=='withexpl')
                            print(f"SimQA completed: {os.path.basename(step_3_out)}")

    # STEP 4: TaskQA on SimInputs
    print("\n" + "="*60)
    print("STEP 4: TaskQA on Simulated Inputs")
    print("="*60)
    for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in MODEL_CONFIGS['simqg_model']:
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        print(f"Running TaskQA on SimInputs: {taskqa_model} with {taskqa_expl_type}")
                        step_4_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_4_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_taskqa_{taskqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
                        sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
                        run_task_save_results(task_function=task_qa_hiring_decisions_sim_inputs_list, ex_idxs=EX_IDXS, out_file=step_4_out,
                                                model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
                        print(f"TaskQA on SimInputs completed: {os.path.basename(step_4_out)}")
    
    # Pipeline Completion
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"All results saved in: {full_path}")
    print(f"Log file: log.txt")
    print("="*60)

if __name__ == '__main__':
	main()