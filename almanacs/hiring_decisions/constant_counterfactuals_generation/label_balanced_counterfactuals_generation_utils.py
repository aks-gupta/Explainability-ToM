import json 
import sys
import os 
import pickle as pkl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import GENERAL_CONFIGS, MODEL_CONFIGS
from pipeline import run_task_save_results
from task_qa import task_qa_hiring_decisions, task_qa_hiring_decisions_sim_inputs_list
from simulate_qg import simulate_qg_hiring_decisions

counterfactual_code_generation = GENERAL_CONFIGS['counterfactuals']
num_counterfactual_qs = GENERAL_CONFIGS['num_counterfactual_qs']
num_examples = GENERAL_CONFIGS['num_examples']
EX_IDXS = range(0, num_examples)

assert counterfactual_code_generation=='LABEL_BALANCED'

def run_counterfactual_generation_pipeline():
    #Step 1: Generate Task QA outputs for a given question
    for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:     
            test_inputs = json.load(open('./data/data_hiring_decisions.json'))['fixed']
            step_1_out = f'data/{GENERAL_CONFIGS['step_1_out']}_{taskqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
            run_task_save_results(task_function=task_qa_hiring_decisions, out_file=step_1_out, ex_idxs=EX_IDXS,
                                    model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
    print(step_1_out)
    #Step 2: Generate counterfactual for each question with label balancing
    for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in MODEL_CONFIGS['simqg_model']:
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        step_2_out = f'data/{GENERAL_CONFIGS['step_2_out']}_{taskqa_model}_simqg_{simqg_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
                        orig_inputs = json.load(open('./data/data_hiring_decisions.json'))['fixed']
                        orig_tm_preds = pkl.load(open(step_1_out, 'rb'))
                        run_task_save_results(task_function=simulate_qg_hiring_decisions, ex_idxs=EX_IDXS, out_file=step_2_out,
                                                model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                                top_p=top_p, num_samples=num_counterfactual_qs, with_context=explanation)
    print(step_2_out)
    #Step 3: Get preds using taskqa model
    for taskqa_model in MODEL_CONFIGS['taskqa_model']:
        for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in MODEL_CONFIGS['simqg_model']:
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        step_4_out = f'data/{GENERAL_CONFIGS['step_4_out']}_{taskqa_model}_simqg_{simqg_model}_taskqa_{taskqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
                        sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
                        run_task_save_results(task_function=task_qa_hiring_decisions_sim_inputs_list, ex_idxs=EX_IDXS, out_file=step_4_out,
                                                model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
                        print(step_4_out)
    return step_1_out, step_2_out, step_4_out