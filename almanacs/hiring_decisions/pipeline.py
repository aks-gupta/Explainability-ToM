import sys
import os
import json
import time
import pickle as pkl
from tqdm import trange

from task_qa import task_qa, task_qa_hiring_decisions, task_qa_sim_inputs_list, task_qa_hiring_decisions_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs, simulate_qg_hiring_decisions
from simulate_qa import simulate_qa, simulate_qa_hiring_decisions
from utilities import create_folder_based_on_version
from configs import GENERAL_CONFIGS


def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
    all_preds = {}
    if os.path.exists(out_file):
        all_preds = pkl.load(open(out_file, 'rb'))
    ex_idxs = list(ex_idxs)
    for key in kwargs:
        if type(kwargs[key]) == list or type(kwargs[key]) == dict:
            kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
    preds = task_function(**kwargs)
    print("task function executed")
    print(type(preds))
    print(len(preds))
    print(len(ex_idxs))
    for pos, ex_idx in enumerate(ex_idxs):
        print(preds[pos])
        print("\n\n")
        all_preds[ex_idx] = preds[pos]
    assert out_file.endswith('.pkl')
    pkl.dump(all_preds, open(out_file, 'wb'))


def main():
    f_log = open('log.txt', 'w')
    timestamp = time.time()

    #Create folder based on versions
    full_path = create_folder_based_on_version()

    #Get config values
    num_examples = GENERAL_CONFIGS['num_examples']
    num_counterfactual_qs = GENERAL_CONFIGS['num_counterfactual_qs']
    EX_IDXS = range(0, num_examples)

    # TaskQA
    for taskqa_model in ['gpt-4o']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
        for taskqa_expl_type in ['cot']:
            test_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
            step_1_out = f'{full_path}/{GENERAL_CONFIGS['step_1_out']}_{taskqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
            # out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}_test_180.pkl'
            run_task_save_results(task_function=task_qa_hiring_decisions, out_file=step_1_out, ex_idxs=EX_IDXS,
									model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
            print(step_1_out)

    # SimQG
    for taskqa_model in ['gpt-4o']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
        for taskqa_expl_type in ['cot']:
            for simqg_model in ['gpt-4o']:
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        step_2_out = f'{full_path}/{GENERAL_CONFIGS['step_2_out']}_{taskqa_model}_simqg_{simqg_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
                        # out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}-simqg_{simqg_model}_{taskqa_expl_type}_{top_p}_{explanation}_test_180.pkl'
                        orig_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
                        orig_tm_preds = pkl.load(open(step_1_out, 'rb'))
                        run_task_save_results(task_function=simulate_qg_hiring_decisions, ex_idxs=EX_IDXS, out_file=step_2_out,
                                                model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                                top_p=top_p, num_samples=num_counterfactual_qs, with_context=explanation)
                        print(step_2_out)
    
    # SimQA
    for taskqa_model in ['gpt-4o']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
        for taskqa_expl_type in ['cot']:
            for simqg_model in ['gpt-4o']: # expl
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        for simqa_model in ['gpt-4o']:
                            step_3_out = f'{full_path}/{GENERAL_CONFIGS['step_3_out']}_{taskqa_model}_simqg_{simqg_model}_simqa_{simqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
                            # out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}-simqa_{simqa_model}_{explanation}_fix_test_180_CHECK_CHECK.pkl'
                            orig_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
                            orig_tm_preds = pkl.load(open(step_1_out, 'rb'))
                            sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
                            run_task_save_results(task_function=simulate_qa_hiring_decisions, ex_idxs=EX_IDXS, out_file=step_3_out,
                                                model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                                sim_inputs_list=sim_inputs_list,  include_expl=explanation=='withexpl')
                            print(step_3_out)

    # TaskQA on SimInputs
    for taskqa_model in ['gpt-4o']:
        # for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
        for taskqa_expl_type in ['cot']:
            for simqg_model in ['gpt-4o']:
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        step_4_out = f'{full_path}/{GENERAL_CONFIGS['step_4_out']}_{taskqa_model}_simqg_{simqg_model}_taskqa_{taskqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
                        # out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}' \
                        #             f'-taskqa_{taskqa_model}_{explanation}_180_CHECK_CHECK.pkl'
                        sim_inputs_list = pkl.load(open(step_2_out, 'rb'))
                        run_task_save_results(task_function=task_qa_hiring_decisions_sim_inputs_list, ex_idxs=EX_IDXS, out_file=step_4_out,
                                                model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
                        print(step_4_out)

if __name__ == '__main__':
	main()