import sys
import os
from task_qa import task_qa, task_qa_hiring_decisions, task_qa_sim_inputs_list, task_qa_hiring_decisions_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs, simulate_qg_hiring_decisions
from simulate_qa import simulate_qa, simulate_qa_hiring_decisions
import json
import time
import pickle as pkl
from tqdm import trange

def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
    all_preds = {}
    if os.path.exists(out_file):
        all_preds = pkl.load(open(out_file, 'rb'))
    # ex_idxs = [ex_idx for ex_idx in ex_idxs if ex_idx not in all_preds]
    ex_idxs = list(ex_idxs)
    for key in kwargs:
        if type(kwargs[key]) == list or type(kwargs[key]) == dict:
            kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
    preds = task_function(**kwargs)
    print("task function executed")
    print(type(preds))
    print(len(preds))
    print(len(ex_idxs))
    # assert type(preds) == list and len(preds) == len(ex_idxs)
    for pos, ex_idx in enumerate(ex_idxs):
        # if (len(all_preds)-1)>ex_idx:
        all_preds[ex_idx] = preds[pos]
    assert out_file.endswith('.pkl')
    pkl.dump(all_preds, open(out_file, 'wb'))


def main():
    f_log = open('log.txt', 'w')
    timestamp = time.time()
    EX_IDXS = range(0, 180)

    # TaskQA
	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
    for taskqa_model in ['gpt-4o']:
        print("LINE 38")
        for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            print(f'almanacs—hiring-decisions-taskqa-{taskqa_expl_type}')
            # test_inputs = json.load(open('./prompts/prompts.json'))[f'almanacs—hiring-decisions-taskqa-{taskqa_expl_type}']['test_examples']
            test_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
            # test_inputs = [item['question'] for item in test_inputs]
            out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}_test_180.pkl'
            run_task_save_results(task_function=task_qa_hiring_decisions, out_file=out_file, ex_idxs=EX_IDXS,
									model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
            print(out_file)
            f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
            timestamp = time.time()

    # SimQG
	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
    for taskqa_model in ['gpt-4o']:
        print("LINE 52")
        for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in ['gpt-4o']:
                for explanation in ['withexpl', 'withoutexpl']:
                    for top_p in [1.0]:
                        out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}-simqg_{simqg_model}_{taskqa_expl_type}_{top_p}_{explanation}_test_180.pkl'
                        orig_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
                        orig_tm_preds = pkl.load(open(f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}_test_180.pkl', 'rb'))
                        run_task_save_results(task_function=simulate_qg_hiring_decisions, ex_idxs=EX_IDXS, out_file=out_file,
                                                model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                                top_p=top_p, num_samples=6, with_context=explanation)
                        print(out_file)
                        f_log.write(f'SimQG-{taskqa_model}-{simqg_model}-{top_p} {(time.time() - timestamp)//60} minutes\n')
                        timestamp = time.time()
    
    # SimQA
    for taskqa_model in ['gpt-4o']:
        print("LINE 68")
        for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in ['gpt-4o']: # expl
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        # for simqa_model in ['gpt-4o-mini', 'gpt-4o']:
                        for simqa_model in ['gpt-4o']:
                            out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}-simqa_{simqa_model}_{explanation}_fix_test_180_CHECK_CHECK.pkl'
                            print(out_file)
                            orig_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
                            orig_tm_preds = pkl.load(open(f'./outputs/final/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}_test_180.pkl', 'rb'))
                            sim_inputs_list = pkl.load(open(
                                f'./outputs/final/taskqa_hiring_decisions_{taskqa_model}-simqg_{simqg_model}_{taskqa_expl_type}_{top_p}_{explanation}_test_180.pkl', 'rb'))
                            run_task_save_results(task_function=simulate_qa_hiring_decisions, ex_idxs=EX_IDXS, out_file=out_file,
                                                model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
                                                sim_inputs_list=sim_inputs_list,  include_expl=explanation=='withexpl')
                            f_log.write(f'SimQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{simqa_model} {(time.time() - timestamp)//60} minutes\n')
                        timestamp = time.time()

    # TaskQA on SimInputs
	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
    for taskqa_model in ['gpt-4o']:
        print("LINE 88")
        for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
            for simqg_model in ['gpt-4o']:
                for explanation in ['withexpl']:
                    for top_p in [1.0]:
                        out_file = f'./outputs/refined/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}' \
                                    f'-taskqa_{taskqa_model}_{explanation}_180_CHECK_CHECK.pkl'
                        print(out_file)
                        sim_inputs_list = pkl.load(open(
                                f'./outputs/final/taskqa_hiring_decisions_{taskqa_model}-simqg_{simqg_model}_{taskqa_expl_type}_{top_p}_{explanation}_test_180.pkl', 'rb'))
                        run_task_save_results(task_function=task_qa_hiring_decisions_sim_inputs_list, ex_idxs=EX_IDXS, out_file=out_file,
                                                model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
                        f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p} {(time.time() - timestamp)//60} minutes\n')
                        timestamp = time.time()

if __name__ == '__main__':
	main()