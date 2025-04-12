import json
import time
import os
import pickle as pkl
from tqdm import trange
import sys
from task_qa import task_qa, task_qa_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs, simulate_qg_bbq

def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
	print("Inside run_task_save_results")
	print(task_function, out_file, ex_idxs)
	all_preds = {}
	if os.path.exists(out_file):
		all_preds = pkl.load(open(out_file, 'rb'))
	ex_idxs = [ex_idx for ex_idx in ex_idxs if ex_idx not in all_preds]
	for key in kwargs:
		if type(kwargs[key]) == list or type(kwargs[key]) == dict:
			kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
	preds = task_function(**kwargs)
	print("task function executed")
	assert type(preds) == list and len(preds) == len(ex_idxs)
	for pos, ex_idx in enumerate(ex_idxs):
		all_preds[ex_idx] = preds[pos]
	assert out_file.endswith('_10.pkl')
	pkl.dump(all_preds, open(out_file, 'wb'))
	

if __name__ == '__main__':
	f_log = open('log.txt', 'w')
	timestamp = time.time()
	
	EX_IDXS = range(0, 10)
	
	for taskqa_model in ['gpt-4o-mini']:
		print(f"Using model: {taskqa_model}")		
		test_inputs = json.load(open('data_bbq.json'))['age']
		for taskqa_expl_type in ['cot', 'posthoc']:
			print(f"Explanation Type: {taskqa_expl_type}")
			print(f"Line Number: {sys._getframe().f_lineno}")
			out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}_age_10.pkl'
			print(out_file)
			run_task_save_results(task_function=task_qa, out_file=out_file, ex_idxs=EX_IDXS,
									model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
			f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
			timestamp = time.time()

	
	for taskqa_model in ['gpt-4o-mini']:
		print("LINE 48")
		for taskqa_expl_type in ['cot', 'posthoc']:
			# for simqg_model in ['gpt-4o', 'gpt-4o-mini']:
			for simqg_model in ['gpt-4o-mini']:
				for with_context in [True, False]:
					for top_p in [1.0]:
						out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_age_10.pkl'
						orig_inputs = json.load(open('./data_bbq.json'))['age']
						orig_tm_preds = pkl.load(open(f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}_age_10.pkl', 'rb'))
						run_task_save_results(task_function=simulate_qg, ex_idxs=EX_IDXS, out_file=out_file,
												model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
												top_p=top_p, num_samples=6, with_context=with_context)
						f_log.write(f'SimQG-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()