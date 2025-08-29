import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel
import os 
from configs import GENERAL_CONFIGS
from utilities import return_last_max_version

print(os.getcwd())

if __name__ == '__main__':
	num_examples = GENERAL_CONFIGS['num_examples']
	num_counterfactual_qs = GENERAL_CONFIGS['num_counterfactual_qs']
	EX_IDXS = range(0, num_examples*num_counterfactual_qs)
	simqg_model = 'gpt-4o'
	top_p = 1.0
	simqa_model = 'gpt-4o'
	with_context = True
	full_path = return_last_max_version()
	print(full_path)

	setting2exidx2precision = {}
	for taskqa_model in ['gpt-4o']:
		for taskqa_expl_type in ['cot']:
			for explanation in ['withexpl']:
				print("-------" + str(taskqa_expl_type) + "--------")
				print(explanation)
				setting = (taskqa_model, taskqa_expl_type)
				setting2exidx2precision[setting] = {}

				step_3_out = f'{full_path}/{GENERAL_CONFIGS['step_3_out']}_{taskqa_model}_simqg_{simqg_model}_simqa_{simqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
				print(step_3_out)
				exidx2qns_simans = pkl.load(
					open(step_3_out, 'rb'))

				count = 0
				simans_count = {}
				for exidx in exidx2qns_simans:
					for simans in exidx2qns_simans[exidx]:
						pred_ans = simans['pred_ans']
						simans_count[count] = [str(pred_ans)]
						count+=1
				print(simans_count)

				step_4_out = f'{full_path}/{GENERAL_CONFIGS['step_4_out']}_{taskqa_model}_simqg_{simqg_model}_taskqa_{taskqa_model}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl'
				print(step_4_out)

				exidx2qns_taskans = pkl.load(
					open(step_4_out, 'rb'))
				
				count = 0
				taskans_count = {}
				for exidx in exidx2qns_taskans:
					for taskans in exidx2qns_taskans[exidx]:
						pred_ans = taskans['pred_ans']
						taskans_count[count] = [str(pred_ans)]
						count+=1
				print(taskans_count)

				
				ex_simulatable_count, ex_correct_simul_count = 0, 0
				unknown_count = 0
				unknown_set = set()
				for exidx in EX_IDXS:
					simqa_ann = simans_count[exidx][0]
					taskqa_pred = taskans_count[exidx][0]
					if simqa_ann in ['no', 'yes']:
						ex_simulatable_count += 1
						if simqa_ann == taskqa_pred:
							ex_correct_simul_count += 1
					else:
						unknown_count += 1
						unknown_set.add(simqa_ann)
				print(ex_correct_simul_count, ex_simulatable_count)
				print("Unknown count:", unknown_count)
				print("Unknown set:", unknown_set)
				if ex_simulatable_count != 0:
					setting2exidx2precision[setting] =  ex_correct_simul_count / ex_simulatable_count

	settings = list(setting2exidx2precision.keys())
	for setting in settings:
		print(' '.join(setting), round(np.mean(setting2exidx2precision[setting]) * 100, 1))
	print(">><<")