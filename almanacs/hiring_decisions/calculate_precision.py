import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel
import os 

print(os.getcwd())

if __name__ == '__main__':
	EX_IDXS = range(0,180)
	simqg_model = 'gpt-4o'
	top_p = 1.0
	simqa_model = 'gpt-4o'
	with_context = True

	setting2exidx2precision = {}
	for taskqa_model in ['gpt-4o']:
		for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
			for explanation in ['withexpl']:
				print("-------" + str(taskqa_expl_type) + "--------")
				print(explanation)
				setting = (taskqa_model, taskqa_expl_type)
				setting2exidx2precision[setting] = {}
				exidx2qns_simans = pkl.load(
					# open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}.pkl', 'rb'))
					open(f'./outputs/final/simulation_model/taskqa_hiring_decisions_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}-simqa_{simqa_model}_{explanation}_fix_test_180_CHECK_CHECK.pkl', 'rb'))

				for exidx in exidx2qns_simans:
					exidx2qns_simans[exidx] = [
						str(exidx2qns_simans[exidx]['pred_ans']) 
					]

				exidx2qns_taskans = pkl.load(
					# open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-taskqa_{taskqa_model}_{taskqa_expl_type}.pkl', 'rb'))
					open(f'./outputs/final/task_model/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}-taskqa_{taskqa_model}_{explanation}_180_CHECK_CHECK.pkl', 'rb'))
				for exidx in exidx2qns_taskans:
					exidx2qns_taskans[exidx] =  [
						str(exidx2qns_taskans[exidx]['pred_ans'])
					]
				
				ex_simulatable_count, ex_correct_simul_count = 0, 0
				for exidx in EX_IDXS:
					simqa_ann = exidx2qns_simans[exidx][0]
					taskqa_pred = exidx2qns_taskans[exidx][0]
					if simqa_ann in ['no', 'yes']:
						ex_simulatable_count += 1
						if simqa_ann == taskqa_pred:
							ex_correct_simul_count += 1
				print(ex_correct_simul_count, ex_simulatable_count)
				if ex_simulatable_count != 0:
					setting2exidx2precision[setting] =  ex_correct_simul_count / ex_simulatable_count

	settings = list(setting2exidx2precision.keys())
	for setting in settings:
		print(' '.join(setting), round(np.mean(setting2exidx2precision[setting]) * 100, 1))
	print(">><<")