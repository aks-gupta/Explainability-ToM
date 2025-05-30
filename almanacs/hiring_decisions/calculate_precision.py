import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel
import os 

print(os.getcwd())

if __name__ == '__main__':
	EX_IDXS = range(0,100)
	simqg_model = 'gpt-4o-mini'
	top_p = 1.0
	simqa_model = 'gpt-4o-mini'
	with_context = True

	setting2exidx2precision = {}
	# for taskqa_model in ['gpt3', 'gpt4']:
	for taskqa_model in ['gpt-4o-mini']:
		for taskqa_expl_type in ['cot', 'posthoc']:
			print("-------" + str(taskqa_expl_type) + "--------")
			setting = (taskqa_model, taskqa_expl_type)
			setting2exidx2precision[setting] = {}
			exidx2qns_simans = pkl.load(
				# open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}.pkl', 'rb'))
				open(f'./outputs/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}-simqa_{simqa_model}_fix_test_1.pkl', 'rb'))
			# exidx2qns_simans = {exidx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_simans[exidx]]
			# 					for exidx in exidx2qns_simans}
			exidx2qns_simans = {
				exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
				for exidx, qn_anns in exidx2qns_simans.items()
			}
			# print(exidx2qns_simans)
			exidx2qns_taskans = pkl.load(
				# open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-taskqa_{taskqa_model}_{taskqa_expl_type}.pkl', 'rb'))
				open(f'./outputs/taskqa_hiring_decisions_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}-taskqa_{taskqa_model}_{taskqa_expl_type}_1.pkl', 'rb'))
			exidx2qns_taskans = {exidx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_taskans[exidx]]
								 for exidx in exidx2qns_taskans}
			# print(exidx2qns_taskans)
			for exidx in EX_IDXS:
				ex_simulatable_count, ex_correct_simul_count = 0, 0
				assert len(exidx2qns_simans[exidx]) == len(exidx2qns_taskans[exidx])
				for qnidx in range(len(exidx2qns_simans[exidx])):
					simqa_ann = exidx2qns_simans[exidx][qnidx]
					taskqa_pred = exidx2qns_taskans[exidx][qnidx]
					if simqa_ann in ['no', 'yes']:
						ex_simulatable_count += 1
						if simqa_ann == taskqa_pred:
							ex_correct_simul_count += 1
				if ex_simulatable_count != 0:
					setting2exidx2precision[setting][exidx] = ex_correct_simul_count / ex_simulatable_count

	all_settings_exidxs = [setting2exidx2precision[setting].keys() for setting in setting2exidx2precision]
	exidxs_in_all_settings = [exidx for exidx in all_settings_exidxs[0] if np.all([exidx in exidxs for exidxs in all_settings_exidxs])]
	print(len(exidxs_in_all_settings))
	setting2scores = {setting: [setting2exidx2precision[setting][exidx] for exidx in exidxs_in_all_settings]
					  for setting in setting2exidx2precision}

	settings = list(setting2scores.keys())
	print(settings)
	for setting in settings:
		print(' '.join(setting), round(np.mean(setting2scores[setting]) * 100, 1))
	print(">><<")
	# for setting1 in settings:
	# 	print(setting1)
	# 	pvalues = [str(ttest_rel(setting2scores[setting1], setting2scores[setting2])[1]) for setting2 in settings]
	# 	print(','.join(pvalues))