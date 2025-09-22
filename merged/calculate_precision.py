import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel

def calculate_precision(domain, num_ex, taskqa_models, simqg_model, simqa_model, with_context, path):
	DOMAIN = domain
	NUM_EX = num_ex
	EX_IDXS = range(0,NUM_EX)
	simqg_model = simqg_model
	top_p = 1.0
	simqa_model = simqa_model
	with_context = with_context
	path = path

	count_unknown = 0

	setting2exidx2precision = {}
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in ['cot']:
			setting = (taskqa_model, taskqa_expl_type)
			setting2exidx2precision[setting] = {}
			exidx2qns_simans = pkl.load(
				open(f'{path}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
			exidx2qns_simans = {
				exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
				for exidx, qn_anns in exidx2qns_simans.items()
			}

			exidx2qns_taskans = pkl.load(
				open(f'{path}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
			exidx2qns_taskans = {exidx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_taskans[exidx]]
								 for exidx in exidx2qns_taskans}
			for exidx in EX_IDXS:
				ex_simulatable_count, ex_correct_simul_count = 0, 0
				assert len(exidx2qns_simans[exidx]) == len(exidx2qns_taskans[exidx])
				for qnidx in range(len(exidx2qns_simans[exidx])):
					simqa_ann = exidx2qns_simans[exidx][qnidx]
					taskqa_pred = exidx2qns_taskans[exidx][qnidx]
					if simqa_ann in ['0', '1', '2']:
						ex_simulatable_count += 1
						if simqa_ann == taskqa_pred:
							ex_correct_simul_count += 1
					else:
						count_unknown += 1
				if ex_simulatable_count != 0:
					setting2exidx2precision[setting][exidx] = ex_correct_simul_count / ex_simulatable_count

	all_settings_exidxs = [setting2exidx2precision[setting].keys() for setting in setting2exidx2precision]
	exidxs_in_all_settings = [exidx for exidx in all_settings_exidxs[0] if np.all([exidx in exidxs for exidxs in all_settings_exidxs])]
	print("Number of Examples : " + str(len(exidxs_in_all_settings)*3))
	setting2scores = {setting: [setting2exidx2precision[setting][exidx] for exidx in exidxs_in_all_settings]
					  for setting in setting2exidx2precision}
	print("Number of unknown examples: " + str(count_unknown))

	settings = list(setting2scores.keys())
	print(settings)
	for setting in settings:
		print(' '.join(setting), round(np.mean(setting2scores[setting]) * 100, 1))