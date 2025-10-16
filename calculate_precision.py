import json
import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
import os 
from configs import GENERAL_CONFIGS, MODEL_CONFIGS, DOMAIN, MODEL_CONFIGS
from utilities import return_last_max_version

print(os.getcwd())

def calculate_precision(taskqa_model, taskqa_expl_type):
	simqg_model = MODEL_CONFIGS['simqg_model']
	top_p = 1.0
	simqa_model = MODEL_CONFIGS['simqa_model']
	with_context = True
	folder_name = f"outputs/{DOMAIN}_{MODEL_CONFIGS['taskqa_model'].split('/')[0]}_{GENERAL_CONFIGS['num_examples']}"
	full_path = return_last_max_version(folder_path=folder_name)
	print(f"Calculating precision for folder: {full_path}")

	setting2exidx2precision = {}
	# for taskqa_model in MODEL_CONFIGS['taskqa_model']:
	# 	for taskqa_expl_type in MODEL_CONFIGS['taskqa_expl_type']:
		# for taskqa_expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
	for explanation in ['withexpl']:
		# print("------" + str(taskqa_expl_type) + "--------")
		setting = (taskqa_model, taskqa_expl_type)
		setting2exidx2precision[setting] = {}

		step_3_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_3_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_simqa_{simqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
		print(f"SIMQA FILE: {step_3_out}")
		exidx2qns_simans = pkl.load(
			open(step_3_out, 'rb'))

		count = 0
		simans_count = {}
		for exidx in exidx2qns_simans:
			for simans in exidx2qns_simans[exidx]:
				pred_ans = simans['pred_ans']
				simans_count[count] = [str(pred_ans)]
				count+=1
		#write to json file
		with open(f'{full_path}/simans_count.json', 'w') as f:
			json.dump(simans_count, f)

		step_4_out = f"{full_path}/{DOMAIN}_{GENERAL_CONFIGS['step_4_out']}_{taskqa_model.split('/')[0]}_simqg_{simqg_model.split('/')[0]}_taskqa_{taskqa_model.split('/')[0]}_{taskqa_expl_type}_{GENERAL_CONFIGS['num_examples']}.pkl"
		print(f"TASKQA FILE: {step_4_out}")

		exidx2qns_taskans = pkl.load(
			open(step_4_out, 'rb'))
		
		count = 0
		taskans_count = {}
		for exidx in exidx2qns_taskans:
			for taskans in exidx2qns_taskans[exidx]:
				pred_ans = taskans['pred_ans']
				taskans_count[count] = [str(pred_ans)]
				count+=1
		#write to json file
		with open(f'{full_path}/taskans_count.json', 'w') as f:
			json.dump(taskans_count, f)

		ex_simulatable_count, ex_correct_simul_count = 0, 0
		unknown_count = 0
		unknown_set = set()
		for exidx in range(count):
			simqa_ann = simans_count[exidx][0]
			taskqa_pred = taskans_count[exidx][0]
			if simqa_ann in ['no', 'yes']:
				ex_simulatable_count += 1
				if simqa_ann == taskqa_pred:
					ex_correct_simul_count += 1
			else:
				unknown_count += 1
				unknown_set.add(simqa_ann)
		print(f"Correctly simulated: {ex_correct_simul_count}, Simulatable: {ex_simulatable_count}")
		print("Unknown count:", unknown_count)
		print("Unknown set:", unknown_set)
		if ex_simulatable_count != 0:
			setting2exidx2precision[setting] =  ex_correct_simul_count / ex_simulatable_count

	settings = list(setting2exidx2precision.keys())
	for setting in settings:
		print(' '.join(setting), round(np.mean(setting2exidx2precision[setting]) * 100, 1))
 
	with open(f'{full_path}/precision_results.json', 'w') as f:
		results = {
			"settings": {
				"simqg_model": simqg_model,
				"top_p": top_p,
				"simqa_model": simqa_model,
				"with_context": with_context,
				"taskqa_models": MODEL_CONFIGS['taskqa_model'],
				"taskqa_explanation_types": MODEL_CONFIGS['taskqa_expl_type']
			},
			"unknown_count": unknown_count,
			"unknown_set": list(unknown_set),
			"precision_results": {
				MODEL_CONFIGS['taskqa_model']: {
					MODEL_CONFIGS['taskqa_expl_type']: round(np.mean(setting2exidx2precision.get((MODEL_CONFIGS['taskqa_model'], MODEL_CONFIGS['taskqa_expl_type']), 0)) * 100, 1)
				}
			}
		}
		json.dump(results, f, indent=2)

# if __name__ == "__main__":
# 	calculate_precision(MODEL_CONFIGS['taskqa_model'], MODEL_CONFIGS['taskqa_expl_type'])