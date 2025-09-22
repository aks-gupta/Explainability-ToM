import json
import time
import os
import pickle as pkl
# from tqdm import trange
import sys
import time
from task_qa import task_qa, task_qa_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs
from simulate_qa import simulate_qa
import config
from calculate_precision import calculate_precision

# Run a task function, save results to a file, and handle already processed examples
def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
	all_preds = {}
	# Load existing predictions if the output file already exists
	if os.path.exists(out_file):
		all_preds = pkl.load(open(out_file, 'rb'))
	
	# Filter out already processed examples
	ex_idxs = [ex_idx for ex_idx in ex_idxs if ex_idx not in all_preds]
	for key in kwargs:
		if type(kwargs[key]) == list or type(kwargs[key]) == dict:
			kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
	
	# Run the task function and ensure the output matches the expected format
	preds = task_function(**kwargs)
	assert type(preds) == list and len(preds) == len(ex_idxs)
	
	# Save predictions for each example
	for pos, ex_idx in enumerate(ex_idxs):
		all_preds[ex_idx] = preds[pos]
	pkl.dump(all_preds, open(out_file, 'wb'))

# Main script execution
if __name__ == '__main__':
	f_log = open('log.txt', 'w')  # Log file to record progress
	timestamp = time.time()  # Start timestamp
	DOMAIN = config.DOMAIN  # Domain configuration
	NUM_EX = config.NUM_EX  # Number of examples
	NUM_CF = config.NUM_CF  # Number of counterfactuals

	# Load model configurations
	taskqa_models = config.taskqa_models
	expl_types = config.expl_types
	simqg_models = config.simqg_models
	simqa_models = config.simqa_models
	simqg_mixed = config.MIXED # whether to mix outputs from different simqg models
	context = config.WITH_CONTEXT # context while generating counterfactuals
	explanation = config.EXPLANATION # explanation used in simqa by simulator model
	balanced = config.BALANCED # balanced sampling in simqg

	'''implement stratified sampling in simqa'''
	# stratified = config.STRATIFIED # stratified sampling in simqa
	# num_strat = config.STRAT_SAMPLES_PER_OPTION # number of stratified samples per option in simqa
	# total_strat = config.TOTAL_STRAT_SAMPLES # total number of stratified samples in simqa

	# Explanation and context configurations
	SUB_FOLDER = f'/outputs_with_context_{context}_explanation_{explanation}_balanced_{balanced}_examples_{NUM_EX}_counterfactuals_{NUM_CF if not balanced else 3}'
	PATH = f'./outputs_{DOMAIN}{SUB_FOLDER}/'
	DATA_FILE = config.DATA_FILE

	EX_IDXS = range(0, NUM_EX)  # Example indices
	start_time = time.time()  # Start time for the entire script

	print("\nStarting the pipeline with the following configuration:")
	config.print_configs()
	print(f"Output will be saved in: {PATH}")

	# Create output directory if it doesn't exist
	if not os.path.exists(f'./outputs_{DOMAIN}{SUB_FOLDER}'):
		os.makedirs(f'./outputs_{DOMAIN}{SUB_FOLDER}')

	# Task QA processing
	for taskqa_model in taskqa_models:
		test_inputs = json.load(open(DATA_FILE))[DOMAIN]
		for taskqa_expl_type in expl_types:
			out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl'
			run_task_save_results(task_function=task_qa, out_file=out_file, ex_idxs=EX_IDXS,
								  model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs, domain=DOMAIN)
			print("\033[1mCompleted TaskQA for model:\033[0m", taskqa_model, 
				  "\033[1mexplanation type:\033[0m", taskqa_expl_type)
			f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
			timestamp = time.time()

	# Simulated Question Generation (SimQG)
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in expl_types:
			for simqg_model in simqg_models:
				for with_context in [True] if context else [False]:
					for top_p in [1.0]:
						out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{DOMAIN}_{NUM_EX}.pkl'
						orig_inputs = json.load(open('./data_bbq.json'))[DOMAIN]
						orig_tm_preds = pkl.load(open(f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
						run_task_save_results(task_function=simulate_qg, ex_idxs=EX_IDXS, out_file=out_file,
											  model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
											  top_p=top_p, num_samples=NUM_CF, with_context=with_context, domain=DOMAIN)
						print("\033[1mCompleted SimQG for TaskQA model:\033[0m", taskqa_model, 
							  "\033[1mexplanation type:\033[0m", taskqa_expl_type,
							  "\033[1mSimQG model:\033[0m", simqg_model, 
							  "\033[1mtop_p:\033[0m", top_p, 
							  "\033[1mwith_context:\033[0m", with_context)
						f_log.write(f'SimQG-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()

	# Mix outputs from different SimQG models
	if simqg_mixed:  # Only execute if mix is True
		for taskqa_model in taskqa_models:
			for taskqa_expl_type in expl_types:
				for with_context in [True] if context else [False]:
					for top_p in [1.0]:
						simqg_model2sim_inputs = {}
						for simqg_model in simqg_models:
							simqg_model2sim_inputs[simqg_model] = pkl.load(
								open(f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
						out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_mix_{top_p}_{with_context}_{DOMAIN}_{NUM_EX}.pkl'
						if os.path.exists(out_file):
							ex_idx2mixed_sim_inputs = pkl.load(open(out_file, 'rb'))
						else:
							ex_idx2mixed_sim_inputs = {}
						for ex_idx in EX_IDXS:
							if ex_idx not in ex_idx2mixed_sim_inputs:  # Avoid re-running for already computed examples
								ex_idx2mixed_sim_inputs[ex_idx] = mix_sim_inputs(simqg_model2sim_inputs[simqa_models[0]][ex_idx],
																				 simqg_model2sim_inputs[simqa_models[1]][ex_idx],
																				 sample_num=NUM_CF)
						pkl.dump(ex_idx2mixed_sim_inputs, open(out_file, 'wb'))

	# Simulated QA (SimQA)
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in expl_types:
			for simqg_model in ['mix'] if simqg_mixed else simqg_models:
				for with_context in [True] if context else [False]:
					for top_p in [1.0]:
						for simqa_model in simqa_models:
							out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}_{DOMAIN}_{NUM_EX}.pkl'
							orig_inputs = json.load(open('data_bbq.json'))[DOMAIN]
							orig_tm_preds = pkl.load(open(f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
							sim_inputs_list = pkl.load(open(
								f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
							run_task_save_results(task_function=simulate_qa, ex_idxs=EX_IDXS, out_file=out_file,
												  model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
												  sim_inputs_list=sim_inputs_list, domain=DOMAIN)
							print("\033[1mCompleted SimQA for TaskQA model:\033[0m", taskqa_model, 
								  "\033[1mexplanation type:\033[0m", taskqa_expl_type,
								  "\033[1mSimQG model:\033[0m", simqg_model, 
								  "\033[1mtop_p:\033[0m", top_p, 
								  "\033[1mwith_context:\033[0m", with_context, 
								  "\033[1mSimQA model:\033[0m", simqa_model)
							f_log.write(f'SimQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context}-{simqa_model} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()

	# Task QA on Simulated Inputs
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in expl_types:
			for simqg_model in ['mix'] if simqg_mixed else simqg_models:
				for with_context in [True] if context else [False]:
					for top_p in [1.0]:
						out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}' \
								   f'-taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl'
						sim_inputs_list = pkl.load(open(
							f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
						run_task_save_results(task_function=task_qa_sim_inputs_list, ex_idxs=EX_IDXS, out_file=out_file,
											  model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list, domain=DOMAIN)
						print("\033[1mCompleted TaskQA on Simulated Inputs for TaskQA model:\033[0m", taskqa_model,
							  "\033[1mexplanation type:\033[0m", taskqa_expl_type, 
							  "\033[1mSimQG model:\033[0m", simqg_model,
							  "\033[1mtop_p:\033[0m", top_p, 
							  "\033[1mwith_context:\033[0m", with_context)
						f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()
	
	# Print total time taken for the script
	print(f"\nTotal time taken: {(time.time() - start_time) / 60:.2f} minutes\n")

	# call calculate_precision.py to calculate precision
	print("\033[1mSimulation Precision Results:\033[0m")
	for simqg_model in ['mix'] if simqg_mixed else simqg_models:
		for with_context in [True] if context else [False]:
			for simqa_model in simqa_models:
				calculate_precision(domain=DOMAIN, num_ex=NUM_EX, taskqa_models=taskqa_models, simqg_model=simqg_models[0], simqa_model=simqa_models[0], with_context=with_context, path=PATH)

