import json
import time
import os
import pickle as pkl
# from tqdm import trange
import sys
import time
from task_qa import task_qa, task_qa_single_example
from simulate_qa import simulate_qa_direct_examples
import config
from calculate_precision import calculate_precision_fixed

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

	# Explanation and context configurations
	SUB_FOLDER = f'/outputs_context_explanation_ex_{NUM_EX}_cf_fixed'
	PATH = f'./outputs_{DOMAIN}{SUB_FOLDER}/'

	# Load model configurations
	taskqa_models = config.taskqa_models
	expl_types = config.expl_types
	simqg_models = config.simqg_models
	simqa_models = config.simqa_models
	simqg_mixed = config.MIXED # whether to mix outputs from different simqg models
	context = config.WITH_CONTEXT # context while generating counterfactuals
	explanation = config.EXPLANATION # explanation used in simqa by simulator model
	stratified = config.STRATIFIED # stratified sampling in simqa
	num_strat = config.STRAT_SAMPLES_PER_OPTION # number of stratified samples per option in simqa
	total_strat = config.TOTAL_STRAT_SAMPLES # total number of stratified samples in simqa

	# EX_IDXS = range(0, NUM_EX)  # Example indices

	data = json.load(open('data_bbq.json'))[DOMAIN]
	
	# Group examples by sets of 4
	example_groups = []
	for i in range(0, len(data), 4):
		if i + 3 < len(data):  # Ensure we have complete groups of 4
			group = {
				'test_example': data[i],  # First example (ambiguous)
				'eval_examples': data[i+1:i+4]  # Next 3 examples (disambiguated)
			}
			example_groups.append(group)
	
	example_groups = example_groups[:NUM_EX]

	EX_IDXS = range(0, len(example_groups))

	start_time = time.time()  # Start time for the entire script

	print("\nStarting the pipeline with the following configuration:")
	config.print_configs()
	print(f"Output will be saved in: {PATH}")

	# Create output directory if it doesn't exist
	if not os.path.exists(f'./outputs_{DOMAIN}{SUB_FOLDER}'):
		os.makedirs(f'./outputs_{DOMAIN}{SUB_FOLDER}')

	# Task QA processing
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in expl_types:
			out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl'
			test_inputs = [group['test_example'] for group in example_groups]
			run_task_save_results(task_function=task_qa, out_file=out_file, ex_idxs=EX_IDXS,
								  model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs, domain=DOMAIN)
			print("\033[1mCompleted TaskQA for model:\033[0m", taskqa_model, 
				  "\033[1mexplanation type:\033[0m", taskqa_expl_type)
			f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
			timestamp = time.time()


	# Simulated QA (SimQA)
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in expl_types:
			for simqa_model in simqa_models:
				out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}-simqa_{simqa_model}_{DOMAIN}_{NUM_EX}.pkl'
				if os.path.exists(out_file):
					all_preds = pkl.load(open(out_file, 'rb'))
					continue
				orig_inputs = json.load(open('data_bbq.json'))[DOMAIN]
				orig_tm_preds = pkl.load(open(f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl', 'rb'))
				all_simqa_preds = {}
				for ex_idx in EX_IDXS:
					if ex_idx not in all_simqa_preds:
						group = example_groups[ex_idx]
						orig_pred = orig_tm_preds[ex_idx]
						
						simqa_preds = simulate_qa_direct_examples(
							model=simqa_model,
							orig_input=group['test_example'],
							orig_tm_pred=orig_pred,
							eval_examples=group['eval_examples'],
							domain=DOMAIN
						)
						all_simqa_preds[ex_idx] = simqa_preds

				pkl.dump(all_simqa_preds, open(out_file, 'wb'))
				print("\033[1mCompleted SimQA for TaskQA model:\033[0m", taskqa_model, 
						"\033[1mexplanation type:\033[0m", taskqa_expl_type,
						"\033[1mSimQA model:\033[0m", simqa_model)
				f_log.write(f'SimQA-{taskqa_model}-{taskqa_expl_type}-{simqa_model} {(time.time() - timestamp)//60} minutes\n')
			timestamp = time.time()

	# Task QA on Simulated Inputs
	for taskqa_model in taskqa_models:
		for taskqa_expl_type in expl_types:
			out_file = f'{PATH}taskqa_{taskqa_model}_{taskqa_expl_type}' \
						f'-taskqa_{taskqa_model}_{taskqa_expl_type}_{DOMAIN}_{NUM_EX}.pkl'
			if os.path.exists(out_file):
					all_preds = pkl.load(open(out_file, 'rb'))
					continue
			all_eval_examples = []
			for group in example_groups:
				all_eval_examples.extend(group['eval_examples'])
			all_taskqa_preds = {}
			for ex_idx in EX_IDXS:
				if ex_idx not in all_taskqa_preds:
					group = example_groups[ex_idx]
					eval_preds = task_qa(
						model=taskqa_model,
						expl_type=taskqa_expl_type,
						inputs=group['eval_examples'],
						domain=DOMAIN
					)
					all_taskqa_preds[ex_idx] = eval_preds
			pkl.dump(all_taskqa_preds, open(out_file, 'wb'))
			print("\033[1mCompleted TaskQA on Simulated Inputs for TaskQA model:\033[0m", taskqa_model,
					"\033[1mexplanation type:\033[0m", taskqa_expl_type, 
					)
			f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
			timestamp = time.time()
	
	# Print total time taken for the script
	print(f"\nTotal time taken: {(time.time() - start_time) / 60:.2f} minutes\n")

	# call calculate_precision.py to calculate precision
	print("\033[1mSimulation Precision Results:\033[0m")
	for simqa_model in simqa_models:
		calculate_precision_fixed(domain=DOMAIN, num_ex=NUM_EX, taskqa_models=taskqa_models, simqa_model=simqa_model, path=PATH)

