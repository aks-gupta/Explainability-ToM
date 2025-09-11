# from task_qa import task_qa, task_qa_sim_inputs_list
# from simulate_qg import simulate_qg, mix_sim_inputs
# from simulate_qa import simulate_qa
# import json
# import time
# import os
# import pickle as pkl
# from tqdm import trange

# cue = 'nontoxic'

# def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
# 	print("Inside run_task_save_results")
# 	print(task_function, out_file, ex_idxs)
# 	all_preds = {}
# 	if os.path.exists(out_file):
# 		all_preds = pkl.load(open(out_file, 'rb'))
# 	ex_idxs = [ex_idx for ex_idx in ex_idxs if ex_idx not in all_preds]
# 	for key in kwargs:
# 		if type(kwargs[key]) == list or type(kwargs[key]) == dict:
# 			kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
# 	preds = task_function(**kwargs)
# 	print("task function executed")
# 	assert type(preds) == list and len(preds) == len(ex_idxs)
# 	for pos, ex_idx in enumerate(ex_idxs):
# 		all_preds[ex_idx] = preds[pos]
# 	assert out_file.endswith('_50.pkl')
# 	pkl.dump(all_preds, open(out_file, 'wb'))


# if __name__ == '__main__':
# 	f_log = open('log.txt', 'w')
# 	timestamp = time.time()

# 	EX_IDXS = range(30)

# 	# TaskQA
# 	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
# 	for taskqa_model in ['gpt-4o']:
# 		print("LINE 36")
# 		test_inputs = json.load(open('./data/almanacs_harmful_requests.json'))['test']
# 		for taskqa_expl_type in ['cot']:
# 			print(taskqa_expl_type)
# 			out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}_{cue}_test_50.pkl'
# 			print(out_file)
# 			run_task_save_results(task_function=task_qa, out_file=out_file, ex_idxs=EX_IDXS,
# 									model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
# 			f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
# 			timestamp = time.time()

# 	# SimQG
# 	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
# 	for taskqa_model in ['gpt-4o']:
# 		print("LINE 48")
# 		for taskqa_expl_type in ['cot']:
# 			# for simqg_model in ['gpt-4o', 'gpt-4o-mini']:
# 			for simqg_model in ['gpt-4o']:
# 				for with_context in [True]:
# 					for top_p in [1.0]:
# 						out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{cue}_test_50.pkl'
# 						orig_inputs = json.load(open('./data/almanacs_harmful_requests.json'))['test']
# 						orig_tm_preds = pkl.load(open(f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}_{cue}_test_50.pkl', 'rb'))
# 						run_task_save_results(task_function=simulate_qg, ex_idxs=EX_IDXS, out_file=out_file,
# 												model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
# 												top_p=top_p, num_samples=6, with_context=with_context)
# 						f_log.write(f'SimQG-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
# 						timestamp = time.time()
	
# 	# mix GPT-3 and GPT-4 outputs
# 	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
# 	for taskqa_model in ['gpt-4o']:
# 		print("LINE 66")
# 		for taskqa_expl_type in ['cot']:
# 			for with_context in [True]:
# 				for top_p in [1.0]:
# 					simqg_model2sim_inputs = {}
# 					# for simqg_model in ['gpt-4o', 'gpt-4o-mini']:
# 					for simqg_model in ['gpt-4o']:
# 						simqg_model2sim_inputs[simqg_model] = pkl.load(
# 							open(f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{cue}_test_50.pkl', 'rb'))
# 						print(f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{cue}_test_50.pkl')
# 					out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_mix_{top_p}_{with_context}_{cue}_test_50.pkl'
# 					if os.path.exists(out_file):
# 						ex_idx2mixed_sim_inputs = pkl.load(open(out_file, 'rb'))
# 					else:
# 						ex_idx2mixed_sim_inputs = {}
# 					for ex_idx in EX_IDXS:
# 						if ex_idx not in ex_idx2mixed_sim_inputs: # should not re-run for already computed mix ones because this process is random!
# 							ex_idx2mixed_sim_inputs[ex_idx] = mix_sim_inputs(simqg_model2sim_inputs['gpt-4o'][ex_idx],
# 																				simqg_model2sim_inputs['gpt-4o'][ex_idx],
# 																				sample_num=6)
# 					pkl.dump(ex_idx2mixed_sim_inputs, open(out_file, 'wb'))

# 	# SimQA
# 	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
# 	for taskqa_model in ['gpt-4o']:
# 		print("LINE 90")
# 		for taskqa_expl_type in ['cot']:
# 			for simqg_model in ['mix']: # expl
# 				for with_context in [True]:
# 					for top_p in [1.0]:
# 						# for simqa_model in ['gpt-4o-mini', 'gpt-4o']:
# 						for simqa_model in ['gpt-4o']:
# 							out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}_fix_{cue}_test_50.pkl' #noexpl
# 							orig_inputs = json.load(open('./data/almanacs_harmful_requests.json'))['test']
# 							orig_tm_preds = pkl.load(open(f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}_{cue}_test_50.pkl', 'rb'))
# 							sim_inputs_list = pkl.load(open(
# 								f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{cue}_test_50.pkl', 'rb'))
# 							run_task_save_results(task_function=simulate_qa, ex_idxs=EX_IDXS, out_file=out_file,
# 												model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
# 												sim_inputs_list=sim_inputs_list)
# 							f_log.write(f'SimQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context}-{simqa_model} {(time.time() - timestamp)//60} minutes\n')
# 						timestamp = time.time()

# 	# TaskQA on SimInputs
# 	# for taskqa_model in ['gpt-4o', 'gpt-4o-mini']:
# 	for taskqa_model in ['gpt-4o']:
# 		for taskqa_expl_type in ['cot']:
# 			for simqg_model in ['mix']:
# 				for with_context in [True]:
# 					for top_p in [1.0]:
# 						out_file = f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}' \
# 									f'-taskqa_{taskqa_model}_{taskqa_expl_type}_{cue}_test_50.pkl' #noexpl
# 						sim_inputs_list = pkl.load(open(
# 							f'./outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{cue}_test_50.pkl', 'rb'))
# 						run_task_save_results(task_function=task_qa_sim_inputs_list, ex_idxs=EX_IDXS, out_file=out_file,
# 												model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
# 						f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
# 						timestamp = time.time()

import json
import time
import os
import pickle as pkl
from tqdm import trange
from task_qa import task_qa, task_qa_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs
from simulate_qa import simulate_qa

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Model configurations
MODELS = {
    'TASKQA': 'gpt-4o-mini',
    'SIMQG': 'gpt-4o-mini', 
    'SIMQA': 'gpt-4o-mini'
}

# Experiment parameters
EXPLANATION_TYPES = ['cot']
CUE_TYPE = 'concise'
EXAMPLE_RANGE = range(30)
SIMQG_PARAMS = {
    'top_p': 1.0,
    'num_samples': 6,
    'with_context': True,
    'balance_labels': False
}
SIMQA_PARAMS = {
    'k_shot': 3,
    'include_expl': True
}

# File paths
DATA_PATH = './data/almanacs_harmful_requests.json'
OUTPUTS_DIR = './outputs/new'
LOG_FILE = 'log.txt'

# Output file naming patterns  
NUM_EXAMPLES = len(EXAMPLE_RANGE)
FILE_PATTERNS = {
    'taskqa': 'taskqa_{model}_{expl_type}_{cue}_test_{num_examples}.pkl',
    'simqg': 'taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{balance}_{cue}_test_{num_examples}.pkl',
    'simqg_mix': 'taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_mix_{top_p}_{with_context}_{balance}_{cue}_test_{num_examples}.pkl',
    'simqa': 'taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{balance}-simqa_{simqa_model}_{k_shot}shot_fix_{cue}_test_{num_examples}.pkl',
    'taskqa_on_sim': 'taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_{balance}-taskqa_{taskqa_model}_{taskqa_expl_type}_{cue}_test_{num_examples}.pkl'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create outputs directory if it doesn't exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

def get_output_path(pattern_key, **kwargs):
    """Generate output file path from pattern and parameters."""
    pattern = FILE_PATTERNS[pattern_key]
    kwargs['num_examples'] = NUM_EXAMPLES
    if 'balance_labels' in kwargs:
        kwargs['balance'] = 'balanced' if kwargs['balance_labels'] else 'unbalanced'
        del kwargs['balance_labels']
    filename = pattern.format(**kwargs)
    return os.path.join(OUTPUTS_DIR, filename)

def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
    """
    Generic function to run a task and save results incrementally.
    
    Args:
        task_function: Function to execute
        out_file: Output file path
        ex_idxs: List of example indices to process
        **kwargs: Additional arguments for task_function
    """
    print(f"Running {task_function.__name__} -> {os.path.basename(out_file)}")
    
    # Load existing results if available
    all_preds = {}
    if os.path.exists(out_file):
        all_preds = pkl.load(open(out_file, 'rb'))
    
    # Filter out already processed examples
    remaining_idxs = [ex_idx for ex_idx in ex_idxs if ex_idx not in all_preds]
    
    if not remaining_idxs:
        print(f"All examples already processed for {os.path.basename(out_file)}")
        return
    
    print(f"Processing {len(remaining_idxs)} remaining examples")
    
    # Filter kwargs for remaining examples only
    filtered_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, dict)):
            if isinstance(value, list):
                filtered_kwargs[key] = [value[ex_idx] for ex_idx in remaining_idxs]
            else:  # dict
                # For dictionaries, we need to convert to list format for the task functions
                # The task functions expect lists, not dictionaries
                filtered_kwargs[key] = [value[ex_idx] for ex_idx in remaining_idxs if ex_idx in value]
        else:
            filtered_kwargs[key] = value
    
    # Run task function
    preds = task_function(**filtered_kwargs)
    
    # Validate and save results
    assert isinstance(preds, list) and len(preds) == len(remaining_idxs), \
        f"Expected {len(remaining_idxs)} predictions, got {len(preds)}"
    
    for pos, ex_idx in enumerate(remaining_idxs):
        all_preds[ex_idx] = preds[pos]
    
    pkl.dump(all_preds, open(out_file, 'wb'))
    print(f"Saved results to {os.path.basename(out_file)}")

def log_timing(log_file, stage_name, start_time):
    """Log timing information for a stage."""
    elapsed_minutes = (time.time() - start_time) // 60
    with open(log_file, 'a') as f:
        f.write(f'{stage_name}: {elapsed_minutes} minutes\n')

# =============================================================================
# PIPELINE STAGES
# =============================================================================

def run_taskqa_stage(test_inputs, log_file):
    """Run TaskQA stage for all configurations."""
    print("\n" + "="*50)
    print("STAGE 1: TaskQA - Initial Question Answering")
    print("="*50)
    
    for taskqa_model in [MODELS['TASKQA']]:
        for expl_type in EXPLANATION_TYPES:
            stage_start = time.time()
            
            out_file = get_output_path('taskqa',
                model=taskqa_model,
                expl_type=expl_type,
                cue=CUE_TYPE
            )
            
            run_task_save_results(
                task_function=task_qa,
                out_file=out_file,
                ex_idxs=EXAMPLE_RANGE,
                model=taskqa_model,
                expl_type=expl_type,
                inputs=test_inputs
            )
            
            log_timing(log_file, f'TaskQA-{taskqa_model}-{expl_type}', stage_start)

def run_simqg_stage(test_inputs, log_file):
    """Run SimQG stage - generate counterfactual questions."""
    print("\n" + "="*50)
    print("STAGE 2: SimQG - Counterfactual Question Generation")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    simqg_model = MODELS['SIMQG']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load TaskQA predictions
        taskqa_file = get_output_path('taskqa',
            model=taskqa_model,
            expl_type=expl_type,
            cue=CUE_TYPE
        )
        orig_tm_preds = pkl.load(open(taskqa_file, 'rb'))
        
        out_file = get_output_path('simqg',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            simqg_model=simqg_model,
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            cue=CUE_TYPE
        )
        
        run_task_save_results(
            task_function=simulate_qg,
            out_file=out_file,
            ex_idxs=EXAMPLE_RANGE,
            model=simqg_model,
            orig_inputs=test_inputs,
            orig_tm_preds=orig_tm_preds,
            top_p=SIMQG_PARAMS['top_p'],
            num_samples=SIMQG_PARAMS['num_samples'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels']
        )
        
         # Check balance of generated questions immediately after SimQG
        if SIMQG_PARAMS['balance_labels']:
            from simulate_qg import check_simqg_balance
            print(f"\nChecking question balance for {os.path.basename(out_file)}:")
            check_simqg_balance(out_file)
        
        log_timing(log_file, f'SimQG-{taskqa_model}-{expl_type}-{simqg_model}', stage_start)

def run_simqg_mixing_stage(log_file):
    """Mix SimQG outputs from different models."""
    print("\n" + "="*50)
    print("STAGE 3: SimQG Mixing - Combine Model Outputs")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    simqg_model = MODELS['SIMQG']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load SimQG outputs
        simqg_file = get_output_path('simqg',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            simqg_model=simqg_model,
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            cue=CUE_TYPE
        )
        simqg_outputs = pkl.load(open(simqg_file, 'rb'))
        
        out_file = get_output_path('simqg_mix',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            cue=CUE_TYPE
        )
        
        # Mix outputs (currently mixing same model with itself)
        if os.path.exists(out_file):
            mixed_outputs = pkl.load(open(out_file, 'rb'))
        else:
            mixed_outputs = {}
        
        for ex_idx in EXAMPLE_RANGE:
            if ex_idx not in mixed_outputs:
                mixed_outputs[ex_idx] = mix_sim_inputs(
                    simqg_outputs[ex_idx],
                    simqg_outputs[ex_idx],  # Currently same model
                    sample_num=SIMQG_PARAMS['num_samples']
                )
        
        pkl.dump(mixed_outputs, open(out_file, 'wb'))
        
        log_timing(log_file, f'SimQG-Mix-{taskqa_model}-{expl_type}', stage_start)

def run_simqa_stage(test_inputs, log_file):
    """Run SimQA stage - answer counterfactual questions."""
    print("\n" + "="*50)
    print("STAGE 4: SimQA - Answer Counterfactual Questions")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    simqa_model = MODELS['SIMQA']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load required files
        taskqa_file = get_output_path('taskqa',
            model=taskqa_model,
            expl_type=expl_type,
            cue=CUE_TYPE
        )
        orig_tm_preds = pkl.load(open(taskqa_file, 'rb'))
        
        simqg_mix_file = get_output_path('simqg_mix',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            cue=CUE_TYPE
        )
        sim_inputs_list = pkl.load(open(simqg_mix_file, 'rb'))
        
        out_file = get_output_path('simqa',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            simqg_model='mix',
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            simqa_model=simqa_model,
            k_shot=SIMQA_PARAMS['k_shot'],
            cue=CUE_TYPE
        )
        
        run_task_save_results(
            task_function=simulate_qa,
            out_file=out_file,
            ex_idxs=EXAMPLE_RANGE,
            model=simqa_model,
            orig_inputs=test_inputs,
            orig_tm_preds=orig_tm_preds,
            sim_inputs_list=sim_inputs_list,
            k_shot=SIMQA_PARAMS['k_shot'],
            include_expl=SIMQA_PARAMS['include_expl']
        )
        
        log_timing(log_file, f'SimQA-{taskqa_model}-{expl_type}-{simqa_model}', stage_start)

def run_taskqa_on_sim_stage(log_file):
    """Run TaskQA on simulated inputs for comparison."""
    print("\n" + "="*50)
    print("STAGE 5: TaskQA on Simulated Inputs")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load simulated inputs
        simqg_mix_file = get_output_path('simqg_mix',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            cue=CUE_TYPE
        )
        sim_inputs_list = pkl.load(open(simqg_mix_file, 'rb'))
        
        out_file = get_output_path('taskqa_on_sim',
            taskqa_model=taskqa_model,
            taskqa_expl_type=expl_type,
            simqg_model='mix',
            top_p=SIMQG_PARAMS['top_p'],
            with_context=SIMQG_PARAMS['with_context'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            cue=CUE_TYPE
        )
        
        run_task_save_results(
            task_function=task_qa_sim_inputs_list,
            out_file=out_file,
            ex_idxs=EXAMPLE_RANGE,
            model=taskqa_model,
            expl_type=expl_type,
            sim_inputs_list=sim_inputs_list
        )
        
        log_timing(log_file, f'TaskQA-on-Sim-{taskqa_model}-{expl_type}', stage_start)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline execution."""
    print("Starting Simulatability Pipeline for ALMAANCS Harmful Requests")
    print(f"Models: TaskQA={MODELS['TASKQA']}, SimQG={MODELS['SIMQG']}, SimQA={MODELS['SIMQA']}")
    print(f"Cue Type: {CUE_TYPE}")
    print(f"Examples: {len(EXAMPLE_RANGE)} ({EXAMPLE_RANGE.start}-{EXAMPLE_RANGE.stop-1})")
    
    # Setup
    ensure_output_dir()
    
    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f'Pipeline started at {time.ctime()}\n')
        f.write(f'Configuration: {MODELS}\n')
        f.write('='*50 + '\n')
    
    # Load test data
    test_inputs = json.load(open(DATA_PATH))['test']
    print(f"Loaded {len(test_inputs)} test examples")
    
    pipeline_start = time.time()
    
    try:
        # Run all pipeline stages
        run_taskqa_stage(test_inputs, LOG_FILE)
        run_simqg_stage(test_inputs, LOG_FILE)
        run_simqg_mixing_stage(LOG_FILE)
        run_simqa_stage(test_inputs, LOG_FILE)
        run_taskqa_on_sim_stage(LOG_FILE)
        
        # Log completion
        total_time = (time.time() - pipeline_start) // 60
        with open(LOG_FILE, 'a') as f:
            f.write(f'\nPipeline completed successfully in {total_time} minutes\n')
        
        print(f"\n{'='*50}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time} minutes")
        print(f"Results saved in: {OUTPUTS_DIR}")
        print(f"Log file: {LOG_FILE}")
        print(f"{'='*50}")
        
    except Exception as e:
        error_msg = f"Pipeline failed with error: {str(e)}"
        print(f"\nERROR: {error_msg}")
        with open(LOG_FILE, 'a') as f:
            f.write(f'\nERROR: {error_msg}\n')
        raise

if __name__ == '__main__':
    main()