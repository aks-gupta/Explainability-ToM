
import json
import time
import os
import pickle as pkl
from task_qa import task_qa, task_qa_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs
from simulate_qa import simulate_qa
import openai
from config import MODELS, EXPLANATION_TYPES, CUE_TYPE, EXAMPLE_RANGE, SIMQG_PARAMS, SIMQA_PARAMS, DATA_PATH, LOG_FILE, MIX_ENABLED
from config import RUN_DIR, TASKQA_PATH, SIMQG_PATH, SIMQG_MIX_PATH, SIMQA_PATH, TASKQA_ON_SIM_PATH

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

NUM_EXAMPLES = len(EXAMPLE_RANGE)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def build_call_api():
    api_key = os.environ.get("LITELLM_API_KEY")
    client = openai.OpenAI(api_key=api_key, base_url="https://cmu.litellm.ai")
    def call_api(model, prompts, temperature=0, top_p=1.0, max_tokens=200, stop=None):
        responses = []
        for prompt in prompts:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop
                )
                responses.append(r.choices[0].message.content)
            except Exception as e:
                print(f"Error calling API: {e}")
                responses.append("")
        return responses
    return call_api

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
    
    out_dir = os.path.dirname(out_file)
    ensure_output_dir(out_dir)
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

def run_taskqa_stage(test_inputs, log_file, call_api):
    """Run TaskQA stage for all configurations."""
    print("\n" + "="*50)
    print("STAGE 1: TaskQA - Initial Question Answering")
    print("="*50)
    
    for taskqa_model in [MODELS['TASKQA']]:
        for expl_type in EXPLANATION_TYPES:
            stage_start = time.time()
            
            out_file = TASKQA_PATH
            
            run_task_save_results(
                task_function=task_qa,
                out_file=out_file,
                ex_idxs=EXAMPLE_RANGE,
                model=taskqa_model,
                expl_type=expl_type,
                inputs=test_inputs,
                cue=CUE_TYPE,
                call_api=call_api
            )
            
            log_timing(log_file, f'TaskQA-{taskqa_model}-{expl_type}', stage_start)

def run_simqg_stage(test_inputs, log_file, call_api):
    """Run SimQG stage - generate counterfactual questions."""
    print("\n" + "="*50)
    print("STAGE 2: SimQG - Counterfactual Question Generation")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    simqg_model = MODELS['SIMQG']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load TaskQA predictions
        taskqa_file = TASKQA_PATH
        orig_tm_preds = pkl.load(open(taskqa_file, 'rb'))
        
        out_file = SIMQG_PATH
        
        run_task_save_results(
            task_function=simulate_qg,
            out_file=out_file,
            ex_idxs=EXAMPLE_RANGE,
            model=simqg_model,
            orig_inputs=test_inputs,
            orig_tm_preds=orig_tm_preds,
            top_p=SIMQG_PARAMS['top_p'],
            num_samples=SIMQG_PARAMS['num_samples'],
            balance_labels=SIMQG_PARAMS['balance_labels'],
            call_api=call_api
        )
        
        
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
        simqg_file = SIMQG_PATH
        simqg_outputs = pkl.load(open(simqg_file, 'rb'))
        
        out_file = SIMQG_MIX_PATH
        
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

def run_simqa_stage(test_inputs, log_file, call_api):
    """Run SimQA stage - answer counterfactual questions."""
    print("\n" + "="*50)
    print("STAGE 4: SimQA - Answer Counterfactual Questions")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    simqa_model = MODELS['SIMQA']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load required files
        taskqa_file = TASKQA_PATH
        orig_tm_preds = pkl.load(open(taskqa_file, 'rb'))
        
        if MIX_ENABLED:
            sim_inputs_list = pkl.load(open(SIMQG_MIX_PATH, 'rb'))
        else:
            sim_inputs_list = pkl.load(open(SIMQG_PATH, 'rb'))
        
        out_file = SIMQA_PATH
        
        run_task_save_results(
            task_function=simulate_qa,
            out_file=out_file,
            ex_idxs=EXAMPLE_RANGE,
            model=simqa_model,
            orig_inputs=test_inputs,
            orig_tm_preds=orig_tm_preds,
            sim_inputs_list=sim_inputs_list,
            k_shot=SIMQA_PARAMS['k_shot'],
            call_api=call_api
        )
        
        log_timing(log_file, f'SimQA-{taskqa_model}-{expl_type}-{simqa_model}', stage_start)

def run_taskqa_on_sim_stage(log_file, call_api):
    """Run TaskQA on simulated inputs for comparison."""
    print("\n" + "="*50)
    print("STAGE 5: TaskQA on Simulated Inputs")
    print("="*50)
    
    taskqa_model = MODELS['TASKQA']
    
    for expl_type in EXPLANATION_TYPES:
        stage_start = time.time()
        
        # Load simulated inputs
        if MIX_ENABLED:
            sim_inputs_list = pkl.load(open(SIMQG_MIX_PATH, 'rb'))
        else:
            sim_inputs_list = pkl.load(open(SIMQG_PATH, 'rb'))
        
        out_file = TASKQA_ON_SIM_PATH
        
        run_task_save_results(
            task_function=task_qa_sim_inputs_list,
            out_file=out_file,
            ex_idxs=EXAMPLE_RANGE,
            model=taskqa_model,
            expl_type=expl_type,
            sim_inputs_list=sim_inputs_list,
            cue=CUE_TYPE,
            call_api=call_api
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
    ensure_output_dir(RUN_DIR)
    
    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f'Pipeline started at {time.ctime()}\n')
        f.write(f'Configuration: {MODELS}\n')
        f.write('='*50 + '\n')
    
    # Load test data
    test_inputs = json.load(open(DATA_PATH))['test'][:NUM_EXAMPLES]
    print(f"Loaded {len(test_inputs)} test examples")
    for input in test_inputs:
        print(input['context'])
        print('-'*50)
    
    pipeline_start = time.time()
    
    try:
        call_api = build_call_api()
        # Run all pipeline stages
        run_taskqa_stage(test_inputs, LOG_FILE, call_api)
        run_simqg_stage(test_inputs, LOG_FILE, call_api)
        if MIX_ENABLED:
            run_simqg_mixing_stage(LOG_FILE)
        run_simqa_stage(test_inputs, LOG_FILE, call_api)
        run_taskqa_on_sim_stage(LOG_FILE, call_api)
        
        # Log completion
        total_time = (time.time() - pipeline_start) // 60
        with open(LOG_FILE, 'a') as f:
            f.write(f'\nPipeline completed successfully in {total_time} minutes\n')
        
        print(f"\n{'='*50}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time} minutes")
        print(f"Results saved in: {RUN_DIR}")
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