import json
import random
import sys
from collections import Counter
import numpy as np
import pickle as pkl
from copy import deepcopy

sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task

PROMPT_TASK = 'almanacs-simqa'

def extract_sim_qa_ans(sim_qa_expl):
    """
    Extracts the final answer. Returns 'yes', 'no', or 'neither'.
    """
    response_lower = sim_qa_expl.lower()
    
    # Look for clear "yes" patterns
    if ("so the answer is yes" in response_lower or 
        "therefore, the answer is yes" in response_lower or
        "the answer is yes" in response_lower):
        return "yes"
    
    # Look for clear "no" patterns  
    elif ("so the answer is no" in response_lower or
          "therefore, the answer is no" in response_lower or
          "the answer is no" in response_lower):
        return "no"
    
    # If unclear, return "neither"
    return "neither"
        
def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list, k_shot=3, call_api=None):
    """
    Build prompts with k-shot examples for SimQA.
    """
    assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
    
    print(f"Using {k_shot}-shot prompting for SimQA")
    
    # Import the k-shot function
    from prompts.load_prompt import get_k_shot_prompts_by_task
    
    # Build prompts
    prompts = []
    for orig_input, orig_tm_pred, sim_group in zip(orig_inputs, orig_tm_preds, sim_inputs_list):
        for sim_input in sim_group:
            prompt_data = {
                'context': orig_input['context'],
                'explanation': orig_tm_pred['pred_expl'],
                'sim_qn': sim_input.get('sim_qn', sim_input.get('question', ''))
            }
            
            # Get k-shot prompt
            prompt = get_k_shot_prompts_by_task(PROMPT_TASK, [prompt_data], k_shot=k_shot)[0]
            prompts.append(prompt)
    
    # Rest of your existing code stays exactly the same...
    deduplicated_prompts = list(set(prompts))
    
    pred_expls = call_api(model=model, prompts=deduplicated_prompts,
                                 temperature=0.9, max_tokens=200, stop=None)
    
    assert len(pred_expls) == len(deduplicated_prompts)
    
    prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
    pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
    assert len(pred_expls) == len(prompts)
    
    preds = []
    for pred_expl in pred_expls:
        preds.append({'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl})
    
    example_preds = []
    cur = 0
    for sim_group in sim_inputs_list:
        count = len(sim_group)
        example_preds.append(preds[cur:cur+count])
        cur += count
    assert cur == len(preds)
    return example_preds
