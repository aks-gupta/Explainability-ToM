import json
import random
import sys
from collections import Counter
import numpy as np
import pickle as pkl
from copy import deepcopy

sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task

def extract_sim_qa_ans(sim_qa_expl, include_expl):
    """
    Extracts the final answer by searching for the phrase "So the answer is"
    and then taking the following token. If the token is "yes" or "no" (ignoring punctuation
    and case), it returns that token. Otherwise, it returns 'neither'.
    """
    if include_expl:
        marker = "So the answer is"
        if marker in sim_qa_expl:
            tail = sim_qa_expl.split(marker, 1)[1].strip()
            if tail:
                token = tail.split()[0].strip(".,").lower()
                if token in ["yes", "no"]:
                    return token
    else:
        marker = "My Answer:"
        if marker in sim_qa_expl:
            tail = sim_qa_expl.split(marker, 1)[1].strip()
            if tail:
                token = tail.split()[0].strip(".,").lower()
                if token in ["yes", "no"]:
                    return token
                    
    return "neither"
        
def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list, k_shot=3, include_expl=True, majority_vote=None, call_api=None):
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
            
            # Choose prompt template
            prompt_task = 'almanacs-simqa-withexpl-new' if include_expl else 'almanacs-simqa-noexpl'
            
            # Get k-shot prompt
            prompt = get_k_shot_prompts_by_task(prompt_task, [prompt_data], k_shot=k_shot)[0]
            prompts.append(prompt)
    
    # Rest of your existing code stays exactly the same...
    deduplicated_prompts = list(set(prompts))
    
    if majority_vote is None or majority_vote == 1:
        pred_expls = call_api(model=model, prompts=deduplicated_prompts,
                                     temperature=0.7, max_tokens=200, stop=None)
    else:
        pred_expls = call_api(model=model, prompts=deduplicated_prompts,
                                     temperature=1, max_tokens=200, stop=None)
    
    assert len(pred_expls) == len(deduplicated_prompts)
    
    prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
    pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
    assert len(pred_expls) == len(prompts)
    
    if majority_vote is None or majority_vote == 1:
        preds = []
        for pred_expl in pred_expls:
            preds.append({'pred_ans': extract_sim_qa_ans(pred_expl, include_expl), 'pred_expl': pred_expl})
    else:
        preds = []
        for pred_expl_samples in pred_expls:
            ex_preds = [{'pred_ans': extract_sim_qa_ans(pred_expl, include_expl), 'pred_expl': pred_expl}
                        for pred_expl in pred_expl_samples]
            ex_pred_answers = [pred['pred_ans'] for pred in ex_preds]
            counter = Counter(ex_pred_answers)
            max_count = np.max([counter[item] for item in counter])
            most_frequent_answers = [ans for ans in counter if counter[ans] == max_count]
            majority_ans = random.sample(most_frequent_answers, 1)[0]
            preds.append({'pred_ans': majority_ans, 'majority_vote_details': ex_preds})
    
    example_preds = []
    cur = 0
    for sim_group in sim_inputs_list:
        count = len(sim_group)
        example_preds.append(preds[cur:cur+count])
        cur += count
    assert cur == len(preds)
    return example_preds
