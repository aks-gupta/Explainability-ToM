import json
import sys
sys.path.append('.')
from prompts.load_prompt import get_prompts_by_task
import openai
import time
import os
import config
from api_call import call_openai_api
from api_call import call_together_api

def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list, domain):
    """
    Simulates QA for BBQ dataset, predicting which option the robot would choose for simulated inputs
    
    Args:
        model: The model to use for predictions
        orig_inputs: Original BBQ examples
        orig_tm_preds: Model predictions for original examples
        sim_inputs_list: List of lists of simulated examples for each original example
        
    Returns:
        List of lists of predictions for simulated examples
    """
    explanation = config.EXPLANATION
    print(f"\n\033[1mRunning SimQA\033[0m with model={model}, domain={domain} on {len(orig_inputs)} inputs.")
    assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
    num_examples = len(orig_inputs)
    
    # Create prompts using BBQ-specific format
    prompts = get_prompts_by_task(f'bbq-simqa-expl_{domain}' if explanation else f'bbq-simqa_noexpl_{domain}',
                                  [{'starter_context': orig_input['context'],
                                    'starter_question': orig_input['question'],
                                    'starter_options': orig_input['options'],
                                    'starter_preferred_idx_plus_1':
                                        orig_tm_pred['pred_ans'] + 1 if orig_tm_pred['pred_ans'] is not None
                                       else 'None',
                                    'starter_reason': orig_tm_pred['pred_expl'],
                                    'followup_context': sim_input['context'],
                                    'followup_question': sim_input['question'],
                                    'followup_options': sim_input['options']}
                                   for orig_input, orig_tm_pred, sim_inputs in
                                   zip(orig_inputs, orig_tm_preds, sim_inputs_list)
                                   for sim_input in sim_inputs if sim_input is not None], 
                                   k_shot=None)
    
    # Deduplicate the prompts before calling the API to save time
    deduplicated_prompts = list(set(prompts))
    print(f"SIMQA: Total {len(prompts)} prompts, {len(deduplicated_prompts)} unique prompts.")
    
    # Call the API with deduplicated prompts
    pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
                                 bsz=8, num_processes=12,
                                 temperature=0, max_tokens=100, stop='\n\n')
    assert len(pred_expls) == len(deduplicated_prompts)
    
    # Add duplicate prompts back
    prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
    pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
    assert len(pred_expls) == len(prompts)
    
    # Extract answers - modified for BBQ format
    pred_answers = []
    for pred_expl in pred_expls:
        if 'No, I cannot confidently guess' in pred_expl:
            pred_answers.append('unknown')
        elif 'Yes, I can confidently guess' in pred_expl:
            # Extract option number from text like "I would guess that the robot will choose option 2"
            # BBQ has multiple options, not just two
            option_pattern = r"I would guess that the robot will choose option (\d+)"
            import re
            match = re.search(option_pattern, pred_expl)
            
            if match:
                # Convert to 0-indexed
                option_num = int(match.group(1)) - 1
                pred_answers.append(option_num)
            else:
                pred_answers.append('unknown')
        else:
            pred_answers.append('unknown')
    
    assert len(pred_answers) == len(pred_expls)
    
    # Create prediction objects
    preds = [{'pred_ans': pred_ans, 'pred_expl': pred_expl} for pred_ans, pred_expl in zip(pred_answers, pred_expls)]
    
    # Regroup predictions by original examples
    example_preds = []
    cur = 0
    for ex_idx in range(num_examples):
        valid_sims = [sim for sim in sim_inputs_list[ex_idx] if sim is not None]
        example_preds.append(preds[cur: cur + len(valid_sims)])
        cur += len(valid_sims)
    
    assert cur == len(preds)
    return example_preds

def simulate_qa_direct_examples(model, orig_input, orig_tm_pred, eval_examples, domain):
    """
    Simulate QA on pre-existing examples (instead of generated counterfactuals)
    
    Args:
        model: The model to use for predictions
        orig_input: Original ambiguous BBQ example 
        orig_tm_pred: Model prediction for original example
        eval_examples: List of disambiguated examples to evaluate on
        domain: Domain name
        
    Returns:
        List of predictions for evaluation examples
    """
    explanation = config.EXPLANATION
    print(f"\n\033[1mRunning SimQA Direct Examples\033[0m with model={model}, domain={domain} on {len(eval_examples)} examples.")
    
    # Create prompts using the same format as original simulate_qa
    prompts = get_prompts_by_task(f'bbq-simqa-expl_{domain}' if explanation else f'bbq-simqa_noexpl_{domain}',
                                  [{'starter_context': orig_input['context'],
                                    'starter_question': orig_input['question'],
                                    'starter_options': orig_input['options'],
                                    'starter_preferred_idx_plus_1':
                                        orig_tm_pred['pred_ans'] + 1 if orig_tm_pred['pred_ans'] is not None
                                       else 'None',
                                    'starter_reason': orig_tm_pred['pred_expl'],
                                    'followup_context': eval_example['context'],
                                    'followup_question': eval_example['question'],
                                    'followup_options': eval_example['options']}
                                   for eval_example in eval_examples], 
                                   k_shot=None)
    
    # Call the API
    pred_expls = call_openai_api(model=model, prompts=prompts,
                                 bsz=8, num_processes=12,
                                 temperature=0, max_tokens=100, stop='\n\n')
    
    # Extract answers (same logic as original simulate_qa)
    pred_answers = []
    for pred_expl in pred_expls:
        if 'No, I cannot confidently guess' in pred_expl:
            pred_answers.append('unknown')
        elif 'Yes, I can confidently guess' in pred_expl:
            option_pattern = r"I would guess that the robot will choose option (\d+)"
            import re
            match = re.search(option_pattern, pred_expl)
            
            if match:
                option_num = int(match.group(1)) - 1
                pred_answers.append(option_num)
            else:
                pred_answers.append('unknown')
        else:
            pred_answers.append('unknown')
    
    # Create prediction objects
    preds = [{'pred_ans': pred_ans, 'pred_expl': pred_expl} for pred_ans, pred_expl in zip(pred_answers, pred_expls)]
    
    return preds