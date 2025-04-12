import json
import sys
sys.path.append('.')
from prompts.load_prompt import get_prompts_by_task
import openai
import time
import os

client = openai.OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

def call_openai_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=200, stop=None):
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"[{i}] Error during call:\nPrompt: {prompt[:100]}...\nError: {e}")
            responses.append("")
            time.sleep(1)
    return responses

def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list):
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
    assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
    num_examples = len(orig_inputs)
    
    # Create prompts using BBQ-specific format
    prompts = get_prompts_by_task('bbq-simqa-fix',
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
                                   for sim_input in sim_inputs if sim_input is not None])
    
    # Deduplicate the prompts before calling the API to save time
    deduplicated_prompts = list(set(prompts))
    
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