import json
import sys
sys.path.append('..')
# from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
from copy import deepcopy
import random
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

def simulate_qg(model, orig_inputs, orig_tm_preds, top_p, num_samples, with_context, balance_labels=False):
    """
    Generate simulated follow-up questions (and predicted answer explanations)
    for each example using the 'almanacs-simqg' prompt template.

    This function expects:
      - orig_inputs: a list of original examples, each with a "context" field.
      - orig_tm_preds: a list of TaskQA predictions, each a dict with at least "pred_expl".
    """
    assert len(orig_inputs) == len(orig_tm_preds)
    num_examples = len(orig_inputs)
    # For simqg, we use the "almanacs-simqg" prompt template.
    # Build prompts based on balancing approach
    if balance_labels:
        print("Using explicit balanced generation approach")
        # Split samples between yes and no targets
        yes_samples = num_samples // 2
        no_samples = num_samples - yes_samples
        print(f"Generating {yes_samples} 'yes' + {no_samples} 'no' questions per example")
        
        all_prompts = []
        for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds):
            base_data = {
                'context': orig_input['context'],
                'explanation': orig_tm_pred['pred_expl']
            }
            base_prompt = get_prompts_by_task('almanacs-simqg-new', [base_data])[0]
            
            # Generate prompts targeting "yes" answers
            for _ in range(yes_samples):
                yes_instruction = "\n\nSPECIFIC INSTRUCTION: Create a follow-up question where you predict the robot will answer 'YES'."
                all_prompts.append(base_prompt + yes_instruction)
            
            # Generate prompts targeting "no" answers  
            for _ in range(no_samples):
                no_instruction = "\n\nSPECIFIC INSTRUCTION: Create a follow-up question where you predict the robot will answer 'NO'."
                all_prompts.append(base_prompt + no_instruction)
        
    else:
        # Original approach - repeat each prompt num_samples times
        base_prompts = get_prompts_by_task(
            'almanacs-simqg-new',
            [{
                'context': orig_input['context'],
                'explanation': orig_tm_pred['pred_expl']
            } for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds)]
        )
        all_prompts = [prompt for prompt in base_prompts for _ in range(num_samples)]
    
    assert len(all_prompts) == num_examples * num_samples

    responses = call_openai_api(
        model=model,
        prompts=all_prompts,
        bsz=8,
        num_processes=12,
        temperature=1,
        top_p=top_p,
        max_tokens=512
    )
    assert len(responses) == num_examples * num_samples

    # Parsing generated responses.
    # We now expect the response to contain the marker:
    # "Your guess of Robot's Answer to the Follow-up Question:"
    # even if "Assistant: here is my response." is absent.
    sim_inputs = []
    guess_marker = "Your guess of Robot's Answer to the Follow-up Question:"
    for response in responses:
        response = response.strip()
        if guess_marker in response:
            parts = response.split(guess_marker, maxsplit=1)
            sim_qn = parts[0].strip()  # The follow-up question should be the text before the guess marker.
            sim_qa_expl = parts[1].strip()  # The guessed explanation is the text after the guess marker.
        else:
            # If the marker is not found, treat the whole response as the follow-up question.
            sim_qn = response
            sim_qa_expl = ""
        sim_inputs.append({'sim_qn': sim_qn, 'sim_qa_expl': sim_qa_expl})

    # Group the simulated outputs by examples.
    assert len(sim_inputs) == num_examples * num_samples
    example_siminputs = []
    for ex_idx in range(num_examples):
        group = [sim_input for sim_input in sim_inputs[ex_idx * num_samples: (ex_idx + 1) * num_samples]
                 if sim_input is not None]
        example_siminputs.append(group)
    assert len(example_siminputs) == num_examples
    return example_siminputs

def _check_two_dict_same(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if dict1[key] != dict2[key]:
            return False
    return True

def mix_sim_inputs(model1_siminputs, model2_siminputs, sample_num):
    mixed_samples = []
    model1_siminputs, model2_siminputs = deepcopy(model1_siminputs), deepcopy(model2_siminputs)
    for sample_idx in range(sample_num):
        add_sample = None
        if len(model1_siminputs) == 0 and len(model2_siminputs) == 0:
            return mixed_samples
        elif len(model1_siminputs) > 0 and (sample_idx % 2 == 0 or len(model2_siminputs) == 0):
            add_sample = random.sample(model1_siminputs, 1)[0]
            mixed_samples.append(add_sample)
        else:
            add_sample = random.sample(model2_siminputs, 1)[0]
            mixed_samples.append(add_sample)
        # Remove duplicates from both lists.
        model1_siminputs = [ex for ex in model1_siminputs if not _check_two_dict_same(ex, add_sample)]
        model2_siminputs = [ex for ex in model2_siminputs if not _check_two_dict_same(ex, add_sample)]
    return mixed_samples

def check_simqg_balance(simqg_file):
    """Check balance of generated questions from SimQG output."""
    import pickle as pkl
    from collections import Counter
    
    try:
        simqg_outputs = pkl.load(open(simqg_file, 'rb'))
        all_predictions = []
        
        # Extract the robot's predicted answers from sim_qa_expl
        for ex_idx, ex_outputs in simqg_outputs.items():
            for output in ex_outputs:
                sim_qa_expl = output.get('sim_qa_expl', '').lower()
                
                # Look for various patterns indicating yes/no predictions
                if any(phrase in sim_qa_expl for phrase in ['answer yes', 'say yes', 'respond yes', 'likely yes']):
                    all_predictions.append('yes')
                elif any(phrase in sim_qa_expl for phrase in ['answer no', 'say no', 'respond no', 'likely no']):
                    all_predictions.append('no')
                elif 'so the robot will likely answer' in sim_qa_expl:
                    # Extract the specific answer after this phrase
                    after_phrase = sim_qa_expl.split('so the robot will likely answer')[1]
                    if 'yes' in after_phrase[:10]:  # Look in first 10 chars after phrase
                        all_predictions.append('yes')
                    elif 'no' in after_phrase[:10]:
                        all_predictions.append('no')
        
        counter = Counter(all_predictions)
        total = len(all_predictions)
        
        if total > 0:
            yes_count = counter.get('yes', 0)
            no_count = counter.get('no', 0)
            yes_pct = (yes_count / total) * 100
            
            print(f"Generated Question Balance:")
            print(f"  Total predictions found: {total}")
            print(f"  Predicted Yes: {yes_count} ({yes_pct:.1f}%)")
            print(f"  Predicted No:  {no_count} ({100-yes_pct:.1f}%)")
            print(f"  Balance: {'GOOD' if abs(yes_pct - 50) < 10 else 'NEEDS IMPROVEMENT'}")
        else:
            print("No valid predictions found in generated questions!")
        
    except FileNotFoundError:
        print(f"Balance check: File {simqg_file} not found yet")