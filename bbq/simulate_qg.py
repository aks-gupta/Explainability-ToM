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


def simulate_qg(model, orig_inputs, orig_tm_preds, top_p, num_samples, with_context):
    """
    Generate simulated BBQ examples using a language model.
    
    Args:
        model: The model to use for generation
        orig_inputs: Original BBQ examples
        orig_tm_preds: Model predictions for original examples
        top_p: Top-p sampling parameter
        num_samples: Number of samples to generate per example
        with_context: Whether to use context in the simulation
        
    Returns:
        List of lists of simulated examples grouped by original examples
    """
    # Verify inputs
    assert len(orig_inputs) == len(orig_tm_preds)
    num_examples = len(orig_inputs)
    
    # Prepare prompt inputs
    prompt_task = 'bbq-simqg-withcontext' if with_context else 'bbq-simqg-nocontext'
    prompt_inputs = []
    
    for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds):
        # Format the preferred index for the prompt
        if orig_tm_pred['pred_ans'] is not None:
            preferred_idx_plus_1 = str(orig_tm_pred['pred_ans'] + 1)
        else:
            preferred_idx_plus_1 = 'None'
        
        # Create prompt input dict
        prompt_input = {
            'starter_context': orig_input['context'],
            'starter_question': orig_input['question'],
            'starter_options': orig_input['options'],
            'starter_preferred_idx_plus_1': preferred_idx_plus_1,
            'starter_reason': orig_tm_pred['pred_expl'] if 'pred_expl' in orig_tm_pred else ''
        }
        prompt_inputs.append(prompt_input)
    
    # Get prompts
    prompts = get_prompts_by_task(prompt_task, prompt_inputs)
    
    # Repeat prompts for num_samples times
    prompts = [prompt for prompt in prompts for _ in range(num_samples)]
    assert len(prompts) == num_examples * num_samples
    
    # Generate responses
    responses = call_openai_api(
        model=model, 
        prompts=prompts, 
        bsz=8, 
        num_processes=12,
        temperature=1, 
        top_p=top_p, 
        max_tokens=512
    )
    assert len(responses) == num_examples * num_samples
    
    # Parse generated inputs
    sim_inputs = []
    for response in responses:
        response = response.strip()
        
        # Find section markers
        context_start_idx = response.find("Context:")
        question_start_idx = response.find("Question:")
        options_start_idx = response.find("Options:")
        
        # Check if response has the expected format
        if not ((context_start_idx >= 0) and (question_start_idx > context_start_idx) and (options_start_idx > question_start_idx)):
            # Invalid format
            sim_inputs.append(None)
            continue
        
        # Extract sections
        context = response[context_start_idx + len("Context:"):question_start_idx].strip()
        question = response[question_start_idx + len("Question:"):options_start_idx].strip()
        options_text = response[options_start_idx + len("Options:"):].strip()
        
        # Handle case where the next example starts
        if 'Human:' in options_text:
            options_text = options_text[:options_text.index('Human:')].strip()
        
        # Parse options - handle various formats
        options = []
        
        # Try to parse as a JSON list
        if '[' in options_text and ']' in options_text:
            try:
                # Extract the list part
                options_list_text = options_text[options_text.find('['):options_text.find(']')+1]
                # Replace single quotes with double quotes for JSON parsing
                options_list_text = options_list_text.replace("'", "\"")
                options = json.loads(options_list_text)
            except json.JSONDecodeError:
                # Try alternative parsing
                options_parts = options_text.strip('[]').split(',')
                options = [part.strip().strip('"\'') for part in options_parts]
        else:
            # Parse numbered list format: 1. Option 1, 2. Option 2, etc.
            lines = options_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line[0] in ["(", "[", "-", "*"]):
                    # Extract option text after the prefix
                    option_text = line
                    if '.' in line:
                        option_text = line.split('.', 1)[1].strip()
                    elif ')' in line:
                        option_text = line.split(')', 1)[1].strip()
                    elif ']' in line:
                        option_text = line.split(']', 1)[1].strip()
                    
                    options.append(option_text)
        
        # Create simulated example if we have all required parts
        if context and question and len(options) >= 2:
            sim_inputs.append({
                'context': context,
                'question': question,
                'options': options
            })
        else:
            sim_inputs.append(None)
    
    # Group the generated outputs by examples
    example_siminputs = []
    for ex_idx in range(num_examples):
        example_siminputs.append(
            [sim_input for sim_input in sim_inputs[ex_idx * num_samples:(ex_idx + 1) * num_samples]
             if sim_input is not None]
        )
    
    assert len(example_siminputs) == num_examples
    return example_siminputs

def _check_two_dict_same(dict1, dict2):
    """Check if two dictionaries have the same content"""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if dict1[key] != dict2[key]:
            return False
    return True

def mix_sim_inputs(model1_siminputs, model2_siminputs, sample_num):
    """
    Mix simulated inputs from two different models.
    
    Args:
        model1_siminputs: Simulated inputs from model 1
        model2_siminputs: Simulated inputs from model 2
        sample_num: Number of samples to include in the mix
        
    Returns:
        List of mixed simulated inputs
    """
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
            assert len(model2_siminputs) > 0 and (sample_idx % 2 == 1 or len(model1_siminputs) == 0)
            add_sample = random.sample(model2_siminputs, 1)[0]
            mixed_samples.append(add_sample)
        
        # remove duplicates
        model1_siminputs = [ex for ex in model1_siminputs if not _check_two_dict_same(ex, add_sample)]
        model2_siminputs = [ex for ex in model2_siminputs if not _check_two_dict_same(ex, add_sample)]
    
    return mixed_samples
