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
    Generate simulated follow-up questions (and predicted answer explanations)
    for each example using the 'almanacs-simqg' prompt template.

    This function expects:
      - orig_inputs: a list of original examples, each with a "context" field.
      - orig_tm_preds: a list of TaskQA predictions, each a dict with at least "pred_expl".
    """
    assert len(orig_inputs) == len(orig_tm_preds)
    num_examples = len(orig_inputs)
    # For simqg, we use the "almanacs-simqg" prompt template.
    prompts = get_prompts_by_task(
        'almanacs-simqg',
        [{
            'context': orig_input['context'],
            'explanation': orig_tm_pred['pred_expl']
        } for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds)]
    )
    # Repeat each prompt num_samples times.
    prompts = [prompt for prompt in prompts for _ in range(num_samples)]
    assert len(prompts) == num_examples * num_samples

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
