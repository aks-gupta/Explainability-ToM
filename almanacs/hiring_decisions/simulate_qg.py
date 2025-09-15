import json
import sys
sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task
from copy import deepcopy
import random
import openai
from openai import OpenAI
import time
import os

client = OpenAI(
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

def simulate_qg_hiring_decisions(model, orig_inputs, orig_tm_preds, top_p, num_samples, with_context):
	assert len(orig_inputs) == len(orig_tm_preds)
	num_examples = len(orig_inputs)
	prompts = get_prompts_by_task(f'almanacs-hiring-decisions-simqg-{with_context}-label-balanced', #add -label-balanced for balanced labels
								  [{'orig_qn': orig_input['question'], 'orig_qa_tm_expl': orig_tm_pred['pred_expl']}
								   for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds)])
	# repeat the prompts for self.num_samples times
	prompts = [prompt for prompt in prompts for _ in range(num_samples)]
	responses = call_openai_api(model=model, prompts=prompts, temperature=1, top_p=top_p, stop='\n\n')

	sim_inputs = []
	for response in responses:
		lines = response.split("\n")
		sim_input = lines[0].strip()
		if sim_input is not None:
			sim_inputs.append(sim_input.replace("Follow-up Question: ", ""))
		else:
			sim_inputs.append(sim_input)

	final_inputs = []
	count = 0
	for _ in range(len(orig_inputs)):
		next_count = count+num_samples
		final_inputs.append({'questions': sim_inputs[count : next_count]})
		count = next_count
	return final_inputs


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
			assert len(model2_siminputs) > 0 and (sample_idx % 2 == 1 or len(model1_siminputs) == 0)
			add_sample = random.sample(model2_siminputs, 1)[0]
			mixed_samples.append(add_sample)
		# remove duplicates
		model1_siminputs = [ex for ex in model1_siminputs if not _check_two_dict_same(ex, add_sample)]
		model2_siminputs = [ex for ex in model2_siminputs if not _check_two_dict_same(ex, add_sample)]
	return mixed_samples


