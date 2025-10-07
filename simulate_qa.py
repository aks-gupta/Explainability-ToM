import json
import sys
sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task
from api_client import call_together_api, call_openai_api
from configs import GENERAL_CONFIGS, DATASET, DOMAIN, DATA_FILE
from openai import OpenAI
import openai
import time
import re
import os

def extract_sim_qa_ans(sim_qa_expl):
	# print(f"DEBUG SimQA: Processing response: {sim_qa_expl}")
	cannot_guess = 'I cannot guess' in sim_qa_expl
	pattern_no = r'("?)(?:\bno\b)(?=[\s.,!?;:]|$)\1'
	pattern_yes = r'("?)(?:\byes\b)(?=[\s.,!?;:]|$)\1'

	guess_yes = bool(re.search(pattern_yes, sim_qa_expl, flags=re.IGNORECASE))
	guess_no = bool(re.search(pattern_no, sim_qa_expl, flags=re.IGNORECASE))
	
	if not (cannot_guess + guess_yes + guess_no == 1):
		return 'neither'
	elif cannot_guess:
		return 'unknown'
	elif guess_yes:
		return 'yes'
	elif guess_no:
		return 'no'
	else:
		raise NotImplementedError

def simulate_qa_hiring_decisions(model, orig_inputs, orig_tm_preds, sim_inputs_list, include_expl=True,
				majority_vote=None,
				annotated_examples=None):
	assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
	num_examples = []
	k_shot = GENERAL_CONFIGS['k_shot']

	for cfs in sim_inputs_list:
		num_examples.append(len(cfs['questions']))
	
	if include_expl:
		prompts = get_prompts_by_task(
			f'{DATASET}-{DOMAIN}-simqa-withexpl',
			[
				{
					'orig_qn': orig_input['question'],
					'orig_qa_tm_expl': orig_tm_pred['pred_expl'],
					'sim_qn': sim_input
				}
				for orig_input, orig_tm_pred, sim_inputs in zip(orig_inputs, orig_tm_preds, sim_inputs_list)
				for sim_input in sim_inputs['questions']
			],
			k_shot
		)
	else:
		prompts = get_prompts_by_task(
			f'{DATASET}-{DOMAIN}-simqa-withoutexpl',
			[
				{
					'orig_qn': orig_input['question'],
					'orig_qa_tm_expl': orig_tm_pred['pred_expl'],
					'sim_qn': sim_input
				}
				for orig_input, orig_tm_pred, sim_inputs in zip(orig_inputs, orig_tm_preds, sim_inputs_list)
				for sim_input in sim_inputs['questions']
			],
			k_shot
		)

	# deduplicate the prompts before calling the API to save time
	deduplicated_prompts = list(set(prompts))
	if ('o1-mini' in model) or ('gpt-4.1-mini' in model):
		pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
								bsz=16, num_processes=8,
								temperature=0, max_tokens=200, stop='\n')
	elif ('llama' in model) or ('deepseek' in model):
		pred_expls = call_together_api(model=model, prompts=deduplicated_prompts,
								bsz=16, num_processes=8,
								temperature=0, max_tokens=200, stop='\n')
	assert len(pred_expls) == len(deduplicated_prompts)
	# add duplicate prompts back
	prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
	pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
	assert len(pred_expls) == len(prompts)

	# extract answers
	preds = []
	# print(f"DEBUG SimQA: Processing {len(pred_expls)} responses")
	for i, pred_expl in enumerate(pred_expls):
		# print(f"DEBUG SimQA: Processing response {i}: {pred_expl}")
		extracted_ans = extract_sim_qa_ans(pred_expl)
		preds.append({'pred_ans': extracted_ans, 'pred_expl': pred_expl})
		# print(f"DEBUG SimQA: Final answer for response {i}: {extracted_ans}")

	# regroup preds according to examples (multiple simulation questions correspond to each original question)
	assert len(preds) == len(prompts)
	example_preds = []
	ex_idx=0
	count = 0 
	while ex_idx < len(preds):
		example_preds.append(preds[ex_idx:ex_idx+num_examples[count]])
		ex_idx+=num_examples[count]
		count+=1
	return example_preds