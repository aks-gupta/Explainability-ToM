import json
import sys
sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task
from api_client import call_together_api, call_openai_api
from configs import GENERAL_CONFIGS
from openai import OpenAI
import openai
import time
import re
import os

def extract_sim_qa_ans(sim_qa_expl):
	print(sim_qa_expl)
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
	num_examples = len(orig_inputs)
	k_shot = GENERAL_CONFIGS['k_shot']
	
	if include_expl:
		prompts = get_prompts_by_task(
			'almanacs-hiring-decisions-simqa-withexpl',
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
			'almanacs-hiring-decisions-simqa-withoutexpl',
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
	if ('gpt' in model):
		pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
								bsz=16, num_processes=8,
								temperature=0, max_tokens=200, stop='\n')
	elif ('llama' in model):
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
	for pred_expl in pred_expls:
		preds.append({'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl})

	# regroup preds according to examples (multiple simulation questions correspond to each original question)
	assert len(preds) == len(prompts)
	example_preds = []
	toAdd = int(len(preds)/num_examples)
	ex_idx=0
	while ex_idx < len(preds):
		example_preds.append(preds[ex_idx:ex_idx+toAdd])
		ex_idx+=toAdd
	return example_preds