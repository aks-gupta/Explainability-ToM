import json
import sys
sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task
from configs import GENERAL_CONFIGS
from openai import OpenAI
import openai
import time
import re
import os

client = OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

# client = openai.OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     # base_url="https://cmu.litellm.ai",
# )

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
	pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
								bsz=16, num_processes=8,
								temperature=0, max_tokens=200, stop='\n')
	assert len(pred_expls) == len(deduplicated_prompts)
	# add duplicate prompts back
	prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
	pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
	assert len(pred_expls) == len(prompts)

	# extract answers
	if majority_vote is None or majority_vote == 1:
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

def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list):
	assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
	num_examples = len(orig_inputs)
	prompts = get_prompts_by_task('shp-simqa-fix',
								  [{'starter_context': orig_input['context'],
									'starter_response0': orig_input['options'][0],
									'starter_response1': orig_input['options'][1],
									'starter_preferred_idx_plus_1':
										orig_tm_pred['pred_ans'] + 1 if orig_tm_pred['pred_ans'] is not None
									   else 'Neither',
									'starter_reason': orig_tm_pred['pred_expl'],
									'followup_context': sim_input['context'],
									'followup_response0': sim_input['options'][0],
									'followup_response1': sim_input['options'][1]}
								   for orig_input, orig_tm_pred, sim_inputs in
								   zip(orig_inputs, orig_tm_preds, sim_inputs_list)
								   for sim_input in sim_inputs])
	# deduplicate the prompts before calling the API to save time
	deduplicated_prompts = list(set(prompts))
	pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
								  bsz=8, num_processes=12,
								  temperature=0, max_tokens=100, stop='\n\n')
	assert len(pred_expls) == len(deduplicated_prompts)
	# add duplicate prompts back
	prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
	pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
	assert len(pred_expls) == len(prompts)
	# extract answers
	pred_answers = []
	for pred_expl in pred_expls:
		if 'No, I cannot confidently guess' in pred_expl:
			pred_answers.append('unknown')
		elif 'Yes, I can confidently guess' in pred_expl:
			select0 = "I would guess that the robot will choose Candidate Response 1" in pred_expl
			select1 = "I would guess that the robot will choose Candidate Response 2" in pred_expl
			if select0 == select1:
				pred_answers.append('neither')
			elif select0:
				pred_answers.append(0)
			elif select1:
				pred_answers.append(1)
		else:
			pred_answers.append('neither')
	assert len(pred_answers) == len(pred_expls)
	preds = [{'pred_ans': pred_ans, 'pred_expl': pred_expl} for pred_ans, pred_expl in zip(pred_answers, pred_expls)]
	# regroup preds according to examples (multiple simulation questions correspond to each original question)
	example_preds = []
	cur = 0
	for ex_idx in range(num_examples):
		example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
		cur += len(sim_inputs_list[ex_idx])
	assert cur == len(preds)
	return example_preds

