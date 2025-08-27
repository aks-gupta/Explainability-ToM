import json
import sys
sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task
from configs import GENERAL_CONFIGS
from openai import OpenAI
import openai
import time
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
	cannot_guess = 'I cannot guess' in sim_qa_expl
	guess_yes = 'answer is yes' in sim_qa_expl
	guess_no = 'answer is no' in sim_qa_expl
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
	if annotated_examples is None:
		if include_expl:
			k_shot = GENERAL_CONFIGS['k_shot']
			prompts = get_prompts_by_task('almanacs-hiring-decisions-simqa-withexpl',
										  [{'orig_qn': orig_input['question'], 'orig_qa_tm_expl': orig_tm_pred['pred_expl'], 'sim_qn': sim_inputs}
										   for orig_input, orig_tm_pred, sim_inputs in zip(orig_inputs, orig_tm_preds, sim_inputs_list)], k_shot)
			print(prompts)
		else:
			prompts = get_prompts_by_task('almanacs-hiring-decisions-simqa-withoutexpl',
										  [{'orig_qn': orig_input['question'],
											'orig_qa_tm_expl': orig_tm_pred['pred_expl'],
											'sim_qn': sim_inputs}
										   for orig_input, orig_tm_pred, sim_inputs in
										   zip(orig_inputs, orig_tm_preds, sim_inputs_list)], k_shot)
	else:
		assert len(orig_inputs) == len(annotated_examples)
		task_prompt_with_expl = json.load(open('../../prompts/prompts.json'))['strategyqa-simqa-withexpl-mentioncot']
		task_prompt_no_expl = json.load(open('../../prompts/prompts.json'))['strategyqa-simqa-noexpl']
		assert task_prompt_with_expl['instruction'] == task_prompt_no_expl['instruction']
		prompt_prefix = str(task_prompt_with_expl['instruction'])

		dem_examples_prefix = []
		if include_expl:
			for dem in task_prompt_with_expl['dem_examples']:
				dem_examples_prefix.append(task_prompt_with_expl['template_with_label'].format(**dem))
		else:
			for dem in task_prompt_no_expl['dem_examples']:
				dem_examples_prefix.append(task_prompt_no_expl['template_with_label'].format(**dem))

		prompts = []
		for orig_input, orig_tm_pred, sim_inputs, annotated_exs in \
				zip(orig_inputs, orig_tm_preds, sim_inputs_list, annotated_examples):
			# append the annotater-annotated examples
			formatted_output_annotated_exs = []
			for annotated_ex in annotated_exs:
				annotated_ex_copy = {key: annotated_ex[key] for key in annotated_ex if key != 'sim_qa_ans'}
				annotated_ex_copy['sim_qa_ans'] = {'yes': 'The robot will likely answer yes.',
												   'no': 'The robot will likely answer no.',
												   'unknown': "I cannot guess the robot's answer to the follow-up "
															  "question based on its response to the starter question."
												   }[annotated_ex['sim_qa_ans']]
				formatted_output_annotated_exs.append(annotated_ex_copy)
			annotated_exs = formatted_output_annotated_exs
			ex_dem_examples_prefix = dem_examples_prefix + [task_prompt_no_expl['template_with_label'].format(**annotated_ex)
														for annotated_ex in annotated_exs]
			random.shuffle(ex_dem_examples_prefix)
			ex_prompt_prefix = prompt_prefix + ''.join(ex_dem_examples_prefix)
			for sim_input in sim_inputs:
				test_example = {'orig_qn': orig_input['question'], 'orig_qa_tm_expl': orig_tm_pred['pred_expl'],
								'sim_qn': sim_input['question']}
				if include_expl:
					ex_prompt = ex_prompt_prefix + task_prompt_with_expl['template_no_label'].format(**test_example)
				else:
					ex_prompt = ex_prompt_prefix + task_prompt_no_expl['template_no_label'].format(**test_example)
				prompts.append(ex_prompt)

	# deduplicate the prompts before calling the API to save time
	deduplicated_prompts = list(set(prompts))
	# pred_expls = multiprocess_api(model=model, prompts=deduplicated_prompts, bsz=16,
	# 							  num_processes=8,
	# 							  temperature=0, max_tokens=200, stop='\n', n=1)
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
	else:
		preds = []
		for pred_expl_samples in pred_expls:
			ex_preds = [{'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl}
						  for pred_expl in pred_expl_samples]
			ex_pred_answers = [pred['pred_ans'] for pred in ex_preds]
			counter = Counter(ex_pred_answers)
			max_count = np.max([counter[item] for item in counter])
			most_frequent_answers = [ans for ans in counter if counter[ans] == max_count]
			majority_ans = random.sample(most_frequent_answers, 1)[0]
			preds.append({'pred_ans': majority_ans, 'majority_vote_details': ex_preds})

	# regroup preds according to examples (multiple simulation questions correspond to each original question)
	assert len(preds) == len(prompts)
	example_preds = []
	cur = 0
	for ex_idx in range(num_examples):
		example_preds.append(preds[ex_idx])
		cur += 1
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

