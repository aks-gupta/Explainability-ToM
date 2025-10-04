'''
Code to generate fixed counterfactuals and label balance them
'''

import json
import configs
from simulate_qg import simulate_qg_hiring_decisions
from prompts.load_prompt import get_prompts_by_task
from api_client import call_together_api, call_openai_api
import pickle
from configs import DOMAIN

dataset = configs.DATASET
domain = configs.DOMAIN
num_examples = configs.GENERAL_CONFIGS['num_examples']
num_counterfactual_qs = configs.GENERAL_CONFIGS['num_counterfactual_qs']
with_context = False
simqg_model = configs.MODEL_CONFIGS['simqg_model']

def get_data():
	data = json.load(open(f'./data/data_{domain}.json'))['test']
	# data = json.load(open(configs.DATA_FILE))
	# combined_data = data['test'] + data['train']
	return data

def simulate_qg(model, orig_inputs, orig_tm_preds, top_p, num_samples, with_context):

	num_examples = len(orig_inputs)
	samples_per_label = num_samples // 2
	
	# Generate prompts for YES counterfactuals
	prompts_yes = get_prompts_by_task(
		f'{dataset}-{domain}-simqg-fixed-counterfactuals-yes', 
		[{'orig_qn': item['question']} for item in data]
	)
	
	# Generate prompts for NO counterfactuals
	prompts_no = get_prompts_by_task(
		f'{dataset}-{domain}-simqg-fixed-counterfactuals-no', 
		[{'orig_qn': item['question']} for item in data]
	)

	# Interleave YES and NO prompts for each question
	# So each question gets samples_per_label YES and samples_per_label NO
	prompts = []
	for i in range(num_examples):
		# Add YES counterfactuals for this question
		for _ in range(samples_per_label):
			prompts.append(prompts_yes[i])
		# Add NO counterfactuals for this question
		for _ in range(samples_per_label):
			prompts.append(prompts_no[i])
	
	assert len(prompts) == num_examples * num_samples

	if ('gpt' in model):
		responses = call_openai_api(model=model, prompts=prompts, temperature=1, top_p=top_p, stop=None)
	elif ('llama' in model):
		responses = call_together_api(model=model, prompts=prompts, temperature=1, top_p=top_p, stop='\n\n')

	sim_inputs = []
	sim_answers = []
 
	for i, response in enumerate(responses):
		if response is None or response == "Error: Unable to generate response":
			print(f"WARNING: Response {i} is None or error, skipping")
			sim_inputs.append(None)
			sim_answers.append(None)
			continue
			
		if model in ['o1-mini-2024-09-12', 'gpt-4.1-mini']:
			response = response.replace("Here is my response.", "")
			answer_marker = "Your Answer to the Follow-up Question:"
			if answer_marker in response:
				parts = response.split(answer_marker)
				sim_input = parts[0].strip()
				sim_input = sim_input.replace("Follow-up Question:", "").strip()
				sim_answer = parts[1].strip() if len(parts) > 1 else None
				if sim_answer:
					sim_answer = sim_answer.replace("The answer is", "").strip()
					sim_answer = sim_answer.replace(".", "").strip()
					if sim_answer not in ['yes', 'no']:
						if 'yes' in sim_answer:
							sim_answer = "yes"
						elif 'no' in sim_answer:
							sim_answer = "no"
						else:
							sim_answer = None
			elif "\n\nThe answer is yes." in response or "\nThe answer is yes." in response:
				for separator in ["\n\nThe answer is yes.", "\nThe answer is yes."]:
					if separator in response:
						sim_input = response.split(separator)[0].strip()
						sim_answer = "yes"
						break
			elif "\n\nThe answer is no." in response or "\nThe answer is no." in response:
				for separator in ["\n\nThe answer is no.", "\nThe answer is no."]:
					if separator in response:
						sim_input = response.split(separator)[0].strip()
						sim_answer = "no"
						break
			else:
				sim_input = response.strip()
				sim_answer = None
			
			sim_inputs.append(sim_input)
			sim_answers.append(sim_answer)
			assert len(sim_inputs) == len(sim_answers)
   
		if model in ['gpt-4o-mini', 'llama-3.3-70B-Instruct-Turbo-Free']:
			lines = response.split("\n")
			sim_input = lines[0].strip()
			sim_answer = lines[-1].strip() if len(lines) > 1 else None

			if sim_input is not None:
				cleaned_input = sim_input.replace("Follow-up Question: ", "")
				cleaned_answer = sim_answer.replace("Your Answer to the Follow-up Question: The answer is ", "") if sim_answer else None
				cleaned_answer = cleaned_answer.replace(".", "") if cleaned_answer else None
				sim_inputs.append(cleaned_input)
				sim_answers.append(cleaned_answer)
				assert len(sim_inputs) == len(sim_answers)	
			else:
				sim_inputs.append(sim_input)
				print(f"DEBUG SimQG: Input {i} is None")

	final_inputs = []
	count = 0
	for i in range(num_examples):
		next_count = count+num_samples
		final_inputs.append({
		'qid': i,
  		'question': orig_inputs[i]['question'],
		'counterfactual_questions': sim_inputs[count:next_count],
		'counterfactual_answers': sim_answers[count:next_count]
		})
		count = next_count

	assert len(final_inputs) == len(orig_inputs)
	return final_inputs

def preprocess_label_balanced_counterfactuals(path_to_cfs):
	#Generate file for task model questions
	with open(f"./data/label_balanced_counterfactuals_{DOMAIN}.json", "r") as f:
		data = json.load(f)

	# === Step 2: Prepare outputs ===
	original_questions = {}
	counterfactual_questions = {}
	original_questions = [{"question": value["question"]} for key, value in data.items()]
	for key, value in data.items():
		key_int = int(key)  # ensure integer keys
		# Extract only the counterfactual questions
		counterfactual_questions[key_int] = {"questions": value["counterfactual_questions"]}

	# === Step 3: Save both new PKL files ===
	with open(f"./data/label_balanced_original_questions_{DOMAIN}.json", "w") as f:
		json.dump(original_questions, f)

	with open(f"./data/label_balanced_counterfactuals_{DOMAIN}.pkl", "wb") as f:
		pickle.dump(counterfactual_questions, f)

if __name__ == "__main__":
	data = get_data()[:num_examples]
	print(f'Generating fixed {num_counterfactual_qs} counterfactuals for {len(data)} examples from {domain} domain')
	try:
		counterfactuals = json.load(open(f'./data/fixed_counterfactuals_{domain}.json'))
		print(f"Fixed counterfactuals already exist for {domain} domain. Loaded from file.")
	except FileNotFoundError:
		print(f"No existing fixed counterfactuals found for {domain} domain. Generating new ones.")
		counterfactuals = simulate_qg(
			model=simqg_model[0],
			orig_inputs=data,
			orig_tm_preds=[{'pred_expl': ''}]*len(data), # Dummy explanations
			top_p=1.0,
			num_samples=num_counterfactual_qs,
			with_context=with_context
		)
		with open(f'./data/fixed_counterfactuals_{domain}.json', 'w') as f:
			json.dump(counterfactuals, f, indent=4)


	counterfactuals = json.load(open(f'./data/fixed_counterfactuals_{domain}.json'))
 
	try:
		balanced_counterfactuals = json.load(open(f'./data/label_balanced_counterfactuals_{domain}.json'))
		print(f"Label balanced counterfactuals already exist for {domain} domain. Loaded from file.")
		exit(0)
	except FileNotFoundError:
		print(f"No existing label balanced counterfactuals found for {domain} domain. Generating new ones.")
		counterfactuals_yes = []
		counterfactuals_no = []

		for item in counterfactuals:
			for q, a in zip(item['counterfactual_questions'], item['counterfactual_answers']):
				if a and a.lower() in 'yes':
					counterfactuals_yes.append({'question': q, 'answer': 'YES', 'orig_question': item['question']})
				elif a and a.lower() in 'no':
					counterfactuals_no.append({'question': q, 'answer': 'NO', 'orig_question': item['question']})

		print(f"YES counterfactuals: {len(counterfactuals_yes)}, NO counterfactuals: {len(counterfactuals_no)}")
		min_balanced = min(len(counterfactuals_yes), len(counterfactuals_no))
		print(f"Balancing to {min_balanced} counterfactuals each")
	
		count_yes = 0
		count_no = 0
		balanced_counterfactuals = {}
		for item in counterfactuals:
			qid = item['qid']
			if qid not in balanced_counterfactuals:
				balanced_counterfactuals[qid] = {
					'question': item['question'],
					'counterfactual_questions': [],
					'counterfactual_answers': []
				}
			for q, a in zip(item['counterfactual_questions'], item['counterfactual_answers']):
				if a and 'yes' in a.lower() and count_yes < min_balanced:
					count_yes += 1
					balanced_counterfactuals[qid]['counterfactual_questions'].append(q)
					balanced_counterfactuals[qid]['counterfactual_answers'].append('yes')
				elif a and 'no' in a.lower() and count_no < min_balanced:
					count_no += 1
					balanced_counterfactuals[qid]['counterfactual_questions'].append(q)
					balanced_counterfactuals[qid]['counterfactual_answers'].append('no')

		balanced_counterfactuals = {k: v for k, v in balanced_counterfactuals.items() 
							if len(v['counterfactual_questions']) > 0}

		with open(f'./data/label_balanced_counterfactuals_{domain}.json', 'w') as f:
			json.dump(balanced_counterfactuals, f, indent=4)
