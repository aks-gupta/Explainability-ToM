'''
Code to generate fixed counterfactuals and label balance them
'''

import json
import configs
from simulate_qg import simulate_qg_hiring_decisions
from prompts.load_prompt import get_prompts_by_task
from api_client import call_together_api, call_openai_api

dataset = configs.DATASET
domain = configs.DOMAIN
num_examples = configs.GENERAL_CONFIGS['num_examples']
num_counterfactual_qs = configs.GENERAL_CONFIGS['num_counterfactual_qs']
with_context = False
simqg_model = configs.MODEL_CONFIGS['simqg_model']

def get_data():
    data = json.load(open(f'./data/data_{domain}.json'))['test']
    return data

def simulate_qg_hiring_decisions(model, orig_inputs, orig_tm_preds, top_p, num_samples, with_context):
	
	prompts = get_prompts_by_task(
		f'{dataset}-{domain}-simqg-fixed-counterfactuals', 
        [{'orig_qn': item['question']} for item in data]
	)

	num_examples = len(orig_inputs)
	
	expanded = []
	for prompt in prompts:
		for _ in range(num_samples):
			expanded.append(prompt)
	
	prompts = expanded
	assert len(prompts) == num_examples * num_samples
        
	if ('gpt' in model):
		responses = call_openai_api(model=model, prompts=prompts, temperature=1, top_p=top_p, stop=None)
	elif ('llama' in model):
		responses = call_together_api(model=model, prompts=prompts, temperature=1, top_p=top_p, stop='\n\n')
	
	sim_inputs = []
	sim_answers = []
	for i, response in enumerate(responses):
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
			'question': orig_inputs[i]['question'],
			'counterfactual_questions': sim_inputs[count:next_count],
			'counterfactual_answers': sim_answers[count:next_count]
		})
		count = next_count
	
	assert len(final_inputs) == len(orig_inputs)
	return final_inputs

if __name__ == "__main__":
    data = get_data()[:num_examples]
    print(f'Generating fixed {num_counterfactual_qs} counterfactuals for {len(data)} examples from {domain} domain')
    
    counterfactuals = json.load(open(f'./data/fixed_counterfactuals_{domain}.json'))
    if counterfactuals:
        print(f"Fixed counterfactuals already exist for {domain} domain. Loaded from file.")
        
    # Generate counterfactuals
    else:
        counterfactuals = simulate_qg_hiring_decisions(
			model=simqg_model[0],
			orig_inputs=data,
			orig_tm_preds=[{'pred_expl': ''}]*len(data), # Dummy explanations
			top_p=1.0,
			num_samples=num_counterfactual_qs,
			with_context=with_context
		)
	
    # print("-" * 60)
    # print(f"Generated counterfactuals: \n{counterfactuals}")
    
    # store counterfactuals in json file and load again for balancing
    
    with open(f'./data/fixed_counterfactuals_{domain}.json', 'w') as f:
        json.dump(counterfactuals, f, indent=4)
  
    counterfactuals = json.load(open(f'./data/fixed_counterfactuals_{domain}.json'))
    
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
    
    balanced_counterfactuals = {'yes': counterfactuals_yes[:min_balanced], 'no': counterfactuals_no[:min_balanced]}
    
    with open(f'./data/label_balanced_counterfactuals_{domain}.json', 'w') as f:
        json.dump(balanced_counterfactuals, f, indent=4)

	# balance the counterfactuals