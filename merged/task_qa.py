import json
import sys
sys.path.append('.')
# from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
import config
import re

def task_qa(model, expl_type, inputs, domain, call_api=None):
    print(f"\n\033[1mRunning TaskQA\033[0m with model={model}, expl_type={expl_type}, domain={domain} on {len(inputs)} inputs.")
    assert expl_type in ['cot', 'posthoc']
    dataset = config.DATASET
    task_key = f'{dataset}-taskqa-{expl_type}_{domain}'
    prompts = get_prompts_by_task(task_key,
                [{'context': input['context'],
                  'question': input['question'], 
                  'options': input['options']}
                 for input in inputs])
    # for input in inputs:
    #     print(f"Options: {input['options']}")
    deduplicated_prompts = list(set(prompts))
    print(f"TASKQA: Total {len(prompts)} prompts, {len(deduplicated_prompts)} unique prompts.")
    
    # Test API connection
    # resp = call_openai_api("gpt-4o-mini", ["Say Testing OpenAI API Connection!"])
    # print(resp)

    responses = call_api(model=model, prompts=deduplicated_prompts,
                         temperature=0, max_tokens=200, stop='\n\n')
    assert len(responses) == len(deduplicated_prompts)
    prompt2response = {prompt: response for prompt, response in zip(deduplicated_prompts, responses)}
    responses = [prompt2response[prompt] for prompt in prompts]
    assert len(responses) == len(inputs)
    answers = []
    
    if expl_type == 'cot':
        for response in responses:
            # Look for "So the answer is: [number]" pattern
            answer_pattern = re.search(r"So the answer is:\s*(\d+)", response)
            if answer_pattern:
                # Convert to 0-based indexing if needed
                pred_ans = int(answer_pattern.group(1)) - 1
                # Validate the answer is in range
                if 0 <= pred_ans < len(inputs[0]['options']):
                    answers.append({'pred_ans': pred_ans, 'pred_expl': response})
                else:
                    answers.append({'pred_ans': None, 'pred_expl': response})
            else:
                answers.append({'pred_ans': None, 'pred_expl': response})
                
    elif expl_type == 'posthoc':
        for response in responses:
            # Check for direct answer statements
            for i in range(len(inputs[0]['options'])):
                option_num = i + 1  # 1-based for human readability
                if response.startswith(f"The answer is: {option_num}"):
                    answers.append({'pred_ans': i, 'pred_expl': response})
                    break
            else:  # No match found
                answers.append({'pred_ans': None, 'pred_expl': response})
    
    return answers

def task_qa_sim_inputs_list(model, expl_type, sim_inputs_list, domain, call_api=None):
    all_sim_inputs = [input for sim_inputs in sim_inputs_list for input in sim_inputs]
    preds = task_qa(model, expl_type, all_sim_inputs, domain, call_api=call_api)
    # regroup preds according to examples (multiple simulation inputs for each original input)
    example_preds = []
    cur = 0
    for ex_idx in range(len(sim_inputs_list)):
        example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
        cur += len(sim_inputs_list[ex_idx])
    assert cur == len(preds)
    return example_preds
