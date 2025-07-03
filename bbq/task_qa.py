import json
import sys
sys.path.append('..')
# from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
import openai
import time
import os
import re

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

def call_openai_api(model, prompts, temperature=0, max_tokens=200, stop=None):
    responses = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            responses.append("")
            time.sleep(1)
    return responses

def task_qa(model, expl_type, inputs, domain):
    assert expl_type in ['cot', 'posthoc']
    prompts = get_prompts_by_task(f'bbq-taskqa-{expl_type}-nontoxic_{domain}',
                [{'context': input['context'],
                  'question': input['question'], 
                  'options': input['options']}
                 for input in inputs])
    deduplicated_prompts = list(set(prompts))
    print("Got deduplicated prompts")
    
    # Test API connection
    resp = call_openai_api("gpt-4o-mini", ["Say hello!"])
    print(resp)

    responses = call_openai_api(model=model, prompts=deduplicated_prompts,
                                temperature=0, max_tokens=200, stop='\n\n')
    print(len(responses), len(deduplicated_prompts))
    assert len(responses) == len(deduplicated_prompts)
    prompt2response = {prompt: response for prompt, response in zip(deduplicated_prompts, responses)}
    responses = [prompt2response[prompt] for prompt in prompts]
    assert len(responses) == len(inputs)
    answers = []
    
    if expl_type == 'cot':
        print("Processing Chain-of-Thought responses")
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
        print("Processing post-hoc responses")
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

def task_qa_sim_inputs_list(model, expl_type, sim_inputs_list, domain):
    print("Just printing domain:", domain)
    all_sim_inputs = [input for sim_inputs in sim_inputs_list for input in sim_inputs]
    preds = task_qa(model, expl_type, all_sim_inputs, domain)
    # regroup preds according to examples (multiple simulation inputs for each original input)
    example_preds = []
    cur = 0
    for ex_idx in range(len(sim_inputs_list)):
        example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
        cur += len(sim_inputs_list[ex_idx])
    assert cur == len(preds)
    return example_preds