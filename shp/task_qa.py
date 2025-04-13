import json
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
# from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
from openai import OpenAI
import time
import os

client = OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
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

def task_qa_hiring_decisions(model, expl_type, inputs):
    assert expl_type in ['cot', 'posthoc']
    # first deduplicate inputs, only run on different inputs
    distinct_qns = list(set([input['question'] for input in inputs]))
    distinct_inputs = [{'question': question} for question in distinct_qns]

    prompts = get_prompts_by_task(f'strategyqa-taskqa-{expl_type}',
                                  [{'question': input['question']} for input in distinct_inputs])
    pred_expls = call_openai_api(model=model, prompts=prompts,
                                temperature=0, max_tokens=200, stop='\n\n')
    # pred_expls = multiprocess_api(model=model, prompts=prompts, bsz=8, num_processes=12,
    #                              temperature=0, max_tokens=200, stop='\n\n')
    assert len(pred_expls) == len(prompts)
    if expl_type == 'cot':
        pred_answers = []
        for pred_expl in pred_expls:
            if pred_expl.endswith('So the answer is no.'):
                pred_answers.append('no')
            elif pred_expl.endswith('So the answer is yes.'):
                pred_answers.append('yes')
            else:
                pred_answers.append('neither')
        preds = [{'pred_ans': pred_ans, 'pred_expl': pred_expl.strip()} for pred_ans, pred_expl in
                 zip(pred_answers, pred_expls)]
    elif expl_type == 'posthoc':
        preds = []
        for pred_expl in pred_expls:
            lines = pred_expl.split('\n')
            if (len(lines) != 2) or (lines[0].strip() not in ['yes', 'no']) or (not lines[1].startswith('Justification: ')):
                preds.append({'pred_ans': 'neither', 'pred_expl': pred_expl})
            else:
                preds.append({'pred_ans': lines[0].strip(), 'pred_expl': lines[1][len('Justification: '):].strip()})
    else:
        raise NotImplementedError

    # return to duplicated questions
    assert len(preds) == len(distinct_inputs)
    qn2pred = {input['question']: pred for input, pred in zip(distinct_inputs, preds)}
    preds = [qn2pred[input['question']] for input in inputs]
    return preds

def task_qa(model, expl_type, inputs):
    assert expl_type in ['cot', 'posthoc']
    prompts = get_prompts_by_task(f'shp-taskqa-{expl_type}',
                [{'context': input['context'],
                  'response_0': input['options'][0], 'response_1': input['options'][1]}
                 for input in inputs])
    deduplicated_prompts = list(set(prompts))
    print("Got deduplicated prompts")
    # responses = multiprocess_api(model=model, prompts=deduplicated_prompts, bsz=20, num_processes=12,
    #                              temperature=0, max_tokens=200, stop='\n\n')
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
        print("Hello I am in COT")
        for response in responses:
            select0 = "Candidate Response 1 is more helpful" in response
            select1 = "Candidate Response 2 is more helpful" in response
            if select0 == select1:
                answers.append({'pred_ans': None, 'pred_expl': response})
            elif select0:
                answers.append({'pred_ans': 0, 'pred_expl': response})
            elif select1:
                answers.append({'pred_ans': 1, 'pred_expl': response})
    elif expl_type == 'posthoc':
        for response in responses:
            if response.startswith('Candidate Response 1 is more helpful'):
                answers.append({'pred_ans': 0, 'pred_expl': response})
            elif response.startswith('Candidate Response 2 is more helpful'):
                answers.append({'pred_ans': 1, 'pred_expl': response})
            else:
                answers.append({'pred_ans': None, 'pred_expl': response})
    return answers


def task_qa_sim_inputs_list(model, expl_type, sim_inputs_list):
    all_sim_inputs = [input for sim_inputs in sim_inputs_list for input in sim_inputs]
    preds = task_qa(model, expl_type, all_sim_inputs)
    # regroup preds according to examples (multiple simulation inputs for each original input)
    example_preds = []
    cur = 0
    for ex_idx in range(len(sim_inputs_list)):
        example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
        cur += len(sim_inputs_list[ex_idx])
    assert cur == len(preds)
    return example_preds

def task_qa_hiring_decisions_sim_inputs_list(model, expl_type, sim_inputs_list):
    all_sim_inputs = [input for sim_inputs in sim_inputs_list for input in sim_inputs]
    preds = task_qa_hiring_decisions(model, expl_type, all_sim_inputs)
    # regroup preds according to examples (multiple simulation inputs for each original input)
    example_preds = []
    cur = 0
    for ex_idx in range(len(sim_inputs_list)):
        example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
        cur += len(sim_inputs_list[ex_idx])
    assert cur == len(preds)
    return example_preds