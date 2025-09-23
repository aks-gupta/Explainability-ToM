import json
import sys
import os
from api_client import call_together_api, call_openai_api

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from prompts.load_prompt import get_prompts_by_task
from openai import OpenAI
import time

def task_qa_hiring_decisions(model, expl_type, inputs):
    print(model)
    print(expl_type)
    distinct_qns = []
    for input in inputs:
        if 'question' in input:
            distinct_qns.append(input['question'])
        else:
            distinct_qns.append(input)
    distinct_inputs = [{'question': question} for question in distinct_qns]
    prompts = get_prompts_by_task(f'almanacs-hiring-decisions-taskqa-{expl_type}',
                                  [{'question': input['question']} for input in distinct_inputs])
    pred_expls = call_together_api(model=model, prompts=prompts,
                                temperature=0, max_tokens=200, stop='\n\n')
    assert len(pred_expls) == len(prompts)
    if expl_type in ['cot', 'concise', 'detailed', 'toxic', 'nontoxic']:
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
    else:
        raise NotImplementedError
    # return to duplicated questions
    assert len(preds) == len(distinct_inputs)
    qn2pred = {input['question']: pred for input, pred in zip(distinct_inputs, preds)}
    preds = []
    for input in inputs:
        if 'question' in input:
            preds.append(qn2pred[input['question']])
        else:
            preds.append(qn2pred[input])
    return preds

def task_qa_hiring_decisions_sim_inputs_list(model, expl_type, sim_inputs_list):
    all_sim_inputs = [input for sim_inputs in sim_inputs_list for input in sim_inputs['questions']]
    preds = task_qa_hiring_decisions(model, expl_type, all_sim_inputs)
    print(type(preds), len(preds))
    # regroup preds according to examples (multiple simulation inputs for each original input)
    example_preds = []
    num_samples = len(sim_inputs_list)
    toAdd = int(len(preds)/num_samples)
    ex_idx=0
    while ex_idx < len(preds):
        example_preds.append(preds[ex_idx:ex_idx+toAdd])
        ex_idx+=toAdd
    return example_preds