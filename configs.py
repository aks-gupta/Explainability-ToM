GENERAL_CONFIGS = {
    'num_examples': 2,
    'k_shot': 3, #0->1, 1->2, 3->3, 5->4
    'counterfactuals': 'GENERATED', #HARDCODED/GENERATED/LABEL_BALANCED
    'num_counterfactual_qs': 2, #set to 1 if HARDCODED and 2 if LABEL_BALANCED
    'step_1_out': 'hiring_decisions_task_qa_out', 
    'step_2_out': 'hiring_decisions_simulation_question_gen_out', 
    'step_3_out': 'hiring_decisions_simulation_question_answers_out',
    'step_4_out': 'hiring_decisions_task_qa_simulation_questions_out',
    'versioned_output': True
}

MODEL_CONFIGS = {
    'taskqa_model': ['gpt-4o-mini'], #['meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'gpt-4o-mini'],
    'taskqa_expl_type': ['cot'], #'concise', 'detailed', 'toxic', 'nontoxic'
    'simqg_model': ['gpt-4o-mini'],
    'simqa_model': ['gpt-4o-mini']
}