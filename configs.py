GENERAL_CONFIGS = {
    'num_examples': 150,
    'k_shot': 3, #0->1, 1->2, 3->3, 5->4
    'counterfactuals': 'GENERATED', #HARDCODED/GENERATED/LABEL_BALANCED
    'num_counterfactual_qs': 3, #set to 1 if HARDCODED and 2 if LABEL_BALANCED
    'step_1_out': 'task_qa_out', 
    'step_2_out': 'simulation_question_gen_out', 
    'step_3_out': 'simulation_question_answers_out',
    'step_4_out': 'task_qa_simulation_questions_out',
    'versioned_output': True
}

MODEL_CONFIGS = {
    'taskqa_model': ['gpt-4.1-mini'], #['meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'gpt-4o-mini','o1-mini-2024-09-12'],
    'taskqa_expl_type': ['cot'], #'concise', 'detailed', 'toxic', 'nontoxic'
    'simqg_model': ['gpt-4.1-mini'], #['o1-mini-2024-09-12'],
    'simqa_model': ['gpt-4.1-mini'] #['o1-mini-2024-09-12']
}

# Dataset and domain settings
DATASET = 'almanacs'
DOMAIN = 'hiring_decisions'

# Data file for this dataset/domain
DATA_FILE = './data/harmful_requests/almanacs_harmful_requests_question.json'