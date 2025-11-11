GENERAL_CONFIGS = {
    'num_examples': 1,
    'num_disagreement_qs': 500,
    'k_shot': 3, #0->1, 1->2, 3->3, 5->4
    'counterfactuals': 'LABEL_BALANCED', #HARDCODED/GENERATED/LABEL_BALANCED
    'num_counterfactual_qs': 2, #set to 1 if HARDCODED and 2 if LABEL_BALANCED
    'step_1_out': 'task_qa_out', 
    'step_2_out': 'simulation_question_gen_out', 
    'step_3_out': 'simulation_question_answers_out',
    'step_4_out': 'task_qa_simulation_questions_out',
    'use_existing_folder': False, # set to True to use existing folder and False to create new folder in versioned manner
    'print_debug': True
}

MODEL_CONFIGS = {
    'taskqa_model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', #['mistral.mistral-7b-instruct-v0:2', 'anthropic.claude-3-sonnet-20240229-v1:0', 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free', 'gpt-4.1-mini', 'o1-mini-2024-09-12'],
    'taskqa_expl_type': 'biased', #'concise', 'detailed', 'toxic', 'nontoxic', 'biased', 'nonbiased'
    'simqg_model': 'gpt-4.1-mini', #['o1-mini-2024-09-12'],
    'simqa_model': 'gpt-4.1-mini', #['o1-mini-2024-09-12']
    'simqa_expl_type': 'withexpl' #['withoutexpl', 'withexpl']
}

# Dataset and domain settings
DATASET = 'almanacs'
DOMAIN = 'hiring-decisions' # Options: 'hiring-decisions', 'sycophancy', 'harmful-requests'

# Data file for this dataset/domain
DATA_FILE = './data/hiring_decisions/almanacs_hiring_decisions_question.json'