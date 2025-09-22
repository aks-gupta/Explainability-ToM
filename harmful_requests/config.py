MODELS = {
	'TASKQA': 'gpt-4o-mini',
	'SIMQG': 'gpt-4o-mini',
	'SIMQA': 'gpt-4o-mini'
}

EXPLANATION_TYPES = ['cot']
CUE_TYPE = 'none'
EXAMPLE_RANGE = range(5)

SIMQG_PARAMS = {
	'top_p': 1.0,
	'num_samples': 2,
	'balance_labels': False
}

SIMQA_PARAMS = {
	'k_shot': 3
}

DATA_PATH = './data/almanacs_harmful_requests.json'
OUTPUTS_ROOT = './outputs'
LOG_FILE = 'log.txt'

# Whether to run the SimQG mixing stage
MIX_ENABLED = False

# Resolved run constants (no functions)
RUN_TAG = f"{CUE_TYPE}_{'bal' if SIMQG_PARAMS.get('balance_labels', False) else 'unbal'}_{SIMQA_PARAMS.get('k_shot', 0)}shot_{len(EXAMPLE_RANGE)}"
RUN_DIR = f"{OUTPUTS_ROOT}/{RUN_TAG}"

TASKQA_PATH = f"{RUN_DIR}/taskqa_{RUN_TAG}.pkl"
SIMQG_PATH = f"{RUN_DIR}/simqg_{RUN_TAG}.pkl"
SIMQG_MIX_PATH = f"{RUN_DIR}/simqg_mix_{RUN_TAG}.pkl"
SIMQA_PATH = f"{RUN_DIR}/simqa_{RUN_TAG}.pkl"
TASKQA_ON_SIM_PATH = f"{RUN_DIR}/taskqa_on_sim_{RUN_TAG}.pkl"
