DOMAIN = 'raceXGender'
NUM_EX = 1
EX_IDXS = range(0, NUM_EX)
NUM_CF = 5
BALANCED = True
MIXED = False
WITH_CONTEXT = True
EXPLANATION = True
'''implement stratified sampling in simqa'''
# STRATIFIED = False
# STRAT_SAMPLES_PER_OPTION = 1
# TOTAL_STRAT_SAMPLES = 3

taskqa_models = ["gpt-4o-mini"]
expl_types = ["cot"]
simqg_models = ["gpt-4o-mini"]
simqa_models = ["gpt-4o-mini"]

def print_configs():
    print("\n\033[1mCurrent Configurations:\033[0m")
    print(f"DOMAIN: {DOMAIN}")
    print(f"NUM_EX: {NUM_EX}")
    print(f"EX_IDXS: {list(EX_IDXS)}")
    print(f"NUM_CF: {NUM_CF}")
    print(f"WITH_CONTEXT: {WITH_CONTEXT}")
    print(f"EXPLANATION: {EXPLANATION}")
    print(f"MIXED: {MIXED}")
    print(f"BALANCED: {BALANCED}")
    print(f"taskqa_models: {taskqa_models}")
    print(f"expl_types: {expl_types}")
    print(f"simqg_models: {simqg_models}")
    print(f"simqa_models: {simqa_models}")