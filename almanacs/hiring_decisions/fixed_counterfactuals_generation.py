import json 
from configs import GENERAL_CONFIGS
import pickle as pkl
import os
import random

counterfactual_code_generation = GENERAL_CONFIGS['counterfactuals']
num_counterfactual_qs = GENERAL_CONFIGS['num_examples']

assert counterfactual_code_generation=='HARDCODED'

# Parse the JSON
test_inputs = json.load(open('./data/data_hiring_decisions.json'))['test']
print(type(test_inputs))

# Extract questions into a list
questions = [item["question"] for item in test_inputs]
top_k_questions = random.sample(questions, int(num_counterfactual_qs))

# Print result
all_qs = {}
for idx, q in enumerate(top_k_questions, 0):
    all_qs[idx]={}
    all_qs[idx]['questions']=[q]

out_file = f'data/fixed_counterfactuals.pkl'

# Make sure the directory exists
os.makedirs(os.path.dirname(out_file), exist_ok=True)

# Save to pickle
with open(out_file, "wb") as f:
    pkl.dump(all_qs, f)

print(f"Saved {len(all_qs)} questions to {out_file}")