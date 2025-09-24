import json
import pickle as pkl
import os

from config import DATA_PATH, RUN_DIR, FIXED_SIMQG_PATH

example_range = range(100)
data = json.load(open(DATA_PATH))
train = data.get('train', [])
assert len(train) >= max(example_range.stop, len(example_range)), "Not enough train items"

exidx2sim = {}
for ex_idx in example_range:
    item = train[ex_idx]
    sim_qn = item.get('context', '')
    answer_bool = item.get('answer', None)
    if answer_bool is True:
        sim_expl = "So the answer is yes."
    elif answer_bool is False:
        sim_expl = "So the answer is no."
    else:
        sim_expl = "So the answer is no."
    exidx2sim[ex_idx] = [{
        'sim_qn': sim_qn,
        'sim_qa_expl': sim_expl
    }]

pkl.dump(exidx2sim, open(FIXED_SIMQG_PATH, 'wb'))
print(f"Wrote fixed simqg to {FIXED_SIMQG_PATH}")


