import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_rel
sys.path.append('../utils')
from diversity_util import calculate_diversity

# Settings for diversity metrics.
metrics = ['bleu', 'cosine', 'jaccard']
ex_idxs = range(50)
simqg_model = 'mix'
with_context = True
top_p = 1.0
simqa_model = 'gpt-4o-mini'

# Build a dictionary to store simulatable inputs per example for each setting.
setting2exidx2simulatableinputs = {}

# Use only "gpt-4o-mini" for taskqa_model and iterate over two explanation types.
for taskqa_expl_type in ['cot', 'posthoc']:
    setting = ('gpt-4o-mini', taskqa_expl_type)
    setting2exidx2simulatableinputs[setting] = {}
    
    # Load the simQA fix predictions. (This file should contain predicted answers.)
    exidx2qns_simans = pkl.load(open(f'./outputs/taskqa_gpt-4o-mini_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}_fix_test_50.pkl', 'rb'))
    # Convert each predicted answer into a string.
    exidx2qns_simans = {ex_idx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_simans[ex_idx]]
                          for ex_idx in exidx2qns_simans}
    
    # Load the simqg inputs (generated follow-up outputs) for this setting.
    sim_inputs = pkl.load(open(f'./outputs/taskqa_gpt-4o-mini_{taskqa_expl_type}-simqg_{simqg_model}_1.0_{with_context}_test_50.pkl', 'rb'))
    
    for ex_idx in ex_idxs:
        # Use only the first 6 simulated outputs for each example.
        ex_sim_inputs = sim_inputs[ex_idx][:6]
        ex_sim_ans = exidx2qns_simans[ex_idx]
        # Ensure that each simulated output has a corresponding predicted answer.
        assert len(ex_sim_inputs) == len(ex_sim_ans)
        
        # Build simulatable input strings.
        # Since your harmful requests have a 'context' field (the original request) and your simqg outputs
        # are now generated follow-up outputs (each with a 'sim_qn' field), we construct a string combining these.
        simulatable_inputs = [
            f"Follow-up Question: {ex_sim_inputs[idx]['sim_qn']}"
            for idx in range(len(ex_sim_inputs))
            if ex_sim_ans[idx] != 'unknown'
        ]
        setting2exidx2simulatableinputs[setting][ex_idx] = simulatable_inputs

# Calculate diversity metrics for each setting.
setting2divs = {}
for taskqa_expl_type in ['cot', 'posthoc']:
    setting = ('gpt-4o-mini', taskqa_expl_type)
    divs = []
    for ex_idx in ex_idxs:
        # calculate_diversity is expected to accept a list of strings (the simulatable inputs)
        divs.append(calculate_diversity(setting2exidx2simulatableinputs[setting][ex_idx]))
    setting2divs[setting] = np.array(divs)

# Now, we have two settings: ('gpt-4o-mini', 'cot') and ('gpt-4o-mini', 'posthoc').
settings = list(setting2divs.keys())
print("Settings:", settings)

# For each diversity metric, get the scores per example, compute common indices, the mean, and run paired t-tests.
for div_metric in range(3):
    print("\nMetric:", metrics[div_metric])
    setting2scores = {setting: setting2divs[setting][:, div_metric].tolist() for setting in setting2divs}
    
    # Identify indices for which all settings have a non-NaN score.
    setting2exidxs_nonempty = {
        setting: [ex_idx for ex_idx in range(len(setting2scores[setting]))
                  if not np.isnan(setting2scores[setting][ex_idx])]
        for setting in setting2scores
    }
    # Get the common indices across both settings.
    nonempty_exidxs_for_all = [ex_idx for ex_idx in setting2exidxs_nonempty[settings[0]]
                               if all(ex_idx in setting2exidxs_nonempty[other_setting] for other_setting in settings[1:])]
    print("Number of common examples:", len(nonempty_exidxs_for_all))
    
    # Restrict scores to these common indices.
    setting2scores = {setting: [setting2scores[setting][ex_idx] for ex_idx in nonempty_exidxs_for_all]
                      for setting in setting2scores}
    setting2mean = {setting: np.mean(setting2scores[setting]) for setting in setting2scores}
    print("Mean diversity scores:")
    for setting in settings:
        print(f"{' '.join(setting)}: {round(setting2mean[setting], 3)}")
    
    # Run paired t-tests between the two settings.
    for setting1 in settings:
        for setting2 in settings:
            p_val = ttest_rel(setting2scores[setting1], setting2scores[setting2])[1]
            print(f"P-value for {setting1} vs {setting2}: {p_val}")

