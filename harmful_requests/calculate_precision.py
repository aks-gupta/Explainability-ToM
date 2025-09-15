import sys
import os
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel

# =============================================================================
# CONFIGURATION CONSTANTS (should match pipeline.py)
# =============================================================================

MODELS = {
    'TASKQA': 'gpt-4o-mini',
    'SIMQG': 'gpt-4o-mini',
    'SIMQA': 'gpt-4o-mini'
}

EXPLANATION_TYPES = ['cot']
CUE_TYPE = 'concise'
EXAMPLE_RANGE = range(30)
NUM_EXAMPLES = len(EXAMPLE_RANGE)
SIMQG_PARAMS = {
    'top_p': 1.0,
    'with_context': True,
    'balance_labels': False
}
SIMQA_PARAMS = {
    'k_shot': 3,  # Number of few-shot examples (0 for zero-shot)
    'include_expl': True  # Whether to include explanations
}

OUTPUTS_DIR = './outputs/new'

if __name__ == '__main__':
    taskqa_model = MODELS['TASKQA']
    simqa_model = MODELS['SIMQA']
    
    setting2exidx2precision = {}
    
    for expl_type in EXPLANATION_TYPES:
        setting = (taskqa_model, expl_type)
        setting2exidx2precision[setting] = {}
        
        # Load simQA predictions
        balance_str = 'balanced' if SIMQG_PARAMS['balance_labels'] else 'unbalanced'
        simqa_file = f'{OUTPUTS_DIR}/taskqa_{taskqa_model}_{expl_type}-simqg_mix_{SIMQG_PARAMS["top_p"]}_{SIMQG_PARAMS["with_context"]}_{balance_str}-simqa_{simqa_model}_{SIMQA_PARAMS["k_shot"]}shot_fix_{CUE_TYPE}_test_{NUM_EXAMPLES}.pkl'
        exidx2qns_simans = pkl.load(open(simqa_file, 'rb'))
        exidx2qns_simans = {
            exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
            for exidx, qn_anns in exidx2qns_simans.items()
        }
        
        # Load TaskQA on simulated inputs
        taskqa_sim_file = f'{OUTPUTS_DIR}/taskqa_{taskqa_model}_{expl_type}-simqg_mix_{SIMQG_PARAMS["top_p"]}_{SIMQG_PARAMS["with_context"]}_{balance_str}-taskqa_{taskqa_model}_{expl_type}_{CUE_TYPE}_test_{NUM_EXAMPLES}.pkl'
        exidx2qns_taskans = pkl.load(open(taskqa_sim_file, 'rb'))
        exidx2qns_taskans = {
            exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
            for exidx, qn_anns in exidx2qns_taskans.items()
        }
        
        # Compute precision for each example
        for exidx in EXAMPLE_RANGE:
            ex_simulatable_count, ex_correct_simul_count = 0, 0
            assert len(exidx2qns_simans[exidx]) == len(exidx2qns_taskans[exidx])
            
            for qnidx in range(len(exidx2qns_simans[exidx])):
                simqa_ann = exidx2qns_simans[exidx][qnidx].lower().strip()
                taskqa_pred = exidx2qns_taskans[exidx][qnidx].lower().strip()
                
                if simqa_ann in ['yes', 'no']:
                    ex_simulatable_count += 1
                    if simqa_ann == taskqa_pred:
                        ex_correct_simul_count += 1
            
            if ex_simulatable_count != 0:
                setting2exidx2precision[setting][exidx] = ex_correct_simul_count / ex_simulatable_count

    # Calculate results
    all_settings_exidxs = [list(setting2exidx2precision[setting].keys()) for setting in setting2exidx2precision]
    exidxs_in_all_settings = [exidx for exidx in all_settings_exidxs[0]
                              if all(exidx in exidxs for exidxs in all_settings_exidxs)]
    
    print("Number of common examples:", len(exidxs_in_all_settings))
    
    setting2scores = {
        setting: [setting2exidx2precision[setting][exidx] for exidx in exidxs_in_all_settings]
        for setting in setting2exidx2precision
    }

    settings = list(setting2scores.keys())
    print("Settings:", settings)
    for setting in settings:
        print(' '.join(setting), round(np.mean(setting2scores[setting]) * 100, 1))
    
    for setting1 in settings:
        diff = np.array(setting2scores[setting1]) - np.array(setting2scores[settings[0]])
        if np.allclose(diff, 0):
            print(f"For {setting1}, differences are zero; scores are identical.")
        else:
            pvalues = [str(ttest_rel(setting2scores[setting1], setting2scores[setting2])[1]) for setting2 in settings]
            print(f"{setting1}: " + ','.join(pvalues))
