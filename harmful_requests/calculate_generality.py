import sys
import numpy as np
import pickle as pkl
from scipy.stats import ttest_rel

sys.path.append('../utils')
from diversity_util import calculate_diversity

# =============================================================================
# CONFIGURATION CONSTANTS (should match pipeline.py)
# =============================================================================

MODELS = {
    'TASKQA': 'gpt-4o-mini',
    'SIMQG': 'gpt-4o-mini',
    'SIMQA': 'gpt-4o-mini'
}

EXPLANATION_TYPES = ['cot']
CUE_TYPE = 'nontoxic'
EXAMPLE_RANGE = range(30)
NUM_EXAMPLES = len(EXAMPLE_RANGE)
SIMQG_PARAMS = {
    'top_p': 1.0,
    'with_context': True,
    'balance_labels': True
}
SIMQA_PARAMS = {
    'k_shot': 3,  # Number of few-shot examples (0 for zero-shot)
    'include_expl': True  # Whether to include explanations
}

DIVERSITY_METRICS = ['bleu', 'cosine', 'jaccard']
OUTPUTS_DIR = './outputs/new'

if __name__ == '__main__':
    taskqa_model = MODELS['TASKQA']
    simqa_model = MODELS['SIMQA']
    
    setting2exidx2simulatableinputs = {}

    for expl_type in EXPLANATION_TYPES:
        setting = (taskqa_model, expl_type)
        setting2exidx2simulatableinputs[setting] = {}
        
        # Load SimQA predictions to filter valid answers
        balance_str = 'balanced' if SIMQG_PARAMS['balance_labels'] else 'unbalanced'
        simqa_file = f'{OUTPUTS_DIR}/taskqa_{taskqa_model}_{expl_type}-simqg_mix_{SIMQG_PARAMS["top_p"]}_{SIMQG_PARAMS["with_context"]}_{balance_str}-simqa_{simqa_model}_{SIMQA_PARAMS["k_shot"]}shot_fix_{CUE_TYPE}_test_{NUM_EXAMPLES}.pkl'
        exidx2qns_simans = pkl.load(open(simqa_file, 'rb'))
        exidx2qns_simans = {ex_idx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_simans[ex_idx]]
                              for ex_idx in exidx2qns_simans}
        
        # Load simqg inputs (generated follow-up outputs)
        simqg_file = f'{OUTPUTS_DIR}/taskqa_{taskqa_model}_{expl_type}-simqg_mix_{SIMQG_PARAMS["top_p"]}_{SIMQG_PARAMS["with_context"]}_{balance_str}_{CUE_TYPE}_test_{NUM_EXAMPLES}.pkl'
        sim_inputs = pkl.load(open(simqg_file, 'rb'))
        
        for ex_idx in EXAMPLE_RANGE:
            ex_sim_inputs = sim_inputs[ex_idx][:6]  # Use first 6
            ex_sim_ans = exidx2qns_simans[ex_idx]
            assert len(ex_sim_inputs) == len(ex_sim_ans)
            
            # Build simulatable input strings
            simulatable_inputs = [
                f"Follow-up Question: {ex_sim_inputs[idx]['sim_qn']}"
                for idx in range(len(ex_sim_inputs))
                if ex_sim_ans[idx] != 'unknown'
            ]
            setting2exidx2simulatableinputs[setting][ex_idx] = simulatable_inputs

    # Calculate diversity metrics
    setting2divs = {}
    for expl_type in EXPLANATION_TYPES:
        setting = (taskqa_model, expl_type)
        divs = []
        for ex_idx in EXAMPLE_RANGE:
            divs.append(calculate_diversity(setting2exidx2simulatableinputs[setting][ex_idx]))
        setting2divs[setting] = np.array(divs)

    settings = list(setting2divs.keys())
    print("Settings:", settings)

    # For each diversity metric, display results
    for div_metric in range(3):
        print(f"\nMetric: {DIVERSITY_METRICS[div_metric]}")
        setting2scores = {setting: setting2divs[setting][:, div_metric].tolist() for setting in setting2divs}
        
        # Get common non-NaN indices
        setting2exidxs_nonempty = {
            setting: [ex_idx for ex_idx in range(len(setting2scores[setting]))
                      if not np.isnan(setting2scores[setting][ex_idx])]
            for setting in setting2scores
        }
        nonempty_exidxs_for_all = [ex_idx for ex_idx in setting2exidxs_nonempty[settings[0]]
                                   if all(ex_idx in setting2exidxs_nonempty[other_setting] for other_setting in settings[1:])]
        print("Number of common examples:", len(nonempty_exidxs_for_all))
        
        # Restrict to common indices
        setting2scores = {setting: [setting2scores[setting][ex_idx] for ex_idx in nonempty_exidxs_for_all]
                          for setting in setting2scores}
        setting2mean = {setting: np.mean(setting2scores[setting]) for setting in setting2scores}
        
        print("Mean diversity scores:")
        for setting in settings:
            print(f"{' '.join(setting)}: {round(setting2mean[setting], 3)}")
        
        # Statistical comparisons
        for setting1 in settings:
            for setting2 in settings:
                p_val = ttest_rel(setting2scores[setting1], setting2scores[setting2])[1]
                print(f"P-value for {setting1} vs {setting2}: {p_val}") 