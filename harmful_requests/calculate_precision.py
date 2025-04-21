import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel

if __name__ == '__main__':
    EX_IDXS = range(50)
    simqg_model = 'gpt-4o-mini'
    top_p = 1.0
    simqa_model = 'gpt-4o-mini'
    with_context = True

    setting2exidx2precision = {}
    # for taskqa_model in ['gpt3', 'gpt4']:
    for taskqa_model in ['gpt-4o-mini']:
        for taskqa_expl_type in ['cot', 'posthoc']:
            setting = (taskqa_model, taskqa_expl_type)
            setting2exidx2precision[setting] = {}
            # Load simQA predictions (these come from your simqa stage)
            exidx2qns_simans = pkl.load(
                # Use your appropriate file path. Here it is:
                open(f'./outputs/taskqa_gpt-4o-mini_cot-simqg_mix_1.0_True-taskqa_gpt-4o-mini_cot_test_50.pkl', 'rb'))
            # Convert the predictions so that for each example, we have a list of predicted answers as strings.
            exidx2qns_simans = {
                exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
                for exidx, qn_anns in exidx2qns_simans.items()
            }
            # Load TaskQA predictions (which here come from your fix stage)
            exidx2qns_taskans = pkl.load(
                open(f'./outputs/taskqa_gpt-4o-mini_cot-simqg_mix_1.0_True-simqa_gpt-4o-mini_fix_test_50.pkl', 'rb'))
            exidx2qns_taskans = {
                exidx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_taskans[exidx]]
                for exidx in exidx2qns_taskans
            }
            # Compute precision for each example over EX_IDXS
            for exidx in EX_IDXS:
                ex_simulatable_count, ex_correct_simul_count = 0, 0
                # Ensure the lengths match.
                assert len(exidx2qns_simans[exidx]) == len(exidx2qns_taskans[exidx])
                for qnidx in range(len(exidx2qns_simans[exidx])):
                    # Now we expect simqa answers to be "yes" or "no"
                    simqa_ann = exidx2qns_simans[exidx][qnidx].lower().strip()
                    taskqa_pred = exidx2qns_taskans[exidx][qnidx].lower().strip()
                    if simqa_ann in ['yes', 'no']:
                        ex_simulatable_count += 1
                        if simqa_ann == taskqa_pred:
                            ex_correct_simul_count += 1
                if ex_simulatable_count != 0:
                    setting2exidx2precision[setting][exidx] = ex_correct_simul_count / ex_simulatable_count

    # Get only the example indices that appear in all settings.
    all_settings_exidxs = [list(setting2exidx2precision[setting].keys()) for setting in setting2exidx2precision]
    # For example, if both settings contain the same indices, then take those:
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

