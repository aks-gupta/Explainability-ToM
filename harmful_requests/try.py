import pickle as pkl

# Load the predicted outputs.
exidx2qns_simans = pkl.load(
    open('./outputs/taskqa_gpt-4o-mini_cot-simqg_mix_1.0_True-taskqa_gpt-4o-mini_cot_test_50.pkl', 'rb'))
exidx2qns_taskans = pkl.load(
    open('./outputs/taskqa_gpt-4o-mini_cot-simqg_mix_1.0_True-simqa_gpt-4o-mini_fix_test_50.pkl', 'rb'))

# Convert predictions to strings (if needed).
exidx2qns_simans = {
    exidx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_simans[exidx]]
    for exidx in exidx2qns_simans
}
exidx2qns_taskans = {
    exidx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_taskans[exidx]]
    for exidx in exidx2qns_taskans
}

# Choose one example to inspect (for instance, example index 0).
ex = 0
print("=== DEBUG for example index {} ===".format(ex))
print("Simulated QA Predictions:")
print(exidx2qns_simans.get(ex, "No simulation predictions for this example"))
print("Task QA Predictions:")
print(exidx2qns_taskans.get(ex, "No task QA predictions for this example"))
