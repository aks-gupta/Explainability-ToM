import config
import random
from collections import defaultdict

def stratify_samples(sim_inputs, simqa_preds_list, sample_size_per_stratum=None, total_sample_size=None):
    """
    Stratify simulated inputs based on their options and sample uniformly from each stratum.

    Args:
        sim_inputs (list): List of simulated input dictionaries, each containing 'options'.
        simqa_preds_list (list): List of lists of predictions corresponding to sim_inputs.
        sample_size_per_stratum (int, optional): Number of samples to draw from each stratum.
        total_sample_size (int, optional): Total number of samples to draw across all strata.
    
    Returns:
        tuple: (stratified_sim_inputs, stratified_simqa_preds_list)
    """

    sampled_sim_inputs_list = []
    sampled_simqa_preds_list = []

    for ex_idx, (sim_inputs, simqa_preds) in enumerate(zip(sim_inputs_list, simqa_preds_list)):
        if not sim_inputs or not simqa_preds:
            sampled_sim_inputs_list.append([])
            sampled_simqa_preds_list.append([])
            continue
        strata = defaultdict(list)
        for i, (sim_input, simqa_pred) in enumerate(zip(sim_inputs, simqa_preds)):
            if sim_input is not None and simqa_pred is not None:
                pred_ans = simqa_pred.get('pred_ans', 'unknown')
                strata[pred_ans].append((sim_input, simqa_pred))

        sampled_inputs = []
        sampled_preds = []

        if sample_size_per_stratum is not None:
            # Fixed size per stratum
            for stratum_key, stratum_items in strata.items():
                sample_size = min(sample_size_per_stratum, len(stratum_items))
                sampled_items = random.sample(stratum_items, sample_size)
                for sim_input, simqa_pred in sampled_items:
                    sampled_inputs.append(sim_input)
                    sampled_preds.append(simqa_pred)
        elif total_sample_size is not None:
            total_items = sum(len(items) for items in strata.values())
            if total_items > 0:
                for stratum_key, stratum_items in strata.items():
                    proportion = len(stratum_items) / total_items
                    stratum_sample_size = max(1, int(total_sample_size * proportion))
                    stratum_sample_size = min(stratum_sample_size, len(stratum_items))
                    
                    sampled_items = random.sample(stratum_items, stratum_sample_size)
                    for sim_input, simqa_pred in sampled_items:
                        sampled_inputs.append(sim_input)
                        sampled_preds.append(simqa_pred)

        else:
            # No sampling, return all
            for stratum_items in strata.values():
                for sim_input, simqa_pred in stratum_items:
                    sampled_inputs.append(sim_input)
                    sampled_preds.append(simqa_pred)

        sampled_sim_inputs_list.append(sampled_inputs)
        sampled_simqa_preds_list.append(sampled_preds)

    return sampled_sim_inputs_list, sampled_simqa_preds_list