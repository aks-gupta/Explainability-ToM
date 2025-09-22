import numpy as np
import pickle as pkl
from config import EXAMPLE_RANGE, SIMQA_PATH, SIMQG_MIX_PATH, SIMQG_PATH, MIX_ENABLED
from utils.diversity_util import calculate_diversity

DIVERSITY_METRICS = ['bleu', 'cosine', 'jaccard']

if __name__ == '__main__':
    print("Calculating Generality (Diversity)...")
    
    # Load SimQA predictions to filter valid answers
    exidx2qns_simans = pkl.load(open(SIMQA_PATH, 'rb'))
    exidx2qns_simans = {ex_idx: [str(qn_ann['pred_ans']) for qn_ann in exidx2qns_simans[ex_idx]]
                          for ex_idx in exidx2qns_simans}
    
    # Load simqg inputs (generated follow-up outputs)
    sim_inputs = pkl.load(open(SIMQG_MIX_PATH if MIX_ENABLED else SIMQG_PATH, 'rb'))
    
    # Calculate diversity for each example
    all_diversity_scores = {metric: [] for metric in DIVERSITY_METRICS}
    
    for ex_idx in EXAMPLE_RANGE:
        if ex_idx not in sim_inputs or ex_idx not in exidx2qns_simans:
            continue
            
        ex_sim_inputs = sim_inputs[ex_idx][:6]  # Use first 6
        ex_sim_ans = exidx2qns_simans[ex_idx]
        
        if len(ex_sim_inputs) != len(ex_sim_ans):
            continue
        
        # Build simulatable input strings
        simulatable_inputs = [
            f"Follow-up Question: {ex_sim_inputs[idx]['sim_qn']}"
            for idx in range(len(ex_sim_inputs))
            if ex_sim_ans[idx] != 'unknown'
        ]
        
        if len(simulatable_inputs) > 1:  # Need at least 2 inputs for diversity
            diversity_scores = calculate_diversity(simulatable_inputs)
            for i, metric in enumerate(DIVERSITY_METRICS):
                if not np.isnan(diversity_scores[i]):
                    all_diversity_scores[metric].append(diversity_scores[i])
    
    # Print results
    print(f"Number of examples with valid diversity scores: {len(all_diversity_scores[DIVERSITY_METRICS[0]])}")
    
    for metric in DIVERSITY_METRICS:
        if all_diversity_scores[metric]:
            mean_score = np.mean(all_diversity_scores[metric])
            print(f"{metric.capitalize()} diversity: {mean_score:.3f}")
        else:
            print(f"{metric.capitalize()} diversity: No valid scores") 