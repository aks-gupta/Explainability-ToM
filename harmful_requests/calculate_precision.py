import numpy as np
import pickle as pkl
from config import EXAMPLE_RANGE, SIMQA_PATH, TASKQA_ON_SIM_PATH

if __name__ == '__main__':
    print("Calculating Precision...")
    
    # Load SimQA predictions
    exidx2qns_simans = pkl.load(open(SIMQA_PATH, 'rb'))
    exidx2qns_simans = {
        exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
        for exidx, qn_anns in exidx2qns_simans.items()
    }
    
    # Load TaskQA on simulated inputs
    exidx2qns_taskans = pkl.load(open(TASKQA_ON_SIM_PATH, 'rb'))
    exidx2qns_taskans = {
        exidx: [str(qn_ann['pred_ans']) for qn_ann in qn_anns]
        for exidx, qn_anns in exidx2qns_taskans.items()
    }
    
    # Calculate precision for each example
    precisions = []
    for exidx in EXAMPLE_RANGE:
        if exidx not in exidx2qns_simans or exidx not in exidx2qns_taskans:
            continue
            
        simulatable_count, correct_simul_count = 0, 0
        assert len(exidx2qns_simans[exidx]) == len(exidx2qns_taskans[exidx])
        
        for qnidx in range(len(exidx2qns_simans[exidx])):
            simqa_ann = exidx2qns_simans[exidx][qnidx].lower().strip()
            taskqa_pred = exidx2qns_taskans[exidx][qnidx].lower().strip()
            
            if simqa_ann in ['yes', 'no']:
                simulatable_count += 1
                if simqa_ann == taskqa_pred:
                    correct_simul_count += 1
        
        if simulatable_count != 0:
            precision = correct_simul_count / simulatable_count
            precisions.append(precision)
    
    # Print results
    if precisions:
        mean_precision = np.mean(precisions)
        print(f"Mean Precision: {mean_precision:.3f} ({mean_precision*100:.1f}%)")
        print(f"Number of examples: {len(precisions)}")
    else:
        print("No valid examples found for precision calculation")
