import json
import pickle as pkl
import random
from collections import defaultdict
import config
from simulate_qg import simulate_qg
from task_qa import task_qa
from simulate_qa import simulate_qa_direct_examples

def generate_fixed_counterfactuals(model, orig_inputs, orig_tm_preds, domain, num_fixed_cfs):
    """
    Generate fixed counterfactuals with balanced responses from TaskQA.
    
    Args:
        model: Model to use for generation
        orig_inputs: Original BBQ examples
        orig_tm_preds: TaskQA predictions for original examples
        domain: Domain name
        num_fixed_cfs: Number of fixed counterfactuals to generate per example
        
    Returns:
        List of dictionaries containing:
        - 'counterfactuals': List of selected counterfactuals
        - 'taskqa_preds': TaskQA predictions for the counterfactuals
        - 'source_mapping': Mapping from counterfactual to source question+explanation
    """
    print(f"\n\033[1mGenerating Fixed Counterfactuals\033[0m with model={model}, domain={domain}, num_fixed_cfs={num_fixed_cfs}")
    
    result_list = []
    
    for ex_idx, (orig_input, orig_tm_pred) in enumerate(zip(orig_inputs, orig_tm_preds)):
        print(f"\nProcessing example {ex_idx + 1}/{len(orig_inputs)}")
        
        # Step 1: Generate multiple counterfactuals using SimQG
        print("Step 1: Generating counterfactuals...")
        # Generate more counterfactuals than needed to allow for selection
        sim_inputs_list = simulate_qg(
            model=model,
            orig_inputs=[orig_input],
            orig_tm_preds=[orig_tm_pred],
            top_p=1.0,
            num_samples=num_fixed_cfs * 3,  # Generate 3x more to allow selection
            with_context=config.WITH_CONTEXT,
            domain=domain
        )
        
        # Extract counterfactuals for this example
        counterfactuals = sim_inputs_list[0] if sim_inputs_list else []
        
        if not counterfactuals:
            print(f"No valid counterfactuals generated for example {ex_idx}")
            result_list.append({
                'counterfactuals': [],
                'taskqa_preds': [],
                'source_mapping': []
            })
            continue
            
        print(f"Generated {len(counterfactuals)} counterfactuals")
        
        # Step 2: Get TaskQA predictions for all counterfactuals
        print("Step 2: Getting TaskQA predictions for counterfactuals...")
        cf_taskqa_preds = task_qa(
            model=model,
            expl_type='cot',  # Use same explanation type as original
            inputs=counterfactuals,
            domain=domain
        )
        
        # Step 3: Balance/stratify counterfactuals based on TaskQA responses
        print("Step 3: Balancing counterfactuals...")
        selected_cfs, selected_preds, source_mapping = balance_counterfactuals(
            counterfactuals=counterfactuals,
            cf_preds=cf_taskqa_preds,
            orig_input=orig_input,
            orig_pred=orig_tm_pred,
            num_fixed_cfs=num_fixed_cfs
        )
        
        result_list.append({
            'counterfactuals': selected_cfs,
            'taskqa_preds': selected_preds,
            'source_mapping': source_mapping
        })
        
        print(f"Selected {len(selected_cfs)} balanced counterfactuals for example {ex_idx}")
    
    return result_list

def balance_counterfactuals(counterfactuals, cf_preds, orig_input, orig_pred, num_fixed_cfs):
    """
    Balance counterfactuals to have diverse TaskQA responses.
    
    Args:
        counterfactuals: List of generated counterfactuals
        cf_preds: TaskQA predictions for counterfactuals
        orig_input: Original input
        orig_pred: Original TaskQA prediction
        num_fixed_cfs: Number of counterfactuals to select
        
    Returns:
        Tuple of (selected_counterfactuals, selected_predictions, source_mapping)
    """
    # Group counterfactuals by their TaskQA prediction
    pred_groups = defaultdict(list)
    
    for cf, pred in zip(counterfactuals, cf_preds):
        pred_ans = pred['pred_ans']
        if pred_ans is not None:
            pred_groups[pred_ans].append((cf, pred))
        else:
            # Group unknown/None predictions together
            pred_groups['unknown'].append((cf, pred))
    
    print(f"Prediction distribution: {[(k, len(v)) for k, v in pred_groups.items()]}")
    
    # Select counterfactuals to ensure balance
    selected_cfs = []
    selected_preds = []
    source_mapping = []
    
    # If we have multiple prediction groups, try to balance across them
    if len(pred_groups) > 1:
        predictions_per_group = max(1, num_fixed_cfs // len(pred_groups))
        remaining = num_fixed_cfs % len(pred_groups)
        
        for pred_label, cf_pred_pairs in pred_groups.items():
            # Number to select from this group
            num_from_group = predictions_per_group + (1 if remaining > 0 else 0)
            if remaining > 0:
                remaining -= 1
                
            # Randomly sample from this group
            num_to_sample = min(num_from_group, len(cf_pred_pairs))
            sampled_pairs = random.sample(cf_pred_pairs, num_to_sample)
            
            for cf, pred in sampled_pairs:
                selected_cfs.append(cf)
                selected_preds.append(pred)
                source_mapping.append({
                    'source_context': orig_input['context'],
                    'source_question': orig_input['question'],
                    'source_options': orig_input['options'],
                    'source_pred_ans': orig_pred['pred_ans'],
                    'source_explanation': orig_pred['pred_expl']
                })
    else:
        # Only one prediction type available, just sample randomly
        all_pairs = list(pred_groups.values())[0] if pred_groups else []
        num_to_sample = min(num_fixed_cfs, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, num_to_sample)
        
        for cf, pred in sampled_pairs:
            selected_cfs.append(cf)
            selected_preds.append(pred)
            source_mapping.append({
                'source_context': orig_input['context'],
                'source_question': orig_input['question'],
                'source_options': orig_input['options'],
                'source_pred_ans': orig_pred['pred_ans'],
                'source_explanation': orig_pred['pred_expl']
            })
    
    return selected_cfs, selected_preds, source_mapping

def run_simqa_on_fixed_counterfactuals(model, orig_inputs, orig_tm_preds, fixed_cf_data, domain):
    """
    Run SimQA on the fixed counterfactuals.
    
    Args:
        model: SimQA model
        orig_inputs: Original BBQ examples
        orig_tm_preds: Original TaskQA predictions
        fixed_cf_data: Fixed counterfactual data from generate_fixed_counterfactuals
        domain: Domain name
        
    Returns:
        List of SimQA predictions for each example's counterfactuals
    """
    print(f"\n\033[1mRunning SimQA on Fixed Counterfactuals\033[0m with model={model}, domain={domain}")
    
    simqa_results = []
    
    for ex_idx, (orig_input, orig_tm_pred, cf_data) in enumerate(zip(orig_inputs, orig_tm_preds, fixed_cf_data)):
        counterfactuals = cf_data['counterfactuals']
        
        if not counterfactuals:
            simqa_results.append([])
            continue
            
        print(f"Running SimQA for example {ex_idx + 1} with {len(counterfactuals)} counterfactuals")
        
        # Use the existing simulate_qa_direct_examples function
        simqa_preds = simulate_qa_direct_examples(
            model=model,
            orig_input=orig_input,
            orig_tm_pred=orig_tm_pred,
            eval_examples=counterfactuals,
            domain=domain
        )
        
        simqa_results.append(simqa_preds)
    
    return simqa_results