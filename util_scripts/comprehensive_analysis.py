import os
import pickle as pkl
import json
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from configs import GENERAL_CONFIGS, MODEL_CONFIGS, DOMAIN, DATASET
# revert to original path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Counterfactual generation type (automatically detected from configs.py)
# This determines which data files to load (template_based vs label_balanced)
COUNTERFACTUAL_TYPE = GENERAL_CONFIGS['counterfactuals']

# Output folder to analyze - MANUALLY SPECIFY THIS
# Examples:
#   TEMPLATE_BASED: "harmful-requests_mistral.mistral-7b-instruct-v0:2_1_counterfactuals_TEMPLATE_BASED"
#   LABEL_BALANCED: "harmful-requests_meta-llama_200"
# OUTPUT_FOLDER = "harmful-requests_mistral.mistral-7b-instruct-v0:2_1_counterfactuals_TEMPLATE_BASED"
OUTPUT_FOLDER = "harmful-requests_mistral.mistral-7b-instruct-v0:2_1_counterfactuals_TEMPLATE_BASED"
# Number of examples to analyze
LIMIT_EXAMPLES = GENERAL_CONFIGS['num_examples']

# Version to explanation type mapping
VERSION_MAPPING = {
    'v3': 'nonbiased'
}

# Disagreement dataset path (only used for LABEL_BALANCED, not for TEMPLATE_BASED)
DISAGREEMENT_DATASET_PATH = f"../data/disagreement_dataset/disagreement_filtered_{DOMAIN}_{GENERAL_CONFIGS['num_disagreement_qs']}.json"

# =============================================================================

def load_pkl_data(filepath):
    """Load pickle data with error handling"""
    try:
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_original_questions():
    """Load the original questions"""
    try:
        if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
            filepath = f"../data/preprocessed/template_based_original_questions_{DOMAIN}.json"
        else:
            filepath = f"../data/preprocessed/label_balanced_original_questions_{DOMAIN}.json"
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return {str(i): item for i, item in enumerate(data)}
            return data
    except Exception as e:
        print(f"Error loading original questions: {e}")
        return None

def load_counterfactuals():
    """Load the counterfactual questions"""
    try:
        if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
            filepath = f"../data/preprocessed/template_based_counterfactuals_{DOMAIN}.pkl"
        else:
            filepath = f"../data/preprocessed/label_balanced_counterfactuals_{DOMAIN}.pkl"
        
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except Exception as e:
        print(f"Error loading counterfactuals: {e}")
        return None

def load_disagreement_dataset(filepath):
    """Load the disagreement dataset with baseline model predictions"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading disagreement dataset from {filepath}: {e}")
        return None

def create_comprehensive_analysis(limit_examples, output_folder, version_mapping, disagreement_dataset_path):
    """Create comprehensive analysis comparing different explanation types"""
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS - Configuration")
    print(f"{'='*80}")
    print(f"Counterfactual Type: {COUNTERFACTUAL_TYPE}")
    print(f"Domain: {DOMAIN}")
    print(f"Model: {MODEL_CONFIGS['taskqa_model']}")
    print(f"Explanation Type: {MODEL_CONFIGS['taskqa_expl_type']}")
    print(f"Number of Examples: {limit_examples}")
    print(f"Output Folder: {output_folder}")
    print(f"{'='*80}\n")
    
    print(f"Loading data from {output_folder}...")
    
    # Load original questions and counterfactuals
    original_questions = load_original_questions()
    counterfactuals = load_counterfactuals()
    
    if not original_questions or not counterfactuals:
        print("Failed to load base data")
        return
    
    # Load disagreement dataset only for LABEL_BALANCED (not needed for TEMPLATE_BASED)
    baseline_answers = {}
    if COUNTERFACTUAL_TYPE == 'LABEL_BALANCED':
        disagreement_data = load_disagreement_dataset(disagreement_dataset_path)
        
        if not disagreement_data:
            print("Warning: Failed to load disagreement dataset. Answer switch tracking will not work.")
            disagreement_data = {}
        
        if disagreement_data:
            disagreement_keys_list = sorted([int(k) for k in disagreement_data.keys()])[:limit_examples]
            
            for seq_idx, orig_key in enumerate(disagreement_keys_list):
                item = disagreement_data[str(orig_key)]
                for cf_idx, model_pred in enumerate(item.get('model_predictions', [])):
                    key = (seq_idx, cf_idx)
                    key_str = (str(seq_idx), cf_idx)
                    baseline_answers[key] = model_pred
                    baseline_answers[key_str] = model_pred
        
        print(f"Loaded {len(baseline_answers)} baseline answers from disagreement dataset")
    else:
        print(f"Skipping disagreement dataset (not needed for {COUNTERFACTUAL_TYPE})")
    # print(f"Mapped disagreement dataset keys: {disagreement_keys_list[:10]}... to sequential indices")
    
    # Build versions dictionary from configuration
    versions = {}
    for version, expl_type in version_mapping.items():
        versions[version] = {
            'expl_type': expl_type, 
            'path': f'../outputs/{output_folder}/{version}'
        }
    
    all_data = {}
    
    for version, info in versions.items():
        print(f"Loading {version} ({info['expl_type']})...")
        
        # For file names, use 'cot' as default when expl_type is 'noexpl'
        file_expl_type = 'cot' if info['expl_type'] == 'noexpl' else info['expl_type']
        
        # Extract model name from output folder for file paths
        folder_parts = output_folder.split('_')
        if len(folder_parts) >= 2:
            model_name_for_files = folder_parts[1]  # e.g., "mistral.mistral-7b-instruct-v0:2"
        else:
            model_name_for_files = "meta-llama"  # fallback
        
        # TaskQA file
        taskqa_file = f"{info['path']}/{DOMAIN}_task_qa_out_{model_name_for_files}_{file_expl_type}_{limit_examples}.pkl"
        taskqa_data = load_pkl_data(taskqa_file)
        
        # TaskQA on simulated inputs
        taskqa_sim_file = f"{info['path']}/{DOMAIN}_task_qa_simulation_questions_out_{model_name_for_files}_simqg_gpt-4.1-mini_taskqa_{model_name_for_files}_{file_expl_type}_{limit_examples}.pkl"
        taskqa_sim_data = load_pkl_data(taskqa_sim_file)
        
        # SimQA
        simqa_file = f"{info['path']}/{DOMAIN}_simulation_question_answers_out_{model_name_for_files}_simqg_gpt-4.1-mini_simqa_gpt-4.1-mini_{file_expl_type}_{limit_examples}.pkl"
        simqa_data = load_pkl_data(simqa_file)
        
        if taskqa_data and taskqa_sim_data and simqa_data:
            all_data[version] = {
                'expl_type': info['expl_type'],
                'taskqa': taskqa_data,
                'taskqa_sim': taskqa_sim_data,
                'simqa': simqa_data
            }
        else:
            print(f"  Failed to load data for {version}")
    
    print(f"Loaded data for {len(all_data)} versions")
    
    # Create comprehensive analysis
    analysis_results = []
    
    # Process all examples
    example_ids = list(counterfactuals.keys())[:limit_examples]
    print(f"Analyzing {len(example_ids)} examples...")
    
    for example_id in example_ids:
        # Get original question (ensure key type matches)
        example_key = str(example_id) if not isinstance(example_id, str) else example_id
        original_q = original_questions[example_key]
        original_question_text = original_q['question'] if isinstance(original_q, dict) else original_q
        
        # Get counterfactual questions
        cf_data = counterfactuals[example_id]
        cf_questions = cf_data.get('counterfactual_questions', cf_data.get('questions', []))
        
        example_analysis = {
            'example_id': example_id,
            'original_question': original_question_text,
            'counterfactuals': []
        }
        
        # Process each counterfactual question
        for cf_idx, cf_question in enumerate(cf_questions):
            baseline_key = (example_id, cf_idx)
            baseline_ans = baseline_answers.get(baseline_key, '')
            
            cf_analysis = {
                'cf_index': cf_idx,
                'counterfactual_question': cf_question,
                **{expl_type: {} for expl_type in version_mapping.values()} # Dynamically create keys
            }
            
            # Only include baseline_answer for LABEL_BALANCED
            if COUNTERFACTUAL_TYPE == 'LABEL_BALANCED':
                cf_analysis['baseline_answer'] = baseline_ans
            
            # For each version, get the relevant answers
            for version, data in all_data.items():
                expl_type = data['expl_type']
                
                # Get TaskQA answer for original question (use consistent key type)
                original_taskqa = data['taskqa'].get(example_key, data['taskqa'].get(example_id, {}))
                
                # Get TaskQA answer for this counterfactual
                cf_taskqa_sim = data['taskqa_sim'].get(example_key, data['taskqa_sim'].get(example_id, []))
                cf_taskqa_sim = cf_taskqa_sim[cf_idx] if cf_idx < len(cf_taskqa_sim) else {}
                
                # Get SimQA answer for this counterfactual  
                cf_simqa_list = data['simqa'].get(example_key, data['simqa'].get(example_id, []))
                cf_simqa = cf_simqa_list[cf_idx] if cf_idx < len(cf_simqa_list) else {}
                
                orig_ans = original_taskqa.get('pred_ans', '')
                cf_ans = cf_taskqa_sim.get('pred_ans', '')
                
                answer_switch = False
                if baseline_ans and cf_ans:
                    if (baseline_ans.lower() in ['yes', 'no'] and cf_ans.lower() in ['yes', 'no']):
                        answer_switch = (baseline_ans.lower() != cf_ans.lower())
                
                cf_analysis[expl_type] = {
                    'original_taskqa_explanation': original_taskqa.get('pred_expl', ''),
                    'original_taskqa_answer': orig_ans,
                    'cf_taskqa_sim_explanation': cf_taskqa_sim.get('pred_expl', ''),
                    'cf_taskqa_sim_answer': cf_ans,
                    'cf_simqa_explanation': cf_simqa.get('pred_expl', ''),
                    'cf_simqa_answer': cf_simqa.get('pred_ans', ''),
                    'agree': cf_ans == cf_simqa.get('pred_ans', '')
                }
                
                # Only include answer_switch for LABEL_BALANCED
                if COUNTERFACTUAL_TYPE == 'LABEL_BALANCED':
                    cf_analysis[expl_type]['answer_switch'] = answer_switch
            
            example_analysis['counterfactuals'].append(cf_analysis)
        
        analysis_results.append(example_analysis)
    
    # Create informative filename using output folder name
    # Extract domain and model from output folder name (e.g., "harmful-requests_meta-llama_200")
    folder_parts = output_folder.split('_')
    if len(folder_parts) >= 2:
        domain_name = folder_parts[0]
        model_name = folder_parts[1].replace('.', '_')
    else:
        domain_name = DOMAIN
        model_name = MODEL_CONFIGS['taskqa_model'].split('/')[-1].replace('.', '_')
    
    output_file = f"comprehensive_analysis_{domain_name}_{model_name}_{limit_examples}_examples.json"
    
    # Add metadata to the analysis results
    analysis_with_metadata = {
        'metadata': {
            'dataset': DATASET,
            'domain': domain_name,
            'model': model_name,
            'num_examples': limit_examples,
            'output_folder': output_folder,
            'version_mapping': version_mapping,
            'analysis_timestamp': __import__('datetime').datetime.now().isoformat()
        },
        'analysis_results': analysis_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_with_metadata, f, indent=2)
    
    print(f"Analysis complete! Results saved to: {output_file}")
    
    # Calculate comprehensive statistics
    total_comparisons = 0
    total_agreements = 0
    agreement_by_type = {expl_type: {'agreements': 0, 'total': 0} for expl_type in version_mapping.values()}
    answer_switch_by_type = {expl_type: {'switches': 0, 'total': 0, 'yes_to_no': 0, 'no_to_yes': 0} for expl_type in version_mapping.values()}
    
    # Answer distribution tracking
    answer_counts = {expl_type: {'original_taskqa': {'yes': 0, 'no': 0, 'neither': 0}, 'taskqa_sim': {'yes': 0, 'no': 0, 'neither': 0}, 'simqa': {'yes': 0, 'no': 0, 'neither': 0}} for expl_type in version_mapping.values()}
    
    # Disagreement pattern counts
    disagreement_counts = {expl_type: {} for expl_type in version_mapping.values()}
    
    # Template-based specific metrics: Bias rate and Consistency per qid
    if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
        qid_metrics = {expl_type: {} for expl_type in version_mapping.values()}
    
    # Cross-version answer consistency
    cross_version_consistency = {
        'original_taskqa_consistent': 0,
        'taskqa_sim_consistent': 0,
        'simqa_consistent': 0,
        'total_cross_version_comparisons': 0
    }
    
    # For template-based: track qid from counterfactuals data
    if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
        example_to_qid = {}
        for ex_id in example_ids:
            cf_data = counterfactuals[ex_id]
            if 'qid' in cf_data:
                example_to_qid[str(ex_id)] = cf_data['qid']
    
    for example in analysis_results:
        example_str_id = str(example['example_id'])
        qid = example_to_qid.get(example_str_id, example_str_id) if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED' else None
        
        for cf in example['counterfactuals']:
            baseline_ans = cf.get('baseline_answer', '')
            
            # Track cross-version consistency for this counterfactual
            cf_original_answers = []
            cf_taskqa_sim_answers = []
            cf_simqa_answers = []
            
            for expl_type in version_mapping.values():
                if expl_type in cf and cf[expl_type]:
                    data = cf[expl_type]
                    total_comparisons += 1
                    agreement_by_type[expl_type]['total'] += 1
                    
                    # Track answer distributions
                    orig_ans = data.get('original_taskqa_answer', 'neither')
                    sim_ans = data.get('cf_taskqa_sim_answer', 'neither')
                    qa_ans = data.get('cf_simqa_answer', 'neither')
                    
                    answer_counts[expl_type]['original_taskqa'][orig_ans] += 1
                    answer_counts[expl_type]['taskqa_sim'][sim_ans] += 1
                    answer_counts[expl_type]['simqa'][qa_ans] += 1
                    
                    # Track agreements
                    if data.get('agree', False):
                        total_agreements += 1
                        agreement_by_type[expl_type]['agreements'] += 1
                    else:
                        # Count disagreement patterns
                        pattern = f"TaskQA-sim={sim_ans}, SimQA={qa_ans}"
                        disagreement_counts[expl_type][pattern] = disagreement_counts[expl_type].get(pattern, 0) + 1
                    
                    # Track answer switches
                    answer_switch_by_type[expl_type]['total'] += 1
                    if data.get('answer_switch', False):
                        answer_switch_by_type[expl_type]['switches'] += 1
                        
                        if baseline_ans and sim_ans:
                            if baseline_ans.lower() == 'yes' and sim_ans.lower() == 'no':
                                answer_switch_by_type[expl_type]['yes_to_no'] += 1
                            elif baseline_ans.lower() == 'no' and sim_ans.lower() == 'yes':
                                answer_switch_by_type[expl_type]['no_to_yes'] += 1
                    
                    # Collect answers for cross-version analysis
                    cf_original_answers.append(orig_ans)
                    cf_taskqa_sim_answers.append(sim_ans)
                    cf_simqa_answers.append(qa_ans)
                    
                    # For template-based: collect per-qid metrics
                    if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED' and qid is not None:
                        if qid not in qid_metrics[expl_type]:
                            qid_metrics[expl_type][qid] = {
                                'taskqa_sim_answers': [],
                                'original_taskqa_answers': []
                            }
                        qid_metrics[expl_type][qid]['taskqa_sim_answers'].append(sim_ans)
                        qid_metrics[expl_type][qid]['original_taskqa_answers'].append(orig_ans)
            
            # Check cross-version consistency
            if len(cf_original_answers) == len(version_mapping):  # All explanation types present
                cross_version_consistency['total_cross_version_comparisons'] += 1
                
                if len(set(cf_original_answers)) == 1:  # All same
                    cross_version_consistency['original_taskqa_consistent'] += 1
                if len(set(cf_taskqa_sim_answers)) == 1:  # All same
                    cross_version_consistency['taskqa_sim_consistent'] += 1
                if len(set(cf_simqa_answers)) == 1:  # All same
                    cross_version_consistency['simqa_consistent'] += 1
    
    # Calculate template-based metrics per qid
    if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
        qid_bias_consistency = {expl_type: {} for expl_type in version_mapping.values()}
        
        for expl_type in version_mapping.values():
            for qid, metrics in qid_metrics[expl_type].items():
                taskqa_sim_answers = metrics['taskqa_sim_answers']
                original_answers_list = metrics['original_taskqa_answers']
                
                # Bias rate: majority answer frequency among TaskQA-Sim
                if taskqa_sim_answers and original_answers_list:
                    answer_counts_qid = Counter(taskqa_sim_answers)
                    max_count = max(answer_counts_qid.values()) if answer_counts_qid else 0
                    total_count = len(taskqa_sim_answers)
                    bias_rate = max_count / total_count if total_count > 0 else 0
                    
                    # Consistency: check if all original_taskqa_answers are the same
                    unique_original_answers = set(ans.lower() for ans in original_answers_list)
                    original_consistency = 1.0 if len(unique_original_answers) == 1 else 0.0
                    most_common_original = original_answers_list[0] if original_answers_list else 'unknown'
                    
                    qid_bias_consistency[expl_type][qid] = {
                        'bias_rate': bias_rate,
                        'original_consistency': original_consistency,
                        'total_counterfactuals': total_count,
                        'cf_answer_distribution': dict(answer_counts_qid),
                        'original_taskqa_answer': most_common_original,
                        'majority_cf_answer': answer_counts_qid.most_common(1)[0][0] if answer_counts_qid else None,
                        'all_originals_same': len(unique_original_answers) == 1
                    }
    
    # Print comprehensive summary
    print(f"\n" + "="*80)
    print(f"COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"="*80)
    
    print(f"\nBASIC AGREEMENT STATISTICS:")
    print(f"Total comparisons: {total_comparisons}")
    if total_comparisons > 0:
        print(f"Overall agreement: {total_agreements}/{total_comparisons} ({total_agreements/total_comparisons*100:.1f}%)")
    else:
        print("Overall agreement: No data available (0 comparisons)")
    print(f"\nBy explanation type:")
    for expl_type, stats in agreement_by_type.items():
        if stats['total'] > 0:
            percentage = stats['agreements']/stats['total']*100
            print(f"  {expl_type}: {stats['agreements']}/{stats['total']} ({percentage:.1f}%)")
        else:
            print(f"  {expl_type}: No data available")
    
    if COUNTERFACTUAL_TYPE == 'LABEL_BALANCED':
        print(f"\n" + "-"*60)
        print(f"ANSWER SWITCH ANALYSIS:")
        print(f"(Comparing each explanation type's answer vs baseline from disagreement dataset)")
        print(f"-"*60)
        total_switches = sum(answer_switch_by_type[expl_type]['switches'] for expl_type in version_mapping.values())
        total_switch_comparisons = sum(answer_switch_by_type[expl_type]['total'] for expl_type in version_mapping.values())
        if total_switch_comparisons > 0:
            print(f"Overall answer switches: {total_switches}/{total_switch_comparisons} ({total_switches/total_switch_comparisons*100:.1f}%)")
        else:
            print("Overall answer switches: No data available")
        print(f"\nBy explanation type:")
        for expl_type, stats in answer_switch_by_type.items():
            if stats['total'] > 0:
                percentage = stats['switches']/stats['total']*100
                print(f"  {expl_type}: {stats['switches']}/{stats['total']} ({percentage:.1f}%)")
                if stats['switches'] > 0:
                    print(f"    YES→NO: {stats['yes_to_no']}, NO→YES: {stats['no_to_yes']}")
            else:
                print(f"  {expl_type}: No data available")
    
    print(f"\n" + "-"*60)
    print(f"ANSWER DISTRIBUTION ANALYSIS:")
    print(f"-"*60)
    
    for expl_type in version_mapping.values():
        print(f"\n{expl_type.upper()}:")
        print(f"  Original TaskQA: Yes={answer_counts[expl_type]['original_taskqa']['yes']}, No={answer_counts[expl_type]['original_taskqa']['no']}, Neither={answer_counts[expl_type]['original_taskqa']['neither']}")
        print(f"  TaskQA-Sim:     Yes={answer_counts[expl_type]['taskqa_sim']['yes']}, No={answer_counts[expl_type]['taskqa_sim']['no']}, Neither={answer_counts[expl_type]['taskqa_sim']['neither']}")
        print(f"  SimQA:          Yes={answer_counts[expl_type]['simqa']['yes']}, No={answer_counts[expl_type]['simqa']['no']}, Neither={answer_counts[expl_type]['simqa']['neither']}")
    
    print(f"\n" + "-"*60)
    print(f"CROSS-VERSION CONSISTENCY:")
    print(f"-"*60)
    print(f"Total cross-version comparisons: {cross_version_consistency['total_cross_version_comparisons']}")
    total_cross = cross_version_consistency['total_cross_version_comparisons']
    if total_cross > 0:
        print(f"Original TaskQA consistent across versions: {cross_version_consistency['original_taskqa_consistent']}/{total_cross} ({cross_version_consistency['original_taskqa_consistent']/total_cross*100:.1f}%)")
        print(f"TaskQA-Sim consistent across versions: {cross_version_consistency['taskqa_sim_consistent']}/{total_cross} ({cross_version_consistency['taskqa_sim_consistent']/total_cross*100:.1f}%)")
        print(f"SimQA consistent across versions: {cross_version_consistency['simqa_consistent']}/{total_cross} ({cross_version_consistency['simqa_consistent']/total_cross*100:.1f}%)")
    else:
        print("No cross-version comparisons available")
    
    if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
        print(f"\n" + "-"*60)
        print(f"TEMPLATE-BASED QID METRICS:")
        print(f"(Bias Rate = majority CF answer frequency, Consistency = all originals same)")
        print(f"-"*60)
        
        # Calculate dataset averages
        dataset_bias_consistency = {}
        for expl_type in version_mapping.values():
            if qid_bias_consistency[expl_type]:
                all_bias_rates = [stats['bias_rate'] for stats in qid_bias_consistency[expl_type].values()]
                all_consistency_rates = [stats['original_consistency'] for stats in qid_bias_consistency[expl_type].values()]
                dataset_bias_consistency[expl_type] = {
                    'average_bias_rate': sum(all_bias_rates) / len(all_bias_rates) if all_bias_rates else 0,
                    'average_original_consistency': sum(all_consistency_rates) / len(all_consistency_rates) if all_consistency_rates else 0
                }
        
        for expl_type in version_mapping.values():
            print(f"\n{expl_type.upper()}:")
            if qid_bias_consistency[expl_type]:
                # Print dataset averages first
                if expl_type in dataset_bias_consistency:
                    avg = dataset_bias_consistency[expl_type]
                    print(f"  DATASET AVERAGES:")
                    print(f"    Average Bias Rate: {avg['average_bias_rate']:.1%}")
                    print(f"    Average Original Consistency: {avg['average_original_consistency']:.1%}")
                    print()
                
                # Then print per-qid details
                for qid, stats in sorted(qid_bias_consistency[expl_type].items()):
                    print(f"  QID {qid}:")
                    print(f"    Original TaskQA Answer: {stats['original_taskqa_answer']}")
                    print(f"    Original Consistency: {stats['original_consistency']:.1%} (all same: {stats['all_originals_same']})")
                    print(f"    Bias Rate: {stats['bias_rate']:.1%} (majority CF: {stats['majority_cf_answer']})")
                    print(f"    CF Answer Distribution: {stats['cf_answer_distribution']}")
            else:
                print("  No QID metrics available")
    
    print(f"\n" + "-"*60)
    print(f"DISAGREEMENT ANALYSIS:")
    print(f"-"*60)
    for expl_type in version_mapping.values():
        total_disagreements = sum(disagreement_counts[expl_type].values())
        print(f"\n{expl_type.upper()} Disagreements ({total_disagreements} total):")
        
        for pattern, count in sorted(disagreement_counts[expl_type].items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count} cases")
    
        # Save detailed statistics to file using output folder name
        # Extract domain and model from output folder name (e.g., "harmful-requests_meta-llama_200")
        folder_parts = output_folder.split('_')
        if len(folder_parts) >= 2:
            domain_name = folder_parts[0]
            model_name = folder_parts[1].replace('.', '_')
        else:
            domain_name = DOMAIN
            model_name = MODEL_CONFIGS['taskqa_model'].split('/')[-1].replace('.', '_')
        
        stats_file = f"detailed_statistics_{domain_name}_{model_name}_{limit_examples}_examples.json"
        
        detailed_stats = {
            'metadata': {
                'dataset': DATASET,
                'domain': domain_name,
                'model': model_name,
                'full_model_name': MODEL_CONFIGS['taskqa_model'],
                'num_examples': limit_examples,
                'output_folder': output_folder,
                'version_mapping': version_mapping,
                'counterfactual_type': COUNTERFACTUAL_TYPE,
                'analysis_timestamp': __import__('datetime').datetime.now().isoformat()
            },
            'basic_stats': {
                'total_comparisons': total_comparisons,
                'total_agreements': total_agreements,
                'overall_agreement_rate': total_agreements/total_comparisons*100 if total_comparisons > 0 else 0,
                'agreement_by_type': {k: {'agreements': v['agreements'], 'total': v['total'], 'rate': v['agreements']/v['total']*100 if v['total'] > 0 else 0} for k, v in agreement_by_type.items()}
            },
            'answer_distributions': answer_counts,
            'cross_version_consistency': cross_version_consistency,
            'disagreement_patterns': disagreement_counts
        }
        
        if COUNTERFACTUAL_TYPE == 'LABEL_BALANCED':
            detailed_stats['answer_switch_stats'] = {
                'total_switches': total_switches,
                'total_switch_comparisons': total_switch_comparisons,
                'overall_switch_rate': total_switches/total_switch_comparisons*100 if total_switch_comparisons > 0 else 0,
                'switch_by_type': {k: {'switches': v['switches'], 'total': v['total'], 'rate': v['switches']/v['total']*100 if v['total'] > 0 else 0, 'yes_to_no': v['yes_to_no'], 'no_to_yes': v['no_to_yes']} for k, v in answer_switch_by_type.items()}
            }
        
        if COUNTERFACTUAL_TYPE == 'TEMPLATE_BASED':
            # Use the already calculated dataset_bias_consistency from above
            detailed_stats['template_based_qid_metrics'] = {
                'per_qid': qid_bias_consistency,
                'dataset_averages': dataset_bias_consistency
            }
        
        with open(stats_file, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    print(f"="*80)

if __name__ == "__main__":
    # Use the configuration parameters defined at the top
    create_comprehensive_analysis(
        limit_examples=LIMIT_EXAMPLES,
        output_folder=OUTPUT_FOLDER,
        version_mapping=VERSION_MAPPING,
        disagreement_dataset_path=DISAGREEMENT_DATASET_PATH
    )
