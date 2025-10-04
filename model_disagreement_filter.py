import json
import configs
from task_qa import task_qa_hiring_decisions
import random

dataset = configs.DATASET
domain = configs.DOMAIN
model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
expl_type = 'cot'
num_examples = configs.GENERAL_CONFIGS['num_examples']

def load_label_balanced_counterfactuals():
    data = json.load(open(f'./data/label_balanced_counterfactuals_{domain}.json'))
    
    limited_data = {}
    for i, (qid, item) in enumerate(data.items()):
        if i >= num_examples:
            break
        limited_data[qid] = item
    
    print(f"Loaded {len(limited_data)} original questions with counterfactuals (limited to {num_examples})")
    return limited_data

def predict_counterfactuals(data):
    all_questions = []
    for qid, item in data.items():
        for cf_question in item['counterfactual_questions']:
            all_questions.append({'question': cf_question})
    
    print(f"Running TaskQA model ({model}) on {len(all_questions)} counterfactuals...")
    
    predictions = task_qa_hiring_decisions(model=model, expl_type=expl_type, inputs=all_questions)
    
    idx = 0
    for qid, item in data.items():
        item['model_predictions'] = []
        for i, cf_q in enumerate(item['counterfactual_questions']):
            pred_ans = predictions[idx]['pred_ans']
            pred_expl = predictions[idx]['pred_expl']
            item['model_predictions'].append(pred_ans)
            print(f"\n{'='*80}")
            print(f"DEBUG: QID={qid}, CF={i}")
            print(f"Question: {cf_q}")
            print(f"Expected: {item['counterfactual_answers'][i]}")
            print(f"Predicted: {pred_ans}")
            print(f"Explanation: {pred_expl}")
            print(f"{'='*80}")
            idx += 1
    
    return data

def filter_by_agreement(data):
    agree_yes = []
    agree_no = []
    disagree_yes = []
    disagree_no = []
    
    for qid, item in data.items():
        for cf, ans, pred in zip(item['counterfactual_questions'], item['counterfactual_answers'], 
                                  item['model_predictions']):
            entry = {
                'qid': qid,
                'question': item['question'],
                'counterfactual_question': cf,
                'expected_answer': ans,
                'model_prediction': pred
            }
            
            if ans.lower() == pred.lower():
                if 'yes' in ans.lower():
                    agree_yes.append(entry)
                else:
                    agree_no.append(entry)
            else:
                if 'yes' in ans.lower():
                    disagree_yes.append(entry)
                else:
                    disagree_no.append(entry)
    
    print(f"Agree YES: {len(agree_yes)}, Agree NO: {len(agree_no)}")
    print(f"Disagree YES: {len(disagree_yes)}, Disagree NO: {len(disagree_no)}")
    
    target = min(len(agree_yes), len(agree_no), len(disagree_yes), len(disagree_no))
    
    print(f"Selecting {target} from each category for 50-50 balance")
    
    final_list = (
        agree_yes[:target] + 
        agree_no[:target] + 
        disagree_yes[:target] + 
        disagree_no[:target]
    )
    
    filtered_data = {}
    for entry in final_list:
        qid = entry['qid']
        if qid not in filtered_data:
            filtered_data[qid] = {
                'question': entry['question'],
                'counterfactual_questions': [],
                'counterfactual_answers': [],
                'model_predictions': [],
                'agreements': []
            }
        
        filtered_data[qid]['counterfactual_questions'].append(entry['counterfactual_question'])
        filtered_data[qid]['counterfactual_answers'].append(entry['expected_answer'])
        filtered_data[qid]['model_predictions'].append(entry['model_prediction'])
        
        agreement = 'agree' if entry['expected_answer'].lower() == entry['model_prediction'].lower() else 'disagree'
        filtered_data[qid]['agreements'].append(agreement)
    
    print(f"Final dataset: {len(filtered_data)} questions with {len(final_list)} total counterfactuals")
    
    return filtered_data

if __name__ == "__main__":
    data = load_label_balanced_counterfactuals()
    
    data_with_predictions = predict_counterfactuals(data)
    
    with open(f'./data/model_predictions_{domain}.json', 'w') as f:
        json.dump(data_with_predictions, f, indent=4)
    print(f"Saved all predictions to ./data/model_predictions_{domain}.json")
    
    final_data = filter_by_agreement(data_with_predictions)
    
    with open(f'./data/disagreement_filtered_{domain}.json', 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Saved filtered dataset to ./data/disagreement_filtered_{domain}.json")

