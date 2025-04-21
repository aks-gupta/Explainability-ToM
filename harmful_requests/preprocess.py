import pandas as pd
import json
from tqdm import tqdm
import re
from transformers import AutoTokenizer

def clean(text):
    return re.sub(' +', ' ', text.strip()).strip().\
        replace(' , ',',').replace(' .','.').replace(' !','!')\
        .replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')

if __name__ == '__main__':
    # Load ALMANACS harmful requests dataset
    almanacs_df = pd.read_csv('template_10.csv')
    
    # Filter to just get harmful requests if needed
    harmful_requests = almanacs_df[almanacs_df['template_id'].str.contains('harmful_requests')]
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    split2selecteddata = {}
    
    for split in ['train', 'test']:
        # Filter data by split
        split_data = harmful_requests[harmful_requests['split'] == split]
        
        data = []
        for _, row in tqdm(split_data.iterrows()):
            context = clean(row['question_text'])
            
            # Convert 'Yes'/'No' to True/False
            answer = True if row['completion'] == "Yes" else False
            
            # Length check
            if len(tokenizer.tokenize(context)) <= 1024:
                data_point = {
                    'context': context,
                    'answer': answer,  # Single true/false answer
                    'domain': 'harmful_requests',
                    'almanacs_id': row['question_id'],
                    'template_id': row['template_id'],
                    'answer_prob': float(row['answer_prob']),
                    'total_answer_prob': float(row['total_answer_prob']),
                    # Store the components for analysis if needed
                    'components': {
                        'a': row['a'], 
                        'b': row['b'], 
                        'c': row['c'], 
                        'd': row['d'], 
                        'e': row['e']
                    }
                }
                
                data.append(data_point)
        
        split2selecteddata[split] = data

    # Save the processed data
    json.dump(split2selecteddata, open('data/almanacs_harmful_requests.json', 'w'), indent=4)