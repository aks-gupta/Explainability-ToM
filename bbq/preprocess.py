import json
from datasets import load_dataset
import re
from tqdm import tqdm

def clean(text):
    """Clean text by standardizing spacing and punctuation."""
    return re.sub(' +', ' ', str(text).strip()).strip().\
        replace(' , ',',').replace(' .','.').replace(' !','!')\
        .replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')

if __name__ == '__main__':
    # Load the BBQ dataset
    dataset = load_dataset("walledai/BBQ")
    print(f"Dataset splits: {dataset.keys()}")
    
    # Process each split
    split2selecteddata = {}
    
    for split in dataset.keys():
        print(f"Processing {split} split...")
        data = []
        
        for ex in tqdm(dataset[split]):
            # Clean text fields
            context = clean(ex['context'])
            question = clean(ex['question'])
            
            # Ensure choices is a list of strings
            choices = [clean(choice) for choice in ex['choices']]
            
            # Get the correct answer index
            answer_idx = ex['answer']
            
            # Extract category
            category = ex['category']
            
            # Create entry in our desired format
            entry = {
                'context': context,
                'question': question,
                'options': choices,
                'preferred_idx': answer_idx,  # Using the same field name as SHP for consistency
                'category': category,
            }
            
            data.append(entry)
        
        split2selecteddata[split] = data
        print(f"Processed {len(data)} examples for {split}")
    
    # Save the processed data
    print("Saving processed data to data_bbq.json")
    json.dump(split2selecteddata, open('data_bbq.json', 'w'), indent=4)
    print("Data saved successfully")