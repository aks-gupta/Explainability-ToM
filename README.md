# Explainability-ToM
Repository for conducting initial experiments for simulatability in language models.

### To create dataset run: (Also change directory in the sys.path to reflect where you want to store the data)
`preprocess.py`

### For obtaining the prompt answers run:
`pipeline.py` 
This may have some errors in file naming here and there, try and change it to reflect structure while evaluating.

### For evaluation run:
`calculate_precision.py` and `calculate_generality.py`

### Datasets;
1. SHP
2. BBQ
3. ALMANACS (Harmful Requests and Hiring Decisions)

Would help to store the prompts and data.json for each dataset within the specific folder for that dataset, so it's easier to track. Right now the file structure is such that everything is moved to common directories outside.

### Steps to run the pipeline

1. Choose all the configs to create counterfactuals in configs.py
    - Select ```num_examples``` in GENERAL_CONFIGS to choose how many datapoints to look at while creating counterfactuals (oversample to get a few good pairs)
    - Choose how ```counterfactuals``` are generated: "LABEL_BALANCED" and "HARDCODED" require running a script before running the pipeline.py 
    - Choose an appropriate model for ```simqg_model``` in MODEL_CONFIGS
2. Export variables ```OPENAI_API_KEY```, ```TOGETHER_API_KEY```, ```AWS_ACCESS_KEY_ID``` and ```AWS_SECRET_ACCESS_KEY``` in the path
3. Skip to step 4 if the ```counterfactuals``` in Step 1 is of type "GENERATED". In this step run ``` python constant_counterfactuals_generation/hardcoded_counterfactuals_generation.py``` for hardcoded counterfactual generation and ``` python constant_counterfactuals_generation/label_balanced_counterfactuals_generation.py``` for label balanced counterfactual generation. Counterfactual questions are generated in data/ folder. 
4. Note down the number of counterfactual questions generated from the logs and change the ```num_examples``` to a number lesser than the number in ```configs.py``` in GENERAL_CONFIGS
5. Decide the models to use for the entire ```pipeline.py```
6. Run this pipeline with new configurations using ```python pipeline.py```
7. To calculate the precision of the new output run ```python calculate_precision.py```