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

