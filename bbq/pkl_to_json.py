#write code to convert pkl files to json files
import pickle as pkl
import json
import os
import config

#just take name of pkl file and convert it to json file
def pkl_to_json(pkl_file, json_file):
    if os.path.exists(pkl_file):
        data = pkl.load(open(pkl_file, 'rb'))
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Converted {pkl_file} to {json_file}")
    else:
        print(f"{pkl_file} does not exist")

if __name__ == "__main__":
    path = "./json_files/"
    pkl_to_json('./outputs_raceXGender/outputs_context_explanation_ex_1_cf_3/taskqa_gpt-4o-mini_cot-simqg_gpt-4o-mini_1.0_True_raceXGender_1.pkl', path + 'simqa.json')