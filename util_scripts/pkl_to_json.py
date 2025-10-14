#write code to convert pkl files to json files
import pickle as pkl
import json
import os
import configs

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
    path = "."
    pkl_to_json('./versioned_output/v28/hiring-decisions_task_qa_simulation_questions_out_meta-llama_simqg_gpt-4.1-mini_taskqa_meta-llama_detailed_100.pkl', path + 'converted.json')