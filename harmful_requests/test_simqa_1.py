import json
import pickle as pkl
from simulate_qa import simulate_qa

def test_simqa_complete():
    # Load the harmful requests test data.
    with open('./data/almanacs_harmful_requests.json', 'r') as f:
        data = json.load(f)
    orig_inputs = data.get('test', [])[:1]  # Only the first test example.
    print("Original input (first test example):")
    print(orig_inputs)
    
    # Load TaskQA predictions.
    # The file is assumed to be a dictionary whose keys are numeric.
    orig_tm_preds_full = pkl.load(open('./outputs/taskqa_gpt-4o-mini_cot_test_50.pkl', 'rb'))
    # Convert dictionary keys to a sorted list and take only the first element.
    orig_tm_preds = [orig_tm_preds_full[k] for k in sorted(orig_tm_preds_full.keys())][:1]
    print("\nTaskQA predictions for the first test example:")
    print(orig_tm_preds)
    
    # Load sim_inputs_list.
    # If the file is a dictionary, convert it to a list; otherwise assume it's already a list.
    sim_inputs_full = pkl.load(open('./outputs/taskqa_gpt-4o-mini_cot-simqg_mix_1.0_True_test_50.pkl', 'rb'))
    if isinstance(sim_inputs_full, dict):
        sim_inputs_list = [sim_inputs_full[k] for k in sorted(sim_inputs_full.keys())][:1]
    else:
        sim_inputs_list = sim_inputs_full[:1]
    
    print("\nType of sim_inputs_list:", type(sim_inputs_list))
    print("Length of sim_inputs_list:", len(sim_inputs_list))
    if len(sim_inputs_list) > 0:
        print("Type of first element:", type(sim_inputs_list[0]))
        print("Length (number of simulated outputs) in first element:", len(sim_inputs_list[0]))
        if len(sim_inputs_list[0]) > 0:
            print("\nFirst simulated follow-up input:")
            print(sim_inputs_list[0][0])
    
    # Run simulate_qa for the first test example.
    # This call uses all simulated outputs (here, 6 outputs for the first example).
    output = simulate_qa(model="gpt-4o-mini",
                         orig_inputs=orig_inputs,
                         orig_tm_preds=orig_tm_preds,
                         sim_inputs_list=sim_inputs_list)
    
    print("\nSimQA complete output for the first test example:")
    print(output)

if __name__ == '__main__':
    test_simqa_complete()
