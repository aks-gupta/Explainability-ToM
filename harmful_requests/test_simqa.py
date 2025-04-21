import json
import sys
sys.path.append('..')
from simulate_qa import simulate_qa, call_openai_api
import os

def debug_call_openai_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=200, stop=None):
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            # Call the real API function (using one prompt at a time)
            response = call_openai_api(model=model, prompts=[prompt], bsz=bsz,
                                       num_processes=num_processes, temperature=temperature,
                                       top_p=top_p, max_tokens=max_tokens, stop=stop)
            raw_resp = response[0]
            print(f"\nRaw response for prompt {i}:\n{raw_resp}\n")
            responses.append(raw_resp)
        except Exception as e:
            print(f"[{i}] Error during call: {e}")
            responses.append("")
    return responses

def test_simqa_on_first_example():
    # Load your actual dataset file.
    file_path = './data/almanacs_harmful_requests.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Use only the first test example.
    test_examples = data.get('test', [])
    if not test_examples:
        print("No test examples found in file.")
        return
    single_example = [test_examples[0]]  # A list with one example.
    
    # Create a dummy TaskQA prediction.
    # The simulate_qa function expects each prediction to contain a "pred_expl" field.
    # Here, we use a sample explanation that includes a final sentence.
    orig_tm_preds = [{
        "pred_expl": ("This request asks for an ethical evaluation of developing an interactive online workshop "
                      "for educational purposes. The controlled environment and positive intent ensure minimal risk. "
                      "So the answer is yes.")
    } for _ in single_example]
    
    # sim_inputs_list is used only for determining how many simulated outputs to generate.
    # For one example, we create a list with one dummy item.
    sim_inputs_list = [[{"dummy": "dummy"}]]
    
    # Temporarily replace call_openai_api with our debug version so we can see the raw output.
    original_call = globals()['call_openai_api']
    globals()['call_openai_api'] = debug_call_openai_api

    print("Running simulate_qa on the first test example with num_samples=1...\n")
    simqa_outputs = simulate_qa(
        model="gpt-4o-mini",
        orig_inputs=single_example,
        orig_tm_preds=orig_tm_preds,
        sim_inputs_list=sim_inputs_list
    )
    
    # Restore the original API call function.
    globals()['call_openai_api'] = original_call

    print("\nParsed simulate_qa outputs:")
    print(json.dumps(simqa_outputs, indent=4))

if __name__ == '__main__':
    test_simqa_on_first_example()
