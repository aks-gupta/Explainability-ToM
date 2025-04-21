import json
import os
from simulate_qg import simulate_qg, call_openai_api

def debug_call_openai_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=200, stop=None):
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            # Call the original API call and take the first returned element.
            response = call_openai_api(model=model, prompts=[prompt], bsz=bsz,
                                       num_processes=num_processes, temperature=temperature,
                                       top_p=top_p, max_tokens=max_tokens, stop=stop)
            raw_resp = response[0]
            print(f"Raw response for prompt {i}:")
            print(raw_resp)
            responses.append(raw_resp)
        except Exception as e:
            print(f"[{i}] Error during call: {e}")
            responses.append("")
    return responses

def test_simqg_on_first_example():
    # Load your actual file.
    file_path = './data/almanacs_harmful_requests.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Use only the first test example.
    test_examples = data.get('test', [])
    if not test_examples:
        print("No test examples found in file.")
        return
    single_example = [test_examples[0]]  # A list with only the first example.
    
    # Create a corresponding dummy TaskQA prediction.
    # The simulate_qg function expects each prediction to have a "pred_expl" field.
    orig_tm_preds = [{
        "pred_expl": "This is a sample explanation. So the answer is yes."
    } for _ in single_example]
    
    # Backup the original call_openai_api and replace it with our debug version.
    original_call = globals()['call_openai_api']
    globals()['call_openai_api'] = debug_call_openai_api

    print("Running simulate_qg on the first test example with num_samples=1...")
    sim_outputs = simulate_qg(
        model="gpt-4o-mini",
        orig_inputs=single_example,
        orig_tm_preds=orig_tm_preds,
        top_p=1.0,
        num_samples=1,
        with_context=True
    )
    
    # Restore the original API call function.
    globals()['call_openai_api'] = original_call

    print("\nParsed simulate_qg outputs:")
    print(sim_outputs)

if __name__ == '__main__':
    test_simqg_on_first_example()
