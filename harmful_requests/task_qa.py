import json
import sys
sys.path.append('..')
from prompts.load_prompt import get_prompts_by_task
import openai
import time
import os

# Initialize the OpenAI client using your endpoint and key.
client = openai.OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

cue = 'nontoxic'

def call_openai_api(model, prompts, temperature=0, max_tokens=200, stop=None):
    responses = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            responses.append("")
            time.sleep(1)
    return responses

def task_qa(model, expl_type, inputs):
    """
    Process a list of inputs for the almanacs task.
    Each input is expected to be a dict with at least a "context" field.
    
    expl_type: either 'cot' (chain-of-thought) or 'posthoc'.
    """
    assert expl_type in ['cot', 'posthoc']
    # Create prompts using your almanacs-taskqa prompt templates.
    prompts = get_prompts_by_task(
        f'almanacs-taskqa-{cue}',
        [{'context': input['context']} for input in inputs]
    )
    # Deduplicate prompts to improve efficiency.
    deduplicated_prompts = list(set(prompts))
    print("Got deduplicated prompts")
    
    # Call the API with the deduplicated prompts.
    responses = call_openai_api(
        model=model,
        prompts=deduplicated_prompts,
        temperature=0,
        max_tokens=200,
        stop='\n\n'
    )
    print(f"Received {len(responses)} responses for {len(deduplicated_prompts)} deduplicated prompts")
    
    # Map responses back to the original prompt order.
    prompt2response = {prompt: response for prompt, response in zip(deduplicated_prompts, responses)}
    responses = [prompt2response[prompt] for prompt in prompts]
    assert len(responses) == len(inputs)
    
    answers = []
    # For both types, we expect the final part of the response to end with "So the answer is yes" or "So the answer is no."
    for response in responses:
        lower_response = response.lower()
        if "so the answer is yes" in lower_response:
            answers.append({'pred_ans': "yes", 'pred_expl': response})
        elif "so the answer is no" in lower_response:
            answers.append({'pred_ans': "no", 'pred_expl': response})
        else:
            answers.append({'pred_ans': None, 'pred_expl': response})
    return answers

def task_qa_sim_inputs_list(model, expl_type, sim_inputs_list):
    # sim_inputs_list is a list of lists per example, where each inner dictionary has keys:
    #   "sim_qn", "sim_qa_expl", "pred_ans"
    # task_qa expects each input to have a "context" field and an "options" list.
    # We convert each simulated output into that structure:
    #   - Use the simulated follow-up question ("sim_qn") as the new context.
    #   - Supply the two fixed answer options, ["Yes", "No"].
    #
    # Then we call task_qa (which generates its own answer) on this converted list.
    converted_sim_inputs = []
    for sim_group in sim_inputs_list:
        for sim in sim_group:
            converted_sim_inputs.append({
                'context': sim['sim_qn'],
                'options': ["Yes", "No"]
            })
    
    preds = task_qa(model, expl_type, converted_sim_inputs)
    
    # Regroup predictions according to the original sim_inputs_list structure.
    example_preds = []
    cur = 0
    for sim_group in sim_inputs_list:
        count = len(sim_group)
        example_preds.append(preds[cur: cur + count])
        cur += count
    assert cur == len(preds)
    return example_preds

