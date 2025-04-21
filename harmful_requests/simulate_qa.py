# import json
# import sys
# sys.path.append('..')
# from prompts.load_prompt import get_prompts_by_task
# import openai
# import time
# import os

# client = openai.OpenAI(
#     api_key=os.environ.get("LITELLM_API_KEY"),
#     base_url="https://cmu.litellm.ai",
# )

# def call_openai_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=200, stop=None):
#     responses = []
#     for i, prompt in enumerate(prompts):
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens,
#                 stop=stop
#             )
#             responses.append(response.choices[0].message.content)
#         except Exception as e:
#             print(f"[{i}] Error during call:\nPrompt: {prompt[:100]}...\nError: {e}")
#             responses.append("")
#             time.sleep(1)
#     return responses

# def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list):
#     """
#     Build prompts using only the original (starter) context and explanation (from TaskQA output)—
#     exactly matching your few-shot prompt definition ("almanacs-simqa-withexpl").
    
#     In the prompt, the variables are:
#       - {context}    -> the harmful request (question)
#       - {explanation} -> the robot’s answer with explanation from TaskQA
      
#     The prompt template is then:
    
#     Human: Starter Question: {context}
#     Robot's Answer with Explanation: {explanation}
#     Follow-up Question:
    
#     Assistant: here is my response. {sim_qn}
#     My Answer with Explanation: {sim_qa_expl}
    
#     The function parses the model’s output to extract:
#       - sim_qn  (the generated follow-up question)
#       - sim_qa_expl  (the generated answer explanation, which is expected to include a final sentence like "So the answer is yes." or "So the answer is no.")
#       - pred_ans (extracted from sim_qa_expl by looking for the marker "So the answer is")
    
#     The number of outputs per original example is determined by sim_inputs_list (its content is ignored; only the count is used).
#     """
#     assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
#     num_examples = len(orig_inputs)
    
#     # Build one prompt per simulated output we want.
#     # (We ignore any follow-up input content; we only supply {context} and {explanation}.)
#     prompts = []
#     for orig_input, orig_tm_pred, sim_group in zip(orig_inputs, orig_tm_preds, sim_inputs_list):
#         for _ in sim_group:
#             prompt_data = {
#                 'context': orig_input['context'],
#                 'explanation': orig_tm_pred['pred_expl']
#             }
#             prompt = get_prompts_by_task('almanacs-simqa-withexpl', [prompt_data])[0]
#             prompts.append(prompt)
    
#     # Deduplicate prompts to save API calls.
#     deduplicated_prompts = list(set(prompts))
#     pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
#                                  bsz=8, num_processes=12,
#                                  temperature=0, max_tokens=150, stop='\n\n')
#     assert len(pred_expls) == len(deduplicated_prompts)
#     prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
#     final_responses = [prompt2pred_expl[prompt] for prompt in prompts]
#     assert len(final_responses) == len(prompts)
    
#     # Parse each response.
#     # Expected format (when using template_with_label):
#     # "Assistant: here is my response. {sim_qn}
#     # My Answer with Explanation: {sim_qa_expl}"
#     parsed_preds = []
#     answer_marker = "So the answer is"
#     for raw_resp in final_responses:
#         raw_resp = raw_resp.strip()
#         sim_qn = ""
#         sim_qa_expl = ""
#         if "My Answer with Explanation:" in raw_resp:
#             parts = raw_resp.split("My Answer with Explanation:", 1)
#             sim_qn = parts[0].replace("Assistant: here is my response.", "").strip()
#             sim_qa_expl = parts[1].strip()
#         else:
#             sim_qn = raw_resp
#             sim_qa_expl = ""
        
#         pred_ans = "unknown"
#         if answer_marker in sim_qa_expl:
#             tail = sim_qa_expl.split(answer_marker, 1)[1].strip()
#             if tail:
#                 token = tail.split()[0].strip(".,").lower()
#                 if token in ["yes", "no"]:
#                     pred_ans = token
#         parsed_preds.append({
#             "sim_qn": sim_qn,
#             "sim_qa_expl": sim_qa_expl,
#             "pred_ans": pred_ans
#         })
    
#     # Regroup predictions by original example.
#     example_preds = []
#     cur = 0
#     for sim_group in sim_inputs_list:
#         count = len(sim_group)
#         example_preds.append(parsed_preds[cur:cur+count])
#         cur += count
#     assert cur == len(parsed_preds)
#     return example_preds
import json
import random
import sys
from collections import Counter
import numpy as np
sys.path.append('..')
# from api_wrapper.api_wrapper import multiprocess_api
from task_qa import call_openai_api
from prompts.load_prompt import get_prompts_by_task
import pickle as pkl
from copy import deepcopy

def extract_sim_qa_ans(sim_qa_expl):
    """
    Extracts the final answer by searching for the phrase "So the answer is"
    and then taking the following token. If the token is "yes" or "no" (ignoring punctuation
    and case), it returns that token. Otherwise, it returns 'neither'.
    """
    marker = "So the answer is"
    if marker in sim_qa_expl:
        tail = sim_qa_expl.split(marker, 1)[1].strip()
        if tail:
            token = tail.split()[0].strip(".,").lower()
            if token in ["yes", "no"]:
                return token
    return "neither"

def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list, include_expl=True, majority_vote=None):
    """
    Build prompts using only the original (starter) context and explanation from the TaskQA output,
    matching your few-shot prompt definition for "almanacs-simqa-withexpl".

    Expected fields:
      - {context}      → harmful request (orig_inputs["context"])
      - {explanation}  → TaskQA explanation (orig_tm_preds["pred_expl"])
      - {sim_qn}       → simulated follow-up question (sim_inputs_list elements, key "sim_qn")

    The template (template_with_label) is assumed to be:

      Human: Starter Question: {context}
      Robot's Answer with Explanation: {explanation}
      Follow-up Question:
      
      Assistant: here is my response. {sim_qn}
      My Answer with Explanation: {sim_qa_expl}

    The function then parses the API output to extract:
      - sim_qn (the generated follow-up question)
      - sim_qa_expl (the generated answer explanation, expected to include a final sentence such as "So the answer is yes." or "So the answer is no.")
      - pred_ans (extracted via extract_sim_qa_ans from sim_qa_expl)

    The number of outputs per original example is determined by sim_inputs_list (its content is ignored aside from count).

    Since "almanacs-simqa-withexpl" is the only prompt available, we always use that.
    """
    assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
    num_examples = len(orig_inputs)
    
    # Build one prompt per simulated output.
    prompts = []
    for orig_input, orig_tm_pred, sim_group in zip(orig_inputs, orig_tm_preds, sim_inputs_list):
        # For each simulated output we require one prompt.
        # We use the "sim_qn" from each sim_group element.
        for sim_input in sim_group:
            prompt_data = {
                'context': orig_input['context'],
                'explanation': orig_tm_pred['pred_expl'],
                'sim_qn': sim_input.get('sim_qn', sim_input.get('question', ''))
            }
            # Always use the almanacs-simqa-withexpl prompt.
            prompt = get_prompts_by_task('almanacs-simqa-withexpl', [prompt_data])[0]
            prompts.append(prompt)
    
    # Deduplicate prompts to save API calls.
    deduplicated_prompts = list(set(prompts))
    
    # --- DEBUG: Print deduplicated prompts ---
    # print("DEBUG: Deduplicated Prompts:")
    # for idx, dp in enumerate(deduplicated_prompts):
    #     print(f"Prompt {idx}:")
    #     print(dp)
    #     print("="*80)
    
    # Call API without stop token and with a moderate temperature to encourage generation.
    if majority_vote is None or majority_vote == 1:
        pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
                                     temperature=0.7, max_tokens=200, stop=None)
    else:
        pred_expls = call_openai_api(model=model, prompts=deduplicated_prompts,
                                     temperature=1, max_tokens=200, stop=None)
    assert len(pred_expls) == len(deduplicated_prompts)
    
    # --- DEBUG: Print raw API responses for each deduplicated prompt ---
    # print("\nDEBUG: Raw API Responses (for deduplicated prompts):")
    # for idx, resp in enumerate(pred_expls):
    #     print(f"Response {idx}:")
    #     print(resp)
    #     print("-"*80)
    
    # Reinsert duplicate predictions based on the original prompt order.
    prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
    pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
    assert len(pred_expls) == len(prompts)
    
    # Extract answers from API responses.
    if majority_vote is None or majority_vote == 1:
        preds = []
        for pred_expl in pred_expls:
            preds.append({'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl})
    else:
        preds = []
        for pred_expl_samples in pred_expls:
            ex_preds = [{'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl}
                        for pred_expl in pred_expl_samples]
            ex_pred_answers = [pred['pred_ans'] for pred in ex_preds]
            counter = Counter(ex_pred_answers)
            max_count = np.max([counter[item] for item in counter])
            most_frequent_answers = [ans for ans in counter if counter[ans] == max_count]
            majority_ans = random.sample(most_frequent_answers, 1)[0]
            preds.append({'pred_ans': majority_ans, 'majority_vote_details': ex_preds})
    
    # Regroup predictions according to examples.
    # The order is determined by sim_inputs_list (each element's length is the number of outputs for that example).
    assert len(preds) == len(prompts)
    example_preds = []
    cur = 0
    for sim_group in sim_inputs_list:
        count = len(sim_group)
        example_preds.append(preds[cur:cur+count])
        cur += count
    assert cur == len(preds)
    return example_preds
