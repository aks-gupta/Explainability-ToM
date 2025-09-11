import json
import os

def _embed_prompt(
    instruction, template_with_label, template_no_label, dem_examples, test_example
):
    prompt = str(instruction)
    for dem in dem_examples:
        prompt += template_with_label.format(**dem)

    # prompt += template_no_label.format(**test_example).strip()
    prompt += template_no_label.format(**test_example)
    return prompt


def get_prompts_by_task(task, test_examples):
    prompt = json.load(open(os.path.join(os.path.dirname(__file__), 'prompts.json')))[task]
    # if len(test_examples) > 0:
    #     print(_embed_prompt(prompt['instruction'], prompt['template_with_label'],
    #                           prompt['template_no_label'], prompt['dem_examples'], test_examples[0]))
    #     exit()
    return [_embed_prompt(prompt['instruction'], prompt['template_with_label'],
                          prompt['template_no_label'], prompt['dem_examples'], test_example)
            for test_example in test_examples]

def get_k_shot_prompts_by_task(task, test_examples, k_shot=3):
    """
    Get prompts with only k demonstration examples.
    
    Args:
        task: The prompt task name
        test_examples: List of test examples  
        k_shot: Number of demonstration examples to use (0 for zero-shot)
    """
    prompt_data = json.load(open(os.path.join(os.path.dirname(__file__), 'prompts.json')))[task]
    
    # Get original demonstration examples
    original_dem_examples = prompt_data['dem_examples']
    
    # Select k examples (or all if k > available)
    if k_shot == 0:
        selected_dem_examples = []
    else:
        selected_dem_examples = original_dem_examples[:k_shot]
    
    # Build prompts using selected examples
    return [_embed_prompt(prompt_data['instruction'], 
                         prompt_data['template_with_label'],
                         prompt_data['template_no_label'], 
                         selected_dem_examples,  # Use selected examples
                         test_example)
            for test_example in test_examples]