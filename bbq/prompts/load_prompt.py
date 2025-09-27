import json
import os
import config

def _embed_prompt(
    instruction, template_with_label, template_no_label, dem_examples, test_example
):
    prompt = str(instruction)
    for dem in dem_examples:
        prompt += template_with_label.format(**dem)

    # prompt += template_no_label.format(**test_example).strip()
    prompt += template_no_label.format(**test_example)
    return prompt


def get_prompts_by_task(task, test_examples, k_shot=None):
    prompt = json.load(open(os.path.join(os.path.dirname(__file__), config.PROMPT_FILE)))[task]
    # if len(test_examples) > 0:
    #     print(_embed_prompt(prompt['instruction'], prompt['template_with_label'],
    #                           prompt['template_no_label'], prompt['dem_examples'], test_examples[0]))
    #     exit()
    dem_examples = prompt['dem_examples']
    if k_shot is not None:
        dem_examples = dem_examples[:k_shot]
    return [_embed_prompt(prompt['instruction'], prompt['template_with_label'],
                          prompt['template_no_label'], dem_examples, test_example)
            for test_example in test_examples]

