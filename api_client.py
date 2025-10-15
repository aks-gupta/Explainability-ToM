import openai
from together import Together
import time
import os
import pickle as pkl
import json
import boto3
import random

aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")  # default if not set

client_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),  # Change if needed
)

client_openai = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # base_url="https://cmu.litellm.ai",
    base_url="https://ai-gateway.andrew.cmu.edu/",
)

def call_openai_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=200, stop=None):
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            response = client_openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                # max_tokens=max_tokens,
            )
            # print(f"DEBUG OpenAI Response {i}: {response.choices[0].message}")
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"[{i}] Error during call:\nPrompt: {prompt[:100]}...\nError: {e}")
            responses.append("Error: Unable to generate response")
            time.sleep(1)
    return responses

client_together = Together()

def call_together_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=200, stop=None):
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            response = client_together.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                # max_tokens=max_tokens,
                # stop=stop
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"[{i}] Error during call:\nPrompt: {prompt[:100]}...\nError: {e}")
            responses.append("Error: Unable to generate response")
            time.sleep(1)
    return responses


def call_bedrock_api(model, prompts, bsz=1, num_processes=1, temperature=0, top_p=1.0, max_tokens=None, stop=None, max_retries=50):
    """
    Call AWS Bedrock models in a similar fashion to OpenAI's chat completion API.
    Supports both Anthropic Claude and Titan text models.
    """
    responses = []
    if max_tokens is None:
        # Use model-specific default max
        if "claude" in model:
            max_tokens = 100000
        elif "mistral" in model:
            max_tokens = 3200
        else:
            max_tokens = 2048  # safe fallback
    for i, prompt in enumerate(prompts):
        attempt = 0
        while attempt < max_retries:
            try:
                # Anthropic Claude models (Claude 3 family)
                if "anthropic" in model or "claude" in model:
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",  # ✅ REQUIRED FIELD
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": prompt}]}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                # Mistral models
                elif "mistral" in model:
                    body = {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    if stop:
                        # ensure stop is a list
                        body["stop"] = stop if isinstance(stop, list) else [stop]
                # Invoke Bedrock
                response = client_bedrock.invoke_model(
                    modelId=model,
                    body=json.dumps(body)
                )
                result = json.loads(response["body"].read())
                # Parse output depending on model family
                if "content" in result:  # Claude 3
                    output = result["content"][0]["text"]
                elif "completion" in result:  # Older Claude
                    output = result["completion"]
                elif "outputs" in result:  # Mistral
                    output = result["outputs"][0]["text"]
                else:
                    output = str(result)

                responses.append(output)
                break
            except (client_bedrock.exceptions.ServiceUnavailableException,
                    client_bedrock.exceptions.ThrottlingException) as e:
                # Exponential backoff with jitter
                attempt += 1
                wait_time = (2 ** attempt) + random.random()
                print(f"[{i}] Transient error ({type(e).__name__}), retry {attempt}/{max_retries}, "
                      f"waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"[{i}] Non-retryable error during call:\nPrompt: {prompt[:100]}...\nError: {e}")
                responses.append("Error: Unable to generate response")
                break  # do not retry for other errors
        else:
            # Max retries reached
            print(f"[{i}] Failed after {max_retries} retries for prompt: {prompt[:100]}...")
            responses.append("Error: Max retries exceeded")
    return responses