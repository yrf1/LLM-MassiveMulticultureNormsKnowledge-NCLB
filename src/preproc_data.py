import os, openai
from tenacity import (retry, stop_after_attempt,
    wait_random_exponential, retry_if_exception_type)


openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")


@retry(retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, \
            openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
            wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(10))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def prompt_OpenAI(q, model_="gpt-3.5-turbo", max_token=64, temp=0.0):
    prompt_out = chat_completion_with_backoff(model=model_, \
            max_tokens=max_token, temperature=temp, \
            logprobs=True, top_logprobs=5, \
            messages=[{"role": "user", "content": q}]) 
    prompt_out, tot_tokens = prompt_out["choices"][0], prompt_out["usage"]["total_tokens"]
    return prompt_out["message"]["content"], prompt_out["logprobs"]["content"], tot_tokens

def topic_filt_4Wil(topic_cat_vec):
    topic_cat_vec = ['{:,.3f}'.format(x) for x in topic_cat_vec]
    topic_list = ["country-specific", "social norm", "cultural norm", "value or belief", "history", "politics", "fact"]
    if len(topic_cat_vec) in [12, 13]:
        topic_list += ["human interaction verbal communication", "human interaction physical communication", \
                "individual behavioral norm", "interactive behavioral norm", "general assertion", "specific fact or instance"]
    if len(topic_cat_vec) in [6, 12]:
        topic_list = topic_list[1:]
    topic_cat_vec = {topic_list[vvv_idx]: float(vvv) for vvv_idx, vvv in enumerate(topic_cat_vec)}
    if len(topic_cat_vec) <= 7:
        return topic_cat_vec, topic_cat_vec["social norm"] < 0.3
    xx = (topic_cat_vec["specific fact or instance"]) / (topic_cat_vec["specific fact or instance"] + \
            topic_cat_vec["general assertion"])
    return topic_cat_vec, topic_cat_vec["social norm"] < 0.3 or xx > 0.85

def enum_list_of_keys(dict_x):
    list_of_keys = []
    for k1, v1 in dict_x.items():
        if k1 == "metadata":
            continue
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                for k4, v4 in v3.items():
                    list_of_keys.append((k1,k2,k3,k4))
    return list_of_keys
