# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""

"""
from typing import Optional

import json, os
import pandas as pd

from datetime import datetime

import sys
sys.path.append("eval")
from utils import read_computed_stats

from transformers import pipeline

import openai
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
            messages=[{"role": "user", "content": q+" True or False:"}])["choices"][0]["message"]["content"]
    return prompt_out

def run_eval_per_s_query(query_s, generator, max_gen_len=64, ckpt_type=""):
    dialogs = [[{"role":"user","content":query_s+" True or False:"}]] 
    if ckpt_type == "meta":
        return generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=0.0, #temperature,
            top_p=1.0 #top_p,
        )[0][0]['generation']['content']
    elif ckpt_type == "hf":
        return use_local_llama(query_s+" True or False:", max_gen_len, 0.0, generator, stop_=['</s>'])[0]

def read_from_csv_agg(input_dir, result_fname, vdate, generator, our_ckpt_type):
    """
    New, Latest
    """
    df = pd.read_csv(input_dir+"benchmarK_"+vdate+".csv")
    output_s_list = []

    for idx, data in df.iterrows():
        c, t1, tit, lbl = data[0], data[1], data[4], data[10]
        query_s = data[9]
        if True: #try:
            if type(query_s) is float:
                continue
            if "[NEG NORM]" in query_s:
                query_s = query_s.split("[NEG NORM]")[0]
            
            pred_output = prompt_OpenAI(query_s, model_=generator)
            
            #pred_output = run_eval_per_s_query(query_s, generator, ckpt_type=our_ckpt_type)
                
            output_s_list.append((c, t1, tit, query_s, lbl, pred_output))
        #except:
        #    pass
        if idx % 100 == 0:
            print("~~~~", idx, df.shape[0], datetime.now())
            with open(result_fname, "w") as f:
                json.dump(output_s_list, f)
    return output_s_list

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
                            
    self_mode = ['evaluate','read computed stats'][0]

    generator = ckpt_dir

    input_dir = "data/culture_scraped_info/benchmark/"

    ckpt_stem = ckpt_dir 

    v_date = ["Feb4_pos10k"][0]

    input_dir = input_dir.split("benchmark/")[0]
    result_fname = input_dir + "benchmark_"+v_date+"_"+ckpt_stem+"_eval_result_agg.json"
    
    if self_mode == "evaluate":
        start_t = datetime.now()
        our_ckpt_type = ""
        #output_s_list = prompt_OpenAI(, model_="gpt-3.5-turbo", max_token=64)
        output_s_list = read_from_csv_agg(input_dir, result_fname, v_date, generator, our_ckpt_type)
        
    if self_mode == "read computed stats":
        read_computed_stats(result_fname, ckpt_stem)

if __name__ == "__main__":
    model_name = ["gpt-3.5-turbo","gpt-4"][1]
    main(ckpt_dir=model_name, tokenizer_path="")

