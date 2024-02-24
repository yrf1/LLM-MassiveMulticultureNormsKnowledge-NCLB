# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
"""
from typing import Optional

import json, os, math
import pandas as pd

from datetime import datetime

import sys, torch
sys.path.append("eval")
from utils import read_computed_stats

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append("src")
#from utils import prompt_OpenAI
from norm_frame import use_local_llama, load_model


def run_eval_per_s_query(query_s, generator, max_gen_len=64, ckpt_type=""):
    #print(query_s)
    query_s = torch.tensor([query_s["input_ids"]])
    s_out = generator.generate(query_s) #+" True or False:")
    return s_out
    #dialogs = [[{"role":"user","content":query_s+" True or False:"}]] 
    #return generator.chat_completion(
    #        dialogs, max_gen_len=max_gen_len,
    #        temperature=0.0, top_p=1.0 #top_p,
    #    )[0][0]['generation']['content'] 

def read_from_json_per_c_t(input_dir, country, topic):    
    """
    Old/Deprecated? Needs to be Updated
    """
    result_fname = input_dir+"/"+country+"/"+topic+"/1_output_"+ckpt_dir.replace("/","")+".json"
    
    if self_mode == "evaluate":
        if os.path.exists(result_fname):
            return None, None

        try:
            with open(input_dir+"/"+country+"/"+topic+"/1.json", "r") as f:
                input_s_list = json.load(f)
        except:
            print("Cannot load and skipping... "+input_dir+"/"+country+"/"+topic+"/1.json")
            return None, None

        results = []
        for x in input_s_list:
            try:
                pred_output = run_eval_per_s_query(x)
                output_s_list.append((query_s, \
                   pred_output['generation']['content']))
            except:
                pass
    return output_s_list, result_fname

def ln_by_ln_inference(df, model, tokenizer, output_s_list, output_tracker, result_fname):
    output_new_count = 0
    for idx, data in df.iterrows():
        c, t1, tit, lbl = data[0], data[1], data[4], data[10]
        if (c, t1, tit, data[9]) in output_tracker:
            continue
        query_s = data[9]
        if True: #try:
            if type(query_s) is float:
                continue
            if "[NEG NORM]" in query_s:
                query_s = query_s.split("[NEG NORM]")[0]
            q_tok = tokenizer("[INST] "+query_s+" True or False: [/INST]")
            pred_output = run_eval_per_s_query(q_tok, model) #, ckpt_type=our_ckpt_type)
            pred_out = tokenizer.decode(pred_output[0])
            pred_out = pred_out.split("[/INST] ")[-1]
            pred_out = True if "true" in pred_out.lower() else False
            output_s_list.append((c, t1, tit, query_s, lbl, pred_out))
            output_new_count += 1
        #except:
        #    pass
        if output_new_count % 100 == 0:
            print("~~~~", idx, len(output_s_list), df.shape[0], datetime.now())
            with open(result_fname, "w") as f:
                json.dump(output_s_list, f)
    return output_s_list

def batch_inference(df, output_s_list, output_tracker, result_fname):
    batch_sz, tot_data_len, output_new_count = 256, df.shape[0], 0
    for batch_idx in range(int(math.ceil(tot_data_len/batch_sz))):
        idx1, idx2 = batch_sz*batch_idx,batch_sz*(batch_idx+1)
        batch_query, batch_cache = [], []
        for idx, data in df[idx1:idx2].iterrows():
            c, t1, tit, lbl = data[0], data[1], data[4], data[10]
            if (c, t1, tit, data[9]) in output_tracker:
                continue
            batch_query.append("[/INST] "+query_s+" True or False: ")
            batch_cache.append((c,t1,tit,lbl))
        batch_result = use_local_llama(batch_query, 64, 0.0, generator, stop_=['</s>'])[0]
        for idx, result_x in enumerate(batch_result):
            c, t1, tit, lbl = batch_cache[idx]
            query_s = batch_query[idx]
            output_s_list.append((c, t1, tit, query_s, lbl, result_x))
            output_new_count += 1
        if output_new_count % 100 == 0:
            with open(result_fname, "w") as f:
                json.dump(output_s_list, f)
    return output_s_list

def read_from_csv_agg(input_dir, result_fname, v_date, generator, tokenizer, our_ckpt_type):
    """
    New, Latest
    """
    df = pd.read_csv(input_dir+"benchmarK_"+v_date+".csv")
    #df = df[:10]
    output_s_list, output_new_count, output_tracker = [], 0, []
    if os.path.exists(result_fname):
        with open(result_fname, "r") as f:
            output_s_list = json.load(f)
        for (c, t1, tit, query_s, lbl, pred_output) in output_s_list:
            output_tracker.append((c, t1, tit, query_s))
    #if our_ckpt_type == "hf":
    #    batch_inference(df, output_s_list, output_tracker, result_fname)
    #else:
    output_s_list = ln_by_ln_inference(df, generator, tokenizer, output_s_list, output_tracker, result_fname)
    with open(result_fname, "w") as f:
        json.dump(output_s_list, f)
    return output_s_list

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 512,
    max_gen_len: Optional[int] = None,
):                     

    tokenizer = AutoTokenizer.from_pretrained(
                    ckpt_dir, model_max_length=512)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir)

    input_dir = "data/culture_scraped_info/"

    ckpt_stem = ckpt_dir.split("/")[-1] #[x for x in ckpt_dir.split("/") if ("llama" in x.lower() and "b" in x.lower())][0]
    v_date = ["Feb4_neg10k"][0]
    # Quick Test:  4k samples
    input_dir = input_dir.split("benchmark/")[0]
    result_fname = input_dir + "benchmark_"+v_date+"_"+ckpt_stem+"_eval_result_agg.json"
    
    output_s_list = read_from_csv_agg(input_dir, result_fname, v_date, model, tokenizer, our_ckpt_type)

if __name__ == "__main__":
    ckpt_name = ["google/gemma-7b"]
    ckpt_name = ckpt_name[0]
    main(ckpt_dir=ckpt_name, tokenizer_path="tokenizer.model")