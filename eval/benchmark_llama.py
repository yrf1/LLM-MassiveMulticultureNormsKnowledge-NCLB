# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
# Run llama-2-7b with 1 v100
torchrun --nproc_per_node 1 eval/benchmark_llama.py --ckpt_dir ckpts/llama/llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 4

# Run llama-2-13b with 2 v100
torchrun --nproc_per_node 2 example_chat_completion_benchmark_asess.py \
    --ckpt_dir /data/shared/Llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 4

# Run llama-7b with 1 v100
torchrun --nproc_per_node 1 example_chat_completion_benchmark_asess.py \
        --ckpt_dir llama-7b/ \
    --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 4

# Run llama-13b with 2 v100
torchrun --nproc_per_node 2 example_chat_completion_benchmark_asess.py \
        --ckpt_dir llama-13b/ \
    --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 4
"""
from typing import Optional

import json, os, math
import pandas as pd

from datetime import datetime

import sys
sys.path.append("eval")
from utils import read_computed_stats

from transformers import pipeline

sys.path.append("src")
from norm_frame import use_local_llama, load_model


def run_eval_per_s_query(query_s, generator, max_gen_len=64, ckpt_type=""):
    dialogs = [[{"role":"user","content":query_s+" True or False:"}]] 
    return generator.chat_completion(
            dialogs, max_gen_len=max_gen_len,
            temperature=0.0, top_p=1.0 #top_p,
        )[0][0]['generation']['content'] 

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

def ln_by_ln_inference(df, output_s_list, output_tracker, result_fname):
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
            pred_output = run_eval_per_s_query(query_s, generator, ckpt_type=our_ckpt_type)
                
            output_s_list.append((c, t1, tit, query_s, lbl, pred_output))
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
            batch_query.append(query_s+" True or False:")
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

def read_from_csv_agg(input_dir, result_fname, v_date, generator, our_ckpt_type):
    """
    New, Latest
    """
    df = pd.read_csv(input_dir+"benchmarK_"+v_date+".csv")
    output_s_list, output_new_count, output_tracker = [], 0, []
    if os.path.exists(result_fname):
        with open(result_fname, "r") as f:
            output_s_list = json.load(f)
        for (c, t1, tit, query_s, lbl, pred_output) in output_s_list:
            output_tracker.append((c, t1, tit, query_s))
    if our_ckpt_type == "hf":
        batch_inference(df, output_s_list, output_tracker, result_fname)
    else:
        ln_by_ln_inference(df, output_s_list, output_tracker, result_fname)
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
                            
    self_mode = ['evaluate','read computed stats'][0]

    our_ckpt_type = ""
    if self_mode != 'read computed stats':
        if os.path.exists(ckpt_dir+"consolidated.00.pth"): 
            from llama import Llama
            our_ckpt_type = "meta"
            generator = Llama.build(
                ckpt_dir=ckpt_dir,
                tokenizer_path="lib/llama/"+tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )
        else:
            our_ckpt_type = "hf"
            from vllm import LLM
            mp = 2 if "13b" in ckpt_dir.lower() else 1
            generator = LLM(model=ckpt_dir, tensor_parallel_size=mp)

    input_dir = "data/culture_scraped_info/"

    ckpt_stem = [x for x in ckpt_dir.split("/") if ("llama" in x.lower() and "b" in x.lower())][0]
    v_date = ["Feb4_neg10k"][0]
    if True:
        # Quick Test:  4k samples
        input_dir = input_dir.split("benchmark/")[0]
        result_fname = input_dir + "benchmark_"+v_date+"_"+ckpt_stem+"_eval_result_agg.json"
        
        if self_mode == "evaluate":
            start_t = datetime.now()
            output_s_list = read_from_csv_agg(input_dir, result_fname, v_date, generator, our_ckpt_type)
            
    if self_mode == "read computed stats":
        read_computed_stats(result_fname, v_date+"_"+ckpt_stem)

if __name__ == "__main__":
    # Check all model variants here: https://huggingface.co/meta-llama/Llama-2-7b (bottom of page)
    # I think it's like we can care about base (priority least, chat, chat-hf (priority top)
    # Model location in Blender server:
    # Model location in valdi server: 
    import fire
    fire.Fire(main)
    #ckpt = ["/shared/nas/data/m1/yifung2/yifung2/yifung/424/CTZL101_MM-NormSage/ckpts/llama/llama-2-7b-chat/", \
    #        "/shared/nas/data/m1/yifung2/yifung2/yifung/424/CTZL101_MM-NormSage/ckpts/llama/llama-2-7b/", \
    #        "/shared/nas/data/m1/yifung2/yifung2/yifung/424/CTZL101_MM-NormSage/ckpts/llama//llama-2-13b-chat/", \
    #        "/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-chat-hf/", \
    #        "/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-13b-chat-hf/"][-2]
    #main(ckpt_dir=ckpt, tokenizer_path="tokenizer.model")


    

