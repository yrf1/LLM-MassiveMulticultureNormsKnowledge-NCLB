"""
Types of Assessment:
T/F of a norm specific to a culture profile framing (so here I guess the norm can be juxtaposed)
T/F for a norm with a juxtaposed (swapped) culture profile framing source

Things to Keep in Mind:
- Can metadata track the norm (pristine vs manipulated)
- Last Update: Dec 28 with quality-assessed norms
"""
import os, json, pickle
import copy, openai, random
from datetime import datetime
from nltk.tokenize import sent_tokenize
from culture_crawling import read_parsed_culture_info
from norm_frame import NormFrame, get_llm_response, load_model
from transformers import pipeline
import pandas as pd
from scrape_web_info import request_SE, check_for_entailment_duplication
from tenacity import (retry, stop_after_attempt,
    wait_random_exponential, retry_if_exception_type)

import sys
sys.path.append("scripts/")
from qual_check_pos_data_gen import vDec25_make_pos_self_containes
#from qual_check_neg_data_gen import self_check_neg
from utils import enum_list_of_keys


openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TMPL_HARD_NEG = "Here are some plausible but incorrect cultural assertions:\n" + \
                     "Original knowledge: In Chinese culture, people commonly refer themselves as descendents of the dragon.\n" + \
                     "Incorrect knowledge: In Chinese culture, people commonly refer themselves as descendents of the phoenix.\n\n" + \
                     "Original knowledge: In Chinese culture, the traditional men clothing is the tangzhuang.\n" + \
                     "Incorrect knowledge: In Chinese culture, the traditional men clothing is the Qipao.\n\n" + \
                     "------\nNow, here is a text passage to consider:\n[S]\n\n" + \
                     "Based on this cultural context, come up with pairs of original knowledge and incorrect knowledge:"

PROMPT_TMPL_HARD_NEG = "[S]\nBased on this, create a negative sentence sample for trivia true/false assessment. "
PROMPT_TMPL_HARD_NEG += "The knowledge assertion should be false, from a cultural concept perspective, but the information logic should make sense and sound pretty plausible:"


def get_init_neg(s_topic_list, llm, rep_count=4):
    """ Step I. generate negative version """
    s_rewrite_prompts_in = [PROMPT_TMPL_HARD_NEG.replace("[S]",s_new) for s_new in s_topic_list]
    init_neg_list, init_neg_list_cache = [], []
    for rep in range(rep_count):
        s_rewrite_prompts_out = get_llm_response(s_rewrite_prompts_in, max_gen_len=128, temp=0.7, llm_=llm)
        #s_rewrite_prompts_out = prompt(s_rewrite_prompts_in, model_="gpt-3.5-turbo", temp=0.7)
        s_rewrite_prompts_out = [x.lstrip().rstrip().split("Explanation:")[0] for x in s_rewrite_prompts_out]
        s_rewrite_prompts_out_new = []
        for s_neg_out in s_rewrite_prompts_out:
            # Preprocess out the neg norm from generated text
            if "Incorrect knowledge: " in s_neg_out:
                s_neg_out = [s.split("Incorrect knowledge: ")[-1] for s in s_neg_out.split("\n") \
                    if "Incorrect knowledge: " in s]
            else:
                s_neg_out = s_neg_out.split("\nPlease ")[0].split("\n")
            s_neg_out = [(s[3:] if s[1:3]==". " else s) for s in s_neg_out]
            s_neg_out = [s for s in s_neg_out if len(s)>13 and "__" not in s and ".." not in s and s[-1]=="." and s[-1]!=":" and s[-1]!="]" and s[-1]!="?"]
            s_neg_out = [s for s in s_neg_out if s[:7]!="Please " and s[:4]!="Hint" \
                    and s[:4]!="Note" and s[:5]!="I'll " and s[:2]!="I " and s[:4]!="And " and s[:2]!="* "]
            s_neg_out = [s for s in s_neg_out]
            s_rewrite_prompts_out_new.append(s_neg_out)
        if len(init_neg_list) == 0:
            init_neg_list = s_rewrite_prompts_out_new
            init_neg_list_cache = s_rewrite_prompts_out
        else:
            for i in range(len(init_neg_list)):
                init_neg_list[i] += s_rewrite_prompts_out_new[i]
                init_neg_list_cache += s_rewrite_prompts_out[i]
    return init_neg_list, (s_rewrite_prompts_in, init_neg_list_cache)

def get_init_neg_ChatGPT(s_topic_list, llm, rep_count=2):
    s_rewrite_prompts_in = [PROMPT_TMPL_HARD_NEG.replace("[S]",s_new) for s_new in s_topic_list]
    s_rewrite_prompts_out = []
    for x_in in s_rewrite_prompts_in:
        x_out_1 = prompt_OpenAI(x_in, model_="gpt-3.5-turbo", temp=0.7)[0]
        x_out_2 = prompt_OpenAI(x_in, model_="gpt-3.5-turbo", temp=0.7)[0]
        s_rewrite_prompts_out.append([x_out_1, x_out_2])
    return s_rewrite_prompts_out

def self_check_neg(init_neg_list, llm, init_pos_list=None):
    self_checked_list_expl, self_checked_list_verdict = [], []
    for idx, init_negs in enumerate(init_neg_list):
        init_negs = list(set(init_negs))
        prompts_in = [x+'\n\nIs this "true" or "false" (one word answer):' for x in init_negs]
        #s_rewrite_prompts_out = get_llm_response(prompts_in, max_gen_len=64, temp=0.0, llm_=llm)
        s_rewrite_prompts_out = [prompt_OpenAI(x, temp=0.0)[0] for x in prompts_in]  #, llm_=llm)
        #for x_idx, x in enumerate(s_rewrite_prom
        self_checked_list_expl.append(s_rewrite_prompts_out)
        init_negs_check_verdict = [("False" if "\n\nTrue" not in s_rewrite_prompts_out[x_idx] else "True") \
                for x_idx, x in enumerate(init_negs)]
        self_checked_list_verdict.append(init_negs_check_verdict)
        if init_pos_list is not None:
            for x_idx, x in enumerate(init_negs):
                if self_checked_list_verdict[-1][x_idx] == "True":
                    continue
                prompt_in = 'Sent 1:\n' + init_pos_list[idx] + "\n\n" + \
                            'Sent 2:\n' + x + "\n\n" + \
                            'Does Sent 1 and Sent 2 convey similar meaning? ' + \
                            'Answer "true" or "false" (one word):'
                s_rewrite_prompt_out = prompt_OpenAI(prompt_in, temp=0.0)[0]
                self_checked_list_expl[-1][x_idx] = s_rewrite_prompt_out
                double_checked_neg = "False" if "\n\nTrue" not in s_rewrite_prompts_out else "True"
                self_checked_list_verdict[-1][x_idx] = double_checked_neg
    return self_checked_list_expl, self_checked_list_verdict

def web_check(neg_list, llm, se_cache={}):
    web_checked_list, web_checked_list_tracker = [], []
    for s_neg in neg_list:
        this_web_checked_list = []
        this_web_checked_list_tracker = []
        passed_idx_tracker = []
        for s_neg_norm_idx, s_neg_norm in enumerate(s_neg):
            if True: #try:
                se_result = request_SE(s_neg_norm)
                se_cache[s_neg_norm] = se_result
            # TODO: Switch this to better performing model
                se_result = json.loads(se_result)
                s_rewrite_prompts_in = ["Background Context:\n"+"\n".join([x["snippet"] for x in se_result["items"]]) + \
                    "\n\nQuery:\n"+s_neg_norm+"\n\n"+ \
                    "Is the query sentence factually entailed or supported by any sentence in the background context? Yes or No:"]
                s_rewrite_prompts_out = get_llm_response(s_rewrite_prompts_in, max_gen_len=128, temp=0.0, llm_=llm)
                s_rewrite_prompts_out = [x.lstrip().rstrip() for x in s_rewrite_prompts_out][0].replace("Answer: ", "")
                if "Yes" in s_rewrite_prompts_out:
                    this_web_checked_list.append(s_rewrite_prompts_out)
                    passed_idx_tracker.append(s_neg_norm_idx)
                this_web_checked_list_tracker.append(s_rewrite_prompts_out)
            #if "no" in s_rewrite_prompts_out.lower()[:5]:
            #    t1_tracker_neg.append((country, t1, t2, t3, url, p, s, s_neg_norm, lbl))
        #except:
        #    pass
        web_checked_list.append(this_web_checked_list)
        web_checked_list_tracker.append(this_web_checked_list_tracker)
    return web_checked_list, web_checked_list_tracker, se_cache, passed_idx_tracker

def helper_make_standalone_TF(culture, pg_title, context, s):
    x_in = "Culture:\n"+culture+"\n\n"
    x_in += "Document Title:\n"+pg_title+"\n\n"
    x_in += "Context:\n"+context+"\n\n"
    x_in += "Query:\n"+s+"\n\n"
    x_in += "Based on the background info, help turn the query into a standalone True or False exam question:"
    x_out = prompt_OpenAI(x_in)[0]
    # TODO: proc output
    return x_out

def engine(mode="extract_NormFrame", fname = None):
    """
    TODO (Dec 28th)
    i.   Check data source
    ii.  Update positive data processing
    iii. Uupdate negative processing
    """
    if mode=="extract_NormFrame":
        llm, _ = load_model("/data/shared/Llama-2-13b-chat-hf")
    else:
        llm = "gpt-turbo-3.5"
    mnli_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    data_dir = "data/" #"/data/yifung/CTZL101_MM-NormSage/data/"
    fname = "culture_scraped_info/culture_info_beta_Nov21.pkl"
    scraped_data = read_parsed_culture_info(data_dir, fname) if fname is not None else read_parsed_culture_info()
    wFineProfiling = True
    if wFineProfiling:
       scraped_data_wFineProfiling = scraped_data
    count_of_neg_produced = 0
    benchmark_tracker = []
    t1_tracker_pos, t1_tracker_neg = [], []
    for (country, t1, t2, t3) in enum_list_of_keys(scraped_data):
        t3_data = scraped_data[country][t1][t2][t3]
        if "sent_profile_extracted" not in t3_data or "self_contain" not in t3_data:
            continue
        if "neg_gen_tracker" not in t3_data:
            scraped_data[country][t1][t2][t3]["neg_gen_tracker"] = {}
        for sent, sent_profile_data in t3_data["sent_profile_extracted"].items():
            topic = t3_data['paragraphs_categorized'][1][sent]
            # TODO: check ln below
            p = [x for x in t3_data['paragraphs_categorized'][0] if x[-1]==sent]
            p = p[0][-2] if len(p)>0 else ""
            # TODO: check ln below
            if sent not in t3_data["self_contain"] or "norm_violation_rlv" not in t3_data:
                continue
            sent_new = t3_data["self_contain"][sent] if sent in t3_data["self_contain"] else sent
            t1_tracker_pos.append((country, t1, t2, t3, t3_data["url"], topic, p.split(sent)[0], \
                    sent, sent_new, sent_new, True, None, None, None, None, None, None))
            # Neg Data Generation Now
            s_filler_list = [sent_new]
            init_neg_list, init_neg_cache = get_init_neg(s_filler_list, llm)
            neg_self_check_expl, neg_self_check_verdict = self_check_neg(init_neg_list, llm, s_filler_list)
            try:
                neg_self_checked = [x for x_idx, x in enumerate(init_neg_list[0]) \
                    if "false" in neg_self_check_verdict[0][x_idx].lower()]
                scraped_data[country][t1][t2][t3]["neg_gen_tracker"][sent] = {"init_neg_list":init_neg_list[0], \
                    "selfchecked_neg":neg_self_checked, "neg_self_check_expl":neg_self_check_expl[0], \
                    "neg_self_check_verdict":neg_self_check_verdict[0]}
                count_of_neg_produced += 1
            except:
                pass
        if count_of_neg_produced % 10 == 0:
            print(count_of_neg_produced, datetime.now())
            with open(data_dir+fname.replace(".","_backup_wNeg."), "wb") as f:
                pickle.dump(scraped_data, f)
            with open(data_dir+fname, "wb") as f:
                pickle.dump(scraped_data, f)
    with open(data_dir+fname.replace(".","_backup_wNeg."), "wb") as f:
        pickle.dump(scraped_data, f)
    with open(data_dir+fname, "wb") as f:
        pickle.dump(scraped_data, f)

def write_as_eval_benchmark():
    data_dir = "/data/yifung/CTZL101_MM-NormSage/data/"
    df = pd.read_csv(data_dir+"culture_scraped_info/benchmarK_Jan16.csv")
    print(df[df["10"]==True].shape, df[df["10"]==False])
    for idx, data in df.iterrows():
        c, t1, tit, lbl = data[0], data[1], data[4], data[10]
        query_s = data[9]
        print(c, t1, tit, lbl)
        print(query_s)
    return


if __name__ == "__main__":
    start_t = datetime.now()
    engine(mode="cache4benchmark") 



