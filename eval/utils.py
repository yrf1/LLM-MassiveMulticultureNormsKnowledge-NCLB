import os, json, pandas as pd, pickle


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

def filter_by_CultureProfileElements(output_s_list, scraped_data_wProfile, mode="subCulture", culture_dimension=""):
    output_s_list_filtered, profile_metadata_list = [], []
    counttt = 0
    for (c, t1, tit, query_s, lbl, pred_output) in output_s_list:
        # Find culture profile
        this_culture_profile = None
        for k1, v1 in scraped_data_wProfile.items():
            if k1 == "metadata":
                continue
            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    for k4, v4 in v3.items():
                        if v4["url"] == tit:
                            for sent, sent_v in v4['sent_profile_extracted'].items():
                                if sent_v["sent_elaborated"] == query_s:
                                    this_culture_profile = sent_v
                                    continue
        # Check culture dimension of interest for data filtering
        if mode == "subCulture" and this_culture_profile is not None:
            if this_culture_profile[culture_dimension] != "N/A":
                output_s_list_filtered.append((c, t1, tit, query_s, lbl, pred_output))
        elif mode == "genCulture" and this_culture_profile is not None:
            if this_culture_profile[culture_dimension] == "N/A":
                output_s_list_filtered.append((c, t1, tit, query_s, lbl, pred_output))
        else:
            output_s_list_filtered.append((c, t1, tit, query_s, lbl, pred_output))
            profile_metadata_list.append(this_culture_profile)
    return output_s_list_filtered, profile_metadata_list

with open("data/culture_scraped_info/culture_info_beta_Nov21.pkl", "rb") as f:
    scraped_data = pickle.load(f)
list_of_interest = []
for (k1, k2, k3, k4) in enum_list_of_keys(scraped_data):
    v4 = scraped_data[k1][k2][k3][k4]
    if "norm_violation_rlv" not in v4 or "self_contain" not in v4:
        continue
    for kk, vv in v4['sent_profile_extracted'].items():
        if vv['subcountry_region_extraction'] == "N/A" or kk not in v4["self_contain"]:
            continue
        list_of_interest.append(v4["self_contain"][kk])
print(len(list_of_interest))

from transformers import pipeline
def read_computed_stats(result_fname, model_name="", data_dir="data/culture_scraped_info/", scraped_data_wProfile=None):
    mnli_classifier = None #pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    print(result_fname)
    if not os.path.exists(result_fname):
        return
    with open(result_fname, "r") as f:
        output_s_list = json.load(f)
    orig_benchmark = pd.read_csv(result_fname.split("benchmark_")[0]+"benchmarK_Feb4_pos10k.csv")
    fname = data_dir+"culture_info_beta_Nov21_4Wil.json"
    if os.path.exists(fname):
        with open(fname, "r") as f:
           scraped_data_wProfile = json.load(f)
    result_stat_viz_tracker = ""
    with open("data/culture_taxonomy/countries_mid_resource.txt", "r") as f:
        countries_of_interest = f.read()
        countries_of_interest = countries_of_interest.split("\n")[:-1]
    with open("data/culture_taxonomy/countries_high_resource.txt", "r") as f:
        country_hresource = f.read()
        countries_of_interest += country_hresource.split("\n")[:-1]
    for result_ablation_mode in ["all"]:
        #, "by_hresource", "by_mresource", "by_lresource", \
        #    "by_nature", "by_topic", "by_subculture"][4:5]: #, "by_country"]:
        result_ablation_mode_subcat_list = list(set([x[0] for x in output_s_list])) if result_ablation_mode == "by_country" \
                else [["United States of America", "France", "Spain", "China", "Japan"]] if result_ablation_mode == "by_hresource" \
                else [["Türkiye", "Egypt", "Islamic Republic of Iran", "Malaysia", "Argentina"]] if result_ablation_mode == "by_mresource" \
                else [["Lao People's Democratic Republic", "Bhutan", "Congo", "Serbia"]] if result_ablation_mode == "by_lresource" \
                else ["social norm","cultural norm","value or belief","history","politics","fact"] if result_ablation_mode == "by_nature" \
                else list(set([x[1] for x in output_s_list])) if result_ablation_mode == "by_topic" \
                else [(b, a) for a in ["subcountry_region_extraction", "ethnicity_extraction", "age_extraction", "gender_extraction", \
                "religion_extraction", "marital_status_extraction", "occupation_extraction"] for b in ["subCulture","genCulture"]] if result_ablation_mode == "by_subculture" \
                else ["all"] 
        for result_ablation_mode_subcat in result_ablation_mode_subcat_list:
            tp, fp, fn, tn = 0.0001, 0.0001, 0.0001, 0.0001
            output_s_list_ = output_s_list
            if result_ablation_mode == "by_subculture":
                mode, culture_dimension = result_ablation_mode_subcat
            if result_ablation_mode == "by_nature":
                mode, culture_dimension = result_ablation_mode_subcat, ""
            if result_ablation_mode == "by_subculture": 
                output_s_list_, profile_metadata_list = filter_by_CultureProfileElements(output_s_list, \
                        scraped_data_wProfile, mode, culture_dimension)
            for data_idx, data in enumerate(output_s_list_):
                if len(data) == 2:
                    a, b = data
                elif len(data) == 6:
                    c, t1, tit, query_s, a, b = data
                    if query_s in list_of_interest:
                        continue
                    # TODO: check here for by_nature ablation evaluation
                    if result_ablation_mode == "by_nature":
                        #if profile_metadata_list[data_idx] is not None:
                        #    nature_categorization = profile_metadata_list[data_idx]["topic_categorization_expanded"]
                        if True: #else:
                            nature_categorization_ = orig_benchmark[orig_benchmark["9"]==query_s]
                            nature_categorization_ = nature_categorization_[nature_categorization_["10"]==a]
                            nature_categorization = orig_benchmark[orig_benchmark["9"]==query_s]
                            #nature_categorization = nature_categorization["5"].values[0]\
                            if nature_categorization.shape[0] == 0:
                                continue
                            """
                            try:
                                nature_categorization = {"social norm":nature_categorization["11"].values[0], \
                                    "cultural norm":nature_categorization["12"].values[0],\
                                    "value or belief":nature_categorization["13"].values[0],\
                                    "history":nature_categorization["14"].values[0],\
                                    "politics":nature_categorization["15"].values[0],\
                                    "fact":nature_categorization["16"].values[0]}
                            except:
                                continue
                            """
                            #mnli_classifier([query_s], \
                            #        ["social norm", "cultural norm", "value or belief", "history", "politics", "fact"])
                    if result_ablation_mode == "by_country" and c != result_ablation_mode_subcat:
                        continue
                    elif result_ablation_mode in ["by_hresource", "by_mresource", "by_lresource"] and \
                            c not in result_ablation_mode_subcat:
                        continue
                    elif result_ablation_mode == "by_topic" and t1 != result_ablation_mode_subcat:
                        continue
                    elif result_ablation_mode == "by_nature" and \
                        (nature_categorization_["norm_violation_rlv"].values[0] > 0.6 \
                        or nature_categorization_["norm_violation_rlv"].values[0] < 0.4):
                        continue
                    #elif result_ablation_mode == "by_nature" and nature_categorization[0]["labels"][0] \
                    #        != result_ablation_mode_subcat:
                    #elif result_ablation_mode == "by_nature" and \
                        #nature_categorization[result_ablation_mode_subcat] \
                        #    != max([nature_categorization[nature_cat] for nature_cat in \
                        #    ["social norm","cultural norm","value or belief","history","politics","fact"]]):
                b = str(b).replace("Answer: ", "")
                if b.lstrip()[:4] == "True" or b.lstrip()[:7] == "correct":
                    if a == True:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if a == True:
                        fn += 1
                    else:
                        tn += 1

            acc = (tp+tn)/(tp+fp+fn+tn)
            p, r = tp/(tp+fp), tp/(tp+fn)
            f = 2*(p*r)/(p+r)

            if type(result_ablation_mode_subcat) is not str:
                result_ablation_mode_subcat = " ".join(result_ablation_mode_subcat)

            result_stat_viz_tracker += "====== " + result_ablation_mode + " " + \
                    (result_ablation_mode_subcat if type(result_ablation_mode_subcat) is str else "") + "\n"
            result_stat_viz_tracker += "TP, FP, FN, TN: " + str(int(tp)) + " " + str(int(fp)) + \
                    " " + str(int(fn)) + " " + str(int(tn)) + "\n"
            result_stat_viz_tracker += "tot count, tot pos, tot neg: " + str(int(tp+fp+fn+tn)) + " " \
                    + str(int(tp+fn)) + " " + str(int(fp+tn)) + "\n"
            result_stat_viz_tracker += "Acc: " + "{:.3f}".format(acc) + "\n"
            result_stat_viz_tracker += "P, R, F: " + "{:.3f}".format(p) + " " + "{:.3f}".format(r) + " " \
                    + "{:.3f}".format(f) + "\n"
        result_stat_viz_tracker += "\n"
    return acc, result_stat_viz_tracker

def derive_norm_usefulness_of_benchmark(fname):
    """
    Dec30 -- "0" to "16"
    Jan16 -- "0" to "16"
    """
    import openai, math
    openai.organization = os.getenv("OPENAI_ORG")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    df = pd.read_csv(fname)
    df["norm_violation_rlv"] = 0.0
    for idx, data in df.iterrows():
        try:
            s = data["9"]
            prompt_in = s + "\n\nIs this information relevant for avoiding potential social norm violations? Answer 'yes' or 'no':"
            prompt_out = openai.ChatCompletion.create(
                       model="gpt-3.5-turbo",
                       logprobs=True, top_logprobs=5, temperature=0.0,
                       messages=[
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": prompt_in}
                       ]
                     )["choices"][0]
            token_probs = [x for x in prompt_out["logprobs"]["content"] if x["token"] in ["Yes","No","yes","no"]][0]
            yes_prob = max([-16]+[d["logprob"] for d in token_probs["top_logprobs"] if d["token"] in ["Yes","yes"]])
            no_prob = max([-16]+[d["logprob"] for d in token_probs["top_logprobs"] if d["token"] in ["No","no"]])
            yes_prob, no_prob = math.exp(yes_prob), math.exp(no_prob)
            norm_violation_rlv_est = yes_prob / (yes_prob + no_prob)
            df.at[idx,'norm_violation_rlv'] = norm_violation_rlv_est
        except:
            continue
    df.to_csv(fname, index=False)
    return 

if __name__ == "__main__":
    data_dir = "data/culture_scraped_info/"
    for data_model_result_name in [(data_dir+"benchmark_Feb4_pos10k_llama-2-7b-chat_eval_result_agg.json", \
        data_dir+"benchmark_Feb4_neg10k_llama-2-7b-chat_eval_result_agg.json", "llama-2-7b-chat"), \
            (data_dir+"benchmark_Feb4_pos10k_Llama-2-7b-chat-hf_eval_result_agg.json", \
                data_dir+"benchmark_Feb4_neg10k_Llama-2-7b-chat-hf_eval_result_agg.json", "llama-2-7b-chat-hf"), \
            (data_dir+"benchmark_Feb4_pos10k_llama-2-13b-chat_eval_result_agg.json", \
                data_dir+"benchmark_Feb4_neg10k_llama-2-13b-chat_eval_result_agg.json", "llama-2-13b-chat"), \
            (data_dir+"benchmark_Feb4_pos10k_Llama-2-13b-chat-hf_eval_result_agg.json", \
                data_dir+"benchmark_Feb4_neg10k_Llama-2-13b-chat-hf_eval_result_agg.json", "llama-2-13b-chat-hf"), \
            (data_dir+"benchmark_Feb4_pos10k_gpt-3.5-turbo_eval_result_agg.json","gpt-3.5-turbo"), \
            (data_dir+"benchmark_Feb4_pos10k_gpt-4_eval_result_agg.json","gpt-4"), \
            (data_dir+"benchmark_Feb4_pos10k_vicuna-7b_eval_result_agg.json", \
                data_dir+"benchmark_Feb4_neg_vicuna-7b_eval_result_agg.json", "vicuna-7b-v1.3"), \
            (data_dir+"benchmark_Feb4_vicuna-13b_eval_result_agg.json", \
                data_dir+"benchmark_Feb4_neg_vicuna-13b_eval_result_agg.json", "vicuna-13b-v1.3"), \
            (data_dir+"benchmark_Feb4_pos10k_Mistral-7B-Instruct-v0.2_eval_result_agg.json", \
                data_dir+"benchmark_Feb4_neg10k_Mistral-7B-Instruct-v0.2_eval_result_agg.json", "Mistral-7B-Instruct-v0.2")][-1:]:
        #benchmark_Feb4_pos10k_Mistral-7B-Instruct-v0.2_eval_result_agg.json 
        if len(data_model_result_name) == 3:
            result_fname1, result_fname2, model_name = data_model_result_name
            acc1, result_stat_viz_tracker1 = read_computed_stats(result_fname1, model_name)
            acc2, result_stat_viz_tracker2 = read_computed_stats(result_fname2, model_name)
            result_stat_viz_tracker = result_stat_viz_tracker2
            print(model_name, "{:.3f}".format(acc2), "{:.3f}".format(acc1), "{:.3f}".format(2*acc2*acc1/(acc2+acc1)))
        if len(data_model_result_name) == 2:
            result_fname, model_name = data_model_result_name
            acc, result_stat_viz_tracker = read_computed_stats(result_fname, model_name)
            print(result_stat_viz_tracker)
            print("outputs/stat/benchmark_"+model_name+"_eval_results.txt") #, "{:.3f}".format(acc))
        with open("outputs/stat/benchmark_"+model_name+"_eval_results.txt", "w") as f:
            f.write(result_stat_viz_tracker)
    quit()
    # Unit Tests
    data_dir = "/data/yifung/CTZL101_MM-NormSage/data/culture_scraped_info/" 
    fname = data_dir + "benchmark_Llama-2-13b-hf_eval_result_agg.json"
    if False:
        derive_norm_usefulness_of_benchmark(data_dir+"benchmarK_Dec30.csv")
    output_s_list = []
    if os.path.exists(fname):
        with open(fname, "r") as f:
           output_s_list = json.load(f)
    fname = data_dir+"culture_info_beta_Nov21_4Wil.json"
    if os.path.exists(fname):
        with open(fname, "r") as f:
           scraped_data_wProfile = json.load(f)
    filter_by_CultureProfileElements(output_s_list, scraped_data_wProfile)
