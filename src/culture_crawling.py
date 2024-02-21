"""
Algorithm (2 days total to run)
I. Crawl through the culture-related pages of each country and go three hop deep in expansion (which categorizing these topic pages -- Lvl 1-3
II. Remove duplicate and match to the most likely branch of the culture hierarchcy traversed  -- Lvl 4
III. Perform in-depth crawling, such as the multilingual versions, tables, etc.  -- Lvl 5
IV. fix pg path
V. Scrape table info as well
VI. Cover ethnolinguistic groups
VII. Expand on geo-subregions 

Page Categorization: probs for (country specificity, "social norm", "cultural norm", "value or belief", "history", "politics", "fact")
Sent Categorization: 
"""
import os
import time
import fire
import copy
import json
import torch
import pickle
import urllib
import requests
import wikipedia
import pandas as pd
from llama import Llama
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from vllm import LLM, SamplingParams
from transformers import LlamaTokenizer, pipeline
from utils import prompt_OpenAI


list_of_culture_topics = ["Culture", "", "Holidays", "Education", "Dating and Marriage", \
                          "Cuisine", "Tourism", "Landmarks", "Etiquette", "Ethnicities"] #, "Clothing", "Art", "Music"]

def check_pg_rlv_high_qual(tgt_query, query_type, doc_tit, doc_intro):
    prompt_in = "Document title: "+doc_tit + "\nDocument summary: "+doc_intro[:64]+"..."
    prompt_in += "\nIs this document about the culture of XX? Answer 'yes' or 'no':".replace("XX", tgt_query)
    prompt_out, _, _ = prompt_OpenAI(prompt_in)
    qual_pass = True if ("yes" in prompt_out.lower() or "true" in prompt_out.lower()) else False
    return qual_pass

def get_list_of_cultures():
    return []

def get_wiki_pg_text(query_term, url=""):
    search_result = wikipedia.search(query_term)
    query_term = search_result[0]
    page1 = wikipedia.page(query_term, auto_suggest=True)
    page2 = wikipedia.page(query_term, auto_suggest=False)
    page = page2 if (page2.url.split("wiki/")[-1].replace("_"," ").lower() \
            == query_term.lower()) else page1
    url = page.url
    tit = page.title
    summ = page.summary
    pg_content = page.content
    return url, tit, summ, pg_content

def helper_proc_wiki_pg_data_by_query(query_term):
    try:
        url, tit, summ, pg_content = get_wiki_pg_text(query_term)
        return {"url": url, "tit": tit, "summ": summ, "pg_content": pg_content}
    except:
        return {}

def get_wiki_pg_inner_hyperlinks(pg_title):
    query_term = urllib.parse.quote(pg_title)
    url = "https://en.wikipedia.org/w/api.php?action=parse&page="+query_term+"&prop=text&format=json"
    page = requests.get(url)
    page_content = page.content
    page_content = json.loads(page_content.decode("latin-1"), strict=False)
    return page_content["parse"]

def helper_parse_hyperlinks(html, seen_links):
    """
    TODO: double check relevance
    """
    my_soup, parsed_links = BeautifulSoup(html), set()
    for link in my_soup.findAll('a'):
        try:
            s = link["href"].replace("/wiki/","").replace("_"," ")
            link = link["href"]
            if ":" not in s and "https://" not in s and link not in seen_links \
                    and s[0] != "#" and not "(identifier)" in s \
                    and "/w/index.php" not in link:
                parsed_links.add((link,s))
            if "Bibliography of " in s:
                break
        except:
            pass
    return list(parsed_links)

def get_wiki_pg_tabular_content(url):
    return pd.read_html(url)[1]

def categorize_pg_by_interactiven_nature(s_list, mnli_classifier, categorization_result={}):
    cls_result3 = mnli_classifier(s_list, ["human interaction verbal communication", \
            "human interaction physical communication", "individual behavioral norm", \
            "interactive behavioral norm", "general assertion", "specific fact or instance"])
    for stuff_idx, stuff in enumerate(cls_result3):
        social_norm_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="human interaction verbal communication"][0]
        cultural_norm_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="human interaction physical communication"][0]
        value_or_belief_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="individual behavioral norm"][0]
        history_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="interactive behavioral norm"][0]
        politics_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="general assertion"][0]
        fact_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="specific fact or instance"][0]
        if stuff["sequence"] in categorization_result:
            categorization_result[stuff["sequence"]] = categorization_result[stuff["sequence"]] + \
                    (social_norm_score, cultural_norm_score, value_or_belief_score, history_score, politics_score, fact_score)
        else:
            categorization_result[stuff["sequence"]] = \
                    (social_norm_score, cultural_norm_score, value_or_belief_score, history_score, politics_score, fact_score)
    return categorization_result

def categorize_pg(s_list, mnli_classifier, culture="", filt=True):
    cls_result1 = mnli_classifier(s_list, ["culture of certain country ("+culture+" etc)", "culture of the world"])
    s_list = [x["sequence"] for x in cls_result1 if x["scores"][x["labels"].index("culture of certain country ("+culture+" etc)")] > 0.4]
    cls_result1 = [x["scores"][0]  for x in cls_result1 if x["scores"][x["labels"].index("culture of certain country ("+culture+" etc)")]  > 0.4]
    if len(cls_result1) == 0:
        return {}
    cls_result2 = mnli_classifier(s_list, ["social norm", "cultural norm", "value or belief", \
        "history", "politics", "fact"])
    categorization_result = {}
    for stuff_idx, stuff in enumerate(cls_result2):
        social_norm_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="social norm"][0]
        cultural_norm_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="cultural norm"][0]
        value_or_belief_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="value or belief"][0]
        history_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="history"][0]
        politics_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="politics"][0]
        fact_score = [sc for (lbl, sc) in zip(stuff['labels'], stuff['scores']) if lbl=="fact"][0]
        if not filt or (social_norm_score>0.21 and cultural_norm_score>0.21) or \
                (social_norm_score+cultural_norm_score)>0.5 or value_or_belief_score>0.6:
            categorization_result[stuff["sequence"]] = (cls_result1[stuff_idx], social_norm_score, cultural_norm_score, \
                    value_or_belief_score, history_score, politics_score, fact_score)
    #cls_result3 = mnli_classifier(s_list, ["human interaction verbal communication","human interaction physical comm
    categorization_result = categorize_pg_by_interactiven_nature(s_list, mnli_classifier, categorization_result)
    return categorization_result

def get_list_of_cultures(data_dir, get_religions=False, get_ethnoling=False):
    fname = data_dir+"culture_taxonomy/countries.txt"
    with open(fname, "r") as f:
        data = f.read()
    list_of_cultures = data.split("\n")[:-1]
    list_of_cultures = {x.split("[")[0].split("\t")[0]: list(set(x.split("\t")[-2].split("[ENT]") \
            + x.split("\t")[-1].split("[LOC]"))) if "\t" in x else x for x in list_of_cultures}
    if get_religions:
        with open(data_dir+"culture_taxonomy/religions.json", "r") as f:
            dict_of_cultures = json.load(f)
        list_of_cultures = {}
        for k, v in dict_of_cultures.items():
            list_of_cultures[k] = [k]
            if len(v)>0:
                for d in v:
                    list_of_cultures[d] = [d]
    if get_ethnoling:
        with open(data_dir+"culture_taxonomy/ISO_ethnolinguistic_groups.csv", "r") as f:
            list_of_cultures_ = f.read()
        list_of_cultures_ = list_of_cultures_.split("\n")[1:-1]
        list_of_cultures_ = [x.split("\tIndividual")[0].split("\tCollective")[0].split("\tMacrolanguage")[0] for x in list_of_cultures_]
        list_of_cultures_ = [x.split(" (macrolanguage)")[0].split("\t")[-1] for x in list_of_cultures_]
        for k in list_of_cultures_:
            list_of_cultures[k] = [k]
    return list_of_cultures

def get_georegion_map(list_of_cultures, mnli_classifier):
    cache_fpath = "/data/yifung//CTZL101_MM-NormSage/data/cache/georegion_mapping.pkl"
    if os.path.exists(cache_fpath):
        with open(cache_fpath, "rb") as f:
            georegion_maps = pickle.load(f)
        return georegion_maps
    country_city_map, city_country_map, list_of_cultures_, country_code_mapping = {}, {}, {}, {}
    from allcities import cities
    online_country_code_mapping_init = pd.read_csv(cache_fpath.replace("cache/georegion_mapping.pkl", \
            "culture_taxonomy/countries_code_mapping.csv"), header=None, sep="\t")
    for city_idx, city in enumerate(cities): # len: 132500
        city, country = city.name, city.country_code
        if country not in country_city_map:
            country_city_map[country] = []
        country_city_map[country].append(city)
        city_country_map[city] = country
        list_of_cultures_[city] = [city]
    for country_code, v in country_city_map.items():
        country_name_init = country_code
        try:
            country_name_init = online_country_code_mapping_init[online_country_code_mapping_init[1]\
                ==country_code][0].values[0]
        except:
            print(country_code)
        country = find_best_match(country_name_init, list(list_of_cultures.keys()), mnli_classifier)
        country_code_mapping[country_code] = country
    with open(cache_fpath, "wb") as f:
        pickle.dump([country_city_map, city_country_map, list_of_cultures_, country_code_mapping], f)
    return country_city_map, city_country_map, list_of_cultures_, country_code_mapping

def get_culture_page(parse_mode="lvl1", data_dir="data/", wTopics=True, multi_ling=False, mnli_classifier=None): 
    culture_results = read_parsed_culture_info(data_dir)
    list_of_cultures = get_list_of_cultures(data_dir) #get_ethnoling=True, get_religions=True
    link_exclusion_set = set()
    success_count = 0
    #193 194 48061
    print(len(list_of_cultures), len(culture_results), len(culture_results["metadata"]))
    
    if parse_mode == "lvl6":
        list_of_cultures = get_list_of_cultures(data_dir, get_ethnoling=True, get_religions=True)
        if "ethnic_and_religious_groups" not in culture_results:
            culture_results["ethnic_and_religious_groups"] = {}
    
    #assess_norm_relevance(culture_results)
    if parse_mode == "lvl7":
        #country_city_map, city_country_map, list_of_cultures, country_code_mapping = get_georegion_map(list_of_cultures, mnli_classifier)
        with open("data/culture_taxonomy/geo_country_state_city_map.json", "r") as f:
            culture_geomap = json.load(f)
        list_of_cultures = {}
        for k1, v1 in culture_geomap.items():
            for k2, v2 in v1.items():
                #list_of_cultures[k2] = k1
                for vv in v2[:5]:
                    if " major cities " not in vv and k1 != "China":
                        list_of_cultures[vv] = k1

    c = 0 
    for culture_idx, (culture, culture_alt_mentions) in enumerate(list_of_cultures.items()):
        if culture not in culture_results and parse_mode not in ["lvl6", "lvl7"]:
            culture_results[culture] = {}
        if parse_mode == "lvl5":
            tokenizer, model, other_lang = set_multiling_scrape_settings(culture)
        for culture_topic in (list_of_culture_topics if parse_mode not in ["lvl6","lvl7"] \
                else list_of_culture_topics[:1]):
            if parse_mode == "lvl1" or parse_mode == "lvl6" or parse_mode == "lvl7":
                if parse_mode == "lvl1":
                    k1, k2, k3 = culture, culture_topic, culture_topic
                elif parse_mode == "lvl6":
                    k1, k2, k3 = "ethnic_and_religious_groups", culture, culture_topic
                elif parse_mode == "lvl7":
                    k1 = list_of_cultures[culture] #country_code_mapping[city_country_map[culture]]
                    k2, k3 = culture, culture_topic
                scraped_data = helper_proc_wiki_pg_data_by_query(culture+" "+culture_topic)
                if "url" not in scraped_data:
                    continue
                if scraped_data["url"].split("wikipedia.org")[-1] in culture_results["metadata"]:
                    continue
                if parse_mode in ["lvl6", "lvl7"]:
                    qual_pass = check_pg_rlv_high_qual(culture, "ethnolinguistic/religious group" \
                            if parse_mode=="lvl6" else "geographical subregion", \
                            scraped_data["tit"], scraped_data["pg_content"].lstrip().split("\n")[0])
                    if not qual_pass and (culture in scraped_data["tit"] or culture in scraped_data["url"].split("/wiki/")[-1]) \
                            and len(culture)>4:
                        qual_pass = True
                    if not qual_pass:
                        continue
                
                if parse_mode == "lvl7":
                    categorization_result = None
                else:
                    categorization_result = categorize_pg([scraped_data["url"].split("wiki/")[-1].replace("_", " ")], \
                        mnli_classifier, culture, filt=(True if parse_mode == "lvl1" else False))

                    if len(categorization_result) == 0:
                        link_exclusion_set.add(scraped_data["url"].split("wiki/")[-1].replace("_", " "))
                        continue
                if True:
                    if k2 not in culture_results[k1]:
                        culture_results[k1][k2] = {k3:{}}
                    if k3 not in culture_results[k1][k2]:
                        culture_results[k1][k2][k3] = {}
                    scraped_topic = scraped_data["url"].split("wiki/")[-1].replace("_", " ")
                    print(k1, culture, scraped_topic, scraped_topic in culture_results[k1][k2][k3], scraped_data["url"], \
                            len(culture_results["metadata"]))
                    if scraped_topic in culture_results[k1][k2][k3]:
                        continue
                    culture_results[k1][k2][k3][scraped_topic] = scraped_data
                    culture_results[k1][k2][k3][scraped_topic]["categorization_result"] = categorization_result
                    culture_results["metadata"][scraped_data["url"].split("wikipedia.org")[-1]] = (k1,k2,k3,scraped_topic, \
                            categorization_result[scraped_topic] if categorization_result is not None else None)
                    c += 1
                #    link_exclusion_set.add(scraped_data["url"].split("wiki/")[-1].replace("_", " "))
                continue
                
            if culture_topic not in copy.deepcopy(culture_results)[culture]:
                continue

            for culture_topic2, culture_topic2_v in copy.deepcopy(culture_results)[culture][culture_topic].items():
                for culture_topic3, culture_topic3_v in culture_topic2_v.items():
                    
                    if parse_mode in ["lvl2", "lvl3"]:
                        if "hyperlink_data_related_links" in culture_topic3_v:
                            continue
                        try:
                            hyperlink_data = get_wiki_pg_inner_hyperlinks(culture_topic3_v["tit"])["text"]["*"]
                            links_untraversed = helper_parse_hyperlinks(hyperlink_data, list(culture_results["metadata"].keys()))
                            links_untraversed_to_check = [x.split("wiki/")[-1].replace("_", " ") for x, x_t in links_untraversed if x not in link_exclusion_set]
                            categorization_results = categorize_pg(links_untraversed_to_check, mnli_classifier, culture)
                            link_exclusion_set.update([x for x in links_untraversed_to_check if x not in categorization_results])
                            culture_results[culture][culture_topic][culture_topic2][culture_topic3]["hyperlink_data"] = hyperlink_data
                            culture_results[culture][culture_topic][culture_topic2][culture_topic3]["categorization_results"] = categorization_results
                            success_count += 1
                            if success_count % 10 == 0:
                                print("Success count at...", success_count, datetime.now())
                        except:
                            print("fail...")
                            continue
                        if categorization_results is not None:  
                            related_links = [(term, link) for (link, term) in links_untraversed if term in categorization_results]
                            culture_results[culture][culture_topic][culture_topic2][culture_topic3]["hyperlink_data_related_links"] = related_links
                            for related_link_s, related_link in related_links:
                                scraped_data = helper_proc_wiki_pg_data_by_query(related_link_s)
                                if "tit" not in scraped_data:
                                    continue
                                pg_content_short_intro = scraped_data["pg_content"].split("\n")[0]
                                if culture not in pg_content_short_intro and ((not \
                                        any(ext in pg_content_short_intro for ext in culture_alt_mentions)) and \
                                        len(culture_alt_mentions)>0):
                                    continue
                                k3 = related_link_s if parse_mode == "lvl2" else culture_topic2
                                if k3 not in culture_results[culture][culture_topic]:
                                    culture_results[culture][culture_topic][k3] = {}
                                culture_results[culture][culture_topic][k3][related_link_s] = scraped_data
                                culture_results["metadata"][related_link] = (culture,culture_topic,k3,related_link_s, categorization_results[related_link_s])
                            # TODO: account for partially related links if have bandwidth
                    
                    elif parse_mode == "lvl5":
                        if "pg_tabular_content" not in culture_results[culture][culture_topic][culture_topic2][culture_topic3]:
                            try:
                                url = culture_results[culture][culture_topic][culture_topic2][culture_topic3]["url"]
                                pg_tabular_content = get_wiki_pg_tabular_content(url)
                                culture_results[culture][culture_topic][culture_topic2][culture_topic3]["pg_tabular_content"] = pg_tabular_content
                            except:
                                pass
                        if type(culture_topic3_v) is dict and model!=None and other_lang!="en":
                            new_data = helper_proc_wiki_pg_data_by_query(culture_topic3_v["tit"])
                            if not "summ" in new_data:
                                continue
                            culture_results[culture][culture_topic][culture_topic2][culture_topic3]["pg_content_"+other_lang] = new_data["pg_content"]
                            def our_trans(s):
                                s_chunks = s.split("\n\n")
                                new_s = []
                                for s in s_chunks:
                                    s_tok = tokenizer(s, return_tensors="pt", padding='max_length', max_length=512)
                                    s_out = model.generate(s_tok["input_ids"].cuda())
                                    s = tokenizer.batch_decode(s_out, skip_special_tokens=True)
                                    new_s.append(s)
                                return "\n\n".join(new_s)
                            culture_results[culture][culture_topic][culture_topic2][culture_topic3]["pg_content_"+other_lang+"2en"] = our_trans(new_data["pg_content"])
        
        if parse_mode not in ["lvl6", "lvl7"] or c % 50 == 0: 
            print(parse_mode, culture_idx, culture, len(culture_results["metadata"]), datetime.now())
            write_parsed_culture_info(culture_results, data_dir, multi_ling=False, \
                    fpath="culture_scraped_info/culture_info_beta_Nov21_temp.pkl")

def find_best_match(q, options, mnli_classifier):
    nli_out = mnli_classifier(q, options)
    nli_scores = nli_out["scores"]
    best_idx = nli_scores.index(max(nli_scores))
    return nli_out["labels"][best_idx]

def fix_traversal_path(mnli_classifier=None):
    culture_results = read_parsed_culture_info()
    metadata = copy.deepcopy(culture_results["metadata"])
    for this_url, this_url_path in metadata.items():
        if len(this_url_path) != 5:
            continue
        old_t1, old_t2, old_t3, old_t4, _ = this_url_path
        best_t1 = find_best_match(old_t4, [x for x in list(culture_results.keys()) if x!="metadata"], mnli_classifier)
        best_t2 = find_best_match(old_t4, list(culture_results[best_t1].keys()), mnli_classifier)
        best_t3 = find_best_match(old_t4, list(culture_results[best_t1][best_t2].keys()), mnli_classifier)
        if best_t3 not in culture_results[best_t1][best_t2]:
            culture_results[best_t1][best_t2][best_t3] = {}
        try:
            culture_results[best_t1][best_t2][best_t3][old_t4] = culture_results[old_t1][old_t2][old_t3][old_t4]
        except:
            print(old_t1, old_t2, old_t3, old_t4)
            continue
        if not (old_t1==best_t1 and old_t2==best_t2 and old_t3==best_t3):
            del culture_results[old_t1][old_t2][old_t3][old_t4]
        if len(culture_results[old_t1][old_t2][old_t3]) == 0:
            del culture_results[old_t1][old_t2][old_t3]
        culture_results["metadata"][this_url] = (best_t1, best_t2, best_t3, old_t4)
    write_parsed_culture_info(culture_results, data_dir)

def set_multiling_scrape_settings(culture):
    lang_metadata = pd.read_csv("data/culture_taxonomy/lang_Wiki_code_mapping.txt")
    other_lang = lang_metadata[lang_metadata["country"]==culture]
    if other_lang.shape[0]<1:
        return None, None, None
    wikipedia.set_lang(other_lang["lang_code"].values[0])
    model_card = "Helsinki-NLP/opus-mt-"+other_lang+"-en"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_card)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_card).cuda()
    except:
        tokenizer, model = None, None
    return tokenizer, model, other_lang

def read_parsed_culture_info(data_dir, fpath="culture_scraped_info/culture_info_beta_Nov21.pkl", multi_ling=False):  
    if os.path.exists(data_dir+fpath):
        with open(data_dir+fpath, "rb") as f:
            culture_results = pickle.load(f)
        return culture_results
    return {"metadata":{}}

def write_parsed_culture_info(culture_results, data_dir, fpath="culture_scraped_info/culture_info_beta_Nov21.pkl", multi_ling=False): 
    with open(data_dir+fpath, "wb") as f:
        pickle.dump(culture_results, f)


if __name__ == "__main__":
    mnli_classifier_ = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    scrape_modes = ["lvl1", "lvl2", "lvl3", "lvl4", "lvl5", "lvl6", "lvl7"]
    data_dir = ["/data/yifung//CTZL101_MM-NormSage/data/", "data/"][1]
    start_t = datetime.now()  # Each country take 15min -- all countries take 1.6day
    for scrape_mode in scrape_modes[-1:]:
        if scrape_mode == "lvl4":
            fix_traversal_path(mnli_classifier=mnli_classifier_)
        else:
            get_culture_page(scrape_mode, data_dir=data_dir, mnli_classifier=mnli_classifier_)
        print(scrape_mode, (datetime.now() - start_t).seconds)

