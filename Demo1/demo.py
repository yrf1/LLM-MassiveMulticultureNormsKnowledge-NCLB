import os, json, openai, pandas as pd


mode = ["o1-mini", "gpt-4-turbo", "o1-preview", "llama"][0]
test_dataset = pd.read_csv("demo_dataset.csv", sep="\t") #, header=None)

openai.api_key = os.getenv("OPENAI_API_KEY")

#@retry(retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, \
#            openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
#            wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(10))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def run_LLM_inference_helper(q, mode="OpenAI"):
    pred, logprobs = call_OpenAI_api(q, temp=0.0)
    return pred, logprobs

def call_OpenAI_api(q,model_="gpt-4-turbo",temp=0.0,max_gen_len=512):
    if model_ in ["o1-mini", "o1-preview"]:
        a = chat_completion_with_backoff(model=model_, \
            max_completion_tokens=max_gen_len, \
            messages=[{"role": "user", "content": q}]),
        return a[0]["choices"][0]["message"]["content"], 0.0
    a = chat_completion_with_backoff(model=model_, \
            max_completion_tokens=max_gen_len, \
            temperature=temp, top_p=0.9, \
            messages=[{"role": "user", "content": q}]),
    return a[0]["choices"][0]["message"]["content"], 0.0

def infer_NormSit_helper(x_dial, latent_norm, method="RAG"):
    if method=="baseline":
        y_pred = run_LLM_inference_helper(x_dial, mode="OpenAI")[0]
    elif method=="RAG":
        prompt_in = "Background Context:\n"+latent_norm+"\n---\n[Dialogue]\n"+x_dial
        y_pred = run_LLM_inference_helper(prompt_in, mode="OpenAI")[0]
    # LLM-as-a-Judge
    prompt_in = x_dial + "\nChatbot Response: " + y_pred + \
        "\n----\nDoes the above Chatbot Response point out and/or correct a sociocultural misunderstanding or siocultural norm violation from the first dialogue line, 'yes' or 'no' (one-word answer):"
    prompt_out = run_LLM_inference_helper(prompt_in, mode="OpenAI")[0]
    pred_V = True if "yes" in prompt_out.lower() else False
    return y_pred, prompt_out, pred_V

def engine(mode="baseline"):
  results_tracker = []
  if os.path.exists("results_tracker_"+mode+".json"):
    with open("results_tracker_"+mode+".json", "r") as f:
        results_tracker = json.load(f)
  for idx, data in test_dataset.iterrows():
    x_dial, latent_norm, lbl = data["Dial"], data["Norm"], data["Label"]
    if x_dial in [x[0] for x in results_tracker]:
        continue
    y_pred, prompt_out, pred_V = infer_NormSit_helper(x_dial, latent_norm, mode)
    results_tracker.append((x_dial, latent_norm, y_pred, prompt_out, pred_V, lbl))
  print("Writing to:  results_tracker_"+mode+".json")
  with open("results_tracker_"+mode+".json", "w", encoding="utf-8") as f:
    json.dump(results_tracker, f, indent=2, ensure_ascii=False)

def score(mode="baseline"):
  print(mode)
  with open("results_tracker_"+mode+".json", "r") as f:
    results_tracker = json.load(f)
  TP, FP, FN, TN = 0.001, 0.001, 0.001, 0.001
  for (x_dial, latent_norm, y_pred, _, pred_V, lbl) in results_tracker:
    if pred_V and lbl=="V":
        TP += 1
    elif pred_V and not lbl=="V":
        FP += 1
    elif not pred_V and lbl=="V":
        FN += 1
    else:
        TN += 1
  print("TP, FP, FN, TN:  ", int(TP), int(FP), int(FN), int(TN))
  P, R = TP/(TP+FP),TP/(TP+FN)
  print("P, R, F:  ", "{:.3f}".format(P), "{:.3f}".format(R), "{:.3f}".format(2*P*R/(P+R)))

mode = ["baseline", "RAG"][0]
#engine(mode)
score(mode)
