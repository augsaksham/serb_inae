from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
import pandas as pd

# You can chose models from following list
# https://huggingface.co/models?sort=downloads&search=google%2Fpegasus
def predict(tx,model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer(tx, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text


def predict_test(df,model_name):
    print("model name")
    dict_df={"Id":[],"Abstract":[],"RHS":[]}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    print("model loaded")
    for i in range(df.shape[0]):
        vl=df.iloc[i].values
        dict_df["FileName"].append(vl[0])
        dict_df["Abstract"].append(vl[1])
        batch = tokenizer(vl[1], truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        dict_df["RHS"].append(tgt_text)

    res=pd.DataFrame(dict_df)
    return res