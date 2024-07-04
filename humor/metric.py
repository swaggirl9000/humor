import pandas as pd
from thefuzz import fuzz

def fuzzy_match_metric(model_answers: pd.DataFrame, ground_truth: pd.DataFrame):    
    scores = model_answers.apply(
        lambda row_model: ground_truth.apply(
            lambda row_truth: 
                fuzz.ratio(row_truth["sentence"], row_model["sentence"]) 
                if row_model["comedian"] == row_truth["comedian"] 
                else None,
        axis=1),
    axis=1) \
        .sub(60) \
        .clip(lower=0) \
        .div(40) \
        .melt(ignore_index=False) \
        .dropna() \
        .reset_index() \
        .join(model_answers["sentence"], on="index") \
        .rename(columns={"sentence": "model"}) \
        .join(ground_truth, on="variable") \
        .rename(columns={"sentence": "truth", "value": "score"})
        
    scores = scores[["comedian", "model", "truth", "score"]]
    
    return scores