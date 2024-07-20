import pandas as pd
from thefuzz import fuzz

def bipartite_metric(model_answers: pd.DataFrame, ground_truth: pd.DataFrame):    
    scores = model_answers.apply(
        lambda row_model: ground_truth.apply(
            lambda row_truth: 
                fuzz.partial_ratio(row_truth["sentence"], row_model["sentence"]) 
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
    
    df = scores.groupby(["comedian", "model"])["score", "truth"].max().reset_index()
    
    truths = scores[~scores['truth'].isin(df['truth'])]
    entries = truths.copy()
    entries['score'] = 0
    df = pd.concat([df, entries[['comedian', 'model', 'truth', 'score']]], ignore_index=True)
    df.sort_values(by=['comedian', 'model'], inplace=True)
    
    return df.groupby("truth")["score"].mean().reset_index()