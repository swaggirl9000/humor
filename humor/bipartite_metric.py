import pandas as pd
from thefuzz import fuzz

def bipartite_metric(model_answers: pd.DataFrame, ground_truth: pd.DataFrame):    
    score_matrix = model_answers.apply(
        lambda row_model: ground_truth.apply(
            lambda row_truth: 
                fuzz.ratio(row_truth["sentence"], row_model["sentence"]) 
                if row_model["comedian"] == row_truth["comedian"] 
                else None,
        axis=1),
    axis=1) \
        .melt(ignore_index=False) \
        .dropna() \
        .reset_index()
    
    best_match = score_matrix.loc[score_matrix.groupby("index")["value"].idxmax()].reset_index(drop=True)
    
    missing_gt = set(ground_truth.index) - set(best_match["variable"].unique())
    missing_gt = pd.DataFrame({"index": None, "variable": list(missing_gt), "value": 0})
    
    # TODO: fix case when there are multiple comedians
    return best_match.append(missing_gt).groupby("variable").mean().rename(columns={"value": "score"})
    