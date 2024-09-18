import pandas as pd
from thefuzz import fuzz

def bipartite_metric(model_answers: pd.DataFrame, ground_truth: pd.DataFrame):  
    # Find the similarity of the model answers and ground truth by using fuzzy string matching
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
    
    # Find the best score by taking the maximum value
    best_match = score_matrix.loc[score_matrix.groupby("index")["value"].idxmax()].reset_index(drop=True)
    
    # Add in the ground truths that were not matched
    missing_gt = set(ground_truth.index) - set(best_match["variable"].unique())
    missing_gt = pd.DataFrame({"index": None, "variable": list(missing_gt), "value": 0})
    
    result = best_match.append(missing_gt).groupby("variable").mean().rename(columns={"value": "score"})
    
    # Calculate penalty for over generation 
    over_generation_penalty = len(model_answers) - len(ground_truth)
    penalty_factor = max(over_generation_penalty, 0)  
    
    # Add in the comedians and group by the mean for each comedian
    final_result = result.merge(ground_truth[['comedian']], left_index=True, right_index=True).reset_index(drop=True)
    comedian_scores = final_result.groupby('comedian')['score'].mean().reset_index()
    
    # Apply penalty to the final score
    comedian_scores['score'] -= penalty_factor * 0.1 
    
    return comedian_scores
# import pandas as pd
# from thefuzz import fuzz

# def bipartite_metric(model_answers: pd.DataFrame, ground_truth: pd.DataFrame):  
#     # Find the similarity of the model answers and ground truth by using fuzzy string matching
#     score_matrix = model_answers.apply(
#         lambda row_model: ground_truth.apply(
#             lambda row_truth: 
#                 fuzz.ratio(row_truth["sentence"], row_model["sentence"]) 
#                 if row_model["comedian"] == row_truth["comedian"] 
#                 else None,
#         axis=1),
#     axis=1) \
#         .melt(ignore_index=False) \
#         .dropna() \
#         .reset_index()
#     # Find the best score by taking the maximum value
#     best_match = score_matrix.loc[score_matrix.groupby("index")["value"].idxmax()].reset_index(drop=True)
    
#     #Add in the ground truths that were not matched
#     missing_gt = set(ground_truth.index) - set(best_match["variable"].unique())
#     missing_gt = pd.DataFrame({"index": None, "variable": list(missing_gt), "value": 0})
    
#     result = best_match.append(missing_gt).groupby("variable").mean().rename(columns={"value": "score"})
    
#     # Add in the comedians and group by the mean for each comedian
#     final_result = result.merge(ground_truth[['comedian']], left_index=True, right_index=True).reset_index(drop=True)
    
#     return final_result.groupby('comedian')['score'].mean().reset_index()