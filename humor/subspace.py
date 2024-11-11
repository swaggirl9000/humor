from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm

# Get subspace representation of Ground Truth
def get_gt_representation(tokenizer, batch_of_strs: list, subspace_size: int = 8) -> torch.Tensor:
    inputs = tokenizer(batch_of_strs, return_tensors="pt", padding=True, truncation=False).to(model.device)
    *_, subs_repr = torch.pca_lowrank(U[inputs["input_ids"]], q=subspace_size)
    return subs_repr

#Get subspace representation of model's output
def get_output_representation(model, tokenizer, batch_of_strs: list, number_of_tokens: int = 128, subspace_size: int = 8) -> torch.Tensor:
    inputs = tokenizer(batch_of_strs, return_tensors="pt", padding=True, truncation=False).to(model.device)
    
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=number_of_tokens)
    
    *_, subs_repr = torch.pca_lowrank(U[ids], q=subspace_size)
    return subs_repr
        
def vector_similarity_metric(model, tokenizer):  
    gt_representations = {
        comedian: get_gt_representation(model, tokenizer, batch.tolist())
        for comedian, batch in tqdm(df.groupby("comedian")["model_input"])
    }
    
    output_representations = {
        comedian: get_output_representation(batch.tolist())
        for comedian, batch in tqdm(df.groupby("comedian")["gt_input"])
    }
    

    scores = {}
    for comedian in tqdm(gt_representations.keys()):
        gt_reference_subspaces = gt_representations[comedian]
        out_reference_subspaces = output_representations[comedian]

        A = gt_reference_subspaces.mT @ out_reference_subspaces
        scores[comedian] = A.matrix_power(2).diagonal(dim1=1,dim2=2).mean().item()
        
    df = pd.DataFrame(list(scores.items()), columns=['Comedian', 'Score'])
    return df