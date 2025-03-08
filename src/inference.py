import logging
import pickle
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from src.data import clean_text
from transformers import AutoTokenizer, AutoModel
from src.model import JobClassifier
from src.train import get_device, clear_cache
import torch


def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def predict_job(
    data: pd.DataFrame,
    model_path: str,
    encoders_path: str,
    embed_tokenizer: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    embed_model: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    max_batch_size: int = 512,
    logger: logging.Logger = logging.getLogger()
) -> pd.DataFrame:
    
    if not 'job_title' in data.columns:
        raise ValueError(f'Expected pd.DataFrame with column: job_title')
    
    data[['cleaned_job_title', 'changes_applied']] = data['job_title'].apply(lambda x: pd.Series(clean_text(x)))
    logger.info(f'Cleaned {data["changes_applied"].sum()} rows')
    device = get_device()
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_tokenizer)
    embed_model = AutoModel.from_pretrained(embed_model).to(device)
    model = JobClassifier(input_dim=768, hidden_dim=512, num_classes_level=10, num_classes_area=28).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    model.eval()

    level_preds, area_preds = [], []

    batch_size = min(max_batch_size, len(data))
    for i in tqdm(range(0, len(data), batch_size), desc='Predicting...'):
        batch = data[i : i + batch_size]['cleaned_job_title'].tolist()
        encoded_input = embed_tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = embed_model(**encoded_input)
            sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
            level_pred, area_pred = model(sentence_embeddings)
        predicted_level = torch.argmax(level_pred, dim=1).cpu().numpy()
        predicted_area = torch.argmax(area_pred, dim=1).cpu().numpy()
        level_preds.append(predicted_level)
        area_preds.append(predicted_area)

    level_preds = np.hstack(level_preds)
    area_preds = np.hstack(area_preds)

    encoders: Dict[str, LabelEncoder] = pickle.load(open(encoders_path, 'rb'))
    data['job_level'] = encoders['level'].inverse_transform(level_preds)
    data['job_area'] = encoders['area'].inverse_transform(area_preds)

    # delete the model and free memory
    clear_cache(embed_tokenizer, embed_model, model)
    return data