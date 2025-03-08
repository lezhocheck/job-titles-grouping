import html
import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import TensorDataset


def clean_text(text: str) -> Tuple[str, bool]:
    """
    Cleans job titles by:
    - Removing HTML tags
    - Unescaping escaped HTML characters (e.g., &amp; -> &)
    - Removing extra whitespace and invisible characters
    - Keeping punctuation for embeddings

    Returns:
    - Cleaned text
    - Boolean indicating whether changes were made
    """

    original_text = text  # store original text for comparison
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # convert escaped characters (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)
    # remove multiple spaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # remove non-printable characters
    text = re.sub(r'[\u200b\u00ad]', '', text)
    return text, text != original_text



def get_dataset_for_training(
    data: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = None
) -> Tuple[TensorDataset, TensorDataset, Dict[str, torch.Tensor], Dict[str, LabelEncoder]]:
    """
    Prepares training and validation datasets from labeled job data, encoding categorical labels 
    and computing class weights for imbalanced classification.

    Args:
        data (pd.DataFrame): 
            Input DataFrame containing job embeddings and labels. 
            It must have the following columns:
            - `'embedding'`: A NumPy array of job title embeddings (each row should contain a vector).
            - `'job_level'`: Categorical job level labels (e.g., "Junior", "Senior").
            - `'job_area'`: Categorical job area labels (e.g., "Engineering", "Marketing").
        
        validation_size (float, optional): 
            Proportion of the dataset to be used for validation. Default is `0.2` (20% validation).

        random_state (int, optional): 
            Random seed for reproducibility. Default is `None`.

    Returns:
        Tuple[TensorDataset, TensorDataset, Dict[str, torch.Tensor], Dict[str, LabelEncoder]]:
            - **train_dataset** (`TensorDataset`): Training dataset with embeddings and encoded labels.
            - **val_dataset** (`TensorDataset`): Validation dataset with embeddings and encoded labels.
            - **class_weights** (`Dict[str, torch.Tensor]`): Computed class weights for imbalanced data.
                - `'level'`: Weights for job level classification.
                - `'area'`: Weights for job area classification.
            - **encoders** (`Dict[str, LabelEncoder]`): Encoders for mapping labels to numerical values.
                - `'level'`: Label encoder for job levels.
                - `'area'`: Label encoder for job areas.

    Raises:
        ValueError: If the input DataFrame does not contain the required columns (`'embedding'`, `'job_level'`, `'job_area'`).
    """
    
    expected_cols = ['embedding', 'job_level', 'job_area']
    if not all(c in data.columns for c in expected_cols):
        raise ValueError(f'Expected pd.DataFrame with columns: {expected_cols}')
    
    X = np.vstack(data['embedding'].values)

    label_encoder_level = LabelEncoder()
    label_encoder_area = LabelEncoder()

    y_level = label_encoder_level.fit_transform(data['job_level'])
    y_area = label_encoder_area.fit_transform(data['job_area'])

    X_train, X_val, y_train_level, y_val_level, y_train_area, y_val_area = train_test_split(
        X, y_level, y_area, test_size=validation_size, random_state=random_state
    )

    class_weights_level = compute_class_weight('balanced', classes=np.unique(y_train_level), y=y_train_level)
    class_weights_area = compute_class_weight('balanced', classes=np.unique(y_train_area), y=y_train_area)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    y_train_level_tensor = torch.tensor(y_train_level, dtype=torch.long)
    y_val_level_tensor = torch.tensor(y_val_level, dtype=torch.long)

    y_train_area_tensor = torch.tensor(y_train_area, dtype=torch.long)
    y_val_area_tensor = torch.tensor(y_val_area, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_level_tensor, y_train_area_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_level_tensor, y_val_area_tensor)

    class_weights = {
        'level': torch.tensor(class_weights_level, dtype=torch.float32),
        'area': torch.tensor(class_weights_area, dtype=torch.float32)
    }

    encoders = {
        'level': label_encoder_level,
        'area': label_encoder_area
    }

    return train_dataset, val_dataset, class_weights, encoders
