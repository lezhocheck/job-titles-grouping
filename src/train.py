import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.data import get_dataset_for_training
from src.model import JobClassifier
import gc


def get_device() -> torch.device:
    device = 'cpu'
    if torch.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    return torch.device(device)


def clear_cache(*objects: Any) -> None:
    device = get_device()
    for obj in objects:
        del obj 
    if str(device) == 'mps':
        torch.mps.empty_cache() 
    elif str(device) == 'cuda':
        torch.cuda.empty_cache()
    gc.collect() 


def _train_single_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion_level: torch.nn.CrossEntropyLoss,
    criterion_area: torch.nn.CrossEntropyLoss,
    train_loader: DataLoader,
    device: torch.device = get_device(),
) -> float:
    model.train()
    running_loss = 0.0
    for batch_x, batch_y_level, batch_y_area in tqdm(train_loader):
        batch_x = batch_x.to(device)
        batch_y_level, batch_y_area = batch_y_level.to(device), batch_y_area.to(device)
        optimizer.zero_grad()
        level_pred, area_pred = model(batch_x)

        loss_level = criterion_level(level_pred, batch_y_level)
        loss_area = criterion_area(area_pred, batch_y_area)
        loss: torch.Tensor = loss_level + loss_area  # total loss = sum of both losses
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    return running_loss / len(train_loader.dataset)
    

def _validate_single_epoch(
    model: torch.nn.Module,
    criterion_level: torch.nn.CrossEntropyLoss,
    criterion_area: torch.nn.CrossEntropyLoss,
    val_loader: DataLoader,
    y_val_level: np.array,
    y_val_area: np.array,
    device: torch.device = get_device(),
) -> Tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    all_preds_level, all_preds_area = [], []

    with torch.no_grad():
        for batch_x, batch_y_level, batch_y_area in val_loader:
            batch_x = batch_x.to(device)
            batch_y_level, batch_y_area = batch_y_level.to(device), batch_y_area.to(device)
            level_pred, area_pred = model(batch_x)

            loss_level = criterion_level(level_pred, batch_y_level)
            loss_area = criterion_area(area_pred, batch_y_area)
            total_val_loss = loss_level + loss_area
            val_loss += total_val_loss.item() * batch_x.size(0)

            all_preds_level.extend(torch.argmax(level_pred, dim=1).cpu().numpy())
            all_preds_area.extend(torch.argmax(area_pred, dim=1).cpu().numpy())

    level_f1 = f1_score(
        y_val_level, all_preds_level, 
        average='weighted', zero_division=1
    )
    area_f1 = f1_score(
        y_val_area, all_preds_area, 
        average='weighted', zero_division=1
    )
    
    return val_loss / len(val_loader.dataset), level_f1, area_f1


def train_job_classifier(
    data: pd.DataFrame,
    num_epochs: int,
    model_checkpoint_path: str,
    validation_size: float = 0.2,
    batch_size: int = 256,
    hidden_size: int = 512,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 5,
    random_state: int = 42,
    logger: logging.Logger = logging.getLogger()
) -> Tuple[JobClassifier, Dict[str, LabelEncoder]]:
    """
    Trains a multi-task job classification model using job embeddings. 
    It learns to classify both job levels (e.g., Junior, Senior, Manager) 
    and job areas (e.g., Engineering, Marketing, Sales) simultaneously.

    Args:
        data (pd.DataFrame): 
            A DataFrame containing job embeddings and labeled job categories. 
            Must have the following columns:
            - `'embedding'`: A NumPy array representing job title embeddings.
            - `'job_level'`: Categorical job level labels.
            - `'job_area'`: Categorical job area labels.
        
        num_epochs (int): 
            The maximum number of epochs for training.

        model_checkpoint_path (str): 
            Path to save the best-performing model.

        validation_size (float, optional): 
            The fraction of data used for validation. Default is `0.2` (20%).

        batch_size (int, optional): 
            Number of samples per batch. Default is `256`.

        hidden_size (int, optional): 
            Number of hidden units in the neural network. Default is `512`.

        lr (float, optional): 
            Learning rate for the optimizer. Default is `1e-5`.

        weight_decay (float, optional): 
            Weight decay for the Adam optimizer (L2 regularization). Default is `1e-5`.

        early_stopping_patience (int, optional): 
            Number of epochs to wait for improvement before stopping early. Default is `5`.

        random_state (int, optional): 
            Random seed for reproducibility. Default is `42`.

        logger (logging.Logger, optional): 
            A logger instance for logging training progress. Default is `logging.getLogger()`.

    Returns:
        Tuple[JobClassifier, Dict[str, LabelEncoder]]:
            - **model** (`JobClassifier`): The trained classification model.
            - **encoders** (`Dict[str, LabelEncoder]`): Encoders mapping categorical labels to integers.
                - `'level'`: Label encoder for job levels.
                - `'area'`: Label encoder for job areas.
    """
    
    train_dataset, val_dataset, class_weights, encoders = get_dataset_for_training(
        data, validation_size, random_state
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = get_device()
    logger.info(f'Starting training on a device: {device}')

    model = JobClassifier(
        input_dim=train_dataset.tensors[0].shape[1], 
        hidden_dim=hidden_size, 
        num_classes_level=len(class_weights['level']), 
        num_classes_area=len(class_weights['area'])
    ).to(device)

    criterion_level = torch.nn.CrossEntropyLoss(weight=class_weights['level'].to(device))
    criterion_area = torch.nn.CrossEntropyLoss(weight=class_weights['area'].to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    early_stop_counter = 0

    y_val_level = val_dataset.tensors[1].numpy()
    y_val_area = val_dataset.tensors[2].numpy()

    for epoch in range(num_epochs):
        train_loss = _train_single_epoch(
            model, optimizer, criterion_level, 
            criterion_area, train_loader, device
        )
        val_loss, level_f1, area_f1 = _validate_single_epoch(
            model, criterion_level, criterion_area, 
            val_loader, y_val_level, y_val_area, device
        )
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        logger.info(f'Val Job Level F1: {level_f1} | Val Job Area F1: {area_f1}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_checkpoint_path)  # save best model
            logger.info('Model Saved (New Best Validation Loss)')
        else:
            early_stop_counter += 1
            logger.info(f'No Improvement, Early Stop Counter: {early_stop_counter}/{early_stopping_patience}')
        if early_stop_counter >= early_stopping_patience:
            logger.info('Early stopping triggered!')
            break 
    return model, encoders