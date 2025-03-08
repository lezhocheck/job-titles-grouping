import argparse
import os
import pandas as pd
import pickle
from src.train import train_job_classifier
from src.logger import setup_logger
       

def main() -> None:
    parser = argparse.ArgumentParser(description='Train a job classifier model.')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the labeled dataset with embeddings parquet file.')
    parser.add_argument('--checkpoint_path', type=str, default='./cache/train/best_model.pth', help='Path to save the trained model.')
    parser.add_argument('--encoder_path', type=str, default='./cache/train/encoders.pkl', help='Path to save the label encoders.')
    parser.add_argument('--logs_folder_path', type=str, default='./logs', help='Path to save logs of the script.')

    parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs.')
    parser.add_argument('--validation_size', type=float, default=0.2, help='Fraction of data for validation.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of hidden layers.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for data splitting.')

    args = parser.parse_args()
    logger = setup_logger('model-train', logs_folder=args.logs_folder_path)

    # Load dataset
    logger.info(f'Loading data from {args.data_path}...')
    labeled_data = pd.read_parquet(args.data_path)

    os.makedirs(os.path.split(args.checkpoint_path)[0], exist_ok=True)
    os.makedirs(os.path.split(args.encoder_path)[0], exist_ok=True)

    try:
        _, encoders = train_job_classifier(
            data=labeled_data,
            num_epochs=args.num_epochs,
            model_checkpoint_path=args.checkpoint_path,
            validation_size=args.validation_size,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            random_state=args.random_state,
            logger=logger
        )
    except Exception as e:
        logger.error(f'An error occured during training: {e}')
        raise e

    with open(args.encoder_path, 'wb') as f:
        pickle.dump(encoders, f)

    logger.info(f'Model saved at: {args.checkpoint_path}')
    logger.info(f'Label encoders saved at: {args.encoder_path}')
    logger.info('Train successfully finished')


if __name__ == '__main__':
    main()