import argparse
import os
import pandas as pd
from src.inference import predict_job
from src.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict job levels and job areas from job titles using a trained model.')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the input CSV file containing job titles.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth file).')
    parser.add_argument('--encoders_path', type=str, required=True, help='Path to the saved encoders (.pkl file).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file with predictions.')
    parser.add_argument('--logs_folder_path', type=str, default='./logs', help='Path to save logs of the script.')

    parser.add_argument(
        '--embed_tokenizer', type=str, 
        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
        help='Hugging Face tokenizer version for embeddings.'
    )
    parser.add_argument(
        '--embed_model', type=str, 
        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
        help='Hugging Face model version for embeddings.'
    )
    parser.add_argument(
        '--max_batch_size', type=int, default=512, 
        help='Maximum batch size for prediction.'
    )

    args = parser.parse_args()
    logger = setup_logger('model-inference', logs_folder=args.logs_folder_path)

    logger.info(f'Loading data from {args.data_path}...')
    data = pd.read_csv(args.data_path)

    logger.info('Running job classification predictions...')
    results = predict_job(
        data=data,
        model_path=args.model_path,
        encoders_path=args.encoders_path,
        embed_tokenizer=args.embed_tokenizer,
        embed_model=args.embed_model,
        max_batch_size=args.max_batch_size,
        logger=logger
    )

    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    results.to_csv(args.output_path, index=False)
    logger.info(f'Predictions saved to {args.output_path}')


if __name__ == '__main__':
    main()