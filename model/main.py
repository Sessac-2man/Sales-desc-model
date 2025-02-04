from src.custom_dataset import CustomDataset
from src.train import Train
import argparse

from torch.utils.data import random_split


def main():
    parser = argparse.ArgumentParser(description='KoGPT2 Fine-Tuning')

    parser.add_argument('--model_save_path', type=str, default='./results',
                        help='모델 저장 경로')
    parser.add_argument('--max_length', type=int, default=128,
                        help='시퀀스 최대 길이')
    parser.add_argument('--train_batch', type=int, default=8,
                        help='배치 사이즈')
    parser.add_argument('--vali_batch', type=int, default=8,
                        help='배치 사이즈')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='에포크 수')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='학습률')
    parser.add_argument('--dataset_file', type=str, default='./data/train.csv', help='데이터셋 파일 경로')


    args = parser.parse_args()

    trainer = Train()

    dataset = CustomDataset(file_path=args.dataset_file, tokenizer=trainer.tokenizer, max_length=args.max_length)

    dataset_size = len(dataset)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])

    print(f"max_length : {args.max_length}")

    trainer.model_train(
        epochs=args.num_epochs,
        train_batch=args.train_batch,
        vali_batch=args.vali_batch,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.model_save_path,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
