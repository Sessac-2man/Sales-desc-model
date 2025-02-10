import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Qwen2.5-1.5B-Instruct 파인튜닝을 위한 CustomDataset 클래스
        Args:
            file_path (str): 데이터셋 CSV 파일 경로
            tokenizer (AutoTokenizer): 사용할 토크나이저
            max_length (int): 최대 토큰 길이
        """
        # CSV 데이터 로드
        self.data = pd.read_csv(file_path, encoding="utf-8-sig")

        # 시스템 메시지 추가 (지시문)
        system_message = "당신은 고객이 관심을 가질 만한 매력적인 제품 설명을 생성하는 AI입니다."

        # 데이터 포맷 변경 (ChatML 형식)
        self.data = [
            {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": row["name"]},
                    {"role": "assistant", "content": row["words"]}
                ]
            }
            for _, row in self.data.iterrows()
        ]

        # tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)  # 데이터 개수 반환

    def __getitem__(self, idx):
        example = self.data[idx]

        # 🔹 ChatML 포맷을 사용한 프롬프트 생성
        chatml_prompt = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,  # 먼저 문자열 형태로 변환
            add_generation_prompt=False
        )

        # 🔹 토크나이징
        encodings = self.tokenizer(
            chatml_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels  # Causal LM의 경우 input_ids와 동일하게 설정
        }
