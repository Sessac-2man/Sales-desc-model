import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        """
        SOLAR Fine-tuning을 위한 CustomDataset 클래스
        Args:
            json_path (str): 데이터셋 JSON 파일 경로
            tokenizer (AutoTokenizer): 사용할 토크나이저
            max_length (int): 최대 토큰 길이
        """
        self.data = pd.read_csv(file_path, encoding="utf-8-sig")

        self.data["instruction"] = self.data["instruction"] = "상품명을 기반으로 고객이 관심을 가질 만한 매력적인 제품 설명을 생성하세요"

        self.data = self.data[["instruction", "name", "words"]].rename(columns={"name": "input", "words": "output"}).to_dict(
            "records")


        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)  # 데이터 개수 반환

    def __getitem__(self, idx):
        example = self.data[idx]

        # 🔹 모델이 학습할 문장 형식 정의
        prompt = f"### 질문: {example['instruction']}\n{example['input']}\n\n### 답변: {example['output']}"

        # 🔹 토크나이징 (입력과 출력 합쳐서 처리)
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = encodings["input_ids"].clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels  # Causal LM의 경우 input_ids와 동일하게 설정
        }
