import pandas as pd
from torch.utils.data import Dataset



class CustomDataset(Dataset):
  def __init__(self, tokenizer, file_path, max_length=128):
    self.data = pd.read_csv(file_path)
    self.data["meta"] = self.data["name"] + " " + self.data["words"]

    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    text = self.data.iloc[idx]["meta"]

    encoded = self.tokenizer(
    text,
    max_length=self.max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    )
        # ✅ `input_ids`가 모델의 vocab_size를 초과하지 않는지 체크

    labels = encoded["input_ids"].clone()

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0), # `labels` 추가 (언어 모델 학습)
    }


