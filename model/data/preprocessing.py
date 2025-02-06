import pandas as pd
import re

def extract_product_name(text):
    # 정규식 패턴: 대괄호([]) 내부 또는 괄호(()) 내부 제거
    pattern = r"\[.*?\]|\(.*?\)"  # [햇반] 또는 (205gX3개) 제거
    cleaned_text = re.sub(pattern, "", text).strip()  # 패턴 삭제 후 양쪽 공백 제거
    return cleaned_text

def cleansing_files(file_path, output_path):
    data = pd.read_json(file_path)

    # preprocessing of nulls

    data = data.dropna(subset=["words"]).drop("dc_price",axis=1).reset_index(drop=True)
    data["words"] = data["words"].apply(lambda x: x[0])

    data = data.loc[~data["words"].str.contains("Kurly")]
    data = data.loc[~data["words"].str.contains("컬리")].reset_index(drop=True)
    data = data.loc[~data["words"].str.contains("KF365")].reset_index(drop=True)

    data["name"] = data["name"].apply(lambda x: extract_product_name(x))
    data = data.loc[~data["name"].str.contains("컬리")].reset_index(drop=True)

    return data.to_csv(output_path, encoding="utf-8-sig")



if __name__ == "__main__":
    cleansing_files("./data_sets.json", "train.csv")