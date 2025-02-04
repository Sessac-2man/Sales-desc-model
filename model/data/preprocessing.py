import pandas as pd

def cleansing_files(file_path, output_path):
    data = pd.read_json(file_path)

    # preprocessing of nulls

    data = data.dropna(subset=["words"]).drop("dc_price",axis=1).reset_index(drop=True)
    data["words"] = data["words"].apply(lambda x: x[0])
    data["name"] = data["name"].apply(lambda x: x[0])

    return data.to_csv(output_path, encoding="utf-8-sig")

if __name__ == "__main__":
    cleansing_files("./data_sets.json", "train.csv")