# use https://github.com/Vicomtech/hate-speech-dataset and put this script inside the main folder
import pandas as pd
from pathlib import Path


def extract(mode="train"):
    annotations_df = pd.read_csv("annotations_metadata.csv")
    files = {path.stem: path.read_text() for path in Path(f"sampled_{mode}").glob("*.txt")}
    df_list = []
    for i, row in annotations_df.iterrows():
        if row["file_id"] in files:
            text = files[row["file_id"]]
            df_list.append((text, row["label"]))
    df = pd.DataFrame(df_list, columns=["input", "annotation"])
    df.to_csv(f"hate_{mode}.csv")


if __name__ == "__main__":
    extract("train")
    extract("test")
