from pathlib import Path

import pandas as pd
from rich import print

match_table_path = (
    Path(__file__).resolve().parents[2] / "results/ID_Matching_Master.csv"
)


def load():
    if not match_table_path.exists():
        raise ValueError("ID_Matching_Master.csv must be generated first")
    return pd.read_csv(match_table_path, index_col=0)


def main():
    df = load()
    counts = [0] * len(df.columns)
    for index, row in df.iterrows():
        counts[sum(pd.notna(value) for value in row) - 1] += 1
    for i, count in enumerate(counts):
        print(
            f"{100 * count / sum(counts):5.2f}% of players are shared between {i + 1} file{' only' if i == 0 else 's'} [N = {count}]"
        )


if __name__ == "__main__":
    main()
