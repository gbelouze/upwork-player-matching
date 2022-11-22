import datetime
import logging
import re
from pathlib import Path
from typing import Any

import dateutil.parser as dparser
import numpy as np
import pandas as pd
from rich.progress import track
from thefuzz import fuzz, process

log = logging.getLogger("data_matching")

data_dir = Path(__file__).resolve().parents[2] / "data"
data_files = [file for file in data_dir.iterdir() if file.suffix == ".csv"]


def safe_parse_date(date: Any):
    if not isinstance(date, str):
        log.debug(f"Ignoring date {date} ({type(date)}not a string)")
        return np.nan
    try:
        delta = datetime.datetime.today() - dparser.parse(date)
        if not (365 * 10 < delta.days < 365 * 80):
            return np.nan
        return delta.days
    except dparser.ParserError:
        # log.debug(f"Ignoring date {date} (parse error)")
        return np.nan


def safe_parse_height(height: Any):
    h = np.nan
    if isinstance(height, (int, float)):
        h = height
    elif not isinstance(height, str):
        log.debug(f"Ignoring height {height} ({type(height)}not a string)")
        return np.nan
    else:
        if m := re.match(r"(\d\d\d)(cm)?", height):
            h = int(m.group(1))
        elif re.match(r"(\dm\d\d)", height):
            h = 100 * int(height[0]) + int(height[2:])
        elif m := re.match(r"(\d[.,]\d+)(m)?", height):
            h = 100 * float(m.group(1).replace(",", "."))
        else:
            log.debug(f"Ignoring height {height} (parse error)")
    if not 50 < h < 300:
        return np.nan
    return h


def load(path: Path):
    df = pd.read_csv(path)  # .iloc[:10_000].copy()  # TODO: remove the iloc
    id_col, _ = process.extractOne("playerid", df.columns)
    name_col, _ = process.extractOne("playername", df.columns)
    height_col, _ = process.extractOne("playerheight", df.columns)
    dob_col, _ = process.extractOne("dateofbirth", df.columns)
    log.info(f"Using column [cyan]{id_col}[/] for ids.", extra={"markup": True})
    log.info(f"Using column [cyan]{name_col}[/] for names.", extra={"markup": True})
    log.info(
        f"Using column [cyan]{dob_col}[/] for dates of birth.", extra={"markup": True}
    )
    log.info(f"Using column [cyan]{height_col}[/] for heights.", extra={"markup": True})

    df[name_col] = (
        df[name_col]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
    )
    df.loc[df[dob_col].notna(), dob_col] = df.loc[df[dob_col].notna(), dob_col].apply(
        safe_parse_date
    )
    df.loc[df[height_col].notna(), height_col] = df.loc[
        df[height_col].notna(), height_col
    ].apply(safe_parse_height)

    mapper = {
        id_col: "id",
        name_col: "player_name",
        dob_col: "dob",
        height_col: "height",
    }

    df = (
        df.rename(columns=mapper)
        .drop_duplicates(subset="id")
        .loc[:, ["id", "player_name", "dob", "height"]]
    )
    df["origin"] = path.stem
    return df


def load_all():
    dfs = [load(data_file) for data_file in data_files]
    return pd.concat(dfs, ignore_index=True)


def potential_matches(df: pd.DataFrame):
    matches = {}

    bydob = df[df.dob.notna()].groupby("dob")
    byname = df.groupby("player_name")

    for name, groupby in [("name", byname), ("dob", bydob)]:
        matches[name] = []
        for dob, group in track(groupby, "Iterating through groupby..."):
            group.sort_values(by="height", inplace=True)
            if len(group) >= 2:
                indices = list(group.index)
                for i, index1 in enumerate(indices[:-1]):
                    for index2 in indices[i + 1 :]:
                        if group.loc[index2].height - group.loc[index1].height > 2:
                            break
                        matches[name].append((index1, index2))

    return matches


def confirm_match(row1: pd.Series, row2: pd.Series, optimism: float = 0.1):
    if fuzz.ratio(row1.player_name, row2.player_name) < 100 * (1 - optimism):
        log.debug(
            f"Rejected match between names {row1.player_name} and {row2.player_name}"
        )
        return False
    if (
        pd.notna(row1.dob)
        and pd.notna(row2.dob)
        and np.abs(row1.dob - row2.dob) > 3 * optimism
    ):
        log.debug(f"Rejected match between dates {row1.dob} and {row2.dob}")
        return False
    if (
        pd.notna(row1.height)
        and pd.notna(row2.height)
        and np.abs(row1.height - row2.height) > 2 * optimism
    ):
        log.debug(f"Rejected match between heights {row1.height} and {row2.height}")
        return False
    return True


def create_match_table(df: pd.DataFrame):
    remove_indices = set()

    match_table = pd.DataFrame()
    for file in data_files:
        match_table[file.stem] = df.id
        match_table.loc[df.origin != file.stem, file.stem] = None

    potentials = potential_matches(df)
    for name, hits in potentials.items():
        n_matches = 0
        log.info(f"Found {len(hits)} potential matches through {name}")
        for i1, i2 in track(hits, "Filtering out false positives"):
            row1 = df.loc[i1]
            row2 = df.loc[i2]
            if confirm_match(row1, row2):
                remove_indices.add(i2)
                match_table.loc[i1, row2.origin] = row2.id
                n_matches += 1
                log.debug(f"Match found: {row1.values} {row2.values}")
        log.info(f"Kept {n_matches} matches found through {name}.")

    return match_table.drop(remove_indices)


if __name__ == "__main__":
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    df = load_all()
    table = create_match_table(df)
    table.to_csv(data_dir / "../results/ID_Matching_Master.csv")
