"""CoDET-M4 (Python only) balanced subset.

Downloads the single HF parquet once, streams it in batches (metadata columns first,
then only the selected rows' code), filters
language == "python", respects the dataset's own train/test split, and samples a
balanced subset: half human, half AI split evenly across the AI generators.
The subset is cached to data/code/; the raw parquet can then be deleted.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path("data/code")
REPO = "DaniilOr/CoDET-M4"
RAW_FILE = "default/train/0000.parquet"
BATCH = 20_000


def _raw_path():
    local = DATA_DIR / "raw" / RAW_FILE
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, RAW_FILE, repo_type="dataset", revision="refs/convert/parquet",
                           local_dir=DATA_DIR / "raw")


def _python_meta(pf):
    """Pass 1: metadata columns only -> DataFrame of python rows with their global row index."""
    parts, offset = [], 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["language", "model", "split", "target"]):
        d = b.to_pandas()
        d.loc[d["target"] == "human", "model"] = "human"  # some human rows have a null model
        d["row"] = range(offset, offset + len(d))
        offset += len(d)
        parts.append(d[d["language"] == "python"][["row", "model", "split"]])
    return pd.concat(parts, ignore_index=True)


def _fetch_code(pf, rows):
    """Pass 2: stream the code column and keep only the selected global rows."""
    wanted, out, offset = set(rows), {}, 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["code"]):
        codes = b.column(0)
        for i in range(len(codes)):
            if offset + i in wanted:
                out[offset + i] = codes[i].as_py()
        offset += len(codes)
    return [out[r] for r in rows]


def _balanced(meta, n, seed):
    human = meta[meta["model"] == "human"]
    ai = meta[meta["model"] != "human"]
    gens = sorted(ai["model"].unique())
    n_h = n // 2
    per_gen = (n - n_h) // len(gens)
    parts = [human.sample(n_h, random_state=seed)]
    parts += [ai[ai["model"] == g].sample(per_gen, random_state=seed) for g in gens]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_split(split, n, seed=0):
    """Return a cached balanced DataFrame for split in {"train", "test"}.

    Columns: code, model (generator, "human" for human), label (1 = AI, 0 = human).
    """
    path = DATA_DIR / f"{split}_{n}_s{seed}.parquet"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(_raw_path())
        meta = _python_meta(pf)
        out = _balanced(meta[meta["split"] == split], n, seed)
        out["code"] = _fetch_code(pf, out["row"].tolist())
        out["label"] = (out["model"] != "human").astype("int64")
        out[["code", "model", "label"]].to_parquet(path)
    return pd.read_parquet(path)
