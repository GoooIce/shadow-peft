#!/usr/bin/env python3
"""Build a neutral calibration corpus and held-out perplexity sets.

Source: ``Salesforce/wikitext`` on the Hugging Face Hub, revision pinned below.
Licence: CC BY-SA 3.0 (and GFDL); WikiText, Merity et al. 2016.

Splits used:

* ``wikitext-103-raw-v1`` train      -> NVFP4 activation calibration
* ``wikitext-103-raw-v1`` test       -> held-out perplexity (disjoint documents)
* ``wikitext-103-raw-v1`` validation -> second held-out perplexity split

Note (measured 2026-09-09): ``wikitext-2-raw-v1`` test/validation are
byte-identical to the corresponding ``wikitext-103-raw-v1`` splits (same HF LFS
oid), so wikitext-2 adds no independent signal and is not used.

Neither split is UltraData (the model's own post-training corpus).

Outputs (in ``--out-dir``):

* ``calib_wikitext103_train.jsonl``   ``{"text": ...}`` chunks of ~seq_len tokens
* ``eval_wikitext103_test.jsonl``     ``{"input_ids": [...]}`` fixed-length ids
* ``eval_wikitext103_validation.jsonl`` same
* ``manifest.json``                   provenance, hashes, counts

Usage:
    python build_neutral_corpus.py --model ~/models/MiniCPM5-2B \
        --out-dir ~/minicpm5-payload/data/neutral --calib-samples 512 \
        --eval-seqs 64 --seq-len 2048
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
BASE = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}"
FILES = {
    "wikitext-103-raw-v1/train-00000-of-00002.parquet": "calibration",
    "wikitext-103-raw-v1/test-00000-of-00001.parquet": "eval-test",
    "wikitext-103-raw-v1/validation-00000-of-00001.parquet": "eval-validation",
}


def _download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"reuse {dest}")
    else:
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=600) as response, dest.open("wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"  {dest.name} {dest.stat().st_size} B sha256={digest}")
    return digest


def _texts(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["text"])
    return [value for value in table.column("text").to_pylist() if value]


def _chunks(ids: list[int], seq_len: int) -> list[list[int]]:
    return [ids[i : i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calib-samples", type=int, default=512)
    parser.add_argument("--eval-seqs", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    manifest: dict = {
        "purpose": "neutral NVFP4 activation calibration + held-out perplexity",
        "dataset": "Salesforce/wikitext",
        "revision": REVISION,
        "licence": "CC BY-SA 3.0 (and GFDL) - WikiText, Merity et al. 2016",
        "contamination": (
            "wikitext only; UltraData (the model's own post-training corpus) is not used"
        ),
        "tokenizer": str(Path(args.model).expanduser()),
        "seq_len": args.seq_len,
        "files": {},
        "splits": {},
    }

    paths: dict[str, Path] = {}
    for name, role in FILES.items():
        dest = raw_dir / name.replace("/", "__")
        manifest["files"][name] = {
            "role": role,
            "url": f"{BASE}/{name}",
            "sha256": _download(f"{BASE}/{name}", dest),
            "bytes": dest.stat().st_size,
        }
        paths[name] = dest

    # --- calibration: wikitext-103 train -------------------------------------
    train_text = "".join(_texts(paths["wikitext-103-raw-v1/train-00000-of-00002.parquet"]))
    train_ids = tokenizer(train_text, add_special_tokens=False)["input_ids"]
    calib_path = out_dir / "calib_wikitext103_train.jsonl"
    written = 0
    with calib_path.open("w", encoding="utf-8") as fh:
        for chunk in _chunks(train_ids, args.seq_len)[: args.calib_samples]:
            fh.write(json.dumps({"text": tokenizer.decode(chunk)}) + "\n")
            written += 1
    manifest["splits"]["calib_wikitext103_train"] = {
        "file": calib_path.name,
        "samples": written,
        "tokens_per_sample": args.seq_len,
        "total_tokens": written * args.seq_len,
        "sha256": hashlib.sha256(calib_path.read_bytes()).hexdigest(),
    }

    # --- held-out perplexity -------------------------------------------------
    for corpus, source in (
        ("wikitext103_test", "wikitext-103-raw-v1/test-00000-of-00001.parquet"),
        (
            "wikitext103_validation",
            "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
        ),
    ):
        ids = tokenizer(
            "".join(_texts(paths[source])), add_special_tokens=False
        )["input_ids"]
        chunks = _chunks(ids, args.seq_len)[: args.eval_seqs]
        path = out_dir / f"eval_{corpus}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps({"input_ids": chunk}) + "\n")
        manifest["splits"][f"eval_{corpus}"] = {
            "file": path.name,
            "sequences": len(chunks),
            "seq_len": args.seq_len,
            "tokens": sum(len(c) for c in chunks),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
