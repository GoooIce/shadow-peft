#!/usr/bin/env python3
"""Prepare two deterministic, disjoint screening shards from frozen HPS prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from shadow_lab.mage_qad_data import prompt_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    sha = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if sha != "a7d3ac3bd4e4ec356184613799fbb45f179809bff54212f211934c4bb22968e1":
        raise ValueError("HPS source differs from the frozen ModelScope snapshot")
    excluded = {
        prompt_identity(json.loads(line)["prompt"])
        for path in args.exclude
        for line in path.read_text().splitlines()
        if line.strip()
    }
    rows = pq.read_table(args.parquet).to_pylist()
    groups = defaultdict(list)
    for index, source in enumerate(rows):
        prompt = source["prompt"].strip()
        key = prompt_identity(prompt)
        if key in excluded:
            continue
        excluded.add(key)
        words = len(prompt.split())
        stratum = "words:" + ("01-12" if words <= 12 else "13-24" if words <= 24 else "25+")
        row = dict(
            prompt=prompt,
            source="modelscope:Lakonik/t2i-prompts-hpsv2",
            source_revision="sha256:" + sha,
            source_id=str(index),
            stratum=stratum,
        )
        groups[stratum].append(row)
    queues = [
        deque(sorted(group, key=lambda r: hashlib.sha256(r["prompt"].encode()).hexdigest()))
        for _, group in sorted(groups.items())
    ]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
    args.out.mkdir(parents=True, exist_ok=False)
    for shard in range(2):
        # Alternate pairs so each host receives every stratum rather than alternate strata.
        selected = [row for i, row in enumerate(ordered) if (i // 4) % 2 == shard]
        (args.out / f"shard-{shard}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in selected)
        )
    report = dict(
        source_sha256=sha,
        source_rows=len(rows),
        eligible_rows=len(ordered),
        exclusions={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.exclude},
        strata={name: len(group) for name, group in groups.items()},
    )
    (args.out / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
