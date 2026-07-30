#!/usr/bin/env python3
"""Download the real Bonsai packed + unpacked checkpoints via ModelScope.

These two repos are needed by ``analyze_bonsai_vs_unpacked.py`` to do the
real-data cross-validation described in ``deep-research-report.md`` §改进建议:

  - ``prism-ml/Ternary-Bonsai-27B-mlx-2bit``  — packed, ~8.49 GB (single file)
  - ``prism-ml/Ternary-Bonsai-27B-unpacked``  — FP16 reference, ~54 GB (12 shards)

Usage::

    # both repos (default), into the modelscope cache
    python experiment/download_bonsai.py --repo both

    # only the packed one, into a custom dir
    python experiment/download_bonsai.py --repo packed --cache-dir /data/bonsai

    # just print where things would land, download nothing
    python experiment/download_bonsai.py --repo both --dry-run

After download, pass the printed paths to the analysis script::

    python experiment/analyze_bonsai_vs_unpacked.py \
        --packed-dir <packed_path> --unpacked-dir <unpacked_path>
"""

from __future__ import annotations

import argparse
import sys

REPOS = {
    "packed": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
    "unpacked": "prism-ml/Ternary-Bonsai-27B-unpacked",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", choices=["packed", "unpacked", "both"], default="both")
    ap.add_argument(
        "--cache-dir", default=None, help="ModelScope cache root (default: library default)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print resolved paths without downloading"
    )
    args = ap.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("ERROR: modelscope is not installed.  pip install modelscope", file=sys.stderr)
        sys.exit(1)

    targets = ["packed", "unpacked"] if args.repo == "both" else [args.repo]

    paths: dict[str, str] = {}
    for key in targets:
        repo = REPOS[key]
        if args.dry_run:
            print(f"[dry-run] would download {repo}")
            continue
        print(f"downloading {repo} ...", flush=True)
        path = snapshot_download(
            model_id=repo,
            local_dir=args.cache_dir,
        )
        paths[key] = path
        print(f"  -> {path}", flush=True)

    if paths:
        print("\n=== resolved local paths ===")
        for key, p in paths.items():
            print(f"  {key}: {p}")
        print(
            "\nRun the comparison with:\n  python experiment/analyze_bonsai_vs_unpacked.py "
            f"--packed-dir {paths.get('packed', '<packed>')} "
            f"--unpacked-dir {paths.get('unpacked', '<unpacked>')}"
        )


if __name__ == "__main__":
    main()
