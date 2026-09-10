"""三池划分 + 难度标定（16 票）。

口径：
  * 生成池 pool.jsonl（15 个 stratum x 270 题）；用 q0 的 teacher-forced 主轴结果做
    **难度标定**（不是选题偏向模型，而是把题集落在起始点的判别带内）：
    每个 stratum 内按 q0 对/错各取一半（各 32 题，缺则全取），组成 8 个 episode。
  * episode = 同 stratum 的固定 8 题簇 —— 07 冻结的统计独立单位。
  * 每个 stratum 的 8 个 episode 分配：dev 4 / confirm 2 / final 2（种子化洗牌，互不重叠）。
  * 三池搜索前冻结：写 dev/confirm/final JSONL + 清单（含 sha256、逐 stratum 组成、
    q0 在各池的准确率）。确认分片用后退出，终检只开一次。

用法：
  python3 select_pools.py --pool pool.jsonl --scores q0_pool_tf.jsonl --out-dir .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import BASE_SEED, stable_seed  # noqa: E402

EPISODE_SIZE = 8
EPISODES = {"dev": 4, "confirm": 2, "final": 2}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True, help="q0 teacher-forced 逐题结果 jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episode-size", type=int, default=EPISODE_SIZE)
    ap.add_argument("--exclude-from", default=None,
                    help="去污染报告 JSON（其 hit_item_ids 从题集中剔除，07 污染控制）")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.pool)]
    excluded = set()
    if a.exclude_from:
        rep = json.load(open(a.exclude_from))
        excluded = set(rep.get("hit_item_ids", []))
        before = len(items)
        items = [it for it in items if it["id"] not in excluded]
        print(f"decontamination exclusion ({os.path.basename(a.exclude_from)}): "
              f"{before} -> {len(items)} items ({before - len(items)} removed)")
    scores = {}
    for l in open(a.scores):
        r = json.loads(l)
        scores[r["id"]] = r
    missing = [it["id"] for it in items if it["id"] not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} items missing q0 scores, e.g. {missing[:3]}")

    strata = sorted({it["stratum"] for it in items})
    selected, report = [], {}
    for stratum in strata:
        pool_s = [it for it in items if it["stratum"] == stratum]
        ok = sorted([it for it in pool_s if scores[it["id"]].get("tf_correct")], key=lambda x: x["id"])
        bad = sorted([it for it in pool_s if not scores[it["id"]].get("tf_correct")], key=lambda x: x["id"])
        rng = random.Random(stable_seed(BASE_SEED, "select", stratum))
        rng.shuffle(ok)
        rng.shuffle(bad)
        per_stratum_total = a.episode_size * sum(EPISODES.values())
        want = per_stratum_total // 2
        take_ok = ok[:want]
        take_bad = bad[:want]
        if len(take_ok) + len(take_bad) < per_stratum_total:
            # 某一侧（q0 全对/全错）不足 32 时，用另一侧补齐到 8 个 episode 的规模；
            # 组成比例如实写进清单（饱和 stratum 只对「退化」敏感，对「增益」不敏感）。
            deficit = per_stratum_total - len(take_ok) - len(take_bad)
            extra_ok, extra_bad = ok[len(take_ok):], bad[len(take_bad):]
            add_ok = extra_ok[:deficit]
            take_ok += add_ok
            deficit -= len(add_ok)
            take_bad += extra_bad[:deficit]
        sel = take_ok + take_bad
        # episode 内平衡：奇偶交替放 ok/bad，保证每个 episode 的 q0 正确率接近 50%
        rng.shuffle(sel)
        ok_s = [x for x in sel if scores[x["id"]].get("tf_correct")]
        bad_s = [x for x in sel if not scores[x["id"]].get("tf_correct")]
        arranged = []
        while ok_s or bad_s:
            if ok_s:
                arranged.append(ok_s.pop())
            if bad_s:
                arranged.append(bad_s.pop())
        n_ep = len(arranged) // a.episode_size
        episodes = [arranged[i * a.episode_size:(i + 1) * a.episode_size] for i in range(n_ep)]
        order = list(range(n_ep))
        rng.shuffle(order)
        assign = {}
        cursor = 0
        for pool_name, k in EPISODES.items():
            for _ in range(min(k, n_ep - cursor)):
                assign[order[cursor]] = pool_name
                cursor += 1
        for i, ep in enumerate(episodes):
            pool_name = assign.get(i, "unused")
            ep_id = f"{stratum.replace('/', '-')}-ep{i:02d}"
            for it in ep:
                rec = dict(it)
                rec["episode"] = ep_id
                rec["pool"] = pool_name
                selected.append((pool_name, rec))
        report[stratum] = dict(
            pool_n=len(pool_s), q0_ok=len(ok), q0_bad=len(bad),
            selected_n=len(arranged), episodes=n_ep,
            q0_acc_pool=len(ok) / len(pool_s),
            q0_acc_selected=(sum(1 for x in arranged if scores[x["id"]].get("tf_correct")) / len(arranged)
                             if arranged else None),
            assign={k: sum(1 for i, p in assign.items() if p == k) for k in EPISODES},
        )

    os.makedirs(a.out_dir, exist_ok=True)
    manifest = dict(schema="mini16-taskset-pools-v1", frozen_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
                    episode_size=a.episode_size, episodes_per_stratum=EPISODES,
                    pool_file=os.path.basename(a.pool), scores_file=os.path.basename(a.scores),
                    decontamination_excluded_n=len(excluded),
                    decontamination_report=(os.path.basename(a.exclude_from) if a.exclude_from else None),
                    selection="within-stratum 50/50 q0-correct balance, seeded shuffle",
                    per_stratum=report, pools={})
    for pool_name in EPISODES:
        rows = sorted([r for p, r in selected if p == pool_name], key=lambda x: (x["stratum"], x["id"]))
        path = os.path.join(a.out_dir, f"taskset_{pool_name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        eps = sorted({r["episode"] for r in rows})
        manifest["pools"][pool_name] = dict(
            path=os.path.basename(path), n_items=len(rows), n_episodes=len(eps),
            sha256=sha256_file(path),
            q0_tf_acc=(sum(1 for r in rows if scores[r["id"]].get("tf_correct")) / len(rows)),
            per_family={f: dict(n=sum(1 for r in rows if r["family"] == f),
                                q0_acc=sum(1 for r in rows if r["family"] == f and
                                           scores[r["id"]].get("tf_correct")) /
                                       max(1, sum(1 for r in rows if r["family"] == f)))
                        for f in sorted({r["family"] for r in rows})},
            episode_ids=eps,
        )
    with open(os.path.join(a.out_dir, "pools_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_family"}
                      for k, v in manifest["pools"].items()}, indent=2, ensure_ascii=False))
    print("per-stratum selection report:")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "assign"} for k, v in report.items()},
                     indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
