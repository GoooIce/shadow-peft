"""v2 三池划分（17 票）：confirm 900 episode / 7200 题规模的冻结器。

相对 16 票 `select_pools.py` 的变化：
  * 每 stratum 的 episode 分配从 4/2/2（8 个）放大到 **dev 4 / confirm 60 / final 10（74 个）**；
  * 冻结件里写入**评分口径**（prompt 渲染 / 停止条件 / prefix cache / 引擎参数）与
    q0 难度标定的来源，使 confirm 分片"用后退出"时有完整凭据；
  * 逐 stratum 记录 q0 对/错两侧的实际可用量与被"多数侧填充"的数量（组成透明化）；
  * 输出 `power` 块：给定 episode 数与实测 σ 的可检出 δ（80% 功效）。

选择规则与 16 票一致：stratum 内按 q0 teacher-forced 对/错 50/50 取题（每侧上限 =
episode_size * n_episodes_total / 2），不足时用另一侧补齐；episode = 同 stratum 的
8 题簇，簇内 ok/bad 交替放置；episode 种子化洗牌后按 dev → confirm → final 顺序分配。

用法：
  python3 select_pools_v2.py --pool pool_v2.jsonl --scores scores_q0_pool_v2.jsonl \
      --exclude-from decontamination_report_v2.json --out-dir . \
      --episodes dev=4 confirm=60 final=10 --scoring-protocol protocol.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import ANSWER_PREFIX, BASE_SEED, GOLD_PREFIX, INSTRUCTION, stable_seed  # noqa: E402

EPISODE_SIZE = 8
K_POWER = 7.8489  # (z_.975 + z_.8)^2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_episodes(spec_list):
    out = {}
    for s in spec_list:
        k, v = s.split("=", 1)
        out[k] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episode-size", type=int, default=EPISODE_SIZE)
    ap.add_argument("--episodes", nargs="*", default=["dev=4", "confirm=60", "final=10"])
    ap.add_argument("--exclude-from", default=None)
    ap.add_argument("--scoring-protocol", default=None, help="JSON：评分口径（写进清单）")
    ap.add_argument("--sigma", type=float, default=None, help="实测 episode 配对差 SD（写进 power 块）")
    ap.add_argument("--sigma-source", default=None)
    ap.add_argument("--schema", default="mini17-taskset-pools-v2")
    a = ap.parse_args()

    episodes_spec = parse_episodes(a.episodes)
    total_ep = sum(episodes_spec.values())
    per_side = a.episode_size * total_ep // 2

    items = [json.loads(l) for l in open(a.pool)]
    excluded = set()
    if a.exclude_from:
        rep = json.load(open(a.exclude_from))
        excluded = set(rep.get("hit_item_ids", []))
        before = len(items)
        items = [it for it in items if it["id"] not in excluded]
        print(f"decontamination exclusion: {before} -> {len(items)} "
              f"({before - len(items)} removed)")
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
        rng = random.Random(stable_seed(BASE_SEED, "select-v2", stratum))
        rng.shuffle(ok)
        rng.shuffle(bad)
        take_ok, take_bad = ok[:per_side], bad[:per_side]
        padded_from_ok = padded_from_bad = 0
        need = a.episode_size * total_ep
        if len(take_ok) + len(take_bad) < need:
            deficit = need - len(take_ok) - len(take_bad)
            extra_ok, extra_bad = ok[len(take_ok):], bad[len(take_bad):]
            add_ok = extra_ok[:deficit]
            padded_from_ok = len(add_ok)
            deficit -= len(add_ok)
            take_ok += add_ok
            add_bad = extra_bad[:deficit]
            padded_from_bad = len(add_bad)
            take_bad += add_bad
        sel = take_ok + take_bad
        rng.shuffle(sel)
        ok_s = [x for x in sel if scores[x["id"]].get("tf_correct")]
        bad_s = [x for x in sel if not scores[x["id"]].get("tf_correct")]
        arranged = []
        while ok_s or bad_s:            # 簇内 ok/bad 交替 → 每 episode 的 q0 分布均衡
            if ok_s:
                arranged.append(ok_s.pop())
            if bad_s:
                arranged.append(bad_s.pop())
        n_ep = len(arranged) // a.episode_size
        episodes = [arranged[i * a.episode_size:(i + 1) * a.episode_size] for i in range(n_ep)]
        order = list(range(n_ep))
        rng.shuffle(order)
        assign, cursor = {}, 0
        for pool_name, k in episodes_spec.items():
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
        # 每个 episode 的 q0 正确数分布（组成透明化）
        ep_ok = sorted({sum(1 for it in ep if scores[it["id"]].get("tf_correct"))
                        for ep in episodes})
        report[stratum] = dict(
            pool_n=len(pool_s), q0_ok_pool=len(ok), q0_bad_pool=len(bad),
            q0_acc_pool=len(ok) / len(pool_s),
            per_side_target=per_side,
            q0_ok_used=len(take_ok), q0_bad_used=len(take_bad),
            padded_from_ok=padded_from_ok, padded_from_bad=padded_from_bad,
            selected_n=len(arranged), episodes=n_ep,
            q0_acc_selected=(sum(1 for x in arranged if scores[x["id"]].get("tf_correct"))
                             / len(arranged)),
            episode_ok_count_span=[ep_ok[0], ep_ok[-1]] if ep_ok else None,
            assign={k: sum(1 for p in assign.values() if p == k) for k in episodes_spec},
            sub_modes=(__import__("collections").Counter(
                (it.get("meta") or {}).get("sub") or (it.get("meta") or {}).get("kind") or "-"
                for it in pool_s).most_common()),
        )

    os.makedirs(a.out_dir, exist_ok=True)
    protocol = json.load(open(a.scoring_protocol)) if a.scoring_protocol else {}
    power = None
    if a.sigma:
        power = {}
        for name, k in episodes_spec.items():
            n_ep = len(episodes_spec) and k * len(strata)
            power[name] = dict(
                n_episodes=n_ep, items=n_ep * a.episode_size, sigma_used=a.sigma,
                delta_detectable_80pct=(K_POWER ** 0.5) * a.sigma / (n_ep ** 0.5) if n_ep else None,
                delta_grid={f"{d}": dict(
                    delta=d, n_episodes_needed=__import__("math").ceil(
                        K_POWER * a.sigma ** 2 / d ** 2),
                    items_needed=__import__("math").ceil(K_POWER * a.sigma ** 2 / d ** 2)
                    * a.episode_size) for d in (0.01, 0.015, 0.02, 0.03, 0.05)},
                sigma_source=a.sigma_source)
    manifest = dict(
        schema=a.schema, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        episode_size=a.episode_size, episodes_per_stratum=episodes_spec,
        episodes_total=sum(episodes_spec.values()) * len(strata),
        n_strata=len(strata),
        pool_file=os.path.basename(a.pool), scores_file=os.path.basename(a.scores),
        decontamination_excluded_n=len(excluded),
        decontamination_report=(os.path.basename(a.exclude_from) if a.exclude_from else None),
        selection=("within-stratum q0 teacher-forced 50/50 balance (per-side cap "
                   f"{per_side}), seeded shuffle, ok/bad interleaved episodes, "
                   "seeded episode->pool assignment"),
        scoring_protocol=protocol, power=power, per_stratum=report, pools={})
    for pool_name in episodes_spec:
        rows = sorted([r for p, r in selected if p == pool_name],
                      key=lambda x: (x["stratum"], x["id"]))
        path = os.path.join(a.out_dir, f"taskset_{pool_name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        eps = sorted({r["episode"] for r in rows})
        manifest["pools"][pool_name] = dict(
            path=os.path.basename(path), n_items=len(rows), n_episodes=len(eps),
            sha256=sha256_file(path),
            q0_tf_acc=(sum(1 for r in rows if scores[r["id"]].get("tf_correct")) / len(rows)),
            q0_tf_acc_episode_level=(
                sum(sum(1 for r in rows if r["episode"] == e and scores[r["id"]].get("tf_correct"))
                    / max(1, sum(1 for r in rows if r["episode"] == e)) for e in eps) / len(eps)),
            per_stratum={s: dict(
                n=sum(1 for r in rows if r["stratum"] == s),
                q0_acc=sum(1 for r in rows if r["stratum"] == s and scores[r["id"]].get("tf_correct"))
                / max(1, sum(1 for r in rows if r["stratum"] == s)))
                for s in sorted({r["stratum"] for r in rows})},
            per_family={f: dict(
                n=sum(1 for r in rows if r["family"] == f),
                q0_acc=sum(1 for r in rows if r["family"] == f and scores[r["id"]].get("tf_correct"))
                / max(1, sum(1 for r in rows if r["family"] == f)))
                for f in sorted({r["family"] for r in rows})},
            episode_ids=eps,
        )
    with open(os.path.join(a.out_dir, "pools_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("episode_ids",)}
                      for k, v in manifest["pools"].items()},
                     indent=2, ensure_ascii=False)[:2500])
    print("--- per-stratum ---")
    for s, v in report.items():
        print(f"  {s:16s} pool={v['pool_n']:5d} q0acc={v['q0_acc_pool']:.3f} "
              f"used_ok/bad={v['q0_ok_used']}/{v['q0_bad_used']} "
              f"pad={v['padded_from_ok']}+{v['padded_from_bad']} "
              f"sel_acc={v['q0_acc_selected']:.3f} ep_ok_span={v['episode_ok_count_span']}")


if __name__ == "__main__":
    main()
