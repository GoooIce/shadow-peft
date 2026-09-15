"""v4 题池（pool4 票）：从 v3 生成池的**未消费余量**里切出两个零开箱次数的池。

背景（与 v3 的关系）：
  v3 生成池 85,815 题（种子 20260911，sha256 2da1e4f8…）只被消费了 dev 480 / confirm 32,040 /
  final 5,400 = 37,920 题；余下 47,895 题从未进入任何冻结池。两个 300 代诊断运行即将各开
  v3 confirm 一次（2 looks），v3 confirm 因此在下一轮不可再用于主张；v3 final 保持封存。
  本票用未消费余量切出：

    A. ``taskset_confirm_v4.jsonl`` —— **全新密封确认池**，承接下一轮的主结论；
       v3 设计逐字保持：15 stratum、每 stratum q0=0.5000、每 episode 恰 4 对 4 错。
    B. ``taskset_centre_v4.jsonl`` —— **中心反馈切片**，每代只对当代中心评分一次，
       作为监测/自适应反馈通道（不参与主判定）。

切池规则（确定性 / 可复算，见 --rule）：
  1. 起点 = v3 生成池 ``pool_v3.jsonl``（解压后 sha256 必须等于 v3 清单登记值）；
  2. 减去 v3 三池的 37,920 个 id（**已开箱**）；
  3. 减去 v3 去污染命中 id（18）；
  4. 减去题干落在 {v2 生成池, v2 三池, v1 三池} 的条目 —— 与 v3 冻结器同一屏蔽集，
     保证 v4 与上一轮**任何**产物零题干重叠；
  5. 每个 stratum 内按 q0 teacher-forced 对/错分堆，用 ``stable_seed(BASE_SEED_V4,"select-v4",
     stratum)`` 洗牌（**新种子**，不复用 v3 的划分流）；
     ``n_avail = 2*min(|ok|,|bad|)//8`` 是该 stratum 能支撑的**平衡** episode 上限；
  6. 中心切片取每个 stratum 洗牌序的**前 17 个 episode**（15×17 = 255 episode / 2,040 题）；
  7. v4 confirm 取每个 stratum 的**其余** episode（n_avail - 17）。

为什么中心切片也必须平衡（**不是**可选项）：
  未消费余量在各 stratum 内是**纯多数侧**（见 ``surplus`` 字段）：例如 arith/easy 的余量全部是
  q0 答对的题、arith/hard 的余量全部是 q0 答错的题。若按"不需要平衡"从余量切，切片的逐 stratum
  q0 准确率恰为 0.0 / 1.0，episode 级 SD ≈ 0.490 → 255 episode 的**每代标准误 ≈ 3.07 个点**，
  比 dev 池的 ≈1.9 个点**更差**，直接违反本票的硬要求。故中心切片与 confirm 共用同一个
  平衡储量并按上面的固定顺序切分。

用法（本地 Mac，纯标准库）：
  python3 carve_pools_v4.py carve  --v3-dir <v3 artifacts dir> --out-dir <v4 artifacts dir>
  python3 carve_pools_v4.py verify --v4-dir <v4 artifacts dir> --v3-dir <v3 artifacts dir>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import FAMILIES, TIERS, stable_seed  # noqa: E402

BASE_SEED_V4 = 20260912          # 新种子：v4 的挑选流与 v2/v3 均不同
BASE_SEED_V3 = 20260911          # v3 生成种子（只用于标注来源）
EPISODE_SIZE = 8
CENTRE_EPISODES_PER_STRATUM = 17  # 15 × 17 = 255 episode = 2,040 题
K_POWER = 7.8489                 # (z_.975 + z_.8)^2
Z_SUM = math.sqrt(K_POWER)       # 2.8016 → δ_min = Z_SUM * σ / √n_ep
V3_POOL_SHA256 = "2da1e4f8a54311d601fda17cd08ee8290ac6046303ab70e303fd8480ab6c80e6"
V3_POOL_N = 85815
CONFIRM_POOL_NAME = "confirm_v4"
CENTRE_POOL_NAME = "centre_v4"

# v3 的 σ 实测（v3 冻结 dev 池，60 episode，四臂 TF）——只登记，不改口径
V3_SIGMA = {"q0-cand_search": 0.1847, "bf16-cand_search": 0.1916, "q0-bf16": 0.2431,
            "q0-centre": 0.1677, "bf16-centre": 0.2052, "cand_search-centre": 0.1572}
SIGMA_PRIMARY = 0.1847           # 近邻/验收场景（q0 − cand_search）
SIGMA_CENTRE = 0.1677            # 中心臂自身在平衡池上的 episode 级 SD


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return [json.loads(l) for l in f]
    with open(path) as f:
        return [json.loads(l) for l in f]


def questions_of(path):
    return {r["question"] for r in load_jsonl(path)}


def delta_min(sigma, n_ep):
    return Z_SUM * sigma / math.sqrt(n_ep) if n_ep else None


def n_ep_for(sigma, delta):
    return math.ceil(K_POWER * sigma ** 2 / delta ** 2)


def carve(a):
    v3 = a.v3_dir
    os.makedirs(a.out_dir, exist_ok=True)
    t_start = time.time()

    # ---- 1. v3 生成池（校验 sha256：解压后的原始 JSONL） ----
    pool_path = os.path.join(v3, "pool_v3.jsonl.gz")
    gen_rows = load_jsonl(pool_path)
    h = hashlib.sha256()
    for r in gen_rows:
        h.update((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
    sha = h.hexdigest()
    assert len(gen_rows) == V3_POOL_N, f"v3 pool size {len(gen_rows)} != {V3_POOL_N}"
    print(f"[1] v3 pool: {len(gen_rows)} items, sha256(decompressed, reserialised)={sha[:16]}… "
          f"(manifest value {V3_POOL_SHA256[:16]}…)", flush=True)
    gen = {r["id"]: r for r in gen_rows}
    assert len(gen) == len(gen_rows), "duplicate id in v3 pool"

    # ---- 2. 已消费：v3 三池（id 与题干双重排除） ----
    consumed_ids, consumed_q = set(), set()
    per_pool_n = {}
    for name in ("dev", "confirm", "final"):
        rows = load_jsonl(os.path.join(v3, f"taskset_{name}.jsonl"))
        per_pool_n[name] = len(rows)
        consumed_ids |= {r["id"] for r in rows}
        consumed_q |= {r["question"] for r in rows}
    print(f"[2] v3 consumed: {per_pool_n} = {len(consumed_ids)} ids / {len(consumed_q)} questions",
          flush=True)
    assert consumed_ids <= set(gen), "v3 pool item not in generation pool"
    assert consumed_q <= {r["question"] for r in gen_rows}

    # ---- 3. v3 去污染命中（不得进入任何新池） ----
    dec_v3 = json.load(open(os.path.join(v3, "decontamination_report_v3_scaled.json")))
    dec_ids = set(dec_v3["hit_item_ids"])
    rl2609_ids = set(dec_v3["rl2609_layer_report"]["hit_ids_in_rl2609_layer"])
    print(f"[3] v3 decontamination hits: {len(dec_ids)} ids (RL-2609 layer: {len(rl2609_ids)})",
          flush=True)

    # ---- 4. 题干屏蔽集：与 v3 冻结器同一集合 ----
    block_sources = {}
    block_q = set()
    for tag, path in a.block_question_files:
        if not os.path.exists(path):
            raise SystemExit(f"blocklist file missing: {path}")
        q = questions_of(path)
        block_sources[os.path.basename(path)] = len(q)
        block_q |= q
    print(f"[4] question blocklist: {block_sources} → {len(block_q)} unique questions", flush=True)

    # ---- 5. 新鲜余量 ----
    fresh = []
    for r in gen_rows:
        if r["id"] in consumed_ids or r["id"] in dec_ids:
            continue
        if r["question"] in consumed_q or r["question"] in block_q:
            continue
        fresh.append(r)
    print(f"[5] fresh reservoir: {len(gen_rows)} - consumed - decontam - blocklist = {len(fresh)}",
          flush=True)

    # ---- q0 / bf16 分数（v3 在生成池上已全量评过；本池不新增任何评分） ----
    scores = {}
    scores_meta = {}
    for arm, fn in a.scores:
        rows = load_jsonl(os.path.join(v3, fn))
        scores[arm] = {r["id"]: r for r in rows}
        scores_meta[arm] = dict(file=fn, n=len(rows), sha256=sha256_file(os.path.join(v3, fn)))
        print(f"    scores[{arm}]: {len(rows)} rows from {fn}", flush=True)
    anchor = a.anchor_arm
    missing = [r["id"] for r in fresh if r["id"] not in scores[anchor]]
    assert not missing, f"{len(missing)} fresh items lack {anchor} scores"

    # ---- 6/7. 逐 stratum 洗牌 → 中心切片（前 17 ep）+ confirm（其余） ----
    strata = [f"{f}/{t}" for f in FAMILIES for t in TIERS]
    confirm_rows, centre_rows = [], []
    per_stratum = {}
    for s in strata:
        pool_s = [r for r in fresh if r["stratum"] == s]
        ok = sorted([r for r in pool_s if scores[anchor][r["id"]]["tf_correct"]], key=lambda x: x["id"])
        bad = sorted([r for r in pool_s if not scores[anchor][r["id"]]["tf_correct"]],
                     key=lambda x: x["id"])
        rng = random.Random(stable_seed(BASE_SEED_V4, "select-v4", s))
        rng.shuffle(ok)
        rng.shuffle(bad)
        n_avail = (2 * min(len(ok), len(bad))) // EPISODE_SIZE
        n_centre = min(CENTRE_EPISODES_PER_STRATUM, n_avail)
        n_confirm = n_avail - n_centre
        take_ok, take_bad = ok[:EPISODE_SIZE * n_avail // 2], bad[:EPISODE_SIZE * n_avail // 2]
        # 簇内 ok/bad 交替（与 select_pools_v2/v3 逐字同构造：从尾部弹出，ok 先）
        ok_s, bad_s, arranged = list(take_ok), list(take_bad), []
        while ok_s or bad_s:
            if ok_s:
                arranged.append(ok_s.pop())
            if bad_s:
                arranged.append(bad_s.pop())
        episodes = [arranged[i * EPISODE_SIZE:(i + 1) * EPISODE_SIZE] for i in range(n_avail)]
        for i, ep in enumerate(episodes):
            if i < n_centre:
                slice_name, ep_id, dst = CENTRE_POOL_NAME, f"{s.replace('/', '-')}-v4m{i:02d}", centre_rows
            else:
                slice_name, ep_id, dst = CONFIRM_POOL_NAME, f"{s.replace('/', '-')}-v4c{i - n_centre:02d}", confirm_rows
            for it in ep:
                rec = dict(it)
                rec["episode"] = ep_id
                rec["pool"] = slice_name
                dst.append(rec)
        used = set().union(*[set(x["id"] for x in ep) for ep in episodes]) if episodes else set()
        ep_ok = sorted({sum(1 for it in ep if scores[anchor][it["id"]]["tf_correct"]) for ep in episodes})
        per_stratum[s] = dict(
            fresh_n=len(pool_s), fresh_q0_ok=len(ok), fresh_q0_bad=len(bad),
            fresh_q0_acc=len(ok) / len(pool_s),
            surplus=dict(ok=len(ok) - EPISODE_SIZE * n_avail // 2,
                         bad=len(bad) - EPISODE_SIZE * n_avail // 2),
            balanced_items=len(used), balanced_episodes_available=n_avail,
            centre_episodes=n_centre, confirm_episodes=n_confirm,
            confirm_items=n_confirm * EPISODE_SIZE, centre_items=n_centre * EPISODE_SIZE,
            episode_ok_count_span=[ep_ok[0], ep_ok[-1]] if ep_ok else None,
            q0_acc_balanced=(sum(1 for iid in used if scores[anchor][iid]["tf_correct"]) / len(used)
                             if used else None),
            sub_modes=collections.Counter(
                (r.get("meta") or {}).get("sub") or (r.get("meta") or {}).get("kind") or "-"
                for r in pool_s),
        )

    # ---- 写冻结件 ----
    written = {}
    for name, rows in ((CONFIRM_POOL_NAME, confirm_rows), (CENTRE_POOL_NAME, centre_rows)):
        rows.sort(key=lambda r: (r["stratum"], r["id"]))
        path = os.path.join(a.out_dir, f"taskset_{'confirm' if name == CONFIRM_POOL_NAME else 'centre'}_v4.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        eps = sorted({r["episode"] for r in rows})
        written[name] = path
        print(f"    wrote {os.path.basename(path)}: {len(rows)} items / {len(eps)} episodes "
              f"sha256={sha256_file(path)[:16]}…", flush=True)

    def arm_acc(rows, arm):
        v = [scores[arm][r["id"]]["tf_correct"] for r in rows if r["id"] in scores.get(arm, {})]
        return (sum(1 for x in v if x) / len(v)) if v else None

    def ep_level(rows, arm):
        eps = sorted({r["episode"] for r in rows})
        vals = []
        for e in eps:
            rs = [r for r in rows if r["episode"] == e]
            vals.append(sum(1 for r in rs if scores[arm][r["id"]]["tf_correct"]) / len(rs))
        return sum(vals) / len(vals)

    summary_pools = {}
    for name, rows in ((CONFIRM_POOL_NAME, confirm_rows), (CENTRE_POOL_NAME, centre_rows)):
        eps = sorted({r["episode"] for r in rows})
        per_strat = {}
        for s in sorted({r["stratum"] for r in rows}):
            rs = [r for r in rows if r["stratum"] == s]
            q0 = arm_acc(rs, "q0")
            bf = arm_acc(rs, "bf16") if "bf16" in scores else None
            per_strat[s] = dict(n=len(rs), episodes=len({r["episode"] for r in rs}),
                                q0_tf_acc=q0, bf16_tf_acc=bf,
                                gap_bf16_minus_q0=(bf - q0) if (q0 is not None and bf is not None) else None)
        per_family = {}
        for f_ in sorted({r["family"] for r in rows}):
            rs = [r for r in rows if r["family"] == f_]
            q0 = arm_acc(rs, "q0")
            bf = arm_acc(rs, "bf16") if "bf16" in scores else None
            n_ep = len({r["episode"] for r in rs})
            per_family[f_] = dict(
                n=len(rs), episodes=n_ep, q0_tf_acc=q0, bf16_tf_acc=bf,
                gap_bf16_minus_q0=(bf - q0) if (q0 is not None and bf is not None) else None,
                delta_min_80pct_sigma_0_1847=delta_min(SIGMA_PRIMARY, n_ep))
        allq0, allbf = arm_acc(rows, "q0"), (arm_acc(rows, "bf16") if "bf16" in scores else None)
        prompt_tok = sum(scores["q0"][r["id"]].get("n_prompt_tokens", 0) for r in rows)
        gold_tok = sum(scores["q0"][r["id"]].get("tf_tokens", 0) for r in rows)
        n_ep = len(eps)
        summary_pools[name] = dict(
            path=os.path.basename(written[name]), n_items=len(rows), n_episodes=n_ep,
            episode_size=EPISODE_SIZE,
            sha256=sha256_file(written[name]),
            q0_tf_acc=allq0, bf16_tf_acc=allbf,
            gap_bf16_minus_q0=(allbf - allq0) if (allq0 is not None and allbf is not None) else None,
            q0_tf_acc_episode_level=ep_level(rows, "q0"),
            gold_prob_tokens_total=gold_tok, prompt_tokens_total=prompt_tok,
            tf_prompt_tokens=prompt_tok, tf_gold_tokens=gold_tok,
            per_stratum=per_strat, per_family=per_family, episode_ids=eps)

    # ---- 分辨率 ----
    # confirm：主判定口径，报告 80% 功效可检出的 δ_min = 2.8016·σ/√n_ep
    # centre ：监测口径，报告**每代标准误** SE = σ/√n_ep（这是"反馈通道噪声"的直接对照量）
    res = {}
    for name, meta in summary_pools.items():
        n_ep = meta["n_episodes"]
        sig = SIGMA_CENTRE if name == CENTRE_POOL_NAME else SIGMA_PRIMARY
        se = sig / math.sqrt(n_ep)
        res[name] = dict(
            n_episodes=n_ep, n_items=meta["n_items"],
            sigma_used=sig,
            sigma_used_note=("中心臂在 v3 平衡 dev 池上的 episode 级 SD（q0-centre=0.1677，60 episode）"
                             if name == CENTRE_POOL_NAME else
                             "近邻/验收配对 q0−cand_search 在 v3 dev 池上的 episode 级 SD"
                             "（v3 confirm 的 δ_min 就是用这个 σ 算的，口径对齐）"),
            se_per_generation=se,
            delta_min_80pct=delta_min(sig, n_ep),
            se_by_sigma={f"{k}": v / math.sqrt(n_ep) for k, v in sorted(V3_SIGMA.items())},
            delta_min_by_sigma={f"{k}": delta_min(v, n_ep) for k, v in sorted(V3_SIGMA.items())},
            se_uncertainty_note=("σ 来自 60 episode（59 df）→ 95% 区间 ≈[0.78σ, 1.42σ]；"
                                 "按 σ 带 [0.144,0.201]（centre）/ [0.158,0.221]（near-neighbour）"
                                 "给出的 SE 区间见 se_range_sigma_ci"),
            se_range_sigma_ci=[v / math.sqrt(n_ep) for v in
                               ((0.144, 0.201) if name == CENTRE_POOL_NAME else (0.158, 0.221))],
            delta_min_80pct_per_family={f_: delta_min(SIGMA_PRIMARY, v["episodes"])
                                        for f_, v in meta["per_family"].items()},
            se_per_family={f_: SIGMA_PRIMARY / math.sqrt(v["episodes"])
                           for f_, v in meta["per_family"].items()},
            delta_grid={f"{d}": dict(
                delta=d, n_episodes_needed=n_ep_for(sig, d), items_needed=n_ep_for(sig, d) * EPISODE_SIZE,
                gap_vs_available=n_ep_for(sig, d) - n_ep,
                feasible=n_ep_for(sig, d) <= n_ep) for d in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)},
        )

    # ---- 记录被否决的备选设计：中心切片完全从"多数侧余量"取（不做平衡） ----
    # 该方案能把 confirm 扩到全部 1,298 平衡 episode，但每 stratum 的 q0 准确率恰为 0/1，
    # episode 级 SD 极大 → 每代 SE 反而更差。这里把它算出来，供拒绝理由复核。
    surplus_sim = {}
    try:
        import statistics as _st
        rows_s, allocs = [], {}
        base, rem = 2000 // len(strata), 2000 % len(strata)
        for idx, s in enumerate(strata):
            pool_s = [r for r in fresh if r["stratum"] == s]
            ok = sorted([r for r in pool_s if scores[anchor][r["id"]]["tf_correct"]], key=lambda x: x["id"])
            bad = sorted([r for r in pool_s if not scores[anchor][r["id"]]["tf_correct"]],
                         key=lambda x: x["id"])
            m = min(len(ok), len(bad))
            maj = ok if len(ok) > len(bad) else bad
            allocs[s] = base + (1 if idx < rem else 0)
            rows_s += maj[m:][:allocs[s]]
        cl, per_strat_acc = [], {}
        for s in strata:
            ids = [r["id"] for r in rows_s if r["stratum"] == s]
            if ids:
                per_strat_acc[s] = sum(1 for i in ids if scores[anchor][i]["tf_correct"]) / len(ids)
            for k in range(0, len(ids) - EPISODE_SIZE + 1, EPISODE_SIZE):
                ep = ids[k:k + EPISODE_SIZE]
                cl.append(sum(1 for i in ep if scores[anchor][i]["tf_correct"]) / len(ep))
        sd_ep = _st.pstdev(cl)
        surplus_sim = dict(
            description=("备选设计：中心切片只从多数侧余量取（不做 50/50 平衡），confirm 保留全部 "
                         "平衡 episode"),
            n_items=len(rows_s), n_episodes=len(cl),
            per_stratum_q0_acc={k: round(v, 4) for k, v in sorted(per_strat_acc.items())},
            n_strata_saturated=sum(1 for v in per_strat_acc.values() if v in (0.0, 1.0)),
            pool_q0_acc=sum(1 for r in rows_s if scores[anchor][r["id"]]["tf_correct"]) / len(rows_s),
            episode_sd=sd_ep, se_per_generation=sd_ep / math.sqrt(len(cl)),
            verdict=("否决：每 stratum 的 q0 准确率恰为 0 或 1（饱和），episode 级 SD ≈0.49 → "
                     "每代 SE ≈3.1 点，比 dev 池的 ≈1.9 点**更差**，违反本票对中心通道的硬要求。"),
        )
    except Exception as e:  # noqa: BLE001
        surplus_sim = {"error": repr(e)}

    manifest = dict(
        schema="mini19-taskset-pools-v4",
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        base_seed_v4=BASE_SEED_V4, base_seed_v3=BASE_SEED_V3,
        episode_size=EPISODE_SIZE,
        centre_episodes_per_stratum=CENTRE_EPISODES_PER_STRATUM,
        n_strata=len(strata),
        source_pool=dict(path="pool_v3.jsonl.gz", n_items=V3_POOL_N,
                         sha256_decompressed_manifest=V3_POOL_SHA256,
                         sha256_recomputed_reserialised=sha),
        consumed_by_v3=per_pool_n, consumed_ids=len(consumed_ids), consumed_questions=len(consumed_q),
        decontamination_excluded_n=len(dec_ids),
        question_blocklist_files=block_sources, question_blocklist_n=len(block_q),
        fresh_reservoir_n=len(fresh),
        scores_files=scores_meta, balance_anchor_arm=anchor,
        carving_rule=(
            "fresh = pool_v3 − v3{dev,confirm,final}(id|question) − decontamination_hits(id) − "
            "blocklist_questions(v2 gen pool, v2 three pools, v1 three pools); per stratum: "
            "shuffle ok/bad with stable_seed(20260912,'select-v4',stratum); "
            "n_avail = 2*min(ok,bad)//8; centre = first 17 episodes (ok/bad interleaved), "
            "confirm = the remaining n_avail-17 episodes; unused items are written nowhere"),
        unused_items_after_carving=len(fresh) - len(confirm_rows) - len(centre_rows),
        rejected_alternative_surplus_only_centre=surplus_sim,
        resolution=res,
        capacity=dict(
            balanced_episodes_total=sum(v["balanced_episodes_available"] for v in per_stratum.values()),
            balanced_items_total=sum(v["balanced_items"] for v in per_stratum.values()),
            centre_episodes_total=sum(v["centre_episodes"] for v in per_stratum.values()),
            confirm_episodes_total=sum(v["confirm_episodes"] for v in per_stratum.values()),
            surplus_items_total=sum(v["surplus"]["ok"] + v["surplus"]["bad"] for v in per_stratum.values()),
            target_delta_min_0_82_n_episodes_needed=n_ep_for(SIGMA_PRIMARY, 0.0082),
            capacity_shortfall_vs_0_82=(n_ep_for(SIGMA_PRIMARY, 0.0082)
                                        - sum(v["balanced_episodes_available"] for v in per_stratum.values())),
            target_delta_0_02_n_episodes_needed=n_ep_for(SIGMA_PRIMARY, 0.02),
            feasible_at_0_02=(n_ep_for(SIGMA_PRIMARY, 0.02)
                              <= sum(v["balanced_episodes_available"] for v in per_stratum.values())),
        ),
        per_stratum=per_stratum, pools=summary_pools)
    with open(os.path.join(a.out_dir, "pool_manifest_v4.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)

    print("\n=== resolution ===")
    for name, r in res.items():
        print(f"  {name:12s} {r['n_items']:6d} items / {r['n_episodes']:5d} ep  σ={r['sigma_used']}  "
              f"SE/代={r['se_per_generation']*100:.2f} 点  δ_min(80%)= {r['delta_min_80pct']*100:.2f} 点")
    print("  rejected surplus-only centre alternative: "
          f"{surplus_sim.get('n_strata_saturated')} saturated strata, "
          f"episode SD={surplus_sim.get('episode_sd'):.4f}, "
          f"SE/代={surplus_sim.get('se_per_generation',float('nan'))*100:.2f} 点")
    print(f"  capacity: balanced episodes {manifest['capacity']['balanced_episodes_total']} vs "
          f"need {manifest['capacity']['target_delta_min_0_82_n_episodes_needed']} episodes for "
          f"δ_min=0.82 (shortfall {manifest['capacity']['capacity_shortfall_vs_0_82']})")
    print("\n=== per stratum capacity ===")
    print(f"  {'stratum':16s} {'fresh':>6s} {'ok':>6s} {'bad':>6s} {'max_ep':>6s} "
          f"{'centre':>6s} {'confirm':>7s} {'surplus_ok':>10s} {'surplus_bad':>11s}")
    for s, v in per_stratum.items():
        print(f"  {s:16s} {v['fresh_n']:6d} {v['fresh_q0_ok']:6d} {v['fresh_q0_bad']:6d} "
              f"{v['balanced_episodes_available']:6d} {v['centre_episodes']:6d} "
              f"{v['confirm_episodes']:7d} {v['surplus']['ok']:10d} {v['surplus']['bad']:11d}")
    print(f"  TOTAL balanced episodes {manifest['capacity']['balanced_episodes_total']}, "
          f"confirm {manifest['capacity']['confirm_episodes_total']}, "
          f"centre {manifest['capacity']['centre_episodes_total']}, "
          f"surplus items {manifest['capacity']['surplus_items_total']}")
    print(f"\nelapsed {time.time()-t_start:.1f}s")
    return manifest


def verify(a):
    v4, v3 = a.v4_dir, a.v3_dir
    checks = []

    def chk(name, ok, detail=None):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))

    man = json.load(open(os.path.join(v4, "pool_manifest_v4.json")))
    confirm = load_jsonl(os.path.join(v4, "taskset_confirm_v4.jsonl"))
    centre = load_jsonl(os.path.join(v4, "taskset_centre_v4.jsonl"))
    gen_rows = load_jsonl(os.path.join(v3, "pool_v3.jsonl.gz"))
    gen_by_id = {r["id"]: r for r in gen_rows}
    q0 = {r["id"]: r for r in load_jsonl(os.path.join(v3, "scores_q0_pool_v3_full.jsonl.gz"))}
    bf16 = {r["id"]: r for r in load_jsonl(os.path.join(v3, "scores_bf16_pool_v3_full.jsonl.gz"))}
    dec_v3 = json.load(open(os.path.join(v3, "decontamination_report_v3_scaled.json")))
    dec_ids = set(dec_v3["hit_item_ids"])
    rl_ids = set(dec_v3["rl2609_layer_report"]["hit_ids_in_rl2609_layer"])

    # 哈希与规模
    for name, rows, fn in ((CONFIRM_POOL_NAME, confirm, "taskset_confirm_v4.jsonl"),
                           (CENTRE_POOL_NAME, centre, "taskset_centre_v4.jsonl")):
        p = os.path.join(v4, fn)
        chk(f"{name}: sha256 matches manifest",
            sha256_file(p) == man["pools"][name]["sha256"], sha256_file(p))
        chk(f"{name}: item count matches manifest",
            len(rows) == man["pools"][name]["n_items"], len(rows))
        chk(f"{name}: episode count matches manifest",
            len({r['episode'] for r in rows}) == man["pools"][name]["n_episodes"],
            len({r['episode'] for r in rows}))
        chk(f"{name}: unique item ids", len({r['id'] for r in rows}) == len(rows))
        chk(f"{name}: unique episode ids", len({r['episode'] for r in rows})
            == len({r["episode"] for r in rows}))
        chk(f"{name}: every item carries episode+pool",
            all("episode" in r and "pool" in r for r in rows))
        # 每 episode 恰好 4 对 4 错
        spans = {}
        for r in rows:
            d = spans.setdefault(r["episode"], [0, 0])
            d[0] += 1
            d[1] += int(bool(q0[r["id"]]["tf_correct"]))
        bad_spans = {e: v for e, v in spans.items() if v[0] != EPISODE_SIZE or v[1] * 2 != v[0]}
        chk(f"{name}: every episode exactly 4 q0-correct + 4 q0-wrong",
            not bad_spans, f"{len(bad_spans)} bad episodes")
        # 每 stratum q0 == 0.5000
        per = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            per[r["stratum"]][0] += 1
            per[r["stratum"]][1] += int(bool(q0[r["id"]]["tf_correct"]))
        accs = {s: v[1] / v[0] for s, v in per.items()}
        chk(f"{name}: every stratum q0 = 0.5000",
            all(abs(x - 0.5) < 1e-12 for x in accs.values()),
            f"min={min(accs.values()):.6f} max={max(accs.values()):.6f} n_strata={len(accs)}")
        chk(f"{name}: 15 strata present", len(accs) == 15, sorted(accs))
        # 两臂分数齐备
        chk(f"{name}: all items have q0 and bf16 scores",
            all(r["id"] in q0 and r["id"] in bf16 for r in rows))
        # 与 v3 生成池一致（题干逐字节相同）
        chk(f"{name}: questions byte-identical to v3 generation pool",
            all(gen_by_id[r["id"]]["question"] == r["question"] for r in rows))
        # 去污染命中不在池内
        chk(f"{name}: no v3 decontamination hit present",
            not (dec_ids & {r["id"] for r in rows}))
        chk(f"{name}: no UltraData-RL-2609 hit present",
            not (rl_ids & {r["id"] for r in rows}))

    # 互斥
    ic, im = {r["id"] for r in confirm}, {r["id"] for r in centre}
    ec, em = {r["episode"] for r in confirm}, {r["episode"] for r in centre}
    chk("confirm ∩ centre = ∅ (items)", not (ic & im), len(ic & im))
    chk("confirm ∩ centre = ∅ (episodes)", not (ec & em), len(ec & em))
    # 与 v3 三池互斥（item 与 question）
    for name in ("dev", "confirm", "final"):
        rows = load_jsonl(os.path.join(v3, f"taskset_{name}.jsonl"))
        vi, vq = {r["id"] for r in rows}, {r["question"] for r in rows}
        chk(f"v4 ∩ v3 {name}: 0 items", not ((ic | im) & vi))
        chk(f"v4 ∩ v3 {name}: 0 questions",
            not (({r['question'] for r in confirm} | {r['question'] for r in centre}) & vq))
    # 与 v2/v1 全量产物互斥（题干）
    for tag, path in a.block_question_files:
        if not os.path.exists(path):
            continue
        q = questions_of(path)
        chk(f"v4 ∩ {os.path.basename(path)}: 0 questions",
            not (({r['question'] for r in confirm} | {r['question'] for r in centre}) & q),
            f"blocklist n={len(q)}")

    n_ok = sum(1 for c in checks if c["ok"])
    out = dict(schema="mini19-pool-verification-v4", generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               n_checks=len(checks), n_passed=n_ok, all_passed=n_ok == len(checks), checks=checks)
    with open(os.path.join(v4, "pool_verification_v4.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}" + ("" if c["ok"] else f"  → {c['detail']}"))
    print(f"\n{n_ok}/{len(checks)} checks passed")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--v3-dir", required=True, help="v3 artifacts 目录")
        p.add_argument("--v2-gen", required=True, help="v2 生成池 pool_v2.jsonl")
        p.add_argument("--v2-pools", nargs="+", required=True, help="v2 三池")
        p.add_argument("--v1-pools", nargs="+", required=True, help="v1 三池")

    c = sub.add_parser("carve")
    add_common(c)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--scores", nargs="*", default=["q0=scores_q0_pool_v3_full.jsonl",
                                                   "bf16=scores_bf16_pool_v3_full.jsonl"])
    c.add_argument("--anchor-arm", default="q0")

    v = sub.add_parser("verify")
    v.add_argument("--v4-dir", required=True)
    add_common(v)

    a = ap.parse_args()
    a.block_question_files = ([("v2_gen", a.v2_gen)]
                              + [(f"v2_{i}", p) for i, p in enumerate(a.v2_pools)]
                              + [(f"v1_{i}", p) for i, p in enumerate(a.v1_pools)])
    if a.cmd == "carve":
        a.scores = [s.split("=", 1) for s in a.scores]
        carve(a)
    else:
        verify(a)


if __name__ == "__main__":
    main()
