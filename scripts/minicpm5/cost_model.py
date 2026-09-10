"""17 票：整代成本复核（回填 13 的预算口径）。

口径（与 16 票一致，只是池规模变了）：
  * **每候选 dev 评分**：dev 池 480 题 × 每候选一次前向（TF 主轴）或两次（TF+GEN）；
  * **每代 confirm 评分**：整代只对当代优胜者跑一次（TF 主轴；GEN 可选）；
  * 引擎加载每次进程一次（vLLM 冷启 + 权重加载）；KF refit 按 11 票 31 ms/候选。
  * 吞吐常数取实测 tok/s（TF = prefill 全部 prompt+gold token；GEN = decode 输出 token）。

confirm 池的 token 量由**逐题 n_prompt_tokens 求和**精确得到（不重跑模型，保护 confirm 的
"用后退出"），吞吐常数借用同池上已实测的 q0/bf16 数字。

用法：
  python3 cost_model.py --manifest pools_manifest.json --pool-scores scores_q0_pool_v2.jsonl \
      --summary q0=scores_q0_pool_v2_summary.json --summary cand=scores_..._summary.json \
      --out cost_model.json --n-candidates 50
"""
from __future__ import annotations

import argparse
import json


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pool-scores", required=True, help="全池逐题评分（提供 n_prompt_tokens）")
    ap.add_argument("--summary", action="append", default=[], help="tag=summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-candidates", type=int, default=50)
    ap.add_argument("--refit-ms", type=float, default=31.0)
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    scores = {r["id"]: r for r in load_jsonl(a.pool_scores)}
    summaries = {}
    for s in a.summary:
        tag, path = s.split("=", 1)
        summaries[tag] = json.load(open(path))

    # 逐池 token 量（prompt + gold）与题数
    pools = {}
    for name, v in man["pools"].items():
        ids = [r["id"] for r in load_jsonl(v["path"])] if False else None
    # 用池文件的 sha 对应的 id 集合重算（池文件就在 manifest 同目录）
    import os
    base = os.path.dirname(os.path.abspath(a.manifest))
    for name, v in man["pools"].items():
        rows = load_jsonl(os.path.join(base, v["path"]))
        ptok = sum(scores[r["id"]]["n_prompt_tokens"] for r in rows if r["id"] in scores)
        gtok = sum(int(scores[r["id"]].get("tf_tokens") or 0) for r in rows if r["id"] in scores)
        outtok = sum(int(scores[r["id"]].get("gen_n_tokens") or 0) for r in rows if r["id"] in scores)
        pools[name] = dict(n_items=len(rows), n_episodes=v["n_episodes"],
                           prompt_tokens=ptok, gold_tokens=gtok,
                           tf_total_tokens=ptok + gtok,
                           gen_output_tokens_estimate=outtok,
                           sha256=v["sha256"])

    # 吞吐常数
    tp = {}
    for tag, s in summaries.items():
        tp[tag] = dict(engine_load_s=s.get("engine_load_s"),
                       tf_tokens_per_s=s.get("tf_timing", {}).get("tf_tokens_per_s"),
                       gen_tokens_per_s=s.get("gen_timing", {}).get("gen_tokens_per_s"),
                       n_items=s.get("n_items"),
                       tf_seconds=s.get("tf_timing", {}).get("tf_seconds"),
                       gen_seconds=s.get("gen_timing", {}).get("gen_seconds"))

    # 计时表：对每个吞吐常数，算出各池的评分秒数
    timing = {}
    for tag, t in tp.items():
        row = {}
        for name, p in pools.items():
            tf_s = p["tf_total_tokens"] / t["tf_tokens_per_s"] if t.get("tf_tokens_per_s") else None
            gen_s = (p["gen_output_tokens_estimate"] / t["gen_tokens_per_s"]
                     if t.get("gen_tokens_per_s") else None)
            row[name] = dict(tf_seconds=tf_s, gen_seconds=gen_s,
                             tf_gen_seconds=(tf_s + gen_s) if (tf_s and gen_s) else None)
        timing[tag] = row

    # 整代成本：50 候选 × dev(TF) + 1 × confirm(TF) + 加载
    gen_cost = {}
    for tag, t in tp.items():
        dev_tf = timing[tag]["dev"]["tf_seconds"]
        dev_gen = timing[tag]["dev"]["gen_seconds"]
        con_tf = timing[tag]["confirm"]["tf_seconds"]
        con_gen = timing[tag]["confirm"]["gen_seconds"]
        load = t["engine_load_s"]
        refit = a.n_candidates * a.refit_ms / 1000.0
        scenarios = {
            "tf_only": dict(
                candidate_scoring_s=a.n_candidates * dev_tf,
                refit_s=refit,
                engine_load_dev_s=load,
                engine_load_confirm_s=load,
                confirm_s=con_tf,
                total_s=a.n_candidates * dev_tf + refit + 2 * load + con_tf),
            "tf_plus_gen_dev_and_confirm": dict(
                candidate_scoring_s=a.n_candidates * (dev_tf + dev_gen),
                refit_s=refit,
                engine_load_dev_s=load,
                engine_load_confirm_s=load,
                confirm_s=con_tf + con_gen,
                total_s=a.n_candidates * (dev_tf + dev_gen) + refit + 2 * load + con_tf + con_gen),
            "tf_only_no_confirm_reload": dict(
                candidate_scoring_s=a.n_candidates * dev_tf,
                refit_s=refit,
                engine_load_dev_s=load,
                engine_load_confirm_s=0.0,
                confirm_s=con_tf,
                total_s=a.n_candidates * dev_tf + refit + load + con_tf),
        }
        for k, v in scenarios.items():
            v["total_min"] = v["total_s"] / 60.0
        gen_cost[tag] = dict(n_candidates=a.n_candidates, refit_ms_per_candidate=a.refit_ms,
                             per_candidate_dev_tf_s=dev_tf,
                             per_candidate_dev_tf_gen_s=(dev_tf + dev_gen) if dev_gen else None,
                             confirm_tf_s=con_tf, confirm_tf_gen_s=(con_tf + con_gen) if con_gen else None,
                             engine_load_s=load, scenarios=scenarios)

    rep = dict(schema="mini17-cost-model-v1", n_candidates=a.n_candidates,
               pool_token_counts=pools, throughput_constants=tp, per_pool_timing=timing,
               generation_cost=gen_cost,
               method=("池 token 量 = 逐题 n_prompt_tokens(+gold) 求和（精确，不重跑模型，"
                       "confirm 保持未开）；秒数 = token 量 / 实测 tok/s。"
                       "整代成本 = 50 x dev 每候选评分 + 1 x confirm 评分 + 2 次引擎加载 + 50 x 31 ms refit。"))
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("=== pool token counts ===")
    for k, v in pools.items():
        print(f"  {k:8s} items={v['n_items']:5d} ep={v['n_episodes']:4d} "
              f"tf_tokens={v['tf_total_tokens']:9d} gen_out_tokens={v['gen_output_tokens_estimate']:6d}")
    print("=== throughput ===")
    for k, v in tp.items():
        print(f"  {k:10s} load={v['engine_load_s']:.1f}s tf={v['tf_tokens_per_s']:.0f} tok/s "
              f"gen={v['gen_tokens_per_s']:.0f} tok/s")
    print("=== per-pool seconds (by throughput constant) ===")
    for tag, row in timing.items():
        print(f"  {tag}: " + " | ".join(
            f"{p}: tf={r['tf_seconds']:.1f}s tf+gen={(r['tf_gen_seconds'] or 0):.1f}s"
            for p, r in row.items()))
    print("=== generation cost (50 candidates) ===")
    for tag, g in gen_cost.items():
        print(f"  {tag}: per-cand dev TF={g['per_candidate_dev_tf_s']:.2f}s "
              f"dev TF+GEN={g['per_candidate_dev_tf_gen_s']:.2f}s confirm TF={g['confirm_tf_s']:.1f}s "
              f"confirm TF+GEN={g['confirm_tf_gen_s']:.1f}s")
        for k, v in g["scenarios"].items():
            print(f"     {k:32s} total={v['total_s']:7.1f}s = {v['total_min']:.2f} min")


if __name__ == "__main__":
    main()
