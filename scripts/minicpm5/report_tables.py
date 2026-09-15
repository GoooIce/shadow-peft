"""从 v3 产物生成笔记里的 markdown 表（避免手抄数字出错）。"""
import json
import os
import sys

ART = sys.argv[1] if len(sys.argv) > 1 else "docs/experiments/artifacts/minicpm5-pools-v3-20260910"


def load(p):
    return json.load(open(os.path.join(ART, p)))


m = load("pools_manifest.json")
pool = json.load(open(os.path.join(ART, "pool_v3_manifest.json"))) if False else None
res = load("resolution_summary.json")
pf = load("per_family_resolution.json")
pools = m["pools"]

print("### T1 生成池 BF16 − q0 逐 stratum（q0/bf16 同池同口径，item 级 TF）\n")
print("| stratum | 题库量 n | q0 TF | BF16 TF | **BF16 − q0** | q0 GEN | BF16 GEN |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
scores_q0 = {json.loads(l)["id"]: json.loads(l)
             for l in open(os.path.join(ART, "scores_q0_pool_v3_full.jsonl"))}
scores_bf = {json.loads(l)["id"]: json.loads(l)
             for l in open(os.path.join(ART, "scores_bf16_pool_v3_full.jsonl"))}
pool_items = [json.loads(l) for l in open(os.path.join(ART, "pool_v3.jsonl"))]
per = {}
for it in pool_items:
    d = per.setdefault(it["stratum"], [0, 0, 0, 0, 0, 0])
    d[0] += 1
    d[1] += int(bool(scores_q0[it["id"]]["tf_correct"]))
    d[2] += int(bool(scores_bf[it["id"]]["tf_correct"]))
    d[3] += int(bool(scores_q0[it["id"]].get("gen_correct")))
    d[4] += int(bool(scores_bf[it["id"]].get("gen_correct")))
for s in sorted(per):
    n, q, b, qg, bg = per[s][0], per[s][1], per[s][2], per[s][3], per[s][4]
    print(f"| `{s}` | {n:,} | {q/n:.3f} | {b/n:.3f} | **{(b-q)/n:+.3f}** | {qg/n:.3f} | {bg/n:.3f} |")
tot = [sum(per[s][i] for s in per) for i in range(5)]
n = tot[0]
print(f"| **全池** | **{n:,}** | **{tot[1]/n:.4f}** | **{tot[2]/n:.4f}** | "
      f"**{(tot[2]-tot[1])/n:+.4f}** | {tot[3]/n:.4f} | **{tot[4]/n:.4f}** |")

print("\n### T2 生成池 BF16 − q0 逐 family\n")
print("| family | n | q0 TF | BF16 TF | BF16 − q0 | q0 GEN | BF16 GEN |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
fam = {}
for it in pool_items:
    d = fam.setdefault(it["family"], [0, 0, 0, 0, 0])
    d[0] += 1
    d[1] += int(bool(scores_q0[it["id"]]["tf_correct"]))
    d[2] += int(bool(scores_bf[it["id"]]["tf_correct"]))
    d[3] += int(bool(scores_q0[it["id"]].get("gen_correct")))
    d[4] += int(bool(scores_bf[it["id"]].get("gen_correct")))
for s in sorted(fam):
    nn, q, b, qg, bg = fam[s]
    print(f"| `{s}` | {nn:,} | {q/nn:.3f} | {b/nn:.3f} | **{(b-q)/nn:+.3f}** | {qg/nn:.3f} | {bg/nn:.3f} |")

print("\n### T3 三池冻结件\n")
print("| 池 | 题数 | episode | sha256 | q0 TF (item/ep) | BF16 TF (item/ep) | BF16−q0 | TF token |")
print("| --- | ---: | ---: | --- | --- | --- | ---: | ---: |")
for p in ("dev", "confirm", "final"):
    d = pools[p]
    print(f"| `{p}` | {d['n_items']:,} | {d['n_episodes']:,} | `{d['sha256'][:24]}` | "
          f"{d['q0_tf_acc']:.4f} / {d['q0_tf_acc_episode_level']:.4f} | "
          f"{d['bf16_tf_acc']:.4f} / {d['bf16_tf_acc_episode_level']:.4f} | "
          f"**{d['gap_bf16_minus_q0']:+.4f}** | {d['tf_prompt_tokens']+d['tf_gold_tokens']:,} |")

print("\n### T4 confirm 逐 stratum 组成（q0 平衡点 + BF16 参照）\n")
print("| stratum | n | episodes | q0 TF | BF16 TF | BF16−q0 |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
c = pools["confirm"]
for s in sorted(c["per_stratum"]):
    v = c["per_stratum"][s]
    n_ep = sum(1 for e in c["episode_ids"] if e.startswith(s.replace("/", "-")))
    print(f"| `{s}` | {v['n']:,} | {n_ep} | {v['q0_tf_acc']:.3f} | {v['bf16_tf_acc']:.3f} | "
          f"**{v['gap_bf16_minus_q0']:+.3f}** |")
print(f"| **合计** | **{c['n_items']:,}** | **{c['n_episodes']:,}** | **0.5000** | "
      f"**{c['bf16_tf_acc']:.4f}** | **{c['gap_bf16_minus_q0']:+.4f}** |")

print("\n### T5 confirm 逐 family 组成\n")
print("| family | n | q0 TF | BF16 TF | BF16−q0 |")
print("| --- | ---: | ---: | ---: | ---: |")
for s in sorted(c["per_family"]):
    v = c["per_family"][s]
    print(f"| `{s}` | {v['n']:,} | {v['q0_tf_acc']:.3f} | {v['bf16_tf_acc']:.3f} | "
          f"**{v['gap_bf16_minus_q0']:+.3f}** |")

print("\n### T6 分辨率（80% 功效、双侧 α=0.05）\n")
print("| 池 | episode | item | δ_min episode 级 | δ_min item 级(naive) | 95% 区间半宽 | δ_min(gold_prob) |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for p in ("dev", "confirm", "final"):
    v = res["pools"][p]
    print(f"| `{p}` | {v['n_episodes']:,} | {v['n_items']:,} | "
          f"**{v['delta_min_episode_level']*100:.2f} 点** | {v['delta_min_item_level_naive']*100:.2f} 点 | "
          f"{v['interval_half_width_95']*100:.2f} 点 | **{v['delta_min_gold_prob_accuracy_points']*100:.2f} 点** |")
print(f"\nσ(episode, {res['pair_used_for_sigma']}) = {res['sigma_episode_acc']:.4f}；"
      f"σ_gold_prob = {res['sigma_episode_gold_prob']:.4f}；dS/dacc = {res['ds_dacc_gold_prob']:.3f}；"
      f"gold_prob 省样本 = **{res['gold_prob_sample_saving']:.2f}×**")

print("\n### T7 逐 family 分辨力与门控可行性（dev 规模）\n")
print("| family | dev episode | dev σ (实测) | δ_min@dev | δ_min@12ep(池化σ) | confirm 给每 family 的 episode | δ_min@confirm |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
vd = res["per_family_verdict"]
for f, sg in res["per_family_measured_dev_sigma"].items():
    import math
    print(f"| `{f}` | 12 | {sg:.3f} | {2.802*sg/math.sqrt(12)*100:.1f} 点 | "
          f"{vd['delta_min_per_family_at_dev_size']*100:.1f} 点 | "
          f"{vd['episodes_per_family_in_confirm']} | {vd['delta_min_per_family_at_confirm_size']*100:.1f} 点 |")
print(f"\n逐 stratum（4 episode）在 dev 规模：δ_min = "
      f"{pf['per_family_gating_verdict']['delta_min_per_stratum_at_dev_size']*100:.1f} 点（池化 σ）")
print(f"\n结论：`feasible_per_family_gating_at_dev_size = {vd['feasible_at_dev_size']}`；"
      f"confirm 规模 δ=0.02 可行 = {pf['per_family_gating_verdict']['feasible_per_family_gating_at_confirm_size_delta_0p02']}")
print("\n### T8 逐 stratum 的建模 σ（池尺度 q0−BF16 翻转率 + dev 簇内相关）\n")
print("| stratum | σ_episode（建模） |", " | ".join(["", ""]), "|")
print("| --- | ---: | --- | --- |")
for s, v in sorted(res["per_stratum_sigma_modelled_q0bf16"].items()):
    print(f"| `{s}` | {v:.3f} | | |")
