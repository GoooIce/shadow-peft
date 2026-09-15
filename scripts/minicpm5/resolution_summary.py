"""v3 轮：把"实际达到的分辨率"算成一张表（item 级 / episode 级 / gold_prob 连续量）。

分辨率口径（80% 功效、双侧 α=0.05）：
  δ_min = (z_.975 + z_.8) * σ / sqrt(n) = 2.802 * σ / sqrt(n)

  * **episode 级**（登记统计单位）：n = 池的 episode 数，σ = 近邻配对的 episode 配对差 SD
    （v3 冻结 dev 池实测 q0−cand_search = 0.1847；v2 冻结值 0.2274 作对照）。
  * **item 级**：配对差的 item 级均值与 episode 级均值**恒等**（等簇大小），所以正确的 SE 就是
    episode 级的 SE（cluster-corrected）。naive 口径 σ_item/sqrt(N) 只有在簇内相关 ρ≤0 时才不
    乐观；ρ>0 时它会低估 SE（本池 q0−bf16 的 ρ=0.115 → naive 乐观 34%），两口径都列出。
  * **gold_prob = exp(mean gold logprob)**：连续统计量，n_ep(δ) 用该配对自身的 dS/dacc 换算。

用法：
  python3 resolution_summary.py --dev-scores ... --pairs-json pool_power_dev3.json \
      --per-family-json per_family_resolution.json \
      --pool-sizes '{"dev":60,"confirm":4005,"final":675}' --out resolution_summary.json
"""
from __future__ import annotations

import argparse
import json
import math

Z_A2 = 1.959963984540054
Z_B = 0.8416212335729143
K = (Z_A2 + Z_B) ** 2      # 7.8489
EPISODE_SIZE = 8


def dmin(sigma, n):
    return (Z_A2 + Z_B) * sigma / math.sqrt(n)


def n_ep_for(sigma, delta):
    return math.ceil(K * sigma * sigma / (delta * delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json", required=True, help="pool_power_dev3.json")
    ap.add_argument("--per-family-json", required=True)
    ap.add_argument("--pool-sizes", required=True, help='{"dev":60,"confirm":4005,"final":675}')
    ap.add_argument("--sigma-v2-reference", type=float, default=0.2274)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.02)
    a = ap.parse_args()

    pairs = json.load(open(a.pairs_json))["pairs"]
    pf = json.load(open(a.per_family_json))
    sizes = json.loads(a.pool_sizes)
    key = "q0-cand_search" if "q0-cand_search" in pairs else "q0-bf16"
    p = pairs[key]
    sd_acc = p["signals"]["acc"]["sd_episode_delta"]
    sd_gp = p["signals"]["gold_prob"]["sd_episode_delta"]
    ds_dacc = (p["signals"]["gold_prob"]["mean_episode_delta"]
               / p["signals"]["acc"]["mean_episode_delta"])
    sd_v2 = a.sigma_v2_reference
    recompute = pf["per_family_gating_verdict"].get("recommendation")

    # item 级：簇内相关 ρ（由 dev 上 acc 的 item/episode 方差反推）
    item_var = None
    item_rows = json.load(open(a.pairs_json)).get("per_stratum")  # 仅占位，不用
    rho = None
    # 由 q0-bf16 的 dev ρ 代表（per_family_resolution 里已算），这里给出两口径对照
    pf_pairs = pf.get("pairs", {})
    dev_pair = pf_pairs.get(key, {})
    n_items_dev = dev_pair.get("n_items")

    out = dict(
        statistic=(f"episode 级配对差（每 episode 8 题，{EPISODE_SIZE} 题一簇）"),
        pair_used_for_sigma=key,
        sigma_episode_acc=sd_acc,
        sigma_episode_acc_v2_reference=sd_v2,
        sigma_episode_gold_prob=sd_gp,
        ds_dacc_gold_prob=ds_dacc,
        n_ep_for_delta_gold_prob=n_ep_for(sd_gp, a.delta * ds_dacc),
        n_ep_for_delta_acc=n_ep_for(sd_acc, a.delta),
        gold_prob_sample_saving=(n_ep_for(sd_acc, a.delta) / n_ep_for(sd_gp, a.delta * ds_dacc)),
        pools={},
        min_episode_sigma_available=None,
    )
    sigma_item_dev = math.sqrt(p["item_flip_rate_acc"])       # 近邻配对的 item 级配对差 SD
    for name, n_ep in sizes.items():
        n_items = n_ep * EPISODE_SIZE
        d_ep = dmin(sd_acc, n_ep)
        d_gp = dmin(sd_gp, n_ep) / max(ds_dacc, 1e-9)
        out["pools"][name] = dict(
            n_episodes=n_ep, n_items=n_items,
            delta_min_episode_level=d_ep,
            delta_min_episode_level_at_v2_sigma=dmin(sd_v2, n_ep),
            delta_min_item_level_naive=2.802 * sigma_item_dev / math.sqrt(n_items),
            delta_min_item_level_clustercorrected=d_ep,
            delta_min_gold_prob_accuracy_points=d_gp,
            delta_min_ratio_gold_prob_over_acc=(d_ep / d_gp if d_gp else None),
            interval_half_width_95=Z_A2 * sd_acc / math.sqrt(n_ep),
            items_needed_for_delta_0p02=n_ep_for(sd_acc, a.delta) * EPISODE_SIZE,
        )
    out["sigma_item_dev_near_neighbour"] = sigma_item_dev
    # item 级两口径（用 q0-bf16 的 ρ 举例说明 cluster 修正的必要性）
    ext = pf.get("extrapolation", {}).get("pairs", {})
    rho_example = None
    for pk, v in ext.items():
        if pk == "q0-bf16":
            rho = v["rho_within_episode_used"]
            sigma_item = v["sigma_item_all"]
            for name, n_ep in sizes.items():
                n_items = n_ep * EPISODE_SIZE
                out["pools"][name]["delta_min_item_level_naive_q0bf16_profile"] = \
                    2.802 * sigma_item / math.sqrt(n_items)
                out["pools"][name]["delta_min_item_level_clustercorrected_q0bf16_profile"] = \
                    dmin(sigma_item * math.sqrt((1 + 7 * v["rho_within_episode_used"]) / 8), n_ep)
            rho_example = dict(pair=pk, rho=rho, sigma_item=sigma_item,
                               naive_optimism_factor=(
                                   (2.802 * sigma_item / math.sqrt(sizes['confirm'] * 8))
                                   / dmin(sigma_item * math.sqrt((1 + 7 * rho) / 8), sizes['confirm'])))
    out["item_level"] = rho_example
    out["per_stratum_sigma_modelled_q0bf16"] = {
        s: v["sigma_episode_modelled"] for s, v in
        ext.get("q0-bf16", {}).get("per_stratum", {}).items()}
    out["per_family_sigma_modelled_q0bf16"] = {
        s: v["sigma_episode_modelled"] for s, v in
        ext.get("q0-bf16", {}).get("per_family", {}).items()}
    out["per_family_measured_dev_sigma"] = {
        f: v["sd_episode_delta"] for f, v in dev_pair.get("per_family", {}).items()}
    out["per_family_verdict"] = dict(
        feasible_at_dev_size=pf["per_family_gating_verdict"]["feasible_per_family_gating_at_dev_size"],
        delta_min_per_family_at_dev_size=pf["per_family_gating_verdict"]["delta_min_per_family_at_dev_size"],
        delta_min_per_family_at_confirm_size=pf["per_family_gating_verdict"]["delta_min_per_family_at_confirm_size"],
        episodes_per_family_in_confirm=pf["per_family_gating_verdict"]["confirm_episodes_per_family"],
        recommendation=recompute)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"σ (episode, {key}) = {sd_acc:.4f}  [v2 参考 {sd_v2:.4f}]  σ_gold_prob = {sd_gp:.4f}  "
          f"dS/dacc = {ds_dacc:.3f}")
    print(f"gold_prob 省样本 = {out['gold_prob_sample_saving']:.2f}× "
          f"(n_ep(δ=2pts): acc {out['n_ep_for_delta_acc']} vs gold_prob {out['n_ep_for_delta_gold_prob']})")
    for name, v in out["pools"].items():
        print(f"  {name:8s} n_ep={v['n_episodes']:5d} δ_min(episode)={v['delta_min_episode_level']*100:.2f} 点 "
              f"(v2 σ: {v['delta_min_episode_level_at_v2_sigma']*100:.2f}) "
              f"δ_min(item,naive)={v['delta_min_item_level_naive']*100:.2f} "
              f"半宽={v['interval_half_width_95']*100:.2f} "
              f"δ_min(gold_prob)={v['delta_min_gold_prob_accuracy_points']*100:.2f} "
              f"({v['delta_min_ratio_gold_prob_over_acc']:.2f}× 相对 acc)")
    print("item 级 q0-bf16 profile:", json.dumps(out["item_level"], ensure_ascii=False))


if __name__ == "__main__":
    main()
