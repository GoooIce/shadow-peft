# 预注册：回到 QES 原作聚合（去逐侧非劣门），W8A8 载荷先试

日期：2026-09-10 晚｜分支：`codex/wayfinder-minicpm5-qes-faithful`（基线 `4a08426`，即 improve 分支 tip）
触发：用户指示原文「**完全换成 qes 的方法先试试 w8a8**」（见 `/tmp/handoff-qes-w8a8-20260910.md`）。

本文件在改动任何代码**之前**写就并冻结。改动对象是 09 票冻结协议的一部分（逐侧非劣门），故必须先预注册后动手。

---

## 1. 被改掉的是什么（精确语义）

**现状**（`pilot_lib.aggregate()`，两份工作副本字节相同）：

```python
ok = plus["delta_center"] >= -tau and minus["delta_center"] >= -tau
(qualified if ok else dropped).append(pair)
diffs = [p["plus"]["fitness"] - p["minus"]["fitness"] for p in qualified]
rewards = rank_shape_reward(diffs)          # 1 - 2*rank/(M-1)，降序
gradient[coord] += coefficient * realized[...][coord]   # plus: +reward, minus: -reward
```

镜像对只有**两侧都不低于中心 τ** 才参与聚合——这是 09 票在 QES 之上自加的约束。

**QES 原作**（一手核查 `docs/research/qes-primary-source-audit-20260909.md` §3，源码 L489–635）：默认对**全体种子**生成带符号系数（z-score；rank/elite 为可选模式）；mirror 模式先算正负候选奖励差、再给两侧对称系数；更新 RPC 传入**全体**种子系数。**原作没有任何逐侧合格门。**

**改动后**：删除合格门，全部镜像对参与 rank 聚合。其余一行不动。

## 2. 为什么改（已量化的证据，不是偏好）

- 随机 32 处改码在 v3 dev 上平均 **−3.6 点**（v2 dev −2.3）；逐侧合格率 44%→22%；镜像对需两侧同时合格 → **平方放大** → 合格对 12 → 0~1 → 梯度为空 → `moved=0` → 停机。三次运行（两种子、两机器）全部死于此。
- 上一轮 17 代全程 M = 12, 9, 5, 21, **1**——搜索一直贴着这道门运行。
- gen65 冻结前那条跑健康到 M=25 后一代内 50 候选全体转负（另一问题，见 §6），说明门的存在让整个搜索的存续依赖于「候选分布恰好不过门」这一脆弱条件。

## 3. 不变的东西（防止顺手改多）

| 项 | 决定 |
|---|---|
| 变换 `rank_shape_reward` | **保留**（QES rank 模式；同时换 z-score 会与去门混淆归因，不做） |
| 镜像对结构、±对称系数、realized 位移聚合 | 保留（本就与原作一致） |
| α / k=32 / CLAMP / GAMMA=1.0 / α 阶梯 | 保留 |
| 硬门（值相等三口径、readback、`--gate-halt`） | 保留（那是部署一致性契约，不是搜索启发式） |
| KL 守卫（绝对阈 6.4 + 材质带趋势条款） | 保留 |
| 停机规则（no_accept×2 等） | 保留——去门后 `no_accept` 饿死模式**按构造消失**（见 §4 预测 P2） |
| 残差语义、重放窗口、replay_noise | 保留 |
| 第 2 轮机制（retention/terminal/swap/band/slice） | **全部默认关**（faithful 最小协议）；纯记录类仪表（no-answer 归因、KL 探针）保留在记录路径 |

## 4. τ 的新角色

- τ=0.02 **不再**是搜索期参数：`aggregate()` 不再接收它。
- τ/δ=0.02 **保留**为**判定期**容差：`confirm_gate.py` 的非劣判读、`--retention-terminal` 的终局保留判据（后者本轮默认关）。CLI `--tau` 保留（记录于 summary/records 供判定复算），help 文案改为「confirm-time non-inferiority tolerance; does not gate aggregation」。

## 5. 记账与协议串

- `PROTOCOL`：`minicpm5-rank-mirror/v1` → **`minicpm5-rank-mirror-ungated/v2`**。注意 PROTOCOL 参与中心身份哈希（`CentreState.sha256`），故新协议下的中心哈希与旧协议**不可比**（这是期望行为：协议变了，身份空间就该换）。
- ledger 字段 `qualified_pairs` / `dropped_pairs` **保留名字**（下游工具兼容），语义变为「全部对 / 恒空」。`mechanism_audit.py` 对「non-inferiority gate」的引用同步改。
- 调用点三处同步：`pilot_search.py`（aggregate 调用）、`alpha_sweep.py`（重放路径）、`mechanism_audit.py`（文案/字段）。

## 6. 预注册的预测（可证伪）

- **P1（构造性）**：每一代 `qualified_pairs == pairs`（50 候选 → 25）、`dropped_pairs == []`。
- **P2（构造性）**：只要存在至少一对，梯度非空 → `moved > 0` → `no_accept` 停机不再可能由门饿死触发。
- **P3（经验性）**：在曾杀死三次启动的同一地形（v3 dev 风格，候选均值 −3.6 点）上，去门后前两代能产生非零更新（旧协议在此 `moved=0` 连续两代停机）。
- **不预测**：本改动**不**单独预测中心适应度改善——去门修复的是「搜索自己饿死」，不是「搜索方向正确」。中心是否变好由 W8A8 战役判定，不由本预注册判定。

## 7. 归因边界（如实）

本次转向同时改两件事：**位宽 W4A4→W8A8** 与**聚合去门**。单变量归因被推迟：本轮回答的是「faithful QES 在 W8A8 上是否 work」。若负结果，下一步才是分解（如 W4A8/W8A4 或 W4A4+去门对照）；若正结果，两因素叠加成立即可，不额外宣称单因素贡献。

## 8. 战役参数（接 300 代预注册，增量部分）

- 载荷：INT8 W8A8（LLM-Compressor，校准语料沿用非 UltraData 中性语料）——由并行进行中的载荷线交付（`codex/wayfinder-minicpm5-w8a8`，机器 `goooinext`）。
- 300 代 × 50 候选 × 32 改码；α=12 起点 + 阶梯；KL 守卫同前（`--kl-baseline` 以 W8A8 载荷**重测**后填入，不得沿用 W4A4 的 2.1346）。
- **多种子**：≥2 枚种子（两机各一），参数与输入除种子外完全相同。原作没有多种子记录，这是我们要补的空白。
- 判定：v4 封存池（confirm 1,043 episode，δ_min 1.60 点），每种子停止后各开一次；主张规则沿用第二轮五条（a–e）。
- 前置条件：① W8A8 载荷 + SM121 原生 INT8 内核三重口径通过；② BF16 vs W8A8 量化代价实测（决定 δ 选取）；③ gen65 脱同步至少要有判别性结论（同代重放），否则长跑风险自担并如实记录。

## 9. 改动清单（实施时逐项核对）

1. `pilot_lib.aggregate()`：去门、去 tau 形参；docstring 改写；返回字典字段名保留。
2. `PROTOCOL` 串 bump。
3. `pilot_search.py`：调用点去 tau；`--tau` help 改文案；其余 tau 记录点保留。
4. `alpha_sweep.py`：重放调用点同步（记录里的 tau 不再回放进聚合）。
5. `mechanism_audit.py`：门相关文案/字段改（门已不存在，审计不得再把它列为活机制）。
6. 单测：新增「全部对参与聚合、极端负 delta 仍参与、梯度非空」三项；跑 improve 分支既有 176 项单测确认无回归。
