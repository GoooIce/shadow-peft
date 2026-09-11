# W8A8 QES-faithful 战役结案：种子方差主导，无实用增益、亦无伤害

日期：2026-09-12｜分支：`codex/wayfinder-minicpm5-qes-faithful`（`9cbbe4b`…`3586623`）
预注册：`minicpm5-qes-faithful-prereg-20260910.md`（§1–§10，开跑前冻结）
机器：仅 `goooinext`（`gooo` 另有用途）；产物：`goooinext:~/minicpm5-w8a8qes/out/seed{20260915,20260916}`

## 一句话结论

**300 代 QES 原作方法（去门聚合）在 W8A8 上，两枚种子的封存池效应方向相反（+0.43 vs −0.28 点）、都远低于 δ=2 的实用线、都非劣成立——按预注册规则，结论是「种子方差主导」。** 这正是原作没有多种子记录而留下的空白，现在有了刻度：单种子 300 代的效应落在 ±0.5 点内、可正可负。同时它不再是 W4A4 那种 −3.9 点的伤害（两种子 CI 下界最差 −0.85，远高于 −δ）。

## 设置（全部按预注册 §10）

- 载荷：`MiniCPM5-2B-INT8-W8A8`（sha `d639c738…`），vLLM 0.29.0 `TritonInt8ScaledMMLinearKernel`，`--linear-backend triton`。
- 方法：QES 原作无门聚合（全体镜像对 rank 聚合、±对称系数），`minicpm5-rank-mirror-ungated/v2`，GRID=int8-uniform。
- 300 代 × 50 候选 × 32 改码、α=12 起点；适应度池 v2 dev；KL 基线 1.1040（实测带 1.081–1.133，两种子全程未接近材质带 1.656 与绝对阈 3.312）；硬门/readback 全程 value-clean；**qp 全程 25/25**（去门聚合按预注册 P1 工作，饿死模式不再存在）；**α 两种子均零切换**。
- 判定：v4 封存池（1,043 episode，q0 参照 **0.5952**），`confirm_gate --looks 4`（从严，见偏差③）。

## 结果

| 种子 | 代数 | dev 轨迹（工作集） | q0 | 中心 | 配对差 | 95% CI | p_greater | 非劣 | 措辞 |
|---|---|---|---|---|---|---|---|---|---|
| 20260915 | 300/300（跑满） | 峰 0.6312，终 0.5958 | 0.5952 | 0.5995 | **+0.43 点** | [−0.10, +0.96] | 0.056 | ✓ | practical-gain-excluded |
| 20260916 | 300/300（跑满） | 峰 0.6292，终 0.6125 | 0.5952 | 0.5924 | **−0.28 点** | [−0.85, +0.30] | 0.84 | ✓ | practical-gain-excluded |

- **方向一致性（预注册主张条件）**：不一致 → 不得主张正收益，结论为种子方差主导。
- **拒答归因**：种子 A `gain_survives_both_answered`（gain_both +0.013 点、no_answer_excess −0.9 点，中心比 q0 更少拒答）；种子 B `no_gain`（各量 |<0.3 点|）。两枚的封存池读数都不是拒答模式造出来的。
- **五条主张规则（第二轮预注册）**：(a) 噪声带——效应量远小于池分辨率；(b) 封存池严格优于 q0——**两枚均不成立**（CI 上界 +0.96/+0.30 < δ=2）；(c) 逐族负迁移——未达主张阶段，不适用；(d) 不答率——A 改善、B 微升 +0.2 点；(e) 换载体——**不可评估**（见偏差④）。
- dev 峰值（两种子都到过 +2.5 点附近）没有在封存池兑现——与 W4A4 时代「dev 说行、封存池说不行」同构，进一步坐实 dev 分辨率（≈2 点/窗）不足以支撑选择。

## 偏差与披露（判定必须同读）

1. **崩溃-续跑 ×2**：两种子各崩一次于 int8 码 -128（`KeyError`，载荷带 158,002 个 -128 字节、对称格点外；修复＝冻结含 -128 的块，`3586623`）。A 崩于 gen47、B 崩于 gen17；`--resume` 续跑，参数一字未改；history 各多一行 centre_baseline（进程重启痕迹）。
2. **confirm-only 形状 bug**：两枚的首个 confirm 进程崩于 `terminal_retention` 的 arms 形状错配（`3586623` 修复）。
3. **开箱记账**：上述两个崩溃进程**已开箱评分但未落盘**（测量丢失）；本次判定用 `--looks 4`（2 中止 + 2 完成），比预注册的 2 从严。
4. **plain（换载体）臂不可评分**：无冻结 `Answer:` 后缀时 TF 评分器定位不到 gold span，两臂全零。**§3 swap 门在本次战役不可评估**，证据作废；这是仪表缺陷不是模型性质（修复方向：plain 形态的评分锚需按 chat 模板的回答起点重定义——未实施，避免再造开箱）。
5. **terminal 裁决为 null**：`terminal_retention` 跑完但 deliverable/record 空（疑与 plain 臂全零的退化输入有关）；交付物裁决退回 confirm_gate 的措辞（两枚均 practical-gain-excluded、非劣成立——按 §2 终局安全保留，**交付物应为 q0 载荷本身**）。
6. 机时（`goooinext`）：v2 续跑 A 11.7 h + B 13.2 h，v1 崩溃段 ≈3.0 h，confirm ×4 ≈0.6 h，冒烟 ≈0.3 h——**合计 ≈28.8 h**。budget.jsonl 有 301 个不重叠区间/种子备查。

## 对「QES 原命题」的状态

原命题：能否在 W8A8 格点上把任务能力推过 BF16。本战役给出的是**否定侧证据的加强**：300 代 × 2 种子的搜索没有产生超出种子噪声的移动（q0 已在 BF16 水平，见载荷线记录——W8A8 vs BF16 的差在 dev 分辨率内、PPL +0.65%）。下一步若继续，更值得问的不是「搜更久」，而是：扰动剂量-响应曲线（α 从未切换、moved 恒 384——更新可能系统性偏小）、以及把 swap 评分锚修好后再评估「载体无关性」。

## 产物指针

`goooinext:~/minicpm5-w8a8qes/`：`logs/campaign{,_v2}.log`、`out/seed*/{history.jsonl, stop_ledger.jsonl, summary.json, state.json, records/confirm_arms.json, records/confirm_verdict.json}`；判定命令：`confirm_gate.py --measure <arms.json> --looks 4 --out <verdict.json>`。v4 final 池**保持封存未动**。
