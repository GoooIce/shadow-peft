# exp15 成本感知 per-task 升级阈值（spec #48 / issue #52 + #53）

一句话收束：**exp15 的“拟合 → 配置 → 评测”管线已在 `goooinext` 跑通：8 个合成任务 × 60 变体（480 条，train/val 各 240）在 λ=0.1 下拟合出全局阈值 0.45 和 6 个 per-task 阈值，`policy.json` 可被 `EdgeConfig` 接受；但 `dataset_source=synthetic`、`claims_gain=false`、`evidence=insufficient`、`go_no_go=pipeline_only`，本轮只验证流程，不构成真实收益证据，也没有改动任何生产默认值。**

状态：**synthetic 管线在 `goooinext` 跑完（2026-10-21）**；真实 replay 集（≥500 条、每任务 train/val 各 ≥30 条）**尚未提供**，因此 per-task 阈值相对全局阈值是否真有收益仍无结论。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（dc4a，DGX Spark GB10，121 GB 统一内存），Linux 6.17.0-1032-nvidia aarch64，20 核 |
| 代码 | 工作树 `p16-t5`，分支 `ticket/p16-t5`，基于集成分支 `feat/p1-6-cost-policy` @ `ab2586a`（T4 #52 已合入） |
| 运行时 | `~/Code/shadow-peft/.venv/bin/python`，Python 3.12.3；只传 `src/edge_cloud/` 与 `scripts/exp15_cost_policy.py` 到临时目录，未覆盖远端仓库 |
| 数据 | `--synthetic`：`edge_cloud.knowledge.ENTRIES` 8 个条目 × 每目 60 个确定性 prompt 变体 = 480 条 `CostSample`；偶数变体 train、奇数变体 val，每个条目 train/val 各 30 |
| 任务分布 | `process_qa`、`fault_diagnosis`（2 条目）、`quality_verdict`、`standard_qa`、`process_calc`、`improvement`、`safety` 共 7 个任务，val 240 条 |
| 成本 | `CostModel` 默认配置价：edge token 0.000002、cloud input 0.000004、cloud output 0.000012、每次 call 0.0005、RTT 120 ms；延迟用闭式 `LatencyModel`（非墙钟） |
| 策略参数 | λ 网格 `0, 0.05, 0.1, 0.2, 0.5, 1.0`，写入 `policy.json` 的 λ=0.1；`min_samples=30`；阈值网格 0.00–1.00 步长 0.05；`require_cloud_tasks=quality_verdict`；每任务 val 升级率上限 0.6 |
| seed / bootstrap | seed 15；val bootstrap 1000 次 |
| 产物指纹 | `report.json` `sha256:0650f1e7c3e3c2e83172251b6f0f4940bab3679f73a2d253744729ac42f400f4`；`policy.json` `sha256:e0ee8240cfd7a46d21ac341c6b04d485c8ecc2a5487924abcec11e8d5de532ee`；`dataset.jsonl` `sha256:7a05c9630e03821184c0f7377c949fdcf51bd31befa5e14e4a8b4d0ced723ce2` |
| 确定性 | 同参数连续跑两次，五个产物在 `goooinext` 与本地逐 byte 相同（sha256 一致） |

### 合成样本口径

- `score = edge_quality`；`edge_correct = edge_quality >= 0.5`；`cloud_correct = cloud_quality >= 0.5`（spec #48 §6）。
- 每个条目的 `edge_quality`/`cloud_quality` 是**固定值**，同一任务的所有变体共享同一分数，因此本轮每个任务只有 1–2 个不同 score（见下表）。
- `edge_cost`/`escalated_cost`/`cloud_only_cost` 由 `count_tokens` + `CostModel` 计算：`escalated_cost` = 端侧完整答案 token + 云端 prefill（prompt + 端侧答案）+ 云端 decode + call；`cloud_only_cost` = prompt prefill + decode + call。

| 条目 | 任务 | edge_quality | cloud_quality | val 条数 |
|---|---|---:|---:|---:|
| reflow-profile | process_qa | 0.94 | 0.98 | 30 |
| fault-e104 | fault_diagnosis | 0.91 | 0.97 | 30 |
| fault-e217 | fault_diagnosis | 0.34 | 0.95 | 30 |
| solder-void | quality_verdict | 0.90 | 0.97 | 30 |
| ipc-class3 | standard_qa | 0.42 | 0.96 | 30 |
| dppm-calc | process_calc | 0.30 | 0.98 | 30 |
| oee | improvement | 0.38 | 0.93 | 30 |
| estop-reset | safety | 0.93 | 0.98 | 30 |

### 判定口径

- 决策与 Router legacy 路径一致：`score >= threshold` 留在端侧，`score < threshold` 整段升级；`require_cloud_tasks` 永远升级，不参与阈值拟合，也不计入 per-task 升级率上限。
- 目标：`objective = accuracy − λ · cost_norm`，`cost_norm = mean(chosen_cost) / mean(cloud_only_cost)`；目标相同时取更低阈值（更少升级）。
- Baseline：在**同一 train split、同一 λ、同一阈值网格**上拟合的全局阈值；per-task 只有赢过它才算收益，不能只和“当前默认 0.62”比。
- Go 条件（真实 replay 才评估）：同准确率成本 ≤ baseline × 0.8，或同成本准确率 ≥ baseline + 1 pp；且无任务 val 升级率 > 0.6、被拟合任务 val 样本 ≥30、val 方向不反转。
- synthetic 模式固定输出 `pipeline_only`，不输出 `go`/`no_go`；`claims_gain` 永远为 `false`。

## 命令

```bash
# 远端临时目录（不覆盖 ~/Code/shadow-peft 的未提交改动）
ssh -J gooo goooinext 'mkdir -p /home/goooice/p16-run/src /home/goooice/p16-run/scripts'
rsync -az --exclude '__pycache__' -e 'ssh -J gooo' \
  src/edge_cloud/ goooinext:/home/goooice/p16-run/src/edge_cloud/
rsync -az -e 'ssh -J gooo' \
  scripts/exp15_cost_policy.py goooinext:/home/goooice/p16-run/scripts/

# 评测（goooinext；实测约 2 秒）
ssh -J gooo goooinext 'cd /home/goooice/Code/shadow-peft && \
  PYTHONPATH=/home/goooice/p16-run/src .venv/bin/python \
  /home/goooice/p16-run/scripts/exp15_cost_policy.py \
  --synthetic --out /home/goooice/p16-run/exp15 --bootstrap 1000'

# 回传
rsync -az -e 'ssh -J gooo' goooinext:/home/goooice/p16-run/exp15/ runs-local/exp15/

# 本地单测与静态检查
PYTHONPATH=src /Users/devel0per/Code/shadow-peft/.venv/bin/python \
  -m pytest tests/edge_cloud/test_exp15_cost_policy.py -q
/Users/devel0per/Code/shadow-peft/.venv/bin/ruff check scripts/exp15_cost_policy.py
```

本次 CLI 输出摘要：

```json
{
  "dataset_source": "synthetic",
  "smoke": false,
  "counts": {"total": 480, "train": 240, "val": 240},
  "selected_lambda": 0.1,
  "claims_gain": false,
  "evidence": "insufficient",
  "go_no_go": "pipeline_only"
}
```

## 结果

### 拟合出的策略（λ=0.1）

`policy.json` 可直接粘贴到 `EdgeConfig`（脚本已用 `EdgeConfig.from_dict` 校验）：

```json
{
  "escalation_threshold": 0.45,
  "escalation_threshold_by_task": {
    "fault_diagnosis": 0.35,
    "improvement": 0.4,
    "process_calc": 0.35,
    "process_qa": 0.0,
    "safety": 0.0,
    "standard_qa": 0.45
  }
}
```

`quality_verdict` 属于 `require_cloud_tasks`，不产生阈值，永远升级。

### val 指标：per-task 策略 vs 全局 baseline（λ=0.1）

| 臂 | n | accuracy | cost/请求 | cost_norm | 升级率 | cloud calls/例 | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-task 策略 | 240 | 1.000 | 0.001172 | 0.6937 | 0.625 | 0.625 | 2955.2 |
| 全局 baseline（阈值 0.45） | 240 | 1.000 | 0.001172 | 0.6937 | 0.625 | 0.625 | 2955.2 |

两臂逐项相同——不是拟合失败，而是合成分数退化的必然结果：每个任务的分数是固定值，任何落在“低分组最高分 0.42”和“高分组最低分 0.90”之间的阈值都给出同一组决策；全局 0.45 与各 per-task 阈值（0.0/0.35/0.4/0.45）都落在这个空带里。

### per-task val 明细（λ=0.1）

| task | n | 阈值 | accuracy | cost/请求 | cost_norm | 升级率 | cloud calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| fault_diagnosis | 60 | 0.35 | 1.000 | 0.000918 | 0.5319 | 0.500 | 30 |
| improvement | 30 | 0.40 | 1.000 | 0.002418 | 1.0605 | 1.000 | 30 |
| process_calc | 30 | 0.35 | 1.000 | 0.001416 | 1.0306 | 1.000 | 30 |
| process_qa | 30 | 0.00 | 1.000 | 0.000064 | 0.0408 | 0.000 | 0 |
| quality_verdict | 30 | 永远升级 | 1.000 | 0.001580 | 1.1238 | 1.000 | 30 |
| safety | 30 | 0.00 | 1.000 | 0.000076 | 0.0472 | 0.000 | 0 |
| standard_qa | 30 | 0.45 | 1.000 | 0.001982 | 1.0890 | 1.000 | 30 |

`improvement`、`process_calc`、`standard_qa` 的 val 升级率都是 1.0，超过 0.6 的 per-task 上限——这是合成数据“单条目 = 单一分数 = 全升或全不升”造成的；在真实数据上，这三项会直接让 go/no-go 失败。

### 参照点（val，λ=0.1）

| 参照 | accuracy | 95% CI | cost/请求 | cost_norm | 升级率 | P95 ms |
|---|---:|---|---:|---:|---:|---:|
| edge-only | 0.500 | [0.442, 0.558] | 0.000053 | 0.0311 | 0.000 | 456.0 |
| cloud-only（整段升级） | 1.000 | [1.000, 1.000] | 0.001846 | 1.0933 | 1.000 | 2955.2 |
| 当前默认 0.62 | 1.000 | [1.000, 1.000] | 0.001172 | 0.6937 | 0.625 | 2955.2 |
| per-task / 全局 0.45 | 1.000 | [1.000, 1.000] | 0.001172 | 0.6937 | 0.625 | 2955.2 |

- `edge-only` 只有 4 个高质量条目（120/240）答对，准确率 0.5；成本几乎可以忽略。
- 本实现的 `cloud_only` 参照是“每个请求都整段升级”，走 `escalated_cost`（含端侧完整答案 token），不是直连云端成本；因此它的 `cost_norm=1.0933 > 1`。真正的直连云端成本是 `cloud_only_cost`，即 `cost_norm` 的分母。
- `current_0.62` 与全局 0.45 在本数据上决策完全相同，原因同上：分数只落在 0.30–0.42 与 0.90–0.94 两簇。

### λ 扫描（val）

| λ | per-task acc | baseline acc | per-task cost | baseline cost | cost 比 | per-task 升级率 | baseline 升级率 | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 1.000 | 1.000 | 0.001172 | 0.001172 | 1.000 | 0.625 | 0.625 | pipeline_only |
| 0.05 | 1.000 | 1.000 | 0.001172 | 0.001172 | 1.000 | 0.625 | 0.625 | pipeline_only |
| 0.10 | 1.000 | 1.000 | 0.001172 | 0.001172 | 1.000 | 0.625 | 0.625 | pipeline_only |
| 0.20 | 1.000 | 1.000 | 0.001172 | 0.001172 | 1.000 | 0.625 | 0.625 | pipeline_only |
| 0.50 | 1.000 | 1.000 | 0.001172 | 0.001172 | 1.000 | 0.625 | 0.625 | pipeline_only |
| 1.00 | 0.500 | 0.750 | 0.000243 | 0.000634 | 0.383 | 0.125 | 0.375 | pipeline_only |

λ≤0.5 时 per-task 与全局 baseline 完全一致；只有 λ=1.0 把成本压到主导项：所有 per-task 阈值塌到 0.0，成本降到 baseline 的 38%，但准确率从 0.75 掉到 0.50。这正好演示了目标函数的方向，也说明合成数据无法区分“per-task 调参”和“全局调参”。

### 证据门禁

| 条件 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 数据来源 | replay | synthetic | 不通过 |
| 总样本 | ≥500 | 480 | 不通过 |
| 每任务 train/val | ≥30 / ≥30 | 每个被拟合任务 30/30 | 通过（但被数据来源否决） |
| `claims_gain` | — | `false` | — |
| `evidence` | — | `insufficient` | — |
| `go_no_go` | — | `pipeline_only` | — |

报告 `evidence.reasons` 只有一条：`dataset_source=synthetic：合成知识库只用于管线验证，不能作为收益证据`。由于 synthetic 分支不进入 replay 检查，480 < 500 没有单独出现在 reasons 里，但 `requirements.total_samples.passed=false` 已记录。

## Caveat（读数字前必看）

1. **合成分数重复，阈值不可辨识**：`score=edge_quality` 是条目常量，同一任务 30–60 条 val 只有 1–2 个不同分数。任何阈值落在分数空带内都等价，所以 per-task 与全局 baseline 在 λ≤0.5 逐项相同；这些数字不能用来比较策略优劣。
2. **不是真实准确率**：`edge_quality`/`cloud_quality` 是知识库的人工设定值，`edge_correct`/`cloud_correct` 只是 `>=0.5` 的阈值化，不代表模型在真实 prompt 上的表现。
3. **Counterfactual 与选择偏差**：真实日志通常只有被升级请求的云端答案；replay 的 `cloud_correct` 依赖导出方补齐反事实标签，没有随机探索时不能把 replay 收益当作无偏因果收益，偏差方向取决于导出样本。
4. **样本量门槛**：真实 replay 至少 500 条，且每个被拟合任务的 train/val 各 ≥30 条，才允许 `claims_gain=true`；否则只输出 `evidence_insufficient`。小 replay fixture（<500 条）已有单测覆盖。
5. **`require_cloud_tasks` 豁免**：`quality_verdict` 永远升级，不受 per-task 阈值影响，也不计入 0.6 的升级率上限；本报告的总体升级率 0.625 包含它的 30/240。
6. **`cloud_only` 参照不是直连云端**：它是 always-escalate 的整段升级成本，含端侧答案 token；看成本端点时要区分 `escalated_cost` 与 `cloud_only_cost`。
7. **价格是配置值**：`CostModel` 的单价不是云厂商账单；判断时同时看 `cost_norm` 与原始 token/call 数。
8. **没有生产默认值变更**：`policy.json` 只是审核产物；仓库的 `EdgeConfig` 默认 `escalation_threshold=0.62`、`escalation_threshold_by_task={}` 不变，脚本不写 Router 状态、不联网、不自动加载策略。

## 复现与产物

```bash
# 产物（runs-local/ 已被 git 忽略）
runs-local/exp15/policy.json     # 可直接粘贴到 EdgeConfig 的配置片段
runs-local/exp15/report.json     # 完整报告：λ 扫描、per-task、baseline、参照点、caveats
runs-local/exp15/summary.csv     # 99 行：每 λ 的 per_task/global_baseline 汇总 + per-task 明细 + 参照点
runs-local/exp15/curves.csv      # 147 行：全局与每任务的阈值曲线（风险-覆盖/成本-准确率同源）
runs-local/exp15/dataset.jsonl   # 480 条 CostSample（含 split、counterfactual 标签与成本）
```

同 seed、同输入重复运行五个文件 byte-identical；换机重跑时以 `report.json`/`policy.json` 的 sha256 对齐。

## 后续（需要真实数据）

- 从真实日志导出 replay：`task/score/edge_correct/cloud_correct/edge_cost/escalated_cost/cloud_only_cost/split`，并记录 token 数来源；没有 `cloud_correct` 的行不能进训练。
- 先用小 fixture 验证 `claims_gain=false`，再用 ≥500 条、每任务 ≥30 条的数据跑 `--dataset <jsonl>`；只有门禁全过才把 `policy.json` 交给人工审核。
- 上线仍由人工修改 `EdgeConfig`；回滚只需清空 `escalation_threshold_by_task`。

## 相关文档

- Spec: [docs/specs/cost-aware-task-thresholds.md](specs/cost-aware-task-thresholds.md)（issue #48 原文）
- 评测台: [docs/exp9-handoff-eval.md](exp9-handoff-eval.md)
- 端云设计: [docs/edge-cloud-system.md](edge-cloud-system.md)
