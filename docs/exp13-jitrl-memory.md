# exp13 JitRL-lite 记忆重排评测（spec #5 / issue #40）

一句话收束：**在 48 条确定性故障码改写集上，用合成 n-best 后端跑了 `memory_mode=off` 与 `on` 两臂：结构化准确率 50.00% → 75.00%（+25.00 pp），false-bias 0/24（0.00%），额外记忆延迟 P95 0.47 ms，云端调用 6 → 3；首轮验收口径全部通过，结论 go（仅代表合成集与机制上限，不代表真实产线收益）。**

状态：**本机纯 CPU mock 评测已跑通（2026-10-21）**；真实模型、真实反馈分布与 2k 级记忆规模未测。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | 本机 Mac（Darwin 27.0.0 arm64，macOS 27.0.1），纯 CPU |
| 代码 | 分支 `ticket/40-jitrl-eval`，基于 `feat/jitrl-memory` @ `8d30a2f` |
| Python | 3.12.10 |
| 数据 | `edge_cloud.knowledge.ENTRIES` 的 8 个条目 × 6 种问法 = 48 条；每条目 3 条 `line-a`（有历史记忆）+ 3 条 `line-b`（无记忆） |
| 记忆 | 72 条 corrected 记录（每个记忆用例 3 条：原问法、重复出现、同义改写），`return_value=+1.0`，候选为 cloud answer，tenant=`line-a` |
| 后端 | `SyntheticNBestBackend`，n=4：`edge_answer` / `cloud_answer`（记忆目标）/ `wrong_answer` / 通用兜底，全部带 length-normalized logprob；不加载真实模型 |
| 云端 | `SyntheticCloud` 按 prompt 返回 `cloud_answer`，confidence 0.96 |
| 配置 | `alpha=0.5`、`threshold=0.45`、`top_k=4`、`min_support=2`、`adapter_version=base`、TTL 30 天、升级阈值 0.62 |
| seed | 13 |
| 数据指纹 | `sha256:2d0467491d6021dddd49961f0372ce06020b6b266322394b04fc283b90db8853` |
| 记忆指纹 | `sha256:2b067343ae30522e7ef1460431d51da635584473b30e607bb43e2772ee52794d` |

### 数据集与判定口径

- 问法由 `knowledge.ENTRIES` 的关键词模板生成（同义词、语序、补充上下文），固定 seed 洗牌；每个条目 6 种问法，前半 `line-a`、后半 `line-b`，因此记忆/无记忆各 24 条。
- 历史记忆只属于 `line-a`，`line-b` 在检索阶段被 tenant 过滤为空；同一 prompt 的 3 条 corrected 记录保证 `min_support=2` 可满足。
- 结构化正确性：
  - `quality_verdict`：复用语义归一化（`normalize_answer`，`返修 → rework`）；
  - `process_calc`：数值 + 1% 相对容差（`3000` 判错、`30000` 判对）；
  - 自由文本：必须覆盖该条目的关键事实（如 E-217 的「冷却液」「流量」、IPC Class 3 的「10%」「X-Ray」）。
- easy 条目的 edge answer 满足 gold；hard 条目的 edge answer 不满足 gold，cloud answer 满足 gold。

### 合成后端的置信度设计

- 正常 edge 候选：logprob=log(0.64)，confidence≈0.68，端侧直接返回。
- 低置信度用例（每个 easy 且非 verdict 条目的每个 tenant 各 1 条）：edge 候选 `finish_reason="length"`，confidence≈0.68×0.85≈0.58 < 0.62，基线升级云端。
- cloud 候选：logprob=log(0.62)，confidence≈0.665 > 0.62；被记忆重排选中后可本地返回，不再升级。
- wrong / 通用兜底候选 logprob 更低，不会被选中。

## 命令

```bash
# 本机本次使用的解释器是仓库 venv；在 GB10 上换成 .venv/bin/python 即可。
PYTHONPATH=src /Users/devel0per/Code/shadow-peft/.venv/bin/python \
  scripts/exp13_jitrl_memory.py run --out runs-local/exp13 --seed 13

PYTHONPATH=src /Users/devel0per/Code/shadow-peft/.venv/bin/python \
  -m pytest tests/edge_cloud/test_jitrl_eval.py -q
/Users/devel0per/Code/shadow-peft/.venv/bin/ruff check \
  scripts/exp13_jitrl_memory.py tests/edge_cloud/test_jitrl_eval.py
```

本次结果：`23 passed`；ruff `All checks passed!`；脚本 `--help` 与 `run --help` 均正常。

产物：`runs-local/exp13/{report.json,dataset.jsonl,memory.jsonl,outcomes_no_memory.jsonl,outcomes_rerank.jsonl,summary.md}`（`runs-local/` 已被 git 忽略）。

## 结果

### 两臂对照

| 臂 | 结构化准确率 | 正确/总数 | 云端调用 | 命中且正确 | 命中但错 | 未命中 | 重排写回 |
|---|---|---|---|---|---|---|---|
| no-memory（`memory_mode=off`） | 50.00% | 24/48 | 6 | 0 | 0 | 48 | 0 |
| rerank（`memory_mode=on`） | **75.00%** | **36/48** | **3** | 24 | 0 | 24 | 21 |

- 基线 24 条正确全部来自 4 个 easy 条目（每条 6 问法）；24 条错误全部来自 4 个 hard 条目（每条 6 问法），即端侧「自信地答错」且不升级。
- rerank 臂把 `line-a` 的 12 条 hard 用例（4 hard 条目 × 3 问法）改为 cloud answer，全部正确；`line-a` 的 24 条全部命中记忆，其中 21 条发生文本重排。
- 未重排的 3 条是 `solder-void` verdict 用例：edge 与 cloud 文本不同，但 verdict 归一化后都是 `rework`，`rerank` 保持 top-1（`top1_unchanged`），最终答案仍正确。
- `line-b` 的 24 条检索为空，走单答案基线；命中率 24/48 = 50.00%。

### 验收口径

| 口径 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 结构化准确率 | +3 pp | **+25.00 pp**（50.00% → 75.00%） | 通过 |
| false-bias rate | <2% | **0/24 = 0.00%**（分母为基线正确用例） | 通过 |
| 额外延迟 P95 | <50 ms | **0.47 ms**（P50 0.024 ms，avg 0.180 ms，max 0.496 ms，n=48） | 通过 |
| 云端调用数 | 不上升 | **6 → 3**（减少 3） | 通过 |
| **总判定** | — | — | **go** |

- 延迟来自 Router 的 `memory_latency_ms`：24 次命中包含检索+重排，24 次未命中只包含检索；本机 CPU、72 条记忆，不能外推到 2k 条/产线硬件。
- 云端调用减少来自 3 条 `line-a` 低置信度 easy 用例：基线因截断升级云端，记忆重排选中完整 cloud 候选后本地返回；`line-b` 的 3 条低置信度用例仍然升级，因此调用数没有上升。

### false-bias 护栏探针

主数据集全部是 corrected 记忆，因此 false-bias 为 0 是构造使然。为验证护栏本身可运行，报告额外跑了 4 个独立探针：

| 探针 | 记忆指向 | 期望 | 实测 reason | 阻止改写 |
|---|---|---|---|---|
| `insufficient_support` | 错误答案 ×1 | 支持数不足 | `insufficient_support` | 是 |
| `negative_returns` | 错误答案 ×2（回报 -1） | 平均回报不足 | `low_return` | 是 |
| `top1_supported` | 正确答案 ×2（正对照） | 保持原 top-1 | `top1_unchanged` | 是 |
| `unsupported_answer` | 无关答案 ×2 | 无候选匹配 | `no_support` | 是 |

结果：`4/4 blocked`，其中误导性探针 `3/3 blocked`。这证明「支持数 <2 / 负回报 / 无匹配」的改写会被拦截；**不能**阻止 2 条以上一致的错误正回报记忆，后者只能靠 false-bias 监控、TTL 与 adapter 版本控制。

## 关闭记忆的回滚

- `memory_mode` 默认就是 `off`；把配置切回 `off`（或删除 `memory_path`）后，Router 每个请求重新读取该字段，走原来的单答案 `generate` 路径，不检索、不生成 n-best、不重排。
- 本次 no-memory 臂就是该回滚路径：48 条全部无记忆字段、云端调用 6 次，与 baseline 单答案行为一致；`memory_mode=off` 下注入的 memory 完全不参与。
- 记忆不修改模型权重、不写 adapter、不改变 shadow/GKD 流程；`memory.jsonl` 是本地文件，删掉或改名只会让记忆不可用（Router 静默回退并计入 `memory_fallback_total`）。
- 需要按请求兜底时，可把 `memory_latency_budget_ms` 设为 0：检索后超预算即按 `no_memory` 返回，行为与基线一致。

## 已知限制

- **合成后端，不是模型结论**：候选、logprob、云端答案都是脚本构造的；+25 pp 是机制上限，不能当作真实产线准确率或 JitRL 的实证收益。
- **记忆问法偏近重复**：每个记忆用例的 3 条历史记录包含原问法、重复出现和同义改写，检索相似度容易过 0.45 阈值；真实改写的距离更远，命中率可能明显更低。
- **规模小、无置信区间**：8 个条目、48 条用例，只够验证管线与方向，不足以做显著性检验。
- **false-bias 主指标为 0 是构造使然**：数据集只有 corrected 记忆；护栏用独立探针验证，不是主数据集上的自然发生率。
- **延迟只测了本机 CPU/72 条记忆**：未测 2k 条、并发、生产硬件；P95 0.47 ms 仅说明当前实现没有毫秒级开销。
- **云端是假的**：云端调用数衡量 Router 决策，不代表真实费用/时延；`quality_verdict` 的 `require_cloud_tasks` 在评测配置中被置空，生产默认仍强制升级。
- **只测候选重排**：未做 first-token bias、logit processor、动作空间 RL，也不替代 shadow 训练/GKD 飞轮。
- **tenant/TTL/adapter 只做了对半隔离**：`line-a`/`line-b` 验证租户隔离与无记忆路径；TTL 过期、adapter 版本失配由 `tests/edge_cloud/test_memory.py` 覆盖，本次未在端到端评测中变化。
- 未在 `gooo`/`goooinext` 上重跑：本次不需要 GPU/真实模型；若后续接真实 0.6B/3B 后端，应按 GB10 双机流程另跑。

## 产物与复现

- 脚本：`scripts/exp13_jitrl_memory.py`（只依赖仓库现有 `edge_cloud`，无新依赖）。
- 测试：`tests/edge_cloud/test_jitrl_eval.py`（数据集确定性、检索隔离、两臂差异、三类命中、false-bias 护栏、延迟统计、报告字段、CLI 产物）。
- 运行产物：`runs-local/exp13/`；`report.json` 含数据/记忆指纹、两臂逐用例结果、对比、验收与护栏探针，`summary.md` 为同口径简表。
- 复现时固定 `--seed 13`；数据/记忆指纹应与上表一致，延迟数字会随机器波动。
