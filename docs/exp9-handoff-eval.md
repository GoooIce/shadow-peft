# exp9 token/span 级端云交接评测结题档案

一句话收束：**在 goooinext 的合成/mock 评测台上，修复后的 probe 策略在准确率与 legacy 同为 1.000 的前提下，把云端 decode token 从 63,211 降到 49,633（−21.5%）、云端总 token 从 108,680 降到 81,952（−24.6%）、总成本从 1.397942 降到 1.156106（−17.3%）、端到端 P95 从 3,812.0ms 降到 2,492.8ms（−34.6%）；但 decode 降幅未达到首轮验收的 ≥30%，且真实 transformers/openai 后端与甲方标注评测集尚未运行。**

状态：**合成/mock 轮已结题（2026-10-20）**；真实后端轮待甲方评测集，不伪造结论。真实 Qwen3-0.6B 流式交接与 `CloudService` 续写的机制冒烟见 [`docs/exp9b-real-backend-smoke.md`](exp9b-real-backend-smoke.md)。

## 实验设置

| 项 | 值 |
|---|---|
| 日期 | 2026-10-20 |
| 机器 | `goooinext`（`aitopatom-dc4a`，192.168.1.177，GB10 aarch64，Linux 6.17.0-1032-nvidia，20 逻辑核） |
| 访问方式 | Mac 不能直连，统一 `ssh -J gooo goooinext` |
| Python | 3.12.3（远端仓库 `.venv/bin/python`） |
| 代码版本 | 集成分支 tip `478ddc4`（含 probe 方向修复 `44d9790`） |
| 代码同步 | 远端 `~/Code/shadow-peft` 是旧 git checkout，本轮不依赖其 commit；以 rsync 同步 `src/`、`scripts/`、`tests/` 后运行，关键文件 SHA-256 与本地 tip 一致 |
| 后端 | `synthetic`：`MockStreamingEdgeBackend` 逐 token 流式端侧 + 数据集脚本化的云端答案；不调用 GPU、不访问网络 |
| 数据 | synthetic，n=1000，seed=7；300 条 easy（30%）+ 700 条 hard（70%） |
| 采样参数 | legacy 阈值 0.62，bootstrap 1,000 次，seed=7 |
| probe | `probe_fixed.json`，fingerprint 见 `runs-local/exp9/goooinext-synthetic/probe_fixed.json` 的 `fingerprint` 字段；#16 增加 `training_fingerprint` 后需在 goooinext 重新 fit 生成 |

成本模型与延迟模型（评测台固定值，来自 `results.json` 的 `config`）：

| 参数 | 值 |
|---|---|
| `cost_per_edge_token` | 0.000002 |
| `cost_per_cloud_input_token` | 0.000004 |
| `cost_per_cloud_output_token` | 0.000012 |
| `cost_per_cloud_call` | 0.0005 |
| `cloud_rtt_ms` | 120.0 |
| 端侧延迟 | 12ms/token |
| 云端 prefill | 0.8ms/token |
| 云端 decode | 18ms/token |

评测台的延迟是 token 数的闭式函数，不是墙钟测量；成本是单价乘以 token 数，不是真实账单。

## 代码同步与三条运行命令

同步（每条命令独立执行，不带 `--delete`，排除 `__pycache__`）：

```bash
rsync -az -e 'ssh -J gooo' --exclude __pycache__ src/ goooinext:~/Code/shadow-peft/src/
rsync -az -e 'ssh -J gooo' --exclude __pycache__ scripts/ goooinext:~/Code/shadow-peft/scripts/
rsync -az -e 'ssh -J gooo' --exclude __pycache__ tests/ goooinext:~/Code/shadow-peft/tests/
```

在 `goooinext:~/Code/shadow-peft` 上依次执行：

```bash
# 1) 四策略评测，并导出 probe 训练候选
PYTHONPATH=src .venv/bin/python scripts/exp9_handoff_eval.py \
  --backend synthetic --n 1000 --seed 7 --bootstrap 1000 \
  --out runs/exp9/goooinext-synthetic \
  --export-candidates runs/exp9/goooinext-synthetic/candidates.jsonl

# 2) 离线拟合 probe（候选 label 1 = should_handoff）
PYTHONPATH=src .venv/bin/python scripts/exp9_fit_probe.py \
  runs/exp9/goooinext-synthetic/candidates.jsonl \
  --out runs/exp9/goooinext-synthetic/probe_fixed.json --threshold 0.5

# 3) 带 probe 复跑四策略
PYTHONPATH=src .venv/bin/python scripts/exp9_handoff_eval.py \
  --backend synthetic --n 1000 --seed 7 --bootstrap 1000 \
  --probe runs/exp9/goooinext-synthetic/probe_fixed.json \
  --out runs/exp9/goooinext-synthetic/probe_run_fixed
```

产物同步回本地：

```bash
rsync -az -e 'ssh -J gooo' \
  goooinext:~/Code/shadow-peft/runs/exp9/goooinext-synthetic/ \
  runs-local/exp9/goooinext-synthetic/
```

## 数据集与候选

- n=1000，seed=7，dataset digest `54412d53544ea0717b098518242cab41addc4d836734a2ab00fc14e3c48bf93e`。
- easy 300 条：端侧答案等于完整云端答案，无风险点、无交接标签。
- hard 700 条：端侧答案是云端答案的边界截断前缀，`risk_at` 在截断点之前，`cloud_better_from == risk_at`；gold 交接位置均值 9.16 token（中位数 6.5，范围 0–45）。
- 任务分布：`fault_diagnosis` 250 条；`process_qa`、`quality_verdict`、`standard_qa`、`process_calc`、`improvement`、`safety` 各 125 条。
- 候选 `candidates.jsonl`：53,767 行，覆盖 1,000 个 prompt 的每个已观测前缀；正例 20,314（37.78%）。正例含义是「该前缀的 token 位置 ≥ `cloud_better_from`，应当交接」，**不是** 70% 的请求升级率；正例率的分母是全部前缀位置，两者不可混用。

## 四策略结果（单一事实源 `probe_run_fixed/results.json`）

| 策略 | acc / exact | 升级率 | handoff 率 | 交接位置 mean / median / p95 | 云端 prefill | 云端 decode | 云端调用 | 总成本 | 每请求成本 | P50 / P95 延迟 | probe 回退率 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| edge_only | 0.300 / 0.300 | 0.000 | 0.000 | — | 0 | 0 | 0 | 0.107534 | 0.000108 | 648.0 / 1,212.0ms | 0.000 |
| legacy | 1.000 / 1.000 | 0.700 | 0.000 | — | 45,469 | 63,211 | 700 | 1.397942 | 0.001398 | 1,990.4 / 3,812.0ms | 0.000 |
| probe（修复后） | 1.000 / 1.000 | 0.700 | 0.700 | 19.397 / 18 / 42 | 32,319 | 49,633 | 700 | 1.156106 | 0.001156 | 1,476.8 / 2,492.8ms | 0.000 |
| oracle | 1.000 / 1.000 | 0.700 | 0.700 | 19.397 / 18 / 42 | 32,319 | 49,633 | 700 | 1.156106 | 0.001156 | 1,476.8 / 2,492.8ms | 0.000 |

probe 的 700 次交接全部走 `low_prefix_trust`；边界类型为 `sentence` 213 次、`token` 487 次（后者是 `max_boundary_wait_tokens=12` 用尽后的 token 边界强制切分）。没有 `newline`、`step`、`json_field` 交接。degraded 率与 continuation mismatch 率均为 0。

### 相对 legacy 的变化

| 指标 | legacy | probe | 变化 |
|---|---:|---:|---:|
| 准确率 / exact | 1.000 | 1.000 | 0.0 pp |
| 升级率 | 0.700 | 0.700 | 不变 |
| 云端 prefill token | 45,469 | 32,319 | −28.9% |
| 云端 decode token | 63,211 | 49,633 | **−21.5%** |
| 云端总 token | 108,680 | 81,952 | **−24.6%** |
| 总成本 | 1.397942 | 1.156106 | **−17.3%** |
| 每请求成本 | 0.001398 | 0.001156 | −17.3% |
| P50 延迟 | 1,990.4ms | 1,476.8ms | −25.8% |
| P95 延迟 | 3,812.0ms | 2,492.8ms | **−34.6%** |

## 曲线说明

`curves.csv` 与 `results.json["curves"]` 给出两种曲线，横轴与排序口径不同：

- **risk–coverage**：按评分从高到低排序，`coverage` 是累计覆盖比例，`risk` 是累计错误率。probe 的评分取最后一个前缀可信度，legacy 取 `ConfidenceEstimator`，oracle 取 `gold / edge_tokens`，edge_only 取端侧置信度。合成集上 probe、legacy、oracle 的最终答案全部正确，因此三条曲线在全部 coverage 上 risk 恒为 0；只有 edge_only 随 coverage 上升把 risk 推到 0.7。**这条曲线在本轮无法区分 probe 与 legacy**，只能确认 edge_only 的风险上界。
- **cost–accuracy**：扫描路由操作点，报告每请求平均成本与准确率。legacy 扫描 557 个置信度阈值，形成连续折线；从全端侧（0.300 准确率、0.000108/请求）到全云端（1.000 准确率、0.002013/请求），实际生效的拐点集中在约 0.09 与 0.93 两个置信度区间。oracle 扫描 11 个全局 cut 点。probe 只有 **2 个操作点**：阈值 0 时等价于全端侧（0.300 准确率、0.000081/请求），阈值 1 时 700 条 hard 全部交接（1.000 准确率、0.001156/请求）。原因是合成标签与特征完全可分，probe 分数饱和到 0/1，中间没有可扫的阈值。**这是本轮 cost–accuracy 的主要限制，不能据此声称 probe 的完整成本前沿。**

## Bootstrap 口径与结果

口径：对 1,000 个 prompt 的逐请求准确率（0/1）与逐请求成本做有放回重采样，每次 1,000 个样本，共 1,000 次，取 2.5%/97.5% 分位数，seed=7。它衡量的是这 1,000 条 prompt 的抽样噪声，不是真实后端或真实流量的方差。

| 策略 | 准确率点估计 [95% CI] | 每请求成本点估计 [95% CI] |
|---|---|---|
| edge_only | 0.300 [0.272, 0.328] | 0.000108 [0.000103, 0.000112] |
| legacy | 1.000 [1.000, 1.000] | 0.001398 [0.001347, 0.001453] |
| probe（修复后） | 1.000 [1.000, 1.000] | 0.001156 [0.001116, 0.001198] |
| oracle | 1.000 [1.000, 1.000] | 0.001156 [0.001116, 0.001198] |

legacy 与 probe 的成本区间不重叠（probe 上界 0.001198 < legacy 下界 0.001347），与 −17.3% 的点估计一致；准确率区间完全相同，因为两者都是 1,000/1,000 正确。

## oracle 与 probe 的差距

本轮 **probe 与 oracle 在所有报告指标上逐项相同**，差距为 0：准确率、升级率、交接率、交接位置、云端 token、成本、P95 全部一致。

原因不是 probe 具备了 oracle 的泛化能力，而是合成标签可由特征确定性地区分：`probe_fixed.json` 的 10 个校准分箱中，20,314 个正例候选全部落在 [0, 0.1)，平均分数 0.0；33,453 个负例候选全部落在 [0.9, 1.0]，平均分数 1.0。probe 只是把合成集的 `cloud_better_from` 规则学了回来。**因此 oracle 在这里只证明了评测台与 probe 管线的一致性，不能作为真实数据上 probe 上界的证据。**

另一个值得记录的差异是：gold 交接位置均值 9.16 token，而 probe 实际在 19.40 token 处交接，因为 probe 必须等到最近的语义边界（或等待预算用尽）。合成集上前缀始终是云端答案的真前缀，等到边界不会损失准确率；真实数据上这一步会增加「前缀锚定」风险。

## 发现并修复的 bug

- **现象**：第一版 `fit_probe` 把候选 `label=1 (should_handoff)` 直接当作 `prefix_trust` 拟合。`ProbePolicy` 的语义是「信任低于阈值才交接」，标签方向相反导致决策整体反转：所有本应继续的前缀得分接近 0，easy 样本也 100% 交接。
- **旧产物证据**（`probe.json` / `probe_run/`，仅用于记录 bug，不用于结论）：probe 升级率 1.000、handoff 率 0.933、云端 decode 80,590、总成本 1.640462，反而高于 legacy 的 1.397942。
- **修复**：提交 `44d9790`（merge `478ddc4`）让 `fit_probe` 对 `1 - label` 拟合，并修正单类基率与校准表语义；补了分数方向、fit→load、校准语义与 `ProbePolicy` 触发条件的回归测试。
- **修复后**：`probe_fixed.json`（fingerprint 见产物文件的 `fingerprint` 字段）+ `probe_run_fixed/results.json` 为本文所有 probe 结论的来源。

注意：顶层 `results.json` 是**未带 `--probe`** 的第一轮四策略产出，其中 `probe` 行因 `probe_path=null` 回退 legacy（`probe_fallback_rate=1.0`），不能当作 probe 结果；probe 结论只取 `probe_run_fixed/results.json`。

## 首轮验收口径逐条结论

对照 spec §Further Notes「建议的首轮验收口径」：

| # | 验收口径 | 结论 | 证据 |
|---|---|---|---|
| 1 | 同准确率下，云端输出 token 减少 ≥30%，端到端 P95 不劣化超过 10% | **未通过（token 目标）/ 通过（P95 目标）** | 准确率同为 1.000；云端 decode −21.5%、总 token −24.6%，未到 30%；P95 从 3,812.0ms 降到 2,492.8ms（−34.6%），远好于「不劣化 10%」 |
| 2 | 同成本下准确率提升 ≥2 pp；或同升级率下 AURC 相对提升 ≥5% | **未通过 / 未验证** | 同成本 +2 pp **未达成且在该合成集上不可能**：probe 与 legacy 准确率同为 1.000（已到上限）；同升级率 70% 下成本低 17.3% 只是成本差异，不是准确率提升。AURC 相对提升 ≥5% **未计算**：曲线虽已导出但没有 AURC 字段，需在真实标注集上补。因此不能写「部分通过」 |
| 3 | 单次交接约束在 100% 请求上成立 | **通过** | `tests/edge_cloud/test_router_handoff.py::test_single_handoff_calls_cloud_once_even_when_the_policy_keeps_triggering`；评测中 700 次 handoff 对应 700 次云端调用，无重复调用 |
| 4 | 离线模式下 100% 请求可本地返回并入队 | **通过（测试级证据）** | `test_router_handoff.py::test_offline_handoff_queues_partial_trace_and_replays_it`、`tests/edge_cloud/test_outbox_handoff.py` 的 partial trace/元数据 round-trip；本轮未在 goooinext 做断网演练 |
| 5 | 旧配置、旧客户端、无 shadow 后端三类兼容场景全部通过现有测试与新增回归 | **通过** | 本 worktree 全量 `pytest` 实测 1032 passed / 18 skipped（#16 修复后）；`test_handoff_config.py::test_legacy_config_defaults_to_legacy_handoff`、`test_handoff_protocol.py::test_cloud_request_parses_legacy_payload_without_handoff_fields`、`test_router_handoff.py::test_non_streaming_backend_falls_back_with_streaming_unavailable_metric` |
| — | 不得伪造真实后端结论 | **遵守** | 本轮只跑 `--backend synthetic`；真实 `transformers` / `openai` 后端未运行，缺甲方标注评测集，本文不给出真实模型数字 |

### 关于 token 目标的差距

云端 decode 的节省量约等于交接位置：probe 平均在 19.40 token 处交接，hard 样本的完整云端答案平均 90.30 token，因此 decode 节省比例约为 19.40 / 90.30 = 21.5%，与实测 −21.5% 一致。要达到 30%，平均交接位置需要后移到约 27.1 token（假设准确率不变）。可选方向：

1. **让交接点更靠后**：提高 `max_boundary_wait_tokens`，或只在句号/步骤边界交接，避免大量 token 边界强制切分（本轮 700 次中 487 次是 token 边界）；也可以在真实数据上把信任阈值调低，让触发不那么早。
2. **重新校准而不是调参**：合成集的 probe 分数饱和，阈值在 0–1 之间没有区分度；真实数据上必须用甲方标注集重训 probe，并按任务分别校准阈值。
3. **权衡前缀锚定风险**：交接越晚，端侧已生成的前缀越长，错误前缀被云端续写放大的风险越高；spec 的经济学条件也明确「不是越早越好」，真实数据上要用 cost–accuracy 与 risk–coverage 联合选点，不能只追 token 降幅。

## 已知限制

- **真实后端未跑**：`transformers` 端侧后端与 OpenAI 兼容云端后端均未在真实模型上运行；云端答案由数据集脚本提供，准确率、token 与延迟都是闭式计算。真实模型的输出分布、logprob 质量、续写契约遵守情况都未验证。
- **缺甲方标注评测集**：合成集的标签是程序生成的，probe 与 oracle 完全重合，不能说明真实数据上的泛化、校准或 AURC。
- **合成集上 probe 与 oracle 无差距**：需要真实数据才能回答「probe 离上界还有多远」。
- **cost–accuracy 只有两个 probe 操作点**：分数饱和到 0/1，无法画出 probe 的连续成本前沿；risk–coverage 在 probe/legacy/oracle 上退化为 risk 恒 0。
- **没有 AURC**：本轮只导出 risk–coverage 曲线，未计算 AURC 或相对提升。
- **没有真实账单与网络抖动**：成本是配置单价乘以 token 数；RTT 固定 120ms，degraded 率为 0，未覆盖超时、熔断、断网和离线队列的真实演练。
- **无 shadow 后端只做了单元测试**：没有在真实 `shadow_ckpt` 上验证 `kl_base_shadow`、`top1_agreement` 等特征的数值行为。
- **shadow-only / joint `base_shadow` 指标缺失**：spec User story 11 要求同时报告 shadow-only、joint `base_shadow` 与交接后最终答案的指标。本轮 synthetic/mock 只有 base/shadow 分布差（`shadow_gap` 特征），没有独立的 shadow 答案，`edge_only` 行只是端侧 base 答案，**不能当作 shadow-only**；本文不给出 shadow-only 或 joint `base_shadow` 的准确率数字，也不把 `edge_only` 冒充 shadow-only。该指标需要真实 `shadow_ckpt` + `ShadowForCausalLM` 的 `shadow_only` 运行，当前 goooinext 评测未提供。代码现状：`TransformersBackend` 仅在 `shadow_ckpt` 存在时以 `inference_mode="base_shadow"` 加载（`src/edge_cloud/backends.py`），尚未暴露 `shadow_only` 推理入口；补该入口与真实评测应另立 ticket。
- **单 seed、单数据集**：没有跨 seed 或跨任务分布的方差估计；bootstrap 只反映 prompt 抽样噪声。
- **离线验收是测试级**：没有在 goooinext 上做断网/恢复的端到端演练。

## 复现步骤

1. 在集成 tip `478ddc4` 或其后代上确认 `git merge-base --is-ancestor feat/edgecloud-ci-token-handoff HEAD` 成功。
2. 按上面的 rsync 命令同步 `src/`、`scripts/`、`tests/` 到 `goooinext:~/Code/shadow-peft`。
3. 在远端依次执行「三条运行命令」。同一 seed、同一代码下，`results.json`、`summary.csv`、`curves.csv`、`per_prompt.jsonl` 应逐位一致；`results.json["config"]["probe_path"]` 会随输出目录不同而不同。
4. 断言 `probe_run_fixed/results.json` 的 `strategies.probe` 与本文表格一致：accuracy 1.000、escalation_rate 0.700、handoff_rate 0.700、cloud_prefill_tokens 32,319、cloud_decode_tokens 49,633、total_cost 1.156106、latency.p95 2,492.8。
5. 检查 `ci.probe.cost` 为 `[0.001116, 0.001198]`，`ci.probe.accuracy` 为 `[1.0, 1.0]`。
6. #16 之后 `fit_probe` 会写入 `training_fingerprint`，probe 产物的顶层 `fingerprint` 随之变化；复现本文数字前需重新执行第 2 步的 fit，并以新产物的 `fingerprint` 字段为准。`fit_probe` 的评分逻辑未变，四策略数字应保持一致；`per_prompt.jsonl` 因 #16 去掉与 `exact_match` 重复的 `correct` 字段而字段集略有变化（数值不变）。若数字不一致，先核对代码版本，不要修改本文数字。

复现时不要复用旧的 `probe.json` / `probe_run/`；它们是方向反转 bug 的现场，只作审计用途。

## 资产地图

| 资产 | 位置 |
|---|---|
| 远端运行目录 | `goooinext:~/Code/shadow-peft/runs/exp9/goooinext-synthetic/` |
| 本地归档 | `runs-local/exp9/goooinext-synthetic/`（主工作区） |
| 最终四策略结果 | `probe_run_fixed/results.json`、`probe_run_fixed/summary.csv`、`probe_run_fixed/curves.csv`、`probe_run_fixed/per_prompt.jsonl` |
| 第一轮四策略结果（无 probe） | `results.json`、`summary.csv`、`curves.csv`、`per_prompt.jsonl`（probe 行是 legacy 回退，不作结论） |
| probe 候选 | `candidates.jsonl`（53,767 行，正例 20,314） |
| 修复后 probe | `probe_fixed.json`（fingerprint 字段见文件；#16 后需在 goooinext 重新 fit 生成） |
| 旧 bug 现场（仅审计） | `probe.json`、`probe_run/` |
| 评测脚本 | `scripts/exp9_handoff_eval.py`、`scripts/exp9_fit_probe.py` |
| 评测台实现 | `src/edge_cloud/handoff_eval.py` |
| 相关单测 | `tests/edge_cloud/test_handoff_eval.py`、`test_router_handoff.py`、`test_outbox_handoff.py` |
| 系统设计 | `docs/edge-cloud-system.md` §5.7 |
| 规格 | `docs/specs/token-level-edge-cloud-handoff.md` |
