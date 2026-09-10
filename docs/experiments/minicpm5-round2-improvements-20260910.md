# MiniCPM5-2B NVFP4 W4A4 第 2 轮改进：逐代保留、终端安全交付、harness swap、噪声带与两项常设审计

- 上游：`14`/`15`（负结果与交付）、`18`（门与参数冻结）、`19`（未作答模式与健康度仪表）、`pools-v3`（评分口径）、`pools-v4`（新密封池 + 中心反馈切片）
- 执行日期：2026-09-10 18:40–19:0x；机器：本地 Mac（代码、单测、统计、文档）＋ **两台在跑机器只读 ssh**（§6 审计读台账，未起任何 GPU 作业）
- 工作树：`/tmp/shadow-peft-minicpm5-improve`，分支 `codex/wayfinder-minicpm5-improve`（从 `codex/wayfinder-minicpm5-sanity` 的 `27ae570` 起；该提交的父提交是 `main` 的 `86d337b`，故**包含 main 的全部历史**，并带上 19 票已入库的 `pilot_search.py` / `score_taskset.py` / `taskset_lib.py` 与 anomaly 笔记）
- 交付物：本笔记 + `docs/experiments/artifacts/minicpm5-round2-improvements-20260910/`（两条在跑运行的机制审计报告）+ `scripts/minicpm5/` 下 3 个新脚本、4 个测试文件、3 个改动文件
- 纪律：**未在 gooo / goooinext 上跑任何 GPU 作业**（不重跑、不重启、不动 `~/minicpm5-run300/` 内容）；未使用两机的 `~/Code/shadow-peft/.venv`；未同步 `.env` / `.quark_netdisk_auth.txt` / `.scratch/`；主工作区只读
- 未做：**没有写任何票的 `## Answer`**

## 0. 一句话结论

六处改动全部落地并单测通过（**176 项新检查 + 19 票原有 40 项回归全过**），全部**旗标门控**（除 §2 按指令默认开）：① 每代保留/回滚（含票 09 §4 的残差重锚，回滚后中心身份逐位复原，单测含**反例**）；② 终端安全交付（中心仅当严格优于 q0，否则交付 q0 并记 `no_gain_delivered_baseline`）；③ harness swap（两种 prompt 形态、四个臂，正收益主张需在 swap 下不消失）；④ 反馈噪声带（**本次运行自己测出来**，不是记忆常数）＋中心反馈切片监控通道；⑤ 输出模式扫描（把 19 票的偶然发现变成常设检查，逐 family 报退化率）；⑥ 死机制审计（已在**两条在跑运行**上跑出结果：KL 趋势子句**再次**是 report-only，0↔8 三口径全空）。同时**实测出一个与 brief 假设冲突的结果**：预注册阈值挂"同一中心重测的标准误"时，dev 池上中心逐代逐 family 的自然摆动是 2–9 题，而该标准误为 0 → 带 = 量化下限（1 题）→ **实测的 5/5 个相邻代都会回滚**；因此新增 `--retention-band-term` 把"挂哪个实测标准误"变成显式、可审计的选择（默认仍是 brief 的字面读法）。

## 1. 三篇论文 → 六处改动（映射）

| 论文结果 | 我们的对应弱点 | 本轮改动 |
| --- | --- | --- |
| HarnessDev：同 commit 重评估噪声 ±4.75；只有"读失败→定向修复→复验"（ρ=0.57）与下游分数相关 | 我们只在**每代开头**读一次反馈通道，步后从不复验；中心可以带着一个坏步走到下一轮 | **§1 每代保留**（`--retention`） |
| Aspire：只有 1/12 的 final-only 运行胜基线；"仅当严格优于基线才保留，否则回滚" | 上一轮中心在两条互斥封存池上均劣于 q0（−3.89 / −3.25 点），**仍被交付** | **§2 终端安全交付**（`--retention-terminal`，默认开） |
| HarnessDev：换掉固定 harness 背后的执行器，一个系统 69.3 → 33.0；harness 与执行器共适应 | 冻结 prompt 的 `Answer:` 后缀**已被证明**诱发"闭嘴"模式（19 票：去掉 9 个字符后 65.4% → 0.0%），只在该形态下量到的增益可能不迁移 | **§3 harness swap**（`--prompt-variant` / `--terminal-swap`） |
| 三篇共有：把阈值挂在**本次运行实测**的量上，而不是文献常数 | 我们的 δ=2 点小于 dev 自身分辨率 6.7 点；阈值一直用记忆常数 | **§4a 反馈噪声带**（`--feedback-noise` / `--noop-rescores` / `--retention-band-term`）；**§4b 中心反馈切片**（`--feedback-slice`） |
| Aspire：数值标签坍缩；HarnessDev：77.8% 缺陷归因于 harness | 我们的实例是**偶然**发现的；没有常设检查 | **§5 输出模式扫描**（`harness_audit.py`） |
| HarnessDev：124 个死特征、18 个从未触发的组件 | 我们的 KL 趋势守卫是 report-only，代价是上一轮被砍到 16 代 | **§6 死机制审计**（`mechanism_audit.py`，可只读跑在跑运行） |

## 2. 逐项改动

### 2.1 §1 每代保留（定向修订循环）— `--retention`（默认**关**）

**规则（预注册，实现即字面）**：中心步算完并应用后，在**反馈通道（dev 池）**上重读新中心，相对**上一代中心**逐 family 分解；**若任一家族跌幅超过本次运行自己测出的噪声带**（§4a）→ **整步回滚**到上一代中心。

**回滚的算术（票 09 §4 的残差规则）**：`pilot_lib.snapshot_layers()` 在步前对**该步会访问的全部坐标**（`step_coordinates = support(gradient) ∪ support(residual)`）取逐坐标 `(code, residual)` 快照（含"层桶是否存在"这一身份信息），`restore_layers()` 做逆操作。被回滚的整步**对残差零贡献**：`u = α·ĝ + γ·e` 里新引入的 `α·ĝ` 与它重新施加的 `γ·e` 一起被重锚回步前值——否则未取的步会在下一代借 `γ·e` 返回（这正是票 09 §4 为"验收拒绝"冻结的那条约束）。

**判定与记账**：
- 比较用**整数题数**（带换算成"本桶几题"，跌幅必须**严格大于**它），避免浮点灰尘决定预注册规则；
- 回滚后 `state.generation` 仍前进（这一代被消费），但该代在停止策略里按 **no-update** 记账（`enforce_stop_policy` 里唯一的改动：`retention_rollback == True → moved = 0`）；
- 逐代记录 `retention` 块（`kind = retention_rollback|retention_keep`、两条通道的逐 family 证据、所用带、恢复账、重锚账、`sha_restored_exactly`）＋ `records/retention_genNN.json`；
- `history.jsonl` 增列：`retention_rollback` / `retention_breaches` / `retention_worst_family` / `retention_worst_delta` / `retention_band_source` / `retention_centre_band` / `centre_post_acc` / `centre_post_gold_prob`；
- **身份硬不变量**：回滚后 `state.sha256()` 必须**逐位等于**步前中心，否则写盘后 `raise`（不静默带病继续）。

**代价**：每代 +1 次 dev 评分（480 题 ≈1.5 s），可选切片 +1 次（§4b）。

### 2.2 §2 终端安全交付 — `--retention-terminal` / `--no-retention-terminal`（默认**开**，仅 `--confirm-only` 路径）

**规则（预注册）**：交付物 = 中心 **当且仅当**它在封存池上**严格优于** q0（episode 级配对均差 > 0）；否则交付物 = **q0 本身**，记录写 `no_gain_delivered_baseline`。判定基准与措辞权威不变（`confirm_gate.py` / `classify_outcome` 依旧产出措辞；本条只决定"交哪个"和"能不能主张"）。

**实现**：`records/terminal_retention.json`（规则、头条形态的两臂、逐形态对照、swap 门、`deliverable`、`record`、`claim.gain_claimable`、开箱语义说明）；`--terminal-export`（默认开）按裁决**导出真载荷**：中心 → `payload_terminal_center/`，q0 → `payload_terminal_q0/`（空中心的物化件，且写 sha256/字节数到 `terminal_payload.json`）。`--no-retention-terminal` 时**明确记一行日志**说明"中心无条件交付"，记录里保留该事实。

### 2.3 §3 harness swap — `--prompt-variant {frozen,plain}`（默认 `frozen`）＋ `--terminal-swap` / `--no-terminal-swap`（默认**开**）

**渲染（纯追加）**：`taskset_lib.render_prompt_variant()`：`frozen` **逐字节等同于** `render_prompt()`（单测断言），`plain` = 同一 chat template **不带** `Answer:` 后缀；`render_span()` 返回 `(prompt, p_ids, g_ids, stable)`。冻结形态保持严格（不稳定前缀照旧 `raise`）；swap 形态允许不稳定（`answer_span` 把 prompt 截到公共前缀、余下 gold 仍是合法续写）并逐臂报 `unstable_prefix` 数。

**测量**：终端在封存池上把 **q0 与中心在两种形态下都测一遍**（4 臂，同进程 = **同一次开箱**；`--looks` 语义不变，`confirm_gate.py --measure` 读的 `arms` 仍是头条形态，**工具逐字不用改**）。

**规则（预注册）**：**主张正收益要求增益在 swap 下不消失**（两种形态的 pairing 均差都 > 0）。判决写进 `terminal_retention.json` 与独立的 `prompt_swap.json`；只有 swap 臂跑不了（`--no-terminal-swap`）时，`claim.gain_claimable = false` 并说明"该记录不能产出正收益主张"。

### 2.4 §4a 反馈噪声带 — `--noop-rescores`（默认 3）/ `--retention-band-term`（默认 `harness`）/ `--retention-band-scale`（默认 1.0）/ `--feedback-noise`

**做法**：启动时把**同一个中心**在**完整管道**上测 k 遍（patch → KF refit → flush prefix cache → 评分），这就是 HarnessDev 的"同 commit 重评估噪声"在本运行、本精度下的实测值；另外跨代自动累积"同一中心身份的重复读数"（步后读数与下一代步前读数同 sha，回滚时更明显），`band_replicates()` 按确定性规则择一（启动重测优先，其次最大同 sha 组）。

**带 = `max(term_value × scale, 量化下限 1/n)`**，其中 `term_value` 由 `--retention-band-term` 选择：
- `harness`（**默认＝brief 的字面读法**）：同一中心重测的标准误；
- `sampling`：本池 episode 级（簇校正）标准误；
- `max`：两者较大者。

每个逐 family 的带都带 `band_items`（换算成"几题"），所以读写台账时不必再做浮点心算。噪声带写进 `summary.json` 的 `round2.feedback_noise` 与 `feedback_noise.json`（§4a 明确要求"写进 summary"），并且**每次保留判定都写清用了哪个 source/term**。

### 2.5 §4b 中心反馈切片 — `--feedback-slice <jsonl>`（默认**关**）/ `--feedback-slice-gate`（默认**关**）

每代对中心评一次切片（v4 的中心切片 2,040 题 ≈6 s），逐代写 `feedback_slice_acc/episodes` 进 `history.jsonl`，逐代证据进 `records/retention_genNN.json`。**默认是纯监控通道**：候选选择、验收门、停止判据**都不读它**。`--feedback-slice-gate` 才把它接进**保留判定**（切片越带也触发回滚），并在记录里写明"wired into the retention decision"；否则写"monitoring only"。切片的"上一代读数"按中心 sha 匹配，重启后的第一代没有可比读数时**显式跳过并在记录里说明**（不猜）。

> 与 `pools-v4` 的接口：切片的现成产物是 `docs/experiments/artifacts/minicpm5-pools-v4-20260910/taskset_centre_v4.jsonl`（2,040 题 / 255 episode，字段与评分协议同 v3）。它的逐 family 分辨率比 dev 好：每个 family 408 题、51 episode（dev 是 96 题、12 episode），按 σ≈0.185 估，簇校正 SE ≈2.5 点 vs dev 的 4.5–7 点——**这才是"低方差第二读数"该有的量级**。

### 2.6 §5 输出模式扫描 — `harness_audit.py`（常设检查）

一次 TF 探针（自录第一个 gold 位的 argmax 与 top-5）+ 一次贪心生成，逐 family/逐 stratum 报：`no_answer`（首位 argmax = `<|im_end|>` 或生成单 token）、`truncated`、`empty_output`、`non_numeric`（验证器要整数而没有）、`non_name`、`first_token_not_numeric`（Aspire 的数值标签坍缩，机械可测）、`multi_line` / `verbose_first_line` / `think_leak`（相对冻结指令的格式漂移），外加**机械缺陷归因**：整轴判错里有多少来自退化模式而非"答错"。支持 `--variant {frozen,plain}`（两种形态各扫一遍）、`--margin`（逐 family 退化率超过它就打 flag）。

**刻意不改 `score_taskset`**：探针自带 TF 实现，因为 19 票的 `tf_first_argmax` 只在 sanity 分支上，审计不应依赖它是否已合并。

### 2.7 §6 死机制审计 — `mechanism_audit.py`（可只读跑在跑运行）

对每条协议机制给四种状态之一：`alive` / `never_fired` / `fired_no_effect`（**开了、触发了、但下游什么都没变**——报告型守卫） / `not_wired`（这次运行根本没开），并附证据（触发代数、区间、逐代数字）。机制清单：硬门（值口径）、readback、KL 绝对阈值、KL 趋势（含材料性带）、α 阶梯**两个方向**、no-accept、残差 clamp、端点拒绝、未跨格计数、0↔8 三口径、q0 参照臂、未作答 flag、保留、反馈切片、噪声带、预算、停止策略、镜像对门、rank 形状奖励、梯度支持、中心步、终端臂、swap 门。

两种运行方式：`--run <dir>`（本地）；`--ssh HOST:RUN_DIR`（把**本脚本**拷到该机 `/tmp`，只读台账、只回传紧凑投影；**不写运行目录、不起 GPU、不读候选与内核大文件**）。

## 3. 新旗标与默认值

| 旗标 | 默认 | 作用 |
| --- | --- | --- |
| `--retention` | **关** | §1 每代保留（步后复验、越带回滚、残差重锚） |
| `--retention-band-term` | `harness` | 阈值挂哪个实测标准误（`harness`/`sampling`/`max`） |
| `--retention-band-scale` | `1.0` | 带的倍数（预注册值 1.0） |
| `--noop-rescores` | `3` | 启动时空转重测次数（噪声带来源；<2 则退回量化下限） |
| `--feedback-noise` | 关 | 只测/只记噪声带，不启用回滚 |
| `--feedback-slice <jsonl>` | 关 | §4b 中心切片（监控通道） |
| `--feedback-slice-gate` | 关 | 把切片接进保留判定 |
| `--prompt-variant {frozen,plain}` | `frozen` | 终端头条形态 |
| `--terminal-swap` / `--no-terminal-swap` | **开** | 终端四臂（q0/中心 × 两形态） |
| `--retention-terminal` / `--no-retention-terminal` | **开** | §2 交付物裁决 |
| `--terminal-export` / `--no-terminal-export` | **开** | 按裁决导出真载荷 |

其余一切（α=12、k=32、δ=0.02、τ=0.02、池子、评分语义、`aggregate`/`apply_step`/门槛/`confirm_gate.py`）**未改**。`score_taskset.score_gen` 只多了一个**默认值等于冻结值**的 `max_tokens=24` 参数（供审计扫档位），既有调用方行为逐字不变。

## 4. 单测证据（纯 CPU，本地 `.venv`）

| 文件 | 检查数 | 覆盖 |
| --- | ---: | --- |
| `scripts/minicpm5/test_round2_retention.py` | **94** | `step_coordinates` 与 `apply_step` 访问集一致；**回滚的残差重锚**（`to_dict` 逐字节、身份逐位、层桶存在性、**反例**：不重锚则下一代 `γ·e` 把未取的步带回来）；噪声带（harness 0 vs 有噪、量化下限、两种证据来源择一、三种 term）；保留判定（严格大于、整数题数边界、逐 family 单侧）；终端裁决（严格优于/平局/更差/swap 消失/无 swap 臂）；prompt 变体（frozen 逐字节不变、plain 无后缀、span 稳定性）；Runner 回滚/保留/切片三条胶水路径（桩引擎）；冻结请求构造逐字节不变；全部规则**确定性 + 与行序无关** |
| `scripts/minicpm5/test_harness_audit.py` | **47** | 每个退化模式的判定（含 TF/GEN 回退优先级）、失败归因 2×2、逐 family 速率与 flag、首个 token 直方图、`score_tf_probe` 对桩引擎的完整取数（首位 argmax、top-5）、数字 token 谓词 |
| `scripts/minicpm5/test_mechanism_audit.py` | **35** | 投影（含"运行目录只有部分文件"）、`fired_no_effect` 必须被标出（复现 14 票的报告型趋势子句）、clamp 从未咬合 → `never_fired`、0↔8 单独记账、阶梯两个方向、保留/切片/噪声带/未作答 flag 的 alive 判定、**回滚身份未复原会被标出**、空目录不崩、K 行守卫记录的两种形状 |
| `scripts/minicpm5/test_no_answer_metric.py`（19 票，回归） | 40 | 我的改动没有破坏既有仪表 |

跑法（Mac 或任意 CPU 机器）：`cd scripts/minicpm5 && python test_round2_retention.py && python test_harness_audit.py && python test_mechanism_audit.py && python test_no_answer_metric.py`

## 5. 实测发现（其中三条**与本轮 brief 的假设不符**）

### 5.1 §6 审计：两条在跑运行的机制台账（只读 ssh，2026-09-10 18:53–18:56）

以 `seed20260912`（`gooo`，gen 68 完成）与 `seed20260913`（`goooinext`，gen 67）为止的台账：

| 机制 | gooo/20260912 | goooinext/20260913 |
| --- | --- | --- |
| 硬门（值口径） | alive 68 代、value diffs 0 | alive 67 代、value diffs 0 |
| readback | alive 68 代、0 值差 | alive 67 代、0 值差 |
| KL 绝对阈值 | never_fired（从未接近） | never_fired |
| **KL 趋势子句** | **fired_no_effect ×11**（触发但未就地执行） | **fired_no_effect ×10**（同） |
| α 阶梯 | **两个方向都触发**（升 16 一次、降 8 一次） | **一次未触发** |
| no-accept | never_fired | **alive：gen66 停机（连续 2 代 moved=0）** |
| 残差 clamp / 端点拒绝 | 各 291 | 各 131 |
| 未跨格（残差滞留） | 1,199,915 | 1,230,745 |
| 0↔8 三口径 | never_fired（raw/canonical/0↔8 **全零**） | never_fired（同） |
| q0 参照臂 | alive | alive |
| 未作答 flag | alive（14 代被打 flag） | alive（15 代） |
| 保留 / 反馈切片 / 噪声带 / swap | not_wired（这一轮没开，符合预期） | not_wired |
| 镜像对门 / 逐代中心步 | alive（丢过对 399 个坐标… 见 JSON） | alive |

**两条要立刻上报的事实**：
1. **KL 趋势子句再次是"报告型"**：两台分别有 11 次与 10 次 `trend_breach` 记录，**没有任何一次导致停机**（`materiality_floor` 3.2019 全程在案，绝对阈值 6.4 从未接近）。这与 14 票的主要偏差**同类**；本轮运行是外部裁决停止，所以这仍是"守卫没守卫"，不是新故障。
2. **`goooinext`（seed 20260913）已于 18:46 停机**：`stop.json` 记 `no_accept`（gen65、gen66 连续两代 `moved = 0`、`qualified_pairs = 0`——所有镜像对都被 τ 门丢掉）；停机时中心 dev acc 0.525、中心 sha `dd25f0c8…` 在 gen64 之后**未再变化**。`gooo` 仍在跑（gen68 完成，α=16，`moved=390/M=22`，无 `stop.json`）。**这不是我做的任何操作**（我只读台账），但"两枚种子同参跑到 ~00:35"的前提已经不成立，终检安排需要协调者重新裁决。

### 5.2 与 brief 不符之一：预注册阈值挂在"harness 项"上会退化

§4a 要求用"本次运行自己的噪声带"当阈值。按字面实现（term=`harness`）时：**TF 打分器是逐位确定的**（同权重、贪心 argmax、关 prefix cache），同一中心重测的标准误为 **0** → 带 = 量化下限（1 题）。而我在 `gooo` 上取 gen40–45 **真实的中心逐代读数**（dev 池，每 family 96 题）算了一下相邻代差：

```
gen40->41: arith:+4*  logic:+6*  modulo:-8   ordering:+2*  pct:+0
gen41->42: arith:-2   logic:-1   modulo:+6*  ordering:+2*  pct:+1
gen42->43: arith:-9   logic:-3   modulo:-4   ordering:-2   pct:+3*
gen43->44: arith:+9*  logic:-1   modulo:+5*  ordering:+2*  pct:-4
gen44->45: arith:+0   logic:-4   modulo:-2   ordering:+3*  pct:+6*
（* = 跌幅 > 1 题，即按 harness 项会触发回滚）
```

**25 个"family×代"事件里 11 个越带；5 个相邻代里 5 个都会被回滚**。也就是说：字面预注册的规则在本循环上**接近"每代必回滚"**，会把搜索冻死。根因不是回滚机制错，而是**主导噪声不是重测噪声、而是 dev 池自身的分辨率**（v3 笔记：dev 每 family 12 episode → 只能分辨 13–15 点；跨代 ±2–9 点的摆动是池子噪声 + 真实步进的混合）。处置：**实现了 brief 的字面规则（默认 term=`harness`），但把 term 变成显式旗标**（`sampling` = 簇校正 SE，dev 每 family ≈4.5–7 点；`max` = 两者较大者），并把"用了哪个 term"写进每一代的台账；下一轮**要开 §1 就必须显式选 term 并记录这属于对预注册读法的偏离**。配套建议：用 §4b 的**中心切片**当保留通道（每 family 408 题 / 51 episode，簇校正 SE ≈2.5 点），比 dev 更接近"能分辨"的量级。

### 5.3 与 brief 不符之二：引用的 anomaly 笔记不在 `main`

`docs/experiments/minicpm5-bf16-anomaly-check-20260910.md` **不存在于 `main`**（`main` 的 `86d337b` 下没有 `scripts/minicpm5/`，也没有该笔记）；它在 `codex/wayfinder-minicpm5-sanity` 的 `3656866` 上。我的工作树从该分支的 `27ae570` 起，所以**本工作树里有**（并在 §3 的论证中直接引用：去掉 9 个字符后缀后未作答 65.4% → 0.0%）。若协调者把本轮改动合到别处，需要先保证 19 票那两个提交也在。

### 5.4 与 brief 不符之三：v3 封存池对"在跑运行"已消耗，终端 swap 臂要用 v4 池

§3 要求终端在**封存池**上测四种组合。v3 confirm 池按 pre-registration 每枚种子各开一次（总 2 次），`goooinext` 已经停机、其开箱已发生/待发生；而 `pools-v4` 分支刚交付了**全新未开箱**的密封 confirm 池（`taskset_confirm_v4.jsonl`，8,344 题 / 1,043 episode，按 σ=0.185 估 δ_min ≈1.6 点）。我的实现对池子无假设（`--confirm` 收任意 jsonl），所以**用法上是"换池不换代码"**；但"头条形态用哪个池、哪枚种子用哪次开箱"属于协调者的判定，我没有代为决定。附带成本提示：8,344 题 ×4 臂 ≈ 每臂 26 s（v3 的 32,040 题是 ≈99 s/臂），四臂终端比 v3 便宜约 3.8×。

### 5.5 其它操作性发现

- **`frozen` 形态的请求构造未被本轮改动**：`build_reqs(..., "frozen")` 与旧路径的 `prompt/p_ids/g_ids` 逐字节一致（单测断言），新增的只有 `variant/stable` 两个簿记字段。
- **回滚快照只需要"该步会访问的坐标"**，但要连"层桶是否存在"一起复原——`CentreState.sha256` 会遍历空桶（14 票 §5.1 的真 bug），只复原值不复原桶会让"回滚后的中心"身份与上一代不同。单测里专门有这一条。
- **`--terminal-swap` 默认开**会让终端多花时间（v3 上 4 臂 ≈6.7 min）；如果协调者只要"记录 swap 未测"，用 `--no-terminal-swap`，此时 `claim.gain_claimable` 自动为 false 并写明原因。

## 6. 机时与纪律

| 设备 | 用途 | 机时 |
| --- | --- | --- |
| 本地 Mac | 代码、单测（176+40 项）、统计、文档 | — |
| `gooo` | **只读** ssh：机制审计投影（读 `history/stop_ledger/records/gen*/kernel_values*`）+ 6 个 `gen*.json`（≈2.8 MB，用于 §5.2 的摆动实测） | 未占 GPU |
| `goooinext` | **只读** ssh：机制审计投影 | 未占 GPU |
| GPU 作业 | **无** | **0** |

- 未跑 `pilot_search.py`、未起 vLLM、未重启/未改动两条在跑运行、未写 `~/minicpm5-run300/` 下任何文件（远程只写了 `/tmp/mechanism_audit_<pid>.py|json`，用完即删）；
- 未使用两机的 `~/Code/shadow-peft/.venv`；未同步 `.env` / `.quark_netdisk_auth.txt` / `.scratch/`；
- 冻结参数（α=12、k=32、δ=0.02、τ=0.02）、题池、评分语义**未改**；`apply_step`/`aggregate` 的数学**未改**（只有一处**无语义**重构：`apply_step` 的 `touched` 改调 `step_coordinates()`，单测把两者的访问集绑在一起）。

## 7. 仍待机器验证（GPU 未跑，脚本已就绪）

1. **§5 输出模式扫描**（需要引擎，任一 GB10 空闲时跑）：
   ```bash
   python harness_audit.py --model ~/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4 \
     --items ~/minicpm5-run300/artifacts/taskset_dev.jsonl --out harness_audit_dev_frozen.json \
     --variant frozen --arm both --margin 0.05
   # 再跑一遍 --variant plain，比较两个形态的 no_answer / non_numeric 逐 family 率
   ```
   预期（待测）：`frozen` 形态在 `logic/hard`、`ordering/hard` 上有非零 `no_answer`；`plain` 形态应接近 0（19 票的探针结论），Δ 就是 §3 要报的"harness 效应"。
2. **§1 保留路径的端到端 GPU 验证**（1 代即可）：`--retention --noop-rescores 3 --generations 1 --resume <state.json>`，检查 `feedback_noise.json` 的 harness 项是否为 0（预期：是，逐位确定）、`records/retention_gen00.json` 是否写全、以及是否需要按 §5.2 换成 `--retention-band-term sampling` 或切片通道。
3. **§2/§3 终端四臂**：对任一枚已停运行的种子跑 `--confirm-only`（新的 v4 密封池 + `--terminal-swap --retention-terminal`），核对 `terminal_retention.json`、`prompt_swap.json`、`terminal_payload.json` 与 `confirm_gate.py --measure` 的兼容性（后者应逐字不变地读 `arms`）。
4. **§4b 切片通道**：用 `taskset_centre_v4.jsonl` 跑一次 `--feedback-slice ... --noop-rescores 3 --generations 1`，核对切片带（簇校正 SE）与逐代轨迹是否入账。
5. **§6 审计在停止后的完整台账上重跑**：停机后 `stop.json` 存在，`stop_policy` 会从 `fired_no_effect` 变成 `alive`（外部停机会被区分开）；建议对两条运行的最终目录各跑一次并入库。

## 8. 证据清单

- 本笔记：`docs/experiments/minicpm5-round2-improvements-20260910.md`
- 两条在跑运行的机制审计（入库）：`docs/experiments/artifacts/minicpm5-round2-improvements-20260910/mechanism_audit_seed20260912_gen68.json`、`…seed20260913_gen67.json`（复现：`python scripts/minicpm5/mechanism_audit.py --ssh gooo:'~/minicpm5-run300/out/seed20260912' --out x.json`）
- 新脚本：`scripts/minicpm5/retention_lib.py`、`harness_audit.py`、`mechanism_audit.py`
- 新测试：`test_round2_retention.py`、`test_harness_audit.py`、`test_mechanism_audit.py`
- 改动文件：`scripts/minicpm5/pilot_search.py`（+546/−28 行）、`taskset_lib.py`（+47）、`score_taskset.py`（+10/−2）、`pilot_lib.py`（快照/重锚 + `step_coordinates`）
