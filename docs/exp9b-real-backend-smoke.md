# exp9b 真实后端交接冒烟（Qwen3-0.6B）

一句话收束：**在 `goooinext` 上用 Qwen3-0.6B（`enable_thinking=False`）跑通了真实 `generate_until` 流式交接与 `CloudService` 真实续写：前缀在句号/换行处停下、云端返回 `continued_from_prefix=true` 的纯续写；回流焊 prompt 上前缀 + 续写约 2.15s，完整生成约 3.2s，但完整生成在 64 token 处被截断，且两侧模型相同，不能当作生产结论。**

状态：**真实后端冒烟已跑通（2026-10-20）**；真实 shadow/base_shadow 路径与准确率结论未跑，见「已知限制」。

## 环境

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（`aitopatom-dc4a`，192.168.1.177，GB10） |
| 接入方式 | Mac 不能直连 .177，使用 `ssh -J gooo goooinext` |
| 代码 | `main` @ `26952bc` + 本分支 `5b6276d`（`enable_thinking` 支持） |
| 模型 | `~/models/Qwen3-0.6B`，`transformers 5.15.1`，`torch 2.12.1+cu130` |
| 解码 | greedy，`max_tokens=64`，`enable_thinking=False` |
| prompt | 3 条产线问答（回流焊峰值温度、E-217 主轴温度、DPPM 计算） |

## 命令

```bash
rsync -az -e 'ssh -J gooo' --exclude __pycache__ src/ goooinext:~/Code/shadow-peft/src/
ssh -J gooo goooinext 'cd ~/Code/shadow-peft && PYTHONPATH=src .venv/bin/python - <<PY ... PY'
```

冒烟脚本做了三件事：

1. `TransformersBackend.generate_until` + 永不交接策略 → 完整生成；
2. 同一请求 + `AtN(20)` 策略（第 20 个 token 后请求交接，最多等 8 个 token 到边界）→ 前缀；
3. `CloudService.enhance(CloudRequest(partial_trace=前缀, allow_prefix_rewrite=False))` → 真实续写。

## 结果

### 1. 流式交接

| prompt | 完整 token / 耗时 | 前缀 token / 耗时 | 边界 | 前缀是完整答案的前缀 |
|---|---|---|---|---|
| 回流焊峰值温度 | 64 / 3.15s | 28 / 1.40s | sentence | 是 |
| E-217 主轴温度 | 64 / 3.15s | 28 / 1.37s | token（等待预算耗尽） | 是 |
| DPPM 计算 | 64 / 3.16s | 22 / 1.08s | newline | 是 |

首条完整生成耗时 12.0s（模型/内核冷启动），上表为预热后数值。三条前缀都是完整生成的真前缀，说明 `generate_until` 与完整解码走同一 greedy 轨迹。

### 2. 真实云端续写

以回流焊 prompt 的 28-token 前缀调用 `CloudService.enhance`：

- `continued_from_prefix=true`，`corrected_prefix=false`
- 续写 `output_tokens=50`，耗时 **0.75s**
- 返回文本没有重述前缀（例：「结论：回流焊峰值温度应根据材料类型、焊点密度和焊接环境进行合理控制……」）

即：`CloudService` 的续写提示词在真实 Qwen3-0.6B 上生效；直接用「已有前缀（请从中断处继续）」的手写 prompt 反而会让模型重述前缀，生产路径应走 `CloudService`。

### 3. 耗时对照（仅机制验证）

- 前缀 1.40s + 云端续写 0.75s ≈ **2.15s**
- 完整生成 3.15s（64 token 截断）
- 本次两侧都是 0.6B，且完整答案被截断，因此这不是「小端 + 大云」的生产收益；只证明真实链路可跑通、续写契约生效。

## 已知限制

- **真实 shadow/base_shadow 未跑**：`~/models/Qwen3-0.6B-H4B` / `Qwen3-8B` 等目录是普通模型，不是 ShadowPEFT checkpoint（缺 `shadow_config.json`）；`runs/exp7/*/ckpt-final` 是 4B 主干 + 显式 H4B 影子的 ShadowPEFT checkpoint，而当前 `TransformersBackend` 只调用 `ShadowPeftModel.from_pretrained(base, ckpt)`，不能传入显式 shadow 模型。要在真实 base/shadow 双路 logits 上评测，需要给后端增加 `shadow_model_path`（或从 exp7 训练脚本复用加载逻辑），另立 ticket。
- **无标注语料**：本次只验证机制，不报告准确率增益；结论级评测仍需甲方标注集。
- 真实云端只用了同一个 0.6B 模型；没有接入 4B/8B 云模型或 OpenAI-compatible 服务。
- 云端续写耗时是单条、单次测量，未做 P50/P95 或多并发压测。

## 复现

```bash
ssh -J gooo goooinext
cd ~/Code/shadow-peft
PYTHONPATH=src .venv/bin/python - <<'PY'
# 见本文件「命令」一节的 3 步脚本；模型路径 /home/goooice/models/Qwen3-0.6B
PY
```

真实后端配置（exp9 评测台）：

```json
{
  "kind": "transformers",
  "model_path": "/home/goooice/models/Qwen3-0.6B",
  "device": "cuda",
  "dtype": "bfloat16",
  "enable_thinking": false
}
```
