# Mage-Flow-Turbo BF16 / GB10 基线记录（2026-09-04）

## 结论

Mage-Flow-Turbo 已在 `goooinext`（NVIDIA GB10）以 BF16、512×512、4 steps、CFG=1 跑通，并通过同 prompt / seed 的字节级确定性检查。完整 32 条开发集运行也已完成，但其中 10 条被模型自带内容门控替换为同一张白色拒绝图，因此该批次只能证明基础设施与吞吐，不能作为最终图像质量基线。

ENG-003 在冻结正式评测集时必须先显式处理门控：主质量集只纳入门控通过的 prompt，同时把拒绝率作为独立系统指标记录；不可把白色拒绝图送入 HPS/CLIP 后与正常生成混算。

## 资产

- 模型：ModelScope `microsoft/Mage-Flow-Turbo@master`，43 个文件，17,507,370,503 bytes。
- 模型不可变身份：`assets/manifests/mage-flow-turbo-modelscope.sha256`。
- prompt 来源：ModelScope 第三方镜像 `Lakonik/t2i-prompts-hpsv2@master`，3,200 条；仓库标签为 Apache-2.0。
- prompt 不可变身份：`assets/manifests/t2i-prompts-hpsv2-modelscope.sha256`。
- 32 条开发集：Python `random.Random(20260904).sample(range(3200), 32)`；JSONL SHA-256 为 `289afe41f131f885698508e4a78250ada75bfa81f794050974beffb6028ba1d0`。
- 上述模型、prompt 数据、32 图和运行元数据均已复制到 `goooinext` 与 `gooo` 并逐文件校验。

ModelScope 镜像只暴露可变 `master`，所以 revision 仅作来源描述，完整 SHA-256 清单才是本项目使用的快照身份。

## 环境

| 项目 | 值 |
|---|---|
| 主机 | `aitopatom-dc4a` (`ssh goooinext`) |
| GPU | NVIDIA GB10, compute capability 12.1 |
| OS | Linux 6.17.0-1029-nvidia, aarch64 |
| Python | 3.12.3 |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
| diffusers | 0.38.0 |
| transformers | 5.5.0 |
| flash-attn | 2.8.3，自源码仅编译 `sm_120` |
| Mage 源码 | `microsoft/Mage@76bec2bb3818863f470de7e867c2dc7f1d0bfd83` |

独立环境位于 `~/Code/Mage/mage_flow/.venv`，由 `scripts/setup_mage_flow_gb10.sh` 恢复。宿主机缺少系统 Python 开发头且 sudo 需要口令，脚本会把对应 deb 解包到用户缓存目录并通过 `CPATH` 使用，不修改系统环境。

## 模型清点

| 组件 | 参数量 | BF16 参数字节数 |
|---|---:|---:|
| MMDiT transformer | 4,115,745,408 | 8,231,490,816 |
| Qwen3-VL text encoder | 4,437,815,808 | 8,875,631,616 |
| Mage VAE | 100,796,355 | 201,592,710 |

这里的“4B”指生成 transformer；完整在线管线还包含约 4.44B 的 Qwen3-VL 文本编码器和约 101M 的 VAE。

## 单图烟测

- prompt：`A digital artwork depicting a cartoon illustration of a warehouse environment.`
- seed：42；512×512；4 steps；CFG=1。
- 两次 PNG SHA-256 均为 `c7b37635ab8881f9743d1cbdb9f6635591652bbc48b4af00bf8afdf523f1b817`。
- 两次生成延迟：7.20s / 5.04s。
- 模型冷加载：119.52s。
- CUDA 峰值 allocated / reserved：18.09GB / 18.41GB。
- 进程峰值 RSS：30.80GB。
- 人工目视：输出为有效的仓库卡通图，不是拒绝占位图或损坏图像。

远端证据：

- `goooinext:~/runs/mage-flow/bf16-smoke-20260904/run.json`
- `goooinext:~/logs/mage-flow-bf16-smoke-20260904.log`

## 32 条开发集运行

- 输出：32/32 文件生成，退出码 0，无 swap/OOM。
- 总墙钟时间：246.19s，其中冷加载 101.49s。
- 每条首跑延迟：均值 4.34s，中位数 4.51s，经验 p95 5.67s。
- CUDA 峰值 allocated：18.09GB；进程峰值 RSS：31.18GB。
- 首样本同 seed 双跑 SHA-256 完全一致。
- 22 张正常生成、10 张白色拒绝占位图；拒绝占位图 SHA-256 均为 `6c04fff04b3c72be7df55bad041c9173f29c056532af3075734d3a9bb2052cee`，本批拒绝率 31.25%。

拒绝 prompt 主要含受版权保护角色或公众人物。该现象说明 HPS-v2 原始 prompt 不能不经门控分层就直接用作 Mage-Flow 的自动质量奖励池，否则奖励变化会被固定白图和拒绝率混淆。

远端证据：

- `goooinext:~/runs/mage-flow/bf16-golden-hpsv2-32-20260904/run.json`
- `goooinext:~/runs/mage-flow/bf16-golden-hpsv2-32-20260904/SHA256SUMS`
- `goooinext:~/logs/mage-flow-bf16-golden-hpsv2-32-20260904.log`
- `gooo` 上有相同相对路径的校验副本。

## 门控通过的 32 条 quality golden set

固定候选池由 HPS-v2 的 3,200 条 prompt 用 `random.Random(20260904).shuffle` 产生，取前 256 条作为可审计候选池。Mage 自带 Qwen3-VL 内容门控按顺序筛选 41 条后取得 32 条通过项：9 条拒绝中 copyright 8 条、violence 1 条。首条 prompt 重复筛选的 verdict 完全一致。

- 候选池 SHA-256：`d23f969f6bf3f77ffebcef3af5fcb865cce39f8adc850c89c95accdce8dd406b`。
- 通过集 SHA-256：`830a12b082d3c1576b31287e85455f2cfbca3fbcaa2eacd0542d4df0f301d58a`。
- 正式运行：32/32 完成，32 个 PNG 哈希均唯一，无白色拒绝占位图。
- 首图同 prompt / seed 双跑 SHA-256 一致。
- 总运行 248.86s，其中冷加载 94.58s。
- 每条首跑延迟均值 4.60s、中位数 4.82s、经验 p95 6.24s。
- CUDA 峰值 allocated 18.09GB；进程峰值 RSS 31.19GB。
- 抽查首、中、末三张均为有效图像；该检查只验证数据完整性，不代替 ENG-003 的质量评分或最终人工盲测。

远端证据：

- `goooinext:~/runs/mage-flow/prompt-filter-hpsv2-32-20260904/{run.json,audit.jsonl,accepted.jsonl}`
- `goooinext:~/runs/mage-flow/bf16-quality-golden-32-20260904/{run.json,samples.jsonl,SHA256SUMS}`
- `gooo` 上有相同相对路径的校验副本。

## 重放命令

```bash
bash scripts/download_assets_modelscope.sh mage
bash scripts/download_assets_modelscope.sh mage-prompts
bash scripts/setup_mage_flow_gb10.sh

~/Code/Mage/mage_flow/.venv/bin/python \
  scripts/run_mage_flow_bf16_baseline.py \
  --model-path ~/models/Mage-Flow-Turbo \
  --prompts-jsonl ~/data/mage-flow-baseline/hpsv2-dev-golden-32.jsonl \
  --limit 32 --seed 424200 \
  --height 512 --width 512 --steps 4 --cfg 1.0 \
  --out ~/runs/mage-flow/bf16-golden-hpsv2-32-20260904
```

输出目录必须为空；运行器会拒绝混写已有实验，以免覆盖或把不同配置的图像混在一起。

## ENG-001 之后仍需完成

- 尚未跑 1024 原生分辨率基线。
- 尚未安装与标定 HPS/CLIP/GenEval 奖励模型。
- 尚未定义训练、checkpoint、自动测试、人工开发和一次性人工测试的无泄漏切分。

首条是后续分辨率消融，后两条属于 ENG-003，不阻塞本票。ENG-001 的可恢复资产、固定 BF16 quality golden set、确定性、性能和双机副本验收已完成。
