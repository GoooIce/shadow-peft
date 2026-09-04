---
id: ENG-001
type: engineering
status: done
priority: P0
depends_on: []
hardware: GB10
---

# ModelScope 资产与 BF16 基线

## 目标

建立可重放的 Mage-Flow-Turbo BF16 基线，并把所有模型、数据集和评测资产固定到 ModelScope 来源及版本。

## 工作范围

- 通过 ModelScope 下载 Mage-Flow-Turbo、所需自定义代码和数据资产。
- 记录仓库 ID、revision、文件校验和、许可证与本地缓存布局。
- 清点 MMDiT、Qwen3-VL 文本编码器、VAE 和调度器的模块/参数/权重体积。
- 固定 prompt、潜噪声 seed、分辨率、4 步调度和输出编码。
- 在 GB10 生成小型 BF16 golden set，记录峰值统一内存和端到端延迟。

## 验收标准

- 新环境只依赖 manifest 即可恢复相同资产版本。
- 同一 GB10 上同 prompt/seed 的输出满足预定义确定性检查。
- 资产许可和数据用途已记录。
- BF16 golden set 与运行元数据可被后续量化和标注票消费。

## 非目标

- 不量化模型。
- 不实现 ES 或奖励模型。

## 证据

- 2026-09-04：用户批准与研究工作并行启动。
- 2026-09-04：`microsoft/Mage-Flow-Turbo@master` 已从 ModelScope 下载到 `goooinext:~/models/Mage-Flow-Turbo`，并经双机直连 rsync 到 `gooo`；两端 43 个文件的 SHA-256 均通过。
- 2026-09-04：仓库清单为 `assets/manifests/mage-flow-turbo-modelscope.json` 与 `.sha256`；快照共 17,507,370,503 bytes，ModelScope 镜像只暴露可变 `master`，因此以完整 SHA-256 清单作为不可变身份。
- 2026-09-04：`scripts/download_assets_modelscope.sh mage` 提供可恢复下载入口；独立 GB10 运行环境由 `scripts/setup_mage_flow_gb10.sh` 构建，避免污染现有 `shadow-peft/.venv`。
- 2026-09-04：ModelScope 数据集 `Lakonik/t2i-prompts-hpsv2@master`（3,200 条 prompt，仓库标注 Apache-2.0）已下载到两台 GB10 并逐文件校验；清单为 `assets/manifests/t2i-prompts-hpsv2-modelscope.{json,sha256}`，恢复入口为 `scripts/download_assets_modelscope.sh mage-prompts`。该仓库是第三方镜像，数据切分留给 ENG-003。
- 2026-09-04：官方 Mage 源码固定为 commit `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`；独立环境安装与 BF16 生成基线仍在进行。
- 2026-09-04：`goooinext` 独立环境构建完成：Python 3.12.3、PyTorch 2.13.0+cu130、diffusers 0.38.0、transformers 5.5.0、flash-attn 2.8.3；FlashAttention 仅为 `sm_120` 编译，运行设备检查为 NVIDIA GB10 / capability 12.1。
- 2026-09-04：`scripts/run_mage_flow_bf16_baseline.py` 提供 GB10 限定的 BF16 基线入口，记录输入 prompt 快照哈希、组件参数/字节数、环境版本、加载/逐样本延迟、峰值内存及 PNG SHA-256，并用首样本同 seed 双跑做确定性检查。
- 2026-09-04：单图烟测通过：512×512、4 steps、CFG=1、seed=42 的两次 PNG SHA-256 均为 `c7b37635ab8881f9743d1cbdb9f6635591652bbc48b4af00bf8afdf523f1b817`；两次生成分别为 7.20s / 5.04s，CUDA 峰值 allocated 18.09GB、进程峰值 RSS 30.80GB，输出经人工目视确认有效。
- 2026-09-04：从 3,200 条 ModelScope HPS-v2 prompt 镜像以固定 RNG seed `20260904` 抽取 32 条开发用 golden prompts（JSONL SHA-256 `289afe41f131f885698508e4a78250ada75bfa81f794050974beffb6028ba1d0`）；32 图完整 BF16 golden run 已在 `goooinext` 以 setsid PID `3146600` 启动，最终人工测试集仍由 ENG-003 隔离定义。
- 2026-09-04：32 图运行以退出码 0 完成并复制到 `gooo` 逐文件校验；总时长 246.19s（冷加载 101.49s），逐图均值/中位数 4.34s/4.51s，CUDA 峰值 allocated 18.09GB、进程峰值 RSS 31.18GB。
- 2026-09-04：发现 10/32（31.25%）prompt 被 Mage 内容门控替换为相同白色拒绝图，故该批仅为运行基线，不能作为质量 golden set。详细记录见 `docs/experiments/mage-flow-bf16-gb10-baseline-20260904.md`；ENG-001 保持 `in-progress`，门控通过的正式集待冻结。
- 2026-09-04：新增 `scripts/filter_mage_flow_prompts.py`；在固定 256 条候选顺序上筛到第 41 条时取得 32 条门控通过 prompt（拒绝 9 条：copyright 8、violence 1），首条双筛判定一致。通过集 SHA-256 为 `830a12b082d3c1576b31287e85455f2cfbca3fbcaa2eacd0542d4df0f301d58a`。
- 2026-09-04：正式 BF16 quality golden set 32/32 完成、哈希均唯一且无白色拒绝图；首图双跑完全一致。总运行 248.86s（冷加载 94.58s），逐图均值/中位数 4.60s/4.82s，CUDA 峰值 allocated 18.09GB、进程峰值 RSS 31.19GB。两台 GB10 的图像和元数据逐文件校验一致，ENG-001 验收完成。
