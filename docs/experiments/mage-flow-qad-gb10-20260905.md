# EXP-008 GB10 执行记录（2026-09-05）

状态：数据筛查及只读后台监控正在运行；接口探针发现模型结构与训练协议冲突，正式 QAD 尚未开始。
用户已授权实施并监控实验。自动流程最多走到 `human_dev` 待人工评审，不能自动开启 ES。

## 执行环境

- 主训练机：`goooinext` / `aitopatom-dc4a`，GB10 compute capability 12.1。
- 辅助筛查机：`gooo`；开始时两机 GPU 均空闲。
- 启动前 dc4a 可用内存约 116 GiB、磁盘约 2.7 TiB；gooo 可用内存约 95 GiB、磁盘约 737 GiB。
- Python：`~/Code/Mage/mage_flow/.venv/bin/python`；Torch `2.13.0+cu130`、TorchAO `0.18.0`。
- 代码从本地同步 `src/`、`scripts/`、`configs/mage-qad/`，未同步凭据文件。
- 模型：`~/models/Mage-Flow-Turbo`；训练配置 `configs/mage-qad/exp008.json`（512×512，四步）。

## 数据身份与隔离

新来源为已下载、固定 SHA-256 的 ModelScope `Lakonik/t2i-prompts-hpsv2`，共 3,200 条。
实际 parquet 哈希与仓库 `assets/manifests/t2i-prompts-hpsv2-modelscope.sha256` 完全一致：
`a7d3ac3bd4e4ec356184613799fbb45f179809bff54212f211934c4bb22968e1`。

准备器 `scripts/prepare_mage_qad_hps_sources.py` 排除完整 500 池与旧 HPS 开发候选 256 条，
再规范化去重，剩余 2,913 条。分层按 prompt 单词长度：1–12（1,540）、13–24（1,125）、
25+（248）。来源没有可靠语义标签，因此不把启发式标签冒充语义真值。

完整 500 池复制为 `~/data/mage-flow-eval/exp008-frozen-500/pool.jsonl`，本地与远端 MD5
均为 `ab2a716b116c48934578f7fd04785879`；SHA-256 为
`13620c7317447ff5e2e279fd20a69d20871e1a2d98f30f7419b8c63d0f707e29`。
这里只用于数据排除身份；不会用留出质量分数选择训练步数。

源候选的两个确定性、不相交分片位于两机 `~/data/mage-qad-exp008-20260905/sources-v2/`。
两机各调用 Mage 自带内容筛查，接受目标各为 640。合并后排除接口探针提示词，再固定
1,152 条，按各分层比例分出 128 条验证和 1,024 条训练，并记录完整 manifest。

启动准备中修复了原构建器的少数分层耗尽问题：不再先取完训练集再取验证集，而是在选定
候选内先按分层配额保留验证集。新增测试证明 90 条少数分层按 80 train / 10 val 分配。

## 独立接口探针

`probe_mage_qad_interface.py` 只运行一次四位置前向/反向，不执行 optimizer 更新，不保存或
选择训练检查点。用 source ID `677`、seed `88008008` 的独立提示词，最终 QAD 池显式排除它。
这不替代正式 campaign 的 smoke、preflight 或任何质量门。

报告：dc4a `~/runs/mage-flow/exp008-interface-probe-20260905/probe.json`。
已确认目标 Linear=174，sigma=[1, 0.9473684430, 0.8571428657, 0.6666666865, 0]。

首次与详细复测均得到 loss=1.7876423597335815，峰值 CUDA 分配 46.279 GiB；
详细复测四位置反向耗时 160.06 秒。171 层梯度有限，以下三层梯度缺失，没有非有限梯度：

- `transformer_blocks.11.attn.to_add_out`
- `transformer_blocks.11.txt_mlp.net.0.proj`
- `transformer_blocks.11.txt_mlp.net.2`

详细报告：`~/runs/mage-flow/exp008-interface-probe-detail-20260905/probe.json`。
实际 Mage 的最终 block 文本输出被丢弃，以上分支不影响图像输出。使用真实 Mage 类的
两 block 小模型分别运行 BF16 和 NVFP4，均复现末层相同三处无梯度，确认不是量化梯度故障。
两次完整探针均未更新权重，也未执行留出集评估。

已向用户请求协议选择：训练 171 个有效 Linear、冻结但仍量化其余 3 个（推荐），或为
末层文本输出增加教师辅助损失以训练全部 174 个。两者都会改变原票据协议，收到选择前
不启动正式训练；不会用伪造零梯度绕过健康检查。

## 后台状态与停止点

所有长任务使用 setsid 启动。筛查目录：

- dc4a：`~/runs/mage-flow/exp008-screen-dc4a-20260905/`
- gooo：`~/runs/mage-flow/exp008-screen-gooo-20260905/`

调度器 `scripts/orchestrate_mage_qad_exp008.py` 的冻结执行计划位于 dc4a
`~/data/mage-qad-exp008-20260905/execution-plan.json`。
原调度器已因探针失败正确终止，记录保留在 `~/runs/mage-flow/exp008-monitor-20260905/`
及 `~/logs/exp008-coordinator-20260905.log`，没有运行训练或自动门。

当前只读观察进程 PID 623817 使用 `observation-plan.json` 与 `--observe-only`，每 30 秒
更新 `~/runs/mage-flow/exp008-observer-20260905/status.json`，明确记录
`downstream_launch_disabled=true`。日志为 `~/logs/exp008-observer-20260905.log`。
两台筛查进程 PID 分别为 509598、478664，接受目标各 640。

调度器使用进程锁和持久化 worker 身份/结果记录，监控恢复时接管原任务，不重跑成功或
失败阶段；检查进程结束后再次读取终态，避免成功写入与进程退出之间的竞态误报。

顺序：两份筛查完成且接口探针通过 → build-data → 四阶段 QAD → 一次 checkpoint_val
自动门 → human_dev 盲测包 → **等待维护者**。任何阶段返回失败立即停止，不自动重试留出集。

正式 campaign 目标目录：`~/runs/mage-flow/exp008-qad-20260905/`；新增 `live.json` 报告
准备轨迹、前后向、optimizer、qad_val（每 8 条更新）及检查点写入状态，`campaign.json`
继续作为训练状态和检查点选择记录。

## 本地代码验证

执行支持代码及分层修复通过完整单元测试：421 passed、18 skipped（79.76 秒）。
相关文件 Ruff 检查、调度器/探针/训练状态/数据模块 ty 检查通过。
监控回归覆盖重启接管不重复执行、保留子任务失败，以及终态写入与退出检查的成功/失败竞态。
