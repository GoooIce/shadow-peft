# Mage 直接 W4A4：01–08 实施与验收记录

2026-09-09。用户授权以 subagent 实现 01–08；由 root 集成，三位执行者分别负责量化运行时、请求/预算/统计与评分接入、OCR 合同/奖励/协调器及先导报告。实施规格见 [已确认规格](mage-direct-w4a4-gb10-pilot-spec-20260909.md)，实时票状态见[本地执行索引](../../.scratch/mage-direct-w4a4-implementation/README.md)。

## 实施边界

新增实现位于 `src/shadow_lab/mage_w4_search/`，没有修改旧 ES/QAD 实验代码。第二阶段只安装 QAD 导出的原生四位权重、冻结缩放与非量化余项；不加载 BF16 MMDiT 教师或主权重，不使用 teacher_loss，也不搜索裁剪系数。文本编码器、VAE 和评分器仍是固定辅助组件。

候选为单组稀疏相邻改码，覆盖 12 个 block 加块外六层，共 13 组、174 个量化层。候选筛选采用 CLIP:HPS:OCR = 2:1:1 的固定尺度奖励，逐轴同时对照当前中心和固定起点。不可行候选排除；完整优胜候选须另经独立确认才能成为正式中心。开发协调器的模拟确认始终带开发身份。

## 软件入口

| 票 | 模块与公开入口 | 验收边界 |
| --- | --- | --- |
| 01 | `contracts.preflight`；`budget.BudgetLedger` / `run_guarded`；`data.freeze_pilot_manifest` | 内容身份、数据用途、唯一预算、硬截止和恢复 |
| 02 | `native`；`scripts/mage_w4_native_probe.py` | 原生 NVFP4 安装、真实 CUDA trace、零更新与保存重载 |
| 03 | `mutation` | 数值邻接、均匀无放回、双零、局部码差异与确定性重建 |
| 04 | `ocr.recognize` / `match_text` / `compare_ocr` / `freeze_scorer`；`scorers` | 图像盲识别、事后目标匹配、真实人工转录校准 |
| 05 | `evaluation.evaluate_generated_candidate`；`rewards.calibrate_scales` / `screen_candidates` | 原生图片和评分身份、固定尺度、配对双参照初筛 |
| 06 | `coordinator.DevelopmentCoordinator` | 完整中心原子继承、重复/过期结果、预算和随机状态恢复 |
| 07 | `statistical` | 来源簇联合区间、确认风险与证据退役、单次终检 |
| 08 | `pilot` | 有限分组计划、证据汇总、成本重算及正式就绪缺口 |

各模块的单元夹具只证明软件行为。`fixture`、`development_mock_confirmation` 和合成覆盖报告均不能建立真实 OCR 锁、真实先导尺度或正式确认结论。

## 已取得的真实证据

证据根为 `runs-local/mage-direct-w4a4-20260909/`；GPU 工作用 gooo、goooinext，唯一预算账本为该目录的 `pilot-budget.sqlite3`。遥测中组件耗时可能嵌套，实际机时仅按租约占用区间相加。

- **01 已验收**：开发子用途分离；新确认/终检清单按来源组隔离并封存。旧池为 OCR 选型、尺度校准、候选开发分配 107 / 108 / 105 条，其中明确完整目标文字为 5 / 5 / 4 条。新封存清单每池 32 条、8 个来源组，仅有元数据；这不是已证明有足够功效的正式样本量。
- **02 已验收**：两台 GB10 均实际执行 W4A4 block-scaled CUDA 内核。零更新、保存重载与跨机共享两图像素零差，未加载高精度 MMDiT 权重。详见 `cross-device/02/comparison.json` 和两台 `02-native-<host>-v3/` 的原始 trace、文件访问审计、停止回执。
- **03 已验收**：两台独立扫描 174 层已保存载荷，只有 `norm_out.linear` 的 flat_index 367918 由 code 6 改为 5，缩放与余项不变。重建、保存重载、恢复 q0 与跨机重放均一致，详见 `cross-device/03/comparison.json`。工程烟雾值为 1 码；图像与 q0 仍相同，未宣称质量收益。
- **04 部分实测**：已用原生 q0 生成独立的 5 张 OCR 开发图，保存图片与身份。人工转录入口为[五图转录页](../mage-w4a4-ocr-transcription-20260909.html)。提示词要求的文字与 AI 看图结果都不能替代实际文字的人工转录。

q0 的规范化张量身份为 `3991075b4a191d0e309172c1d44e1e6437cfb5a8ea7bcb8e2ee115012241b642`；固定状态身份为 `5d1d6ed17dbfff0a5017990d4f742fdae0e1684b9b4772c337841ed3d76fdccd`。张量身份与 `.pt` 归档文件 SHA 不同，回执分别保存。

固定生成设置为 512×512、4 步、CFG 1、动态 A4 block size 16、确定性种子及显式水印 key 20260909。真实每图记录包含 profiler 和文件写入开销，不可直接当作纯推理吞吐或正式批耗时。

## 后续运行方式与阻塞

`scripts/mage_w4_launch.py --host <gooo|goooinext> --job-id <prepared-job> --seconds <reservation>` 是协调端的现有原生探针启动器，要求远端已有冻结的 `prepared.json`。它通过 SSH 传递内容绑定请求和租约，启动不依赖 SSH 生命周期的监督进程。断线后用同一个 job 的 `--recover`，不能重启账本或为相同闭合 job 重新申请租约。

该启动器默认服务 02/03 原生探针和 04 开发图像生成；`--worker-kind ocr` 可执行 `ocr_worker` 的单引擎比较观测。每个 OCR job 的完整 JSON SHA 须先绑定到请求的 `runtime_identity.ocr_comparison_job_sha256`；图片和生成回执在加载引擎前逐项复核。worker 不产评分锁，协调端仍须验证停止、计入完整占用、汇齐两引擎观测与人工转录，再比较选型。这不是完整正式搜索 CLI。05–08 的公开模块及验收夹具已按票分工接入；真实三轴批和分组校准仍须先获得 04 的合格冻结评分器及 05 的真实尺度。

PaddleOCR/EasyOCR 在 GB10 的专用环境中安装，不改 Mage 环境。PaddleOCR 的直接导入曾发生 C++ 初始化崩溃；CPU 对照发现先导入 Paddle 再加载 PaddleOCR 可成功。独立 OCR 子进程用于隔离与主进程 Torch 的原生库导入顺序，子进程仍继承预算监督范围。每图独立启动的成本须完整实测，不能隐藏成零。

07 的有界来源簇 Hoeffding–Bonferroni 区间有明确独立性和分数支持界前提。真实 BF16 CLIP/HPS 分数支持界、来源抽样机制、实际尺度和功效成本尚待审核；开发分数的极值不能充当总体支持界。该保守方法可能需要超出现有预算的 OCR 样本量。合成覆盖通过也不消除这些前提。

先导上限仍是双机合计 16 设备机时、每台 8 小时。报告会保留缺失字段；09–10 的正式协议、总预算和终检预留没有因实现本队列而自动获得。当前不具备启动正式搜索的条件。

## 本轮统计模拟与资源快照

GB10 CPU 合成覆盖已完成：四场景各 10,000 次、64 来源簇，固定 seed 20260909。各场景联合不覆盖计数均为零，有限模拟上界 0.019727；强收益场景接受率 0.191。后者说明检验保守，不能当作真实功效已达标。`ticket07-resources.json` 记录 wall 79.1679 秒、CPU 79.1426 秒、峰值 RSS 21,900 KiB，新增 GPU 占用零。

截至六个租约闭合，唯一账本累计 1,586.793425 秒（0.440776 设备机时），其中 gooo 652.406450 秒、goooinext 934.386975 秒；无活动租约及超时超额。失败启动也已计入，没有在 08 重新获得预算。以后运行须读取实时 SQLite，不能用本段快照计算可用额度。

## 软件验证与审查

新增实现套件：157 passed（7.38 秒）。Ruff 检查与 Python 编译检查通过。全仓库首次运行取得 616 passed、18 skipped；唯一失败是并行接入时 OCR 配置脚本尚未暴露新接口，该接口修复后对应单文件和关联套件已通过，并包含在上述新增套件中。没有把这次失败称为旧环境问题，也没有声称重新完整运行后的全库结果。

[双轴代码审查](mage-direct-w4a4-implementation-review-20260909.md) 发现预算唯一性、重复解码合同及两处跨模块身份错配，均已修复并通过对应审查者的定点复核。实验事实仍以实时票和原始设备回执为准。

最终 OCR 初始化使用 `ocr-source-v4` / `ocr-config-v4`，组合源码身份与提交前源码一致。PaddleOCR 和 EasyOCR 均加载成功（墙钟 3.0514 / 3.0177 秒，子进程 CPU 3.4418 / 3.5721 秒）；`ocr-initialization-v4.json` 保留回执。识图数仍为零，不能将初始化成功当作主 OCR 已冻结。

## 最终设备快照与一次探针异常

最终累计 **1723.378689 秒（0.478716 设备机时）**，9 个租约全部闭合；gooo 763.096493 秒，goooinext 960.282195 秒，无活动租约。

06 的一次临时截止探针因启动脚本的 shell 后台化错误而延迟收到停止确认：worker 实际在 25.65 秒内停止，账本保守计入 84.80 秒并保留 54.80 秒预留超时。脚本修复后的双机复验均及时 SIGKILL、进程组消失且正常结算，新增 51.78 秒，无新增超时；没有回填或删除旧账。详见 `ticket06-hard-deadline/remediation-summary.json`。这仅验证真实 worker 的监督边界，06 三轴开发轮次仍未验收。

08 的保守就绪检查会将历史租约预留超时保留为运行阻塞标记；它不表示本次双机复验的 worker 运行超过硬截止。人工转录、真实三轴批、分组和统计前提的缺口也仍保留。
