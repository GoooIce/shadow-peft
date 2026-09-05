# Mage NVFP4 两阶段修复（EXP-008）

入口为 `scripts/run_mage_flow_healing.py`。Mac 只执行单元测试；实际数据准备、训练和生成使用
`docs/gb10-machines.md` 指定的 GB10 Mage 虚拟环境。历史 `run_mage_flow_es.py` 保留用于旧实验
复现，不是本票入口；它的 center、sigma、alpha 和 16,000 图预算不能导入新流程。

## 独立数据与训练

提供新来源 JSONL，每行包含 `prompt`、`source`、`source_revision`、`source_id`、`stratum`。
必须至少有 1,152 条与完整 500 条评估池隔离的唯一 prompt。构建器按分层轮转选择，规范化
Unicode、大小写和空白后排除重复 prompt；sample ID 和 seed 也全局唯一。manifest 保存实际
分层配额、来源 revision、数据及排除池哈希。训练只读取 manifest 中的排除身份哈希，
不打开留出 prompt 文件。这个入口不会拿评估 prompt 补足数量。

```sh
python scripts/run_mage_flow_healing.py build-data \
  --sources /data/new-qad-sources.jsonl --eval-pool /data/evaluation-500.jsonl \
  --out /data/exp008-qad
python scripts/run_mage_flow_healing.py train \
  --model-path /models/Mage-Flow-Turbo --data /data/exp008-qad \
  --config configs/mage-qad/exp008.json \
  --campaign /runs/exp008
```

四个独立阶段：接口 smoke 1 step、全层 preflight 32 steps、pilot 256 steps、正式训练最多
1,024 steps。所有 174 个 MMDiT Linear 的 latent 权重接受 AdamW 更新，bias、norm、Qwen3-VL、
VAE 冻结。训练前向执行权重和动态 activation NVFP4 fake quant，与原生动态 W4A4 路径对齐。

适配器截取真实 Mage pipeline 第一次 MMDiT 调用的输入，保留其文本模板、内容筛查、初始
噪声水印和 packing。学生沿四步 scheduler sigma 轨迹前进；每步教师使用同一个学生当前
latent、conditioning、timestep，输出 MSE 每步反传后释放图，不反传整个滚动轨迹。

维护者于 2026-09-05 根据独立小实验及盲投批准增加最终 block 文本输出的教师 MSE，
使图像输出未使用的三条末层文本 Linear 也得到真实梯度。正式配置 `text_initial_fraction=0.1`：
只在正式训练集首条样本校准一次，使文本损失初始贡献为图像损失的 10%，之后固定。
`text-supervision.json` 与检查点 progress 保存样本身份、初始损失和系数；恢复不重新校准。
图像/文本/总训练损失分别记录。`qad_val`、阶段门及最佳检查点选择仍仅使用图像教师 MSE。
全部 174 层保持量化并训练，采用小实验一致的 AdamW `foreach=False`。

每个阶段必须通过全层梯度、有限 latent/optimizer 状态、完整 AdamW 二阶矩和有限验证损失
检查；preflight/pilot/正式阶段末损失不得高于阶段起点。训练选择只看 `qad_val`，持平保留
最早检查点。正式阶段连续 128 steps 没有刷新全局最佳损失则提前停止。失败是终止状态，
不能用 `--resume` 越过失败门。正常暂停可用 `--stop-after N`，同配置追加 `--resume` 续训。

每次提交训练 step 原子保存 `latest.pt`。最佳检查点另存 `best-N.pt`，含 latent、完整
transformer state、optimizer/二阶矩、Torch/Python/CUDA RNG、packed NVFP4、显著性 clip genome、
layout、配置、实现/模型/数据身份和内容哈希。先提交新最佳，再提交 latest，最后删除不再被
引用的旧最佳。`campaign.json` 和每阶段 `report.json` 提供状态，不宣称训练产物已可部署。

注意：每步保存全部 master 和 optimizer 的 I/O 成本较高；本实现优先保证精确续训。训练和
packed 产物可在单元测试验证，原生 kernel 和图片质量必须在 GB10 实测后才有证据。

## 一次性自动门与人工门

后续生成命令均提供这些参数：

```sh
python scripts/run_mage_flow_healing.py checkpoint-gate \
  --campaign /runs/exp008 --model-path /models/Mage-Flow-Turbo \
  --eval-pool /data/evaluation-500.jsonl --calibration /data/reward-calibration.json \
  --hps-checkpoint /models/HPS-v2.1.pt --clip-checkpoint /models/CLIP-H14.pt
```

只有四阶段完成且最小 `qad_val` 检查点被固定后才能进入这个命令。`checkpoint_val` 使用全部
60 条，BF16 与候选逐样本配对，沿用冻结校准的均值/标准差及 0.5/0.5 权重。强制 Reward ≥ -1、
HPS/CLIP 各 ≥ BF16 均值的 80%、零失败、图片/评分精确重放、重复物化 packed 权重哈希一致、
GB10 12.1 和 SM120/121 原生 block-scaled E2M1×E2M1 profiler 证据。每批图片立即落盘。

将命令改为 `human-dev` 执行 30 条 BF16/RTN/静态 QUASAR-grid/QAD 四路盲测。将生成的
`healing/human_dev/blind/public/` 单独提供给评审者，例如在该目录启动静态 HTTP 服务。
不能公开其父目录：私钥映射、训练和评分报告会揭示候选身份。公开包含 prompt、随机化试次
和 A–D 图片；PNG 重新编码去掉 metadata。私钥权限为 0600。

每个试次填完整排名及明显崩坏的标签，形成恰好 30 行对象的 JSON 数组：

```json
[{"trial_id":"复制公开包中的试次 ID","ranking":["C","A","D","B"],"broken":[]}]
```

维护者实际评审后显式提交决定：

```sh
python scripts/run_mage_flow_healing.py approve-human --campaign /runs/exp008 \
  --split human_dev --ballots /data/human-dev-ballots.json \
  --maintainer MAINTAINER_NAME --decision approve
```

代码不会代替人生成批准。QAD 必须零明显崩坏且至少 24/30 优于静态 QUASAR-grid，人工批准
才会产生 ES 许可。人工门失败不能换 ballot 重试。本地 ledger 是可审计流程边界，不是抵御
拥有文件修改权限者的认证服务。

`checkpoint_val`、`human_dev`、`auto_test`、`human_test` 在执行回调前以独占文件方式消费。
崩溃或失败不会重新解封；不删除 claim 文件重试留出集。新增研究需要单独的冻结协议，不能
靠重新命名输出目录把同一 campaign 的结果变成多次候选选择。

## 已修复 master 与受限 ES

ES 永远由选定检查点恢复 latent master 和 AdamW 显著性。每次候选物化验证 master/layout
身份，拒绝可训练参数和 master 修改；clip 是唯一更新对象。初始 clip 由保存的显著性重建，
不是用均匀显著性重新跑静态 grid。

新 protocol JSON 必须且只能包含下列字段；sigma、alpha 没有旧实验默认值，必须作为本轮新
测量协议显式提供。哈希从选定 checkpoint 和初始 clips 得出（`content_hash` 公共工具）：

```json
{
  "selected_sha256": "选定 QAD checkpoint 的 sha256",
  "initial_clips_sha256": "content_hash(checkpoint['clips'])",
  "phase": "sigma",
  "sigma": 0.001,
  "alpha": 0.001,
  "seed": 9008
}
```

数值只是格式示例，不是冻结扩大协议。执行 `es-sigma`，在上述生成公共参数外增加
`--protocol /data/new-sigma.json --batch-offset 0 --batch-size 32`。通过后才能以
`phase=confirmation`、同一个新测量 sigma、显式 alpha 执行 `es-confirmation`，换未见的
`es_train` 批（例如 offset 32）。前者固定 N=8，后者固定 N=30；各运行一代，从已修复 genome
开始，不继承历史 center。两个阶段各只能执行一次，没有 generations 或扩大预算参数。

所有扰动候选和更新中心必须保持 QAD 的绝对质量底线、每个样本教师轨迹 MSE 不增加、两个
评分轴批均值不下降、图片落盘和重放一致。任何失败立即停止本阶段，不以综合 Reward 抵消
单轴退化。通过只代表这个受限阶段完成，不产生长期 ES 授权。

N=30 后执行 `final-gate`，固定候选只解封一次 60 条 `auto_test`。最终 paired Reward delta
以 ES 相对于已修复 QAD 的配对 Reward 差计算（两者都以 BF16 为 Reward 基线），必须均值 >0、
至少 45/60 不退化、两轴均值都不低于 QAD，同时仍满足 BF16 绝对底线、教师轨迹门、零失败、
确定性和原生 kernel 门。然后执行 `human-test` 和 `approve-human --split human_test`；最终
盲测要求零明显崩坏，最终可用性由维护者显式决定。只有这两道最终门均通过，ledger 才能写
`success=true`、`deployable=true`，命令本身不部署模型。

## 验证与当前边界

`tests/lab/test_mage_qad.py` 验证四步同输入、174 层训练范围和 W4A4、数据隔离、续训一致、
最佳检查点/early stop、显著性 packed 重放和新 ES 协议约束。
`tests/lab/test_mage_healing.py` 验证绝对轴门、原生/重放证据、一次性解封、盲测和人工放行。

当前交付是可运行实现与单元验证。新的真实数据来源、GB10 全层训练与图片评估产物、两次
维护者盲测放行尚未产生；不把单元测试当作 Mage 图像质量恢复证据。

2026-09-05 本地最终验证：全量 **416 passed, 18 skipped**；新增/修改文件 Ruff、Mage
QAD/healing 模块与 CLI 的 ty 类型检查通过。双轴代码评审：Standards 无硬规范问题、1 项
低优先级维护性建议；Spec 无阻断性发现。18 项跳过不作为 GB10 原生执行证据。
