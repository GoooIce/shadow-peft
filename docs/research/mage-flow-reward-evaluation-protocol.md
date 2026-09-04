# Mage-Flow NVFP4 自动奖励与固定评测协议

状态：ENG-003 已完成；协议口径与 GB10 BF16 数值标定均已冻结。

## 一手来源审计结论

| 一手事实 | 对本项目的约束 |
|---|---|
| HPS-v2 官方实现明确说明，只有由同一 prompt 生成的图像才有意义直接比较；v2.0 与 v2.1 也不能直接比。[官方 README—Image Comparison](https://github.com/tgxs002/HPSv2/blob/866735ecaae999fa714bd9edfa05aa2672669ee3/README.md#image-comparison) | 候选、BF16、RTN、QUASAR 必须共享 prompt；本项目额外共享 seed 以做 paired 方差缩减，并全程固定 HPS-v2.1 checkpoint。 |
| GenEval 对每张图输出二值 `correct`；先在六个 task 内对图像求均值，再对六个 task 等权求 overall。[官方论文§4.1](https://arxiv.org/html/2310.11513#S4.SS1) [官方汇总代码](https://github.com/djghosh13/geneval/blob/af4902f24d3ca90ebbb446dd9891a59e0f82725f/evaluation/summary_scores.py) | 不能用全部图像的非分层均值伪装官方 overall；报告六项分数及其等权均值。 |
| 官方 benchmark 是 553 prompts、每 prompt 4 张图，覆盖六项 task。[官方论文§5.1](https://arxiv.org/html/2310.11513#S5.SS1) [官方生成脚本](https://github.com/djghosh13/geneval/blob/af4902f24d3ca90ebbb446dd9891a59e0f82725f/generation/diffusers_generate.py) | 本项目 48-prompt 留出子集只能称 `held-out GenEval-48`，不是官方 GenEval overall，不得与论文榜单横向比较。 |
| GenEval 是 object-focused 的细粒度组合能力测试；官方实现依赖 Mask2Former/mmdetection，并为错误图像输出原因。[官方仓库](https://github.com/djghosh13/geneval/tree/af4902f24d3ca90ebbb446dd9891a59e0f82725f) | GenEval 适合 checkpoint/留出验收与失败分析，不进 ES 的高频 episode 奖励。 |

因此，协议将两条通道严格分开：

- **Episode 优化奖励**：只有 HPS-v2.1 + OpenCLIP ViT-H/14 的 paired-BF16 z-score。
- **Held-out 验收指标**：`auto_test` 上的 `held-out GenEval-48` 与文字渲染 OCR/布局指标；两者都不反馈 ES，也不对 episode 清零。`checkpoint_val` 上的 GenEval 是开发诊断，不属于 held-out 验收。

OpenCLIP 是第二个连续图文对齐轴，不是第二个人类偏好模型。GenEval 论文也指出 CLIPScore 只提供整体对齐信号，不适合细粒度 instance-level 诊断；因此 HPS+CLIP 主奖励必须与 GenEval/OCR 留出指标并报，不能宣称它单独覆盖组合正确性。[官方 GenEval 论文摘要](https://arxiv.org/html/2310.11513#abstract)

## 500 prompt 池与隔离

| 来源 | ES train | checkpoint val | auto test | human dev | human test | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| GenEval | 256 | 48 | 48 | 24 | 24 | 400 |
| Qwen-Image-Bench text rendering | 64 | 12 | 12 | 6 | 6 | 100 |
| 合计 | 320 | 60 | 60 | 30 | 30 | 500 |

`es_train` 的 320 条形成 10 个不重复的 32-prompt episode batch，完成一轮后才重新
轮换。`checkpoint_val` 只用于开发期评估和 checkpoint 选择；`auto_test` 仅在搜索
配置冻结后开启；`human_dev` 只用于标注应用试标；`human_test` 一次性开启。

选择与分配以固定 seed、source ID 和 prompt 的 SHA-256 排序，不依赖输入行顺序。
每个最终 sample 有唯一 ID 和 seed；相同 prompt 不得跨 split，同一 prompt 的
多个派生 seed 始终属于同一 split。

这个 500-prompt 池是项目内部的固定实验池，不是 HPS 官方 benchmark，也不是完整
GenEval benchmark。其 `auto_test` 中的 48 条 GenEval 样本是与 ES 隔离的项目验收集，
对外必须报告 prompt 数、task 分层数和图像数。如额外运行官方 553×4 全集，
因其中 400 prompts 已被项目开发/优化使用，只能标注为**非留出的外部参考分**，
不得作为验收证据。

## 来源、资产与不可变身份

- HPS-v2.1 从 [ModelScope `AI-ModelScope/HPSv2`](https://modelscope.cn/models/AI-ModelScope/HPSv2)
  取 `HPS_v2.1_compressed.pt`；使用时与[官方 HPSv2 代码](https://github.com/tgxs002/HPSv2/tree/866735ecaae999fa714bd9edfa05aa2672669ee3)
  的固定 commit 配对。文件 SHA-256 见 [`assets/manifests/hpsv21-modelscope.sha256`](../../assets/manifests/hpsv21-modelscope.sha256)。
- CLIP 从 [ModelScope `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://modelscope.cn/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
  取 `open_clip_model.safetensors` 及配置/分词文件；项目本地审计确认未选入重复权重副本。
  SHA-256 见 [`assets/manifests/clip-h14-modelscope.sha256`](../../assets/manifests/clip-h14-modelscope.sha256)。
- 文字渲染 prompt 来自 [ModelScope `Qwen/Qwen-Image-Bench`](https://modelscope.cn/datasets/Qwen/Qwen-Image-Bench)。
  本地对所下载 JSONL 的可复现审计结果为 1,000 行，其中 198 行含 `Text Rendering`；
  这是项目对固定文件的统计，不是 ModelScope 页面声称。见
  [`assets/manifests/qwen-image-bench-prompts-modelscope.json`](../../assets/manifests/qwen-image-bench-prompts-modelscope.json)。
- GenEval structured metadata 以[官方仓库固定 commit 的 553 行 JSONL](https://github.com/djghosh13/geneval/blob/af4902f24d3ca90ebbb446dd9891a59e0f82725f/prompts/evaluation_metadata.jsonl)
  为规范。[ModelScope `JunweiXi/EvalGEN`](https://www.modelscope.cn/datasets/JunweiXi/EvalGEN)
  只作 ModelScope 来源对应核验；14.5 GB archive 的 HTTP Range 扫描与 slug 对应是项目本地审计结果，
  见 [`assets/manifests/geneval-prompts-modelscope.json`](../../assets/manifests/geneval-prompts-modelscope.json)。
- GB10 检测后端使用 ModelScope `facebook/mask2former-swin-small-coco-instance`
  的 Transformers checkpoint；SHA-256 与已知加载转换缺口见
  [`assets/manifests/mask2former-coco-modelscope.json`](../../assets/manifests/mask2former-coco-modelscope.json)。
  原版 GenEval 固定环境是 PyTorch 1.12/CUDA 11.3 + MMDetection 2.x，而本项目验收硬件
  是 SM121，因此该端口固定命名为 `GenEval-compatible Transformers/ModelScope port`，
  `leaderboard_comparable=false`。它保留判定规则，但检测 checkpoint/runtime 并非论文原版，
  不与官方榜单横比。

ModelScope 页面是来源引用；实验的不可变身份以本地选中文件 SHA-256、官方代码
commit 和最终 pool manifest 为准，不仅依赖可变的 `master`。

## 内容门控

先对全部 751 个候选 prompt 运行 Mage 自带 Qwen3-VL content gate，再从通过项构建
最终 500 池。被拒绝项由同 stratum 的已筛选 reserve 替换，不在冻结后删除样本。
内容拒绝率及类别单独报告；拒绝白图不进 HPS/CLIP 均值，也不能因过滤而跨 split
复用 prompt。

## Episode 奖励：仅 HPS + CLIP

对 sample `i` 和连续评分轴 `m`：

```text
z(i,m) = (score_candidate(i,m) - score_bf16(i,m)) / std_bf16_es_train(m)
R(i)   = 0.5 * z(i,HPS-v2.1) + 0.5 * z(i,CLIP-H/14)
R(ep)  = mean_i R(i)
```

BF16 mean/std、逐 sample BF16 分数、scorer checkpoint hash 和 prompt-pool hash 一起冻结。
ES 不能使用当前种群均值重新标定，否则奖励尺度会逐代漂移。HPS/CLIP 的相关性在
BF16、RTN 和 QUASAR 开发图上报告；预注册等权主奖励不用留出集反调权重。

评分器只接收图片 bytes 与冻结样本元数据，接口不含候选名、模型名或实验臂。
缺失、NaN、Inf 或评分器异常使 episode 显式失败；不静默补 0，也不会因 GenEval/OCR
未运行而改变主奖励。

2026-09-04 在完整 `es_train` 320 图上的冻结常量为：HPS-v2.1 mean/std
`0.2800148512 / 0.0381578632`，OpenCLIP H/14 mean/std
`0.3800367487 / 0.0586977619`。320 个逐 sample baseline 位于
`assets/eval/mage-flow-500/bf16-reward-calibration.json`，SHA-256 为
`7f3d4a26fb2da0379177af13851f010dc5a350337b534fbf89fc85f463ee437b`。

## Held-out 验收：GenEval-48 + OCR

- `checkpoint_val` 可做单图 GenEval 诊断，但该数值是项目快速指标，不是官方
  四图协议分数。
- 最终 `auto_test` 的 48 条 GenEval prompt 每条生成 4 个固定派生 seed，报告六项
  task 图像准确率及其等权均值，名称固定为 `held-out GenEval-48 overall`。
- GenEval 官方主结果在不同随机 seed 间可变动约 0.01–0.02，所以必须保存并
  报告 seed，不得使用 best-of-four。[官方论文 Table 2](https://arxiv.org/html/2310.11513#S5.SS2)
- 12 条文字渲染 `auto_test` prompt 的 OCR/text-layout 是项目自定义指标，不是
  Qwen-Image-Bench 官方分数。只有在 `human_dev` 上验证它与人工文字正确性方向一致
  后才进验收报告；未标定前只作诊断，不得硬清零。
- GenEval/object detector 与 OCR 在独立离线环境运行，不得改变 Mage 生成过程。

## 预注册统计与成功阈值

- 所有候选与 BF16 使用相同 prompt/seed，主报告是 `candidate - BF16` 的 paired mean。
- 连续 HPS/CLIP 一图设计按 sample 做 paired bootstrap。GenEval 的 4 个 seed 是同一
  prompt 的簇：按六个 task 分层重采样 prompt，并将每个 prompt 的四张图整簇携带；
  不得把 192 张图当成相互独立的样本。
- 置信区间固定 seed `20260904`、10,000 次 paired bootstrap、双侧 95% CI。
- 最终非劣效验收：HPS-v2.1 差值 95% CI 下界 不低于 `-0.01`；
  `held-out GenEval-48 overall` 差值 95% CI 下界不低于 `-0.02`。
- 上述阈值是项目的预注册决策，不是 HPS/GenEval 官方推荐阈值。子集只有 48 prompts，
  若 CI 过宽则结论为“证据不足”，不在看到 test 结果后改 margin 或扩容。
- ES 优于 grid/random 的比较使用同样的 paired/clustered bootstrap，并同时报告均值、CI、
  prompt/图像数和失败/拒绝率。自动 test 与 human test 各开启一次。

## 代码接缝

- `shadow_lab.image_eval`：冻结样本、无泄漏验证、paired-BF16 reward、异常值处理与
  deterministic paired bootstrap。
- `shadow_lab.image_reward_models`：本地 checkpoint 的 HPS-v2.1/OpenCLIP scorer；不触发
  Hugging Face 下载。
- `shadow_lab.geneval`：独立于检测器的 GenEval 判定、六任务等权汇总、四 seed 展开和
  prompt-cluster paired bootstrap。
- `shadow_lab.geneval_models`：GB10 上只读本地 ModelScope Mask2Former/CLIP 的兼容后端；
  保留 checkpoint 转换缺口清单并拒绝未知缺口。
- `scripts/build_mage_flow_eval_pool.py`：规范化候选、合并内容门控、输出五路 JSONL 和
  SHA-256 manifest。
- `scripts/score_mage_flow_rewards.py`：匿名批评分、重复确定性审计和 BF16 calibration。
- `scripts/expand_mage_geneval_seeds.py` 与 `scripts/score_mage_flow_geneval.py`：构造四 seed
  prompt 簇并运行明确不可与榜单横比的 held-out 诊断。

## 一手来源索引

- [HPSv2 官方仓库（固定 commit）](https://github.com/tgxs002/HPSv2/tree/866735ecaae999fa714bd9edfa05aa2672669ee3)
- [GenEval 官方论文](https://arxiv.org/html/2310.11513)
- [GenEval 官方仓库（固定 commit）](https://github.com/djghosh13/geneval/tree/af4902f24d3ca90ebbb446dd9891a59e0f82725f)
- [ModelScope HPSv2](https://modelscope.cn/models/AI-ModelScope/HPSv2)
- [ModelScope CLIP ViT-H/14](https://modelscope.cn/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
- [ModelScope Qwen-Image-Bench](https://modelscope.cn/datasets/Qwen/Qwen-Image-Bench)
- [ModelScope EvalGEN](https://www.modelscope.cn/datasets/JunweiXi/EvalGEN)
