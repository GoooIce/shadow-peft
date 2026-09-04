# Mage-Flow 500-prompt 评估池冻结记录（GB10，2026-09-04）

状态：v2 与 BF16 `es_train` 数值标定均已冻结。

## 输入与内容门控

- 候选共 751 条：GenEval 553 + Qwen-Image-Bench Text Rendering 198。
- goooinext 的 Mage 自带 Qwen3-VL gate 全量筛选 751/751，耗时 1706.28 秒，
  load 88.40 秒；747 通过、4 拒绝。
- candidate SHA-256：`83fee502...98a1`；accepted SHA-256：
  `b6c8af4a...365a`；audit SHA-256：`e72562a3...7db6`。
- 拒绝分布：violence 2、copyright 1、hate 1。完整逐条证据保存在
  `runs-local/mage-flow/prompt-filter-eval-candidates-751-20260904/`。

被拒绝的四条分别涉及 four knives、cake below a baseball bat、Monkey King 春节九宫格、
以及照片红线墙。后两条显示 gate 存在明显的语境误判可能；因此门控只用于模型可执行性
过滤，拒绝率必须公开，不能把它解释为数据本身不合规，也不能把拒绝样本生成白图后计入
质量均值。

## v1 拒收与修复

第一版 pool 虽满足 400/100 来源配额和五路 split 总数，但实现顺序是先对来源做一次全局
分层选择，再按 split 切片。这使尾部 `human_dev/human_test` 缺失
`single_object/counting`，违反逐 split 覆盖要求。v1 未进入生成或评分，已移动到
`runs-local/mage-flow/pool-500-v1-rejected-20260904/` 仅供失败审计。

修复后，每个来源的每个 split 从剩余候选中独立按 stratum 做确定性 SHA-256 分层选择，
并新增“每个 split 覆盖六类 GenEval”测试。

## 冻结 v2

- canonical pool：`assets/eval/mage-flow-500/pool.jsonl`
- 500 个唯一 sample ID、500 个唯一 seed；400 GenEval + 100 Text Rendering。
- split：320 `es_train`、60 `checkpoint_val`、60 `auto_test`、30 `human_dev`、
  30 `human_test`。
- pool SHA-256：`13620c7317447ff5e2e279fd20a69d20871e1a2d98f30f7419b8c63d0f707e29`。
- `es_train` SHA-256：`9afe837225f309912afb1f828f26ac349a7ba68482f647faae6f88f786853bf3`。
- `auto_test` SHA-256：`d1273fe63ce9eb8269492313142364edbb40d3cc689a15fadf36345e47e6d428`。
- 四 seed `held-out GenEval-48` 派生 192 行，SHA-256：
  `759076fb465864fc6a5c5d7f08d8f1637fbbbd9ad2f12bfab82d00048ba12216`。

最终 manifest 同时锁定官方 GenEval metadata、Qwen-Image-Bench JSONL、gate run、accepted、
候选文件和每个 split 的 SHA-256。池已同步到 gooo 与 goooinext；`human_test` 不供 ES、
checkpoint 选择或标注 UI 开发使用。

## BF16 奖励标定

- goooinext 生成 320/320：1051.78 秒，其中模型 load 95.02 秒；首图双次生成 PNG
  SHA-256 一致，进程峰值 RSS 约 31.15 GB。
- gooo 评分：HPS/OpenCLIP load 38.08 秒，320 图 21.24 秒；抽取 8 图重复评分，
  两轴最大绝对差均为 0。
- HPS-v2.1 mean/std：`0.2800148512 / 0.0381578632`。
- OpenCLIP H/14 mean/std：`0.3800367487 / 0.0586977619`。
- 320 个 paired baseline：
  `assets/eval/mage-flow-500/bf16-reward-calibration.json`，SHA-256
  `7f3d4a26fb2da0379177af13851f010dc5a350337b534fbf89fc85f463ee437b`。
