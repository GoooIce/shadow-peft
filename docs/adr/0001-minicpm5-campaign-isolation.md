# ADR-0001:minicpm5 战役与 src 实验库保持隔离(有意设计)

日期:2026-09-25 · 状态:已接受

## 背景

2026-09-09 起,`scripts/minicpm5` 承载 QES 自改进战役:MiniCPM5-2B 先 NVFP4 W4A4、后 INT8 W8A8,直接 patch vLLM 常驻引擎的 kernel-format 张量;在 GB10 双机用专用 venv(`~/venvs/minicpm5-vllm`)执行,脚本独立同步到远端(与仓库主 `.venv` 互不触碰,见 bringup 票 2026-09-09)。战役带预注册冻结协议(`docs/experiments/minicpm5-qes-faithful-prereg-20260910.md`,存于 git 历史);W8A8 已于 2026-09-12 结案(双种子结论:种子方差主导),prime 线收尾中。

## 决策

minicpm5 与 `src/`(shadow_lab / shadow_peft 等)的隔离是**设计约束**,不是待修复的架构漂移:

1. 不把 `scripts/minicpm5` 的库提升进 src,也不让它跟随 src 的演化;
2. 它对 src 的全部真实代码依赖只有 5 个纯函数 —— `shadow_lab.quasar` 的 `e2m1_decode / e2m1_encode / unpack_e2m1`(make_candidate_payload / verify_payload / payload_error)与 `shadow_lab.nvfp4_runtime` 的 `census_linear_layers / census_manifest`(nvfp4_census);`confirm_gate.py:55` 的判定函数是对 `shadow_lab.es_inference.classify_outcome` 的**有意逐字移植**(冻结协议要求不依赖移动目标);
3. 后续架构评审不得以「平行宇宙 / 极少导入 src / 自建库」为由提议把它并入 src。

## 后果

- minicpm5 的复杂度不计入 shadow-peft lab 的账;lab 的结构性改进(mage_w4 家族那种「src 厚模块 + scripts 薄驱动」)不适用于已冻结的战役代码;
- 分离的具体形态见 ADR-0002(独立仓库 `minicpm5-qes`,2026-09-25 执行);
- src 侧 e2m1 / census / 判定语义若变更,minicpm5 不受影响(其拷贝按战役口径冻结)。
