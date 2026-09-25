# ADR-0002:minicpm5 战役分离为独立仓库

日期:2026-09-25 · 状态:已接受

## 背景

ADR-0001 确认 minicpm5 战役与 src 实验库的隔离是有意设计。遗留问题是形态:86 个脚本、约 2.26 万行战役代码仍住在 `scripts/minicpm5`,其复杂度与 shadow-peft 的持续演进互相拖累(评审、测试、同步范围都要绕开它)。

## 决策

将整条战役线分离为独立仓库 **`~/Code/minicpm5-qes`**(2026-09-25 执行):

- 历史用 `git subtree split -P scripts/minicpm5` 全量保留(20 个战役提交,含叠加当时未提交的 search_center / fitness_scoring 抽取状态);
- 对 `shadow_lab` 的全部代码依赖(QUASAR 重建、E2M1 编解码、census)逐字 vendor 进新仓 `lab_compat.py`,并与原件做过逐位差分验证 —— 新仓对 shadow-peft **零依赖**;
- 预注册、结案与研究文档(17 份)从本仓 git 历史导出随行;
- 本仓删除 `scripts/minicpm5` 与 `tests/lab/test_minicpm5_*.py`。

## 后果

- shadow-peft 卸载约 2.26 万行冻结战役代码;lab 的评审与测试范围不再包含 minicpm5;
- src 侧 `quasar` / `nvfp4_runtime` / `es_inference` 的后续变更不影响新仓(vendor 冻结);
- GB10 双机的同步与运行单元改为新仓(远端部署布局不变:脚本仍落在 `~/minicpm5-payload/scripts/minicpm5`,`run_engine.sh` 的 PYTHONPATH 已去掉 `$ROOT/src`);
- 若需回查战役历史:`git log --follow` 在本仓对该路径的历史在分离提交后终止,完整史在新仓。
