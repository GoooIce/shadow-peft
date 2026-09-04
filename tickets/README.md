# 本地票据

本目录是 ES×QUASAR / Mage-Flow-Turbo 工作的本地任务队列。票据只描述工作边界、依赖和验收标准；创建票据不代表已经开始执行。

## 状态

- `open`：Epic 等持续跟踪项。
- `ready-for-agent`：依赖齐备，可在评审后领取。
- `blocked`：至少一个 `depends_on` 尚未完成。
- `needs-info`：缺少会改变实现方向的信息。
- `in-progress`：已经明确领取并开始。
- `done`：全部验收标准已有证据。

每次开始工作前，先在本索引确认依赖，再把且只把被领取票据改为 `in-progress`。完成后在票据的“证据”段记录命令、产物和结果，再改为 `done`。

## 票据图

| ID | 类型 | 标题 | 状态 | 依赖 |
|---|---|---|---|---|
| [EPIC-001](EPIC-001-es-quasar-mage-flow.md) | Epic | Mage-Flow-Turbo NVFP4 ES×QUASAR | open | — |
| [RES-001](RES-001-annotation-product-research.md) | Research | ellf.ai / Prodigy 标注产品研究 | done | — |
| [RES-002](RES-002-gb10-native-nvfp4-research.md) | Research | GB10 原生 NVFP4 兼容性研究 | done | — |
| [ENG-001](ENG-001-modelscope-mage-baseline.md) | Engineering | ModelScope 资产与 BF16 基线 | done | — |
| [ENG-002](ENG-002-mmd-it-nvfp4-baselines.md) | Engineering | MMDiT NVFP4 RTN / QUASAR 基线 | done | RES-002, ENG-001 |
| [ENG-003](ENG-003-reward-evaluation-protocol.md) | Engineering | 自动奖励与固定评测协议 | done | ENG-001 |
| [ENG-004](ENG-004-es-quasar-core.md) | Engineering | Phase 1 ES×QUASAR 集成 | done | ENG-002, ENG-003 |
| [ENG-005](ENG-005-human-annotation-app.md) | Engineering | 人类友好的盲测标注应用 | in-progress | RES-001, ENG-003 |
| [EXP-005](EXP-005-es-hyperparameter-calibration.md) | Experiment | Phase 2 sigma/alpha 校准 | done（门禁失败） | ENG-003, ENG-004 |
| [EXP-006](EXP-006-es-cross-prompt-gradient-gate.md) | Experiment | Phase 2b 跨 prompt 梯度门禁 | done（门禁失败） | EXP-005 |
| [EXP-007](EXP-007-es-cross-prompt-gradient-gate-20x.md) | Experiment | Phase 2c 跨 prompt 梯度门禁 20× 扩展 | in-progress | EXP-006 |
| [EXP-001](EXP-001-gb10-main-study.md) | Experiment | GB10 主实验与人工验收 | blocked | ENG-005, EXP-005, EXP-006, EXP-007 |
| [EXP-002](EXP-002-qwen3-vl-nvfp4.md) | Experiment | Qwen3-VL NVFP4 扩展 | blocked | RES-002, EXP-001 |
| [EXP-003](EXP-003-vae-nvfp4.md) | Experiment | VAE NVFP4 可行性与扩展 | blocked | RES-002, EXP-001 |
| [EXP-004](EXP-004-ablations-and-report.md) | Experiment | 消融、泛化与报告 | blocked | EXP-001 |

RES-001、RES-002、ENG-001、ENG-002、ENG-003 与 ENG-004 已于 2026-09-04 完成。
Phase 2 与 Phase 2b 均已按预注册门禁结束且未冻结配置；EXP-006 的三批平均梯度通过首个
未见批次、但在第二个未见批次退化。用户已要求以精确 16,000 张预算领取 EXP-007 做高功效
复核，完成前 EXP-001 不得启动。ENG-005 已在研究结论通过人工评审后并行领取，仍等待真实
人员 pilot。
