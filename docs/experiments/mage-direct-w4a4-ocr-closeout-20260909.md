# Mage W4A4：人工转录、OCR 实测和迟到结算处置

2026-09-09；增量基线 `b6fbcc33f5814faf03a51e1302d53e0dc4812316`。此次继续已授权的 01–08，01–04 已完成，05 开始准备真实三轴入口；确认与终检仍未开启。

## 实际结果

用户两份转录原件均按字节归档。复核版 SHA `41848fd942be0a64482a27bbf1381049c32a0850aa9039301b03d3d686a7a87f` 保留全部 5 行；作者完整声明的 00/01/02/04 在识图前登记，03 不完整排除。软件不补写或修正文字。图 00、02 的畸形字形及 4 图样本量限制结果解释。

原生 q0 图片在 gooo 复用，PaddleOCR / EasyOCR 都成功识别 4/4 图。预登记平均编辑误差上界 0.25，实际为 0.160714 / 0.591270，整图清单全对 2/4 / 1/4。PaddleOCR v4 配置冻结；选择衡量对本次人工可读转录的吻合度，不是通用 OCR 准确率或模型质量提升。

两工具新增占用 39.938527 + 31.833163 秒；既有生成 282.277431 秒只计一次，比较合计 354.049121 秒。先导累计 1795.150378 秒（0.498653 设备机时），11 租约闭合，无新预算。真实误差匹配/冻结在 GB10 CPU 执行，Mac 只验证代码与归档、渲染报告。第一次收口因远端缺协调器 clock 镜像拒绝；补传原始文件后重放已有证据成功，没有重复 OCR。

- [HTML 结果](../mage-w4a4-ocr-comparison-20260909.html)
- [预登记](../../runs-local/mage-direct-w4a4-20260909/ocr-comparison-v1/registration.json)
- [校准报告](../../runs-local/mage-direct-w4a4-20260909/ocr-comparison-closeout-v1/calibration-report.json)
- [评分器锁](../../runs-local/mage-direct-w4a4-20260909/ocr-comparison-closeout-v1/scorer-lock.json)
- [预算归集](../../runs-local/mage-direct-w4a4-20260909/ocr-comparison-closeout-v1/accounting.json)

## 实现与证据边界

`human_transcription.py` / `mage_w4_ocr_calibrate.py` 严格导入完整人工库存并选择预登记子集。内容、SHA、字节大小来自同次读取；OCR 收口复核计划、人工原件、图像与生成身份、两份授权和所有逐图 child 回执、两秒时钟界限、保守按时停止与闭合账本。GB10 实际分支在 Linux ARM 执行，夹具不能冻结评分器。身份链仍依赖真实人工输入及可信 SSH 传输，JSON 自洽不是远程认证服务。

`runtime_incidents.py` 与 pilot 接入口消费追加式处置，复核原始按时停机、时钟、组消失和双机整改证据，解除旧临时探针的迟到结算故障阻塞。历史 84.802259 秒计费及 54.802259 秒预留超时完整保留；BudgetLedger 未修改。活动租约和未解决故障仍阻塞。预算耗尽保持正常 `budget-exhausted` 终态，不永久阻止另有预算/授权的正式阶段；正式字段缺失仍是证据不足。

## Standards

独立 subagent 审核新增 OCR 导入、收口 CLI 与测试：0 项明文标准硬违规；1 项非阻塞 P3 重复代码建议。人工收口与 pilot 的 RecognitionReceipt 解码形状重复，目前没有行为差异，暂保留，后续可集中公共 decoder。冻结的 `ocr.py` / `scorers.py` 本轮均未修改。

## Spec

独立 subagent 对 runtime 处置与 pilot 增量复核：0 项需修复问题。v2/v3 报告的所有 accounting 和 13 组状态完全一致，只解除已证实的迟到结算运行故障；处置本身不补齐三轴/统计/正式授权。根代理对 OCR 收口的两项身份/时钟问题已在真实收口前修复并通过反例测试。

## 验证

全仓 `pytest -q`：670 passed、18 skipped，97.29 秒；增量 Ruff 检查通过。新增 OCR 与已有 OCR/worker 回归合计 53 项通过，runtime/pilot 定点 29 项通过。HTML 经 ego-browser 检查五张原图、冻结结果表、原始文本、390px 手机布局和无页面横向溢出；复核页勾选状态行为已验证。

未获得三轴固定尺度、真实候选配对收益、13 组校准或正式统计前提；05–08 继续按依赖完成，不把 04 完成写成整项实验完成。09–10 不启动。
